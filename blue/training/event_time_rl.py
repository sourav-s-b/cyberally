"""Strict-guard Qwen Red PPO: decision-time MC versus bootstrapped GAE.

The actor and critic use the tested GPT-Blue v2 geometry and Blue-visible
features, imported at pinned source revision f32ba72. All gradients arise
from actual free investigation choices. Forced/busy/urgent/verify ticks still
contribute native team rewards to each choice's elapsed-time return.

This runner changes neither simulator rewards nor Red implementation. For a
matched experiment use identical args except --gae-lambda 1.0 vs 0.9 and
--out. Qwen's exact installed digest is checked for every episode.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from blue.harness.learning import update
from blue.harness.policy import StrictHarnessPolicy
from blue.harness.redesign import FEATURE_NAMES, VERSION, CentralValue, RedesignHook, SetActor
from blue.harness.scoring import MLScorer, sha256
from blue.training import mappo_guide as mg
from blue.training.harness_redesign import Recorder, ExecutionAudit


RED_SOURCE = ("red/llm_agent.py", "red/planned_agent.py", "red/blue_env.py")
FROZEN_QWEN = {
    "model": "qwen2.5:3b",
    "digest": "357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b",
    "ollama_version": "0.40.0",
}
BLUE_SOURCE = (
    "blue/training/event_time_rl.py", "blue/training/harness_redesign.py",
    "blue/training/mappo_guide.py", "blue/harness/features.py",
    "blue/harness/scoring.py", "blue/harness/policy.py",
    "blue/harness/redesign.py", "blue/harness/numerics.py",
    "blue/harness/learning.py", "blue/harness/diagnostics.py",
    "blue/core/wrapper.py", "blue/policies/ordered.py",
)


class FreeChoiceRecorder(Recorder):
    """Only two-or-more-candidate sweeps are learned decision rows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.singleton_choices = 0

    def __call__(self, env, agent, cands, scored):
        if len(cands) == 1:
            self.singleton_choices += 1
            return cands[0]
        return super().__call__(env, agent, cands, scored)


class FreeChoiceAudit(ExecutionAudit):
    """Match training's learned-decision count at evaluation."""

    def __call__(self, env, agent, cands, scored):
        if len(cands) == 1:
            return cands[0]
        return super().__call__(env, agent, cands, scored)


def source_hashes():
    files = BLUE_SOURCE + RED_SOURCE
    return {name: sha256(name) for name in files}


def runtime():
    return {"python": platform.python_version(),
            "torch_num_threads": torch.get_num_threads(),
            **{name: version(name) for name in
               ("numpy", "scipy", "torch", "scikit-learn", "gym", "gymnasium")}}


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   text=True).strip()


def llm_config(cfg):
    return {"model": cfg["model"], "endpoint": cfg["endpoint"],
            "timeout_seconds": 60.0, "max_requests": cfg["max_requests"]}


def planner_config(cfg):
    return {"max_plans_per_agent": cfg["plans_per_agent"],
            "interval": cfg["plan_interval"]}


def make_env_factory(cfg, expected_model=None):
    if cfg["red_agent"] not in ("llm", "finite"):
        raise ValueError("use llm or finite Red")

    def factory(seed, steps):
        kw = {"seed": seed, "steps": steps,
              "temporal_features": ("ages", "belief"),
              "include_root_session": True}
        if cfg["red_agent"] == "llm":
            from red.blue_env import CC4LLMEnv
            env = CC4LLMEnv(llm_config=llm_config(cfg),
                            planner_config=planner_config(cfg), **kw)
            if expected_model is not None and env.llm_model != expected_model:
                raise RuntimeError("Qwen model digest/version drift")
            return env
        from blue.core.wrapper import CC4MARLEnv
        return CC4MARLEnv(red_agent="finite", **kw)

    return factory


def check_seeds(seeds, other=()):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    if any(7809 <= s <= 8200 for s in seeds):
        raise ValueError("reserved final seeds remain unopened")
    if set(seeds) & set(other):
        raise ValueError("train/eval seed overlap")


def model_info(cfg):
    if cfg["red_agent"] == "finite":
        return None
    # The local client checks the exact installed tag and returns its digest
    # and Ollama version without spending a plan request.
    from red.llm_agent import LLMConfig, OllamaClient
    actual = OllamaClient(LLMConfig(**llm_config(cfg))).check_model()
    if actual != FROZEN_QWEN:
        raise RuntimeError("Qwen tag/digest/Ollama version differs from freeze")
    return actual


def config(args):
    skip = {"mode", "out", "resume", "stop_after", "model_dir", "eval_seeds",
            "policy_replicas", "checkpoint"}
    return {k: v for k, v in vars(args).items() if k not in skip}


def train(args):
    torch.set_num_threads(args.threads)
    cfg = config(args)
    check_seeds(args.train_seeds)
    if args.iters < 1 or args.eps_per_iter < 1 or args.steps < 1:
        raise ValueError("invalid training budget")
    if args.gae_lambda not in (0.9, 1.0):
        raise ValueError("frozen arms require lambda=0.9 or 1.0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    scorer = MLScorer(args.scorer)
    actor = SetActor(args.hidden)
    critic = CentralValue(args.hidden, args.value_scale)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    exact_model = model_info(cfg)
    env_factory = make_env_factory(cfg, exact_model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION, "feature_names": list(FEATURE_NAMES),
        "config": cfg, "scorer_sha256": scorer.sha256,
        "model_info": exact_model, "source_files": source_hashes(),
        "source_commit": git_commit(), "runtime": runtime(),
        "dependency_commit_gpt_blue": "f32ba7284d7000127bacb6f09fab2dcf980d7048",
        "objective": "native team reward; strict coverage guard; sampled PPO",
        "status": "started", "history": [],
    }
    history = []
    if (out / "manifest.json").exists():
        if not args.resume:
            raise FileExistsError("new run directory required")
        old = json.loads((out / "manifest.json").read_text())
        for key in ("version", "feature_names", "config", "scorer_sha256",
                    "model_info", "source_files", "source_commit", "runtime"):
            if old[key] != manifest[key]:
                raise ValueError("resume drift: " + key)
        ck = torch.load(out / "checkpoint.pt", map_location="cpu",
                        weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        actor_opt.load_state_dict(ck["actor_optimizer"])
        critic_opt.load_state_dict(ck["critic_optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        history = ck["history"]
    elif args.resume:
        raise FileNotFoundError("no resumable run")
    manifest["history"] = history
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for iteration in range(len(history), args.iters):
        rec = FreeChoiceRecorder(actor, critic, scorer, args, iteration)
        policy = StrictHarnessPolicy(rec, record_requests=True)
        episodes = []
        for e in range(args.eps_per_iter):
            seed = args.train_seeds[(iteration * args.eps_per_iter + e)
                                    % len(args.train_seeds)]
            rec.policy_seed = args.seed * 100003 + iteration * 1009 + e + 1
            singleton_before = rec.singleton_choices
            result = mg.run_team_episode(policy, seed, args.steps, hook=rec,
                                         recorder=rec, env_factory=env_factory)
            episodes.append({**result,
                             "singleton_forced": (rec.singleton_choices
                                                  - singleton_before),
                             "requests": policy.request_digest.result()})
            print(f"iteration={iteration+1} episode={e+1} seed={seed} "
                  f"return={result['return']:.1f} "
                  f"learned_decisions={result['n_actor']}", flush=True)
        if not rec.rows or not all("duration" in r for r in rec.rows):
            raise RuntimeError("no valid event-time decision rows")
        stats = update(actor, critic, actor_opt, critic_opt, rec.rows,
                       algorithm="mappo", bonus=args.bonus,
                       temperature=args.temp, explore=args.explore,
                       epochs=args.epochs, minibatch=args.minibatch,
                       clip=args.clip, entropy_coef=args.entropy_coef,
                       target_kl=args.target_kl,
                       max_grad_norm=args.max_grad_norm,
                       critic_epochs=args.critic_epochs)
        history.append({"iteration": iteration + 1, "episodes": episodes,
                        "decision_rows": len(rec.rows),
                        "singleton_forced_choices": rec.singleton_choices,
                        "sampled_override_rate": (rec.sampled_disagree
                                                  / max(rec.total, 1)),
                        **stats})
        # Preserve every candidate for the predeclared dev-checkpoint rule.
        checkpoints = out / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        actor_rel = f"checkpoints/actor_{iteration + 1:03d}.th"
        critic_rel = f"checkpoints/critic_{iteration + 1:03d}.th"
        torch.save(actor.state_dict(), out / actor_rel)
        torch.save(critic.state_dict(), out / critic_rel)
        history[-1].update(actor_path=actor_rel,
                           actor_sha256=sha256(out / actor_rel),
                           critic_path=critic_rel,
                           critic_sha256=sha256(out / critic_rel))
        ck = {"actor": actor.state_dict(), "critic": critic.state_dict(),
              "actor_optimizer": actor_opt.state_dict(),
              "critic_optimizer": critic_opt.state_dict(),
              "history": history, "torch_rng": torch.get_rng_state(),
              "numpy_rng": np.random.get_state(),
              "python_rng": random.getstate()}
        torch.save(ck, out / "checkpoint.tmp")
        os.replace(out / "checkpoint.tmp", out / "checkpoint.pt")
        torch.save(actor.state_dict(), out / "actor.th")
        torch.save(critic.state_dict(), out / "critic.th")
        manifest.update(history=history,
                        status="complete" if len(history) == args.iters else "paused",
                        actor_sha256=sha256(out / "actor.th"),
                        critic_sha256=sha256(out / "critic.th"))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"iteration": iteration + 1,
                          "return_mean": float(np.mean(
                              [r["return"] for r in episodes])), **stats}),
              flush=True)
        if args.stop_after and len(history) >= args.stop_after:
            break


def load(model_dir, scorer_path, policy_seed, execution="sample", checkpoint=0):
    d = Path(model_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    cfg = manifest["config"]
    scorer = MLScorer(scorer_path)
    if checkpoint:
        chosen = [row for row in manifest["history"]
                  if row["iteration"] == checkpoint]
        if len(chosen) != 1:
            raise ValueError("checkpoint iteration not found")
        actor_path = d / chosen[0]["actor_path"]
        actor_sha = chosen[0]["actor_sha256"]
    else:
        actor_path = d / "actor.th"
        actor_sha = manifest.get("actor_sha256")
    if (manifest["version"] != VERSION or
            manifest["feature_names"] != list(FEATURE_NAMES) or
            manifest["status"] != "complete" or
            actor_sha != sha256(actor_path) or
            manifest["scorer_sha256"] != scorer.sha256 or
            manifest["source_files"] != source_hashes() or
            manifest["runtime"] != runtime() or
            manifest["model_info"] != model_info(cfg)):
        raise ValueError("checkpoint/source/runtime/model contract mismatch")
    actor = SetActor(cfg["hidden"])
    actor.load_state_dict(torch.load(actor_path, map_location="cpu",
                                     weights_only=True))
    actor.eval()
    hook = RedesignHook(actor, scorer, cfg["bonus"], cfg["ml_inputs"],
                        cfg["temp"], cfg["explore"], policy_seed, execution)
    return StrictHarnessPolicy(hook, record_requests=True), manifest


def evaluate(args):
    torch.set_num_threads(args.threads)
    if not args.model_dir:
        raise ValueError("--model-dir required")
    check_seeds(args.eval_seeds)
    if args.policy_replicas < 1:
        raise ValueError("empty policy replica list")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise FileExistsError("new evaluation directory required")
    results = {"claim": "development evaluation; no held-out superiority claim",
               "model_dir": str(Path(args.model_dir).resolve()),
               "seeds": args.eval_seeds, "policy_replicas": args.policy_replicas,
               "cells": {}, "untrained_mixture": {}, "strict_lancer": {}}
    for seed in args.eval_seeds:
        policy, manifest = load(args.model_dir, args.scorer, seed * 997,
                                checkpoint=args.checkpoint)
        cfg = manifest["config"]
        check_seeds([seed], cfg["train_seeds"])
        if args.steps != cfg["steps"]:
            raise ValueError("horizon drift")
        env_factory = make_env_factory(cfg, manifest["model_info"])
        from blue.policies.ordered import LancerValues, OrderedPolicy
        strict = OrderedPolicy(scorer=LancerValues(fruitless_decay=.5),
                               guard=True)
        results["strict_lancer"][str(seed)] = mg.run_team_episode(
            strict, seed, args.steps, env_factory=env_factory)
        rows = []
        untrained_rows = []
        for replica in range(args.policy_replicas):
            policy_seed = seed * 997 + replica
            # SetActor's head is zero at initialization, so this is the
            # same teacher-mixture sampling path without a trained residual.
            zero_actor = SetActor(cfg["hidden"])
            zero_hook = RedesignHook(
                zero_actor, MLScorer(args.scorer), cfg["bonus"],
                cfg["ml_inputs"], cfg["temp"], cfg["explore"], policy_seed)
            zero_audit = FreeChoiceAudit(zero_hook)
            zero_policy = StrictHarnessPolicy(zero_audit,
                                              record_requests=True)
            zero_result = mg.run_team_episode(
                zero_policy, seed, args.steps, hook=zero_audit,
                recorder=zero_audit, env_factory=env_factory)
            untrained_rows.append({**zero_result,
                                   "policy_seed": policy_seed,
                                   "requests": zero_policy.request_digest.result()})
            policy, _ = load(args.model_dir, args.scorer,
                             policy_seed,
                             checkpoint=args.checkpoint)
            audit = FreeChoiceAudit(policy.hook)
            policy.hook = audit
            cell = mg.run_team_episode(policy, seed, args.steps, hook=audit,
                                       recorder=audit,
                                       env_factory=env_factory)
            rows.append({**cell, "policy_seed": policy_seed,
                         "requests": policy.request_digest.result()})
        results["cells"][str(seed)] = rows
        results["untrained_mixture"][str(seed)] = untrained_rows
        results["model_manifest"] = manifest
        (out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"seed": seed,
                          "strict_lancer": results["strict_lancer"][str(seed)]["return"],
                          "untrained": [r["return"] for r in untrained_rows],
                          "rl": [r["return"] for r in rows]}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "eval"))
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir")
    p.add_argument("--checkpoint", type=int, default=0,
                   help="1-based saved iteration; 0 loads the final actor")
    p.add_argument("--scorer", default="blue/results/event_time_rl/scorer.pkl")
    p.add_argument("--red-agent", choices=("llm", "finite"), default="llm")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434")
    p.add_argument("--max-requests", type=int, default=36)
    p.add_argument("--plans-per-agent", type=int, default=2)
    p.add_argument("--plan-interval", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-seeds", type=int, nargs="+",
                   default=[7706, 7707, 7708, 7709])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[7641, 7642])
    for name, default in (("iters", 2), ("eps-per-iter", 2), ("steps", 400),
                          ("hidden", 64), ("epochs", 4),
                          ("critic-epochs", 4), ("minibatch", 256),
                          ("policy-replicas", 2), ("stop-after", 0),
                          ("threads", 1)):
        p.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", 3e-4), ("critic-lr", 3e-4),
                          ("bonus", .25), ("temp", .5), ("gamma", .99),
                          ("gae-lambda", 1.), ("explore", .2),
                          ("clip", .2), ("entropy-coef", 0.),
                          ("target-kl", .02), ("max-grad-norm", 1.),
                          ("value-scale", 20.)):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--ml-inputs", choices=("zero", "risk", "both"),
                   default="risk")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    train(parsed) if parsed.mode == "train" else evaluate(parsed)
