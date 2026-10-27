"""Restartable RL comparisons for SetActor, candidate attention, and routing.

Runs use the strict Lancer guard, native reward, sampled PPO, and
episode-balanced Discovery/Finite Red. This does not touch the frozen Qwen
event-time runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from blue.harness.attention_actor import CandidateAttentionActor
from blue.harness.learning import update
from blue.harness.policy import StrictHarnessPolicy
from blue.harness.routed_hybrid_actor import SituationRouterActor
from blue.harness.redesign import (CentralValue, FEATURE_NAMES, RedesignHook,
                                   SetActor)
from blue.harness.scoring import MLScorer, sha256
from blue.policies.ordered import LancerValues, OrderedPolicy
from blue.training import event_time_rl as et
from blue.training import mappo_guide as mg


def actor_for(arch, hidden):
    if arch == "set":
        return SetActor(hidden)
    if arch == "attention":
        return CandidateAttentionActor(42, 3)
    if arch == "hybrid":
        return SituationRouterActor(set_hidden=hidden, attention_hidden=42,
                                     attention_heads=3, router_hidden=32)
    if arch == "fixed":
        return SituationRouterActor(set_hidden=hidden, attention_hidden=42,
                                     attention_heads=3, router_hidden=32,
                                     fixed_gate=0.5)
    raise ValueError("unknown architecture")


def runtime(device="cpu"):
    return {"python": platform.python_version(),
            "torch_num_threads": torch.get_num_threads(),
            "device": device,
            "cuda_device_name": (torch.cuda.get_device_name(0)
                                 if device == "cuda" and torch.cuda.is_available()
                                 else None),
            **{name: version(name) for name in
               ("numpy", "scipy", "torch", "scikit-learn", "gym",
                "gymnasium")}}


def source_files():
    paths = (
        "blue/training/architecture_compare.py",
        "blue/training/event_time_rl.py",
        "blue/training/harness_redesign.py",
        "blue/training/mappo_guide.py",
        "blue/harness/attention_actor.py",
        "blue/harness/routed_hybrid_actor.py",
        "blue/harness/redesign.py",
        "blue/harness/learning.py",
        "blue/harness/features.py",
        "blue/harness/policy.py",
        "blue/harness/scoring.py",
        "blue/harness/numerics.py",
        "blue/core/wrapper.py",
        "blue/policies/ordered.py",
    )
    return {p: sha256(p) for p in paths}


def env_factory(red):
    if red not in ("discovery", "finite"):
        raise ValueError("unknown scripted Red")

    def build(seed, steps):
        from blue.core.wrapper import CC4MARLEnv
        return CC4MARLEnv(seed=seed, steps=steps,
                          temporal_features=("ages", "belief"),
                          include_root_session=True, red_agent=red)
    return build


def check_seeds(train, evaluate):
    all_seeds = list(train) + list(evaluate)
    if not train or not evaluate or len(set(all_seeds)) != len(all_seeds):
        raise ValueError("train/evaluation seeds must be nonempty and disjoint")
    if any(7809 <= s <= 8200 for s in all_seeds):
        raise ValueError("reserved final-evaluation seed block")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "eval"))
    p.add_argument("--arch", choices=("set", "attention", "hybrid", "fixed"),
                   required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir")
    p.add_argument("--scorer", default="blue/results/event_time_rl/scorer.pkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-seeds", type=int, nargs="+",
                   default=[7600, 7601, 7602, 7603])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[7604, 7605])
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--eps-per-iter", type=int, default=2)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--critic-epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--critic-lr", type=float, default=0.0003)
    p.add_argument("--bonus", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    # Fixed at Monte Carlo credit to isolate policy-update changes. The later
    # Kaggle Qwen run is a separate Red regime and has no held-out comparison.
    p.add_argument("--gae-lambda", type=float, default=1.0)
    p.add_argument("--explore", type=float, default=0.5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--reference-kl-coef", type=float, default=0.0,
                   help="persistent KL penalty to the frozen zero-residual "
                        "Lancer-mixture policy (policy ablation)")
    p.add_argument("--experiment-tag", default=None,
                   help="manifest experiment name for a preregistered screen")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--value-scale", type=float, default=20.0)
    p.add_argument("--ml-inputs", choices=("zero", "risk", "both"),
                   default="risk")
    p.add_argument("--resume", action="store_true")
    return p


def train(args):
    if args.iters < 1 or args.eps_per_iter < 2 or args.eps_per_iter % 2:
        raise ValueError("use an even episode count for balanced Red regimes")
    if args.gae_lambda != 1.0:
        raise ValueError("this pilot is frozen to Monte Carlo credit (lambda=1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch runtime")
    device = torch.device(args.device)
    check_seeds(args.train_seeds, args.eval_seeds)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    scorer = MLScorer(args.scorer)
    actor = actor_for(args.arch, args.hidden).to(device)
    critic = CentralValue(args.hidden, args.value_scale).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(args).items()
           if k not in ("mode", "out", "model_dir", "resume")}
    cfg["red_schedule"] = "alternating_discovery_finite_each_iteration"
    cfg["actor_parameters"] = sum(p.numel() for p in actor.parameters())
    manifest = {
        "experiment": (args.experiment_tag or
                       ("blue-situation-routed-hybrid-long-v1"
                        if args.arch == "hybrid" else
                        "blue-set-vs-candidate-attention-pilot-v1")),
        "status": "started", "config": cfg,
        "feature_names": list(FEATURE_NAMES),
        "scorer_sha256": scorer.sha256,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_files": source_files(), "runtime": runtime(args.device),
        "claim": ("two-training-seed policy screen; descriptive, no superiority "
                  "inference" if args.experiment_tag else
                  ("two-training-seed hybrid study; descriptive, no superiority "
                   "inference" if args.arch == "hybrid" else
                   "two-seed pilot only; no superiority inference")),
        "history": [],
    }
    history = []
    mp = out / "manifest.json"
    if mp.exists():
        if not args.resume:
            raise FileExistsError("run directory exists; pass --resume")
        old = json.loads(mp.read_text())
        for k in ("experiment", "config", "feature_names", "scorer_sha256",
                  "source_files", "runtime"):
            if old.get(k) != manifest[k]:
                raise ValueError("resume contract drift: " + k)
        manifest["source_commit"] = old["source_commit"]
        ck = torch.load(out / "checkpoint.pt", map_location="cpu",
                        weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        actor_opt.load_state_dict(ck["actor_optimizer"])
        critic_opt.load_state_dict(ck["critic_optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda" and "cuda_rng" in ck:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        history = ck["history"]
    elif args.resume:
        raise FileNotFoundError("no checkpoint to resume")

    manifest["history"] = history
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    for it in range(len(history), args.iters):
        rec = et.FreeChoiceRecorder(actor, critic, scorer, args, it)
        policy = StrictHarnessPolicy(rec, record_requests=True)
        episodes = []
        for e in range(args.eps_per_iter):
            global_ep = it * args.eps_per_iter + e
            red = "discovery" if global_ep % 2 == 0 else "finite"
            seed = args.train_seeds[global_ep % len(args.train_seeds)]
            rec.policy_seed = args.seed * 100003 + it * 1009 + e + 1
            singleton_before = rec.singleton_choices
            r = mg.run_team_episode(
                policy, seed, args.steps, hook=rec, recorder=rec,
                env_factory=env_factory(red))
            episodes.append({**r, "red_agent": red, "seed": seed,
                             "singleton_forced":
                             rec.singleton_choices - singleton_before,
                             "requests": policy.request_digest.result()})
            print(f"{args.arch} rng={args.seed} iter={it+1} "
                  f"episode={e+1} red={red} seed={seed} "
                  f"return={r['return']:.1f} decisions={r['n_actor']}",
                  flush=True)
        if not rec.rows or not all("duration" in row for row in rec.rows):
            raise RuntimeError("empty or malformed decision-time rollout")
        stats = update(
            actor, critic, actor_opt, critic_opt, rec.rows,
            algorithm="mappo", bonus=args.bonus, temperature=args.temp,
            explore=args.explore, epochs=args.epochs,
            minibatch=args.minibatch, clip=args.clip,
            entropy_coef=args.entropy_coef, target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm,
            critic_epochs=args.critic_epochs,
            reference_kl_coef=args.reference_kl_coef)
        history.append({"iteration": it + 1, "episodes": episodes,
                        "decision_rows": len(rec.rows), **stats})
        ck = {"actor": actor.state_dict(), "critic": critic.state_dict(),
              "actor_optimizer": actor_opt.state_dict(),
              "critic_optimizer": critic_opt.state_dict(),
              "history": history, "torch_rng": torch.get_rng_state(),
              "cuda_rng": (torch.cuda.get_rng_state_all()
                           if device.type == "cuda" else None),
              "numpy_rng": np.random.get_state(),
              "python_rng": random.getstate()}
        torch.save(ck, out / "checkpoint.tmp")
        os.replace(out / "checkpoint.tmp", out / "checkpoint.pt")
        torch.save(actor.state_dict(), out / "actor.th")
        torch.save(critic.state_dict(), out / "critic.th")
        manifest.update(
            history=history,
            status="complete" if len(history) == args.iters else "paused",
            actor_sha256=sha256(out / "actor.th"),
            critic_sha256=sha256(out / "critic.th"))
        mp.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"iteration": it + 1,
                          "return_mean": float(np.mean(
                              [x["return"] for x in episodes])), **stats}),
              flush=True)


def evaluate(args):
    if not args.model_dir:
        raise ValueError("--model-dir required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch runtime")
    device = torch.device(args.device)
    check_seeds(args.train_seeds, args.eval_seeds)
    torch.set_num_threads(args.threads)
    d = Path(args.model_dir)
    m = json.loads((d / "manifest.json").read_text())
    if m.get("status") != "complete":
        raise ValueError("cannot evaluate an incomplete pilot checkpoint")
    cfg = m["config"]
    if (cfg["arch"] != args.arch or cfg.get("device", "cpu") != args.device
            or m["scorer_sha256"] != sha256(args.scorer)):
        raise ValueError("architecture/scorer mismatch")
    if m["actor_sha256"] != sha256(d / "actor.th"):
        raise ValueError("actor checkpoint hash mismatch")
    learned = actor_for(args.arch, cfg["hidden"]).to(device)
    learned.load_state_dict(torch.load(d / "actor.th", map_location=device,
                                      weights_only=True))
    learned.eval()
    report = {"claim": "pilot descriptive only",
              "train_manifest": m, "eval_seeds": args.eval_seeds,
              "results": {}}
    if args.arch in ("hybrid", "fixed"):
        report["router_gate"] = {}
    for red in ("discovery", "finite"):
        report["results"][red] = {"strict_lancer": {}, "untrained": {},
                                  "learned": {}}
        if args.arch in ("hybrid", "fixed"):
            report["router_gate"][red] = {}
        for seed in args.eval_seeds:
            strict = OrderedPolicy(scorer=LancerValues(fruitless_decay=.5),
                                   guard=True)
            base = mg.run_team_episode(strict, seed, cfg["steps"],
                                       env_factory=env_factory(red))
            report["results"][red]["strict_lancer"][str(seed)] = base
            policy_seed = seed * 997 + int(cfg["seed"])
            zero = actor_for(args.arch, cfg["hidden"]).to(device)
            zero_hook = RedesignHook(
                zero, MLScorer(args.scorer), cfg["bonus"], cfg["ml_inputs"],
                cfg["temp"], cfg["explore"], policy_seed, "sample")
            zero_audit = et.FreeChoiceAudit(zero_hook)
            zero_policy = StrictHarnessPolicy(zero_audit,
                                              record_requests=True)
            zero_result = mg.run_team_episode(
                zero_policy, seed, cfg["steps"], hook=zero_audit,
                recorder=zero_audit, env_factory=env_factory(red))
            report["results"][red]["untrained"][str(seed)] = zero_result
            hook = RedesignHook(
                learned, MLScorer(args.scorer), cfg["bonus"], cfg["ml_inputs"],
                cfg["temp"], cfg["explore"], policy_seed, "sample")
            audit = et.FreeChoiceAudit(hook)
            trained = StrictHarnessPolicy(audit, record_requests=True)
            if args.arch in ("hybrid", "fixed"):
                learned.begin_gate_capture()
            result = mg.run_team_episode(
                trained, seed, cfg["steps"], hook=audit, recorder=audit,
                env_factory=env_factory(red))
            report["results"][red]["learned"][str(seed)] = result
            if args.arch in ("hybrid", "fixed"):
                gates = learned.end_gate_capture()
                report["router_gate"][red][str(seed)] = {
                    "n_decisions": len(gates),
                    "mean_attention_weight": float(np.mean(gates)) if gates else None,
                    "std_attention_weight": float(np.std(gates)) if gates else None,
                    "min_attention_weight": float(np.min(gates)) if gates else None,
                    "max_attention_weight": float(np.max(gates)) if gates else None,
                }
            print(f"eval arch={args.arch} rng={cfg['seed']} red={red} "
                  f"seed={seed} lancer={base['return']:.1f} "
                  f"untrained={zero_result['return']:.1f} "
                  f"learned={result['return']:.1f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    args = parser().parse_args()
    train(args) if args.mode == "train" else evaluate(args)
