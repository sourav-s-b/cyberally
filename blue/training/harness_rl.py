"""Opt-in PPO path with the SAME feature/ML/coverage harness in train and eval.

New feature width requires new checkpoints. Does not rewrite risk_actor or
load its weights. No V(s)-based action gate and no reward shaping/filtering.
"""
from __future__ import annotations

import argparse
import json
import random
import os
import subprocess
import platform
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

from blue.harness.features import ACTOR_NAMES, VERSION
from blue.harness.policy import HarnessActorHook, HarnessPolicy
from blue.harness.scoring import MLScorer, sha256
from blue.training import mappo_guide as mg


def check_seeds(seeds):
    if not seeds or any(7809 <= s <= 8200 for s in seeds):
        raise ValueError("empty seed list or protected final block")


class HarnessRecorder(mg.JointRecorder):
    def __init__(self, actor, scorer, bonus, gen, temp, ml_inputs, diagnostics=False):
        super().__init__(actor, bonus, gen, temp=temp)
        self.adapter = HarnessActorHook(actor, scorer, bonus, ml_inputs)
        self.sampled_disagree = 0
        self.diagnostics = diagnostics
        self.sensitivity = []

    def reset(self):
        self.adapter.reset()
        self.sensitivity = []

    def observe(self, env, agent):
        self.adapter.observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        import torch as th
        from blue.training.residual import masked_choice
        from blue.policies.ordered import argmax_pick
        F = th.from_numpy(self.adapter.features(env, agent, cands))
        with th.no_grad():
            residual = th.tanh(self.actor(F))
        base = th.tensor([s for s, _ in scored], dtype=th.float32)
        if self.diagnostics:
            from blue.harness.diagnostics import input_sensitivity
            self.sensitivity.append(input_sensitivity(self.actor, F, base, self.bonus))
        logits = (base + self.bonus * residual) / self.temp
        mask = [1.0] * len(cands)
        idx, logp, ent = masked_choice(logits, mask, sample=True, seed_rng=self.gen)
        teacher = argmax_pick(scored)
        greedy = argmax_pick([(float(logits[i]), h) for i, h in enumerate(cands)])
        self.agree += int(greedy == teacher)
        self.sampled_disagree += int(cands[idx] != teacher)
        self.total += 1
        row = {"feats": F.numpy(), "joint": mg.joint_context(env, self._policy_ref, self._jcache),
               "k": len(cands), "mask": mask, "base": base.numpy(),
               "choice": idx, "logp": logp, "ent": ent,
               "tick": env._tick, "agent": agent, "ret": 0.0, "temp": self.temp}
        self.rows.append(row)
        self.pending.setdefault(agent, []).append(len(self.rows) - 1)
        self.per_agent[agent] = self.per_agent.get(agent, 0) + 1
        return cands[idx]


def build_actor(hidden=64):
    from blue.training.residual import build_nets
    return build_nets(feat_dim=len(ACTOR_NAMES), hidden=hidden)[0]


def source_identity():
    pin = Path("snapshot.json")
    if pin.exists():
        return json.loads(pin.read_text())["base_commit"]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def train(args):
    import torch as th
    check_seeds(args.train_seeds)
    if args.iters < 1 or args.eps_per_iter < 1 or args.temp <= 0:
        raise ValueError("invalid training budget")
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    scorer = MLScorer(args.scorer)
    actor = build_actor(args.hidden)
    critic = mg.build_central_critic(hidden=args.hidden)
    opt = th.optim.Adam([*actor.parameters(), *critic.parameters()], lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and not args.resume:
        raise FileExistsError("use a new run directory; never overwrite an experiment")
    manifest = {"version": VERSION, "features": list(ACTOR_NAMES),
                "scorer_sha256": scorer.sha256, "scorer": scorer.path,
                "config": vars(args), "gamma": mg.GAMMA,
                "source_commit": source_identity(),
                "source_status": subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True).splitlines(),
                "env_kw": mg.ENV_KW, "status": "started",
                "runtime": {"python": platform.python_version(), **{
                    p: package_version(p) for p in (
                    "numpy", "scipy", "torch", "scikit-learn", "gym", "gymnasium")}},
                "objective": "native team reward; unfiltered fresh on-policy rollouts",
                "checkpoint_semantics": "new actor geometry; legacy checkpoints incompatible"}
    # Pin actual dirty source bytes, not just HEAD.
    manifest["source_files"] = {str(p): sha256(p) for p in [
        *sorted(Path("blue/harness").glob("*.py")), Path(__file__),
        Path(mg.__file__), Path("blue/training/residual.py"),
        Path("blue/policies/ordered.py"), Path("blue/core/wrapper.py")]}
    hist = []
    start = 0
    if args.resume:
        old = json.loads((out / "manifest.json").read_text())
        keys = ("seed", "iters", "eps_per_iter", "steps", "hidden", "lr", "bonus", "temp", "max_age", "ml_inputs", "train_seeds")
        if any(old["config"][k] != manifest["config"][k] for k in keys) or old["source_files"] != manifest["source_files"] or old["scorer_sha256"] != scorer.sha256:
            raise ValueError("resume configuration/source/artifact drift")
        if old["config"].get("diagnostics", False) != args.diagnostics:
            raise ValueError("resume diagnostic configuration drift")
        if old.get("runtime", manifest["runtime"]) != manifest["runtime"]:
            raise ValueError("resume runtime drift")
        ck = th.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["optimizer"])
        th.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["numpy_rng"]); random.setstate(ck["python_rng"])
        hist = ck["history"]; start = len(hist)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for iteration in range(start, args.iters):
        gen = th.Generator().manual_seed(args.seed * 100003 + iteration + 1)
        rec = HarnessRecorder(actor, scorer, args.bonus, gen, args.temp, args.ml_inputs,
                              diagnostics=args.diagnostics)
        policy = HarnessPolicy(rec, max_age=args.max_age, record_requests=args.diagnostics)
        returns, interventions = [], []
        episode_diagnostics = []
        for e in range(args.eps_per_iter):
            seed = args.train_seeds[(iteration * args.eps_per_iter + e) % len(args.train_seeds)]
            result = mg.run_team_episode(policy, seed, args.steps, hook=rec, recorder=rec, **mg.ENV_KW)
            print(f"iteration={iteration+1} episode={e+1} seed={seed} return={result['return']}", flush=True)
            returns.append(result["return"])
            interventions.append(policy.guard_stats())
            if args.diagnostics:
                diagnostic = {"seed": seed, **policy.request_digest.result(),
                              "sweep_sensitivity": rec.sensitivity}
                episode_diagnostics.append(diagnostic)
                print(f"request_hash={diagnostic['requested_actions_sha256']} "
                      f"iteration={iteration+1} episode={e+1} arm={args.ml_inputs}", flush=True)
        if not rec.rows:
            raise RuntimeError("no trainable sweep decisions; no update performed")
        stats = mg.ppo_central_update(actor, critic, opt, rec.rows, bonus=args.bonus)
        if not all(np.isfinite(v) for v in stats.values()):
            raise RuntimeError("nonfinite PPO update")
        row = {"iteration": iteration + 1, "returns": returns,
               "decisions": rec.total, "greedy_agreement": rec.agree / rec.total,
               "sampled_override_rate": rec.sampled_disagree / rec.total,
               "guard": interventions, **stats}
        hist.append(row)
        if args.diagnostics:
            # Per-decision sensitivity lives outside compact training history.
            (out / f"diagnostics_{iteration+1:03d}.json").write_text(
                json.dumps(episode_diagnostics, indent=2) + "\n")
            row["episode_request_digests"] = [{k: v for k, v in d.items()
                if k != "sweep_sensitivity"} for d in episode_diagnostics]
        th.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "optimizer": opt.state_dict(),
                 "history": hist, "torch_rng": th.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate()}, out / "checkpoint.tmp")
        os.replace(out / "checkpoint.tmp", out / "checkpoint.pt")
        (out / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in hist))
        th.save(actor.state_dict(), out / "actor.th")
        th.save(critic.state_dict(), out / "critic.th")
        print(f"iteration={iteration+1} return={np.mean(returns):+.2f} decisions={rec.total}", flush=True)
        if args.stop_after and len(hist) >= args.stop_after:
            break
    manifest.update(status="complete" if len(hist) == args.iters else "paused", history=hist, actor_sha256=sha256(out / "actor.th"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_hook(model_dir, scorer_path):
    import torch as th
    d = Path(model_dir)
    m = json.loads((d / "manifest.json").read_text())
    scorer = MLScorer(scorer_path)
    expected = list(ACTOR_NAMES)
    if m["version"] != VERSION or m["features"] != expected or m["scorer_sha256"] != scorer.sha256:
        raise ValueError("checkpoint/scorer feature or artifact mismatch")
    if m["status"] != "complete" or sha256(d / "actor.th") != m["actor_sha256"]:
        raise ValueError("checkpoint incomplete or modified")
    c = m["config"]
    actor = build_actor(c["hidden"])
    actor.load_state_dict(th.load(d / "actor.th", map_location="cpu", weights_only=True))
    actor.eval()
    return HarnessActorHook(actor, scorer, c["bonus"], c["ml_inputs"]), m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stop-after", type=int, default=0, help="checkpoint-boundary interruption for recovery checks")
    ap.add_argument("--diagnostics", action="store_true", help="full request hashes and fixed-state ML input sensitivity")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scorer", default="blue/results/gpt_harness/scorer.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-seeds", nargs="+", type=int, default=[7706, 7707, 7708, 7709])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bonus", type=float, default=0.25)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--max-age", type=float, default=80)
    ap.add_argument("--ml-inputs", choices=("both", "risk", "zero"), default="both")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
