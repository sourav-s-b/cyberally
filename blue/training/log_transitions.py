"""OP3 behaviour-logging pass (prereg op3-awdc-logs-8433-20261022).

Records TRANSITIONS of the scripted behaviour heuristics under the
CURRENT wrapper obs contract (foundation-v3, temporal features, red
finite), Blue-visible channels only. The skeleton's prereg finding: no
transition logs exist under this contract; legacy 872-dim logs must NOT
be converted; this pass is the only valid source.

npz keys per prereg: obs, actions, rewards, masks, lengths + meta
(saved separately in the manifest): per-agent obs as issued by
``env.get_obs()`` at decision time, per-agent masks as issued by
``env.get_avail_agent_actions()``, recorded policy actions, native
per-tick team reward ``float(rew[0])``, and the done flag.

Gates built in:
- G1 obs-contract check on EVERY first logged episode's every tick (obs
  per-agent width vs get_env_info(), every action inside its mask)
  before the batch continues.

Usage (train venv, repo root):
  .venv-train/bin/python -m blue.training.log_transitions \
      --out blue/results/op3_awdc_logs_v1/logged.npz \
      --manifest blue/results/op3_awdc_logs_v1/logging_manifest.json

Default recipe (per prereg): 12 train seeds x (guard + lancer + sleep +
random x3 policy seeds) = 72 episodes, ~25 s each, cap 3 h.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TRAIN_SEEDS = [8433, 8434, 8435, 8436, 8437, 8438, 8439, 8440,
               8441, 8442, 8443, 8444]
RANDOM_POLICY_SEEDS = 3


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode()
    except Exception:
        return "unknown"


def build_arm(name, api_seed=None):
    from blue.core.baselines import MaskedRandomBaseline, SleepBaseline
    from blue.policies.ordered import LancerValues, OrderedPolicy
    if name == "guard":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                             max_age=80.0)
    if name == "lancer":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5))
    if name == "sleep":
        return SleepBaseline()
    if name == "random":
        return MaskedRandomBaseline(seed=0 if api_seed is None
                                    else int(api_seed))
    raise ValueError(f"unknown arm {name}")


def log_episode(kind, seed, steps, env_kw):
    """One episode. ``kind`` is the arm family; returns dict of arrays."""
    from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
    policy = build_arm(kind, api_seed=seed if kind == "random" else None)
    env = CC4MARLEnv(seed=seed, steps=steps, **env_kw)
    env.reset(seed=seed)
    policy.reset()
    info = env.get_env_info()
    obs_size = int(info["obs_size_per_agent"][BLUE_AGENTS.index(
        next(iter(BLUE_AGENTS)))])
    n_acts = int(info["n_actions"])
    obs_l, act_l, rew_l, mask_l, done_l = [], [], [], [], []
    for _tick in range(1, steps + 1):
        obs = np.stack([np.asarray(o, dtype=np.float32)
                        for o in env.get_obs()])
        assert obs.shape[0] == len(BLUE_AGENTS), "obs not per-agent"
        assert obs.shape[1] == obs_size, (
            f"obs width {obs.shape[1]} != get_env_info "
            f"{obs_size} (obs-contract violation, G1)")
        masks = [env.get_avail_agent_actions(i) for i in range(5)]
        for i, m in enumerate(masks):
            assert m.shape == (n_acts,)
        obs_l.append(np.asarray(obs, dtype=np.float32))
        mask_l.append(np.stack([np.asarray(m, dtype=np.int8)
                                for m in masks]))
        acts = {}
        for a in BLUE_AGENTS:
            idx = BLUE_AGENTS.index(a)
            act = int(policy.select(env, a))
            assert masks[idx][act], (
                f"{kind} action {act} outside mask (G1, agent {a})")
            acts[a] = act
        act_l.append([acts[a] for a in BLUE_AGENTS])
        _obs2, rew, terminated, truncated, _ = env.step(acts)
        rew_l.append(float(rew[0]))
        done_l.append(bool(terminated or truncated))
        if terminated or truncated:
            break
    env.close()
    return {
        "obs": np.stack(obs_l),
        "actions": np.stack(act_l).astype(np.int16),
        "masks": np.stack(mask_l),
        "rewards": np.asarray(rew_l, dtype=np.float64),
        "done": np.asarray(done_l, dtype=np.int8),
    }


def main():
    from blue.training.mappo_guide import ENV_KW
    ap = argparse.ArgumentParser(description="OP3 behaviour logging")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    cli = ap.parse_args()
    env_kw = dict(ENV_KW)
    env_kw["red_agent"] = "finite"

    jobs = []
    for s in cli.seeds:
        jobs.append(("guard", s, None))
        jobs.append(("lancer", s, None))
        jobs.append(("sleep", s, None))
        for j in range(RANDOM_POLICY_SEEDS):
            jobs.append(("random", s, 1000 * (j + 1) + s))

    order = [b[0] for b in jobs]
    eps = {k: {"obs": [], "actions": [], "rewards": [], "masks": [],
               "done": []} for k in set(order)}
    meta_kinds, meta_seeds, meta_pol_seeds, meta_lens = [], [], [], []
    total_r = 0.0
    blown_budget = False
    prog_t0 = time.time()
    for i, (kind, seed, pol_seed) in enumerate(jobs):
        if prog_t0 is not None and (time.time() - prog_t0) > 3 * 3600:
            blown_budget = True
            print("wallclock cap 3h hit; stopping with partial npz "
                  "(prereg stop rule: no padding)", flush=True)
            break
        r = log_episode(kind, seed, cli.steps, env_kw)
        d = eps[kind]
        for k in d:
            d[k].append(r[k])
        extra_r = r["obs"].nbytes
        total_r += extra_r
        meta_kinds.append(kind)
        meta_seeds.append(seed)
        meta_pol_seeds.append(pol_seed)
        meta_lens.append(len(r["rewards"]))
        print(f"[{i + 1}/{len(jobs)}] {kind:6s} seed={seed} "
              f"pol_seed={pol_seed} T={len(r['rewards'])} "
              f"ret={r['rewards'].sum():+.1f} "
              f"({time.time() - prog_t0:.0f}s)", flush=True)

    flat = {}
    for kind, d in eps.items():
        if not d["obs"]:
            continue
        flat[f"obs_{kind}"] = np.concatenate(d["obs"])
        flat[f"actions_{kind}"] = np.concatenate(d["actions"])
        flat[f"masks_{kind}"] = np.concatenate(d["masks"])
        flat[f"rewards_{kind}"] = np.concatenate(d["rewards"])
        flat[f"done_{kind}"] = np.concatenate(d["done"])
    flat["lengths"] = np.asarray(meta_lens, dtype=np.int32)
    flat["meta_kinds"] = np.asarray(meta_kinds)
    flat["meta_seeds"] = np.asarray(meta_seeds, dtype=np.int64)
    flat["meta_pol_seeds"] = np.asarray(
        [-1 if p is None else p for p in meta_pol_seeds], dtype=np.int64)
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    np.savez_compressed(cli.out, **flat)
    manifest = {
        "purpose": "OP3 behaviour logging: current-contract transitions "
                   "(blue-visible only), native rewards, prereg min "
                   "recipe (guard/lancer/sleep/random on train seeds)",
        "git_commit": git_commit(),
        "env_kw": {k: list(v) if isinstance(v, tuple) else v
                   for k, v in env_kw.items()},
        "steps": cli.steps,
        "windows": "train seeds 8433-8444 (12) + random policy seeds "
                   "1000*s..3000*s (distinct behaviour distributions)",
        "episodes_logged": len(meta_lens),
        "blown_budget": blown_budget,
        "bytes": total_r,
        "kinds": sorted(set(meta_kinds)),
        "pol_seed_offset_per_sim_seed": "1000/2000/3000 * sim_seed",
    }
    with open(cli.manifest, "w") as f:
        json.dump(manifest, f, indent=1)
    sha = hashlib.sha256(open(cli.out, "rb").read()).hexdigest()[:16]
    print("npz bytes:", total_r, "eps:", len(meta_lens), "sha:", sha)
    print("wrote", cli.out, "and", cli.manifest)


if __name__ == "__main__":
    main()
