"""Long-horizon unattended trainer for the risk-feature actor.

Purpose: run longer than a laptop session allows, and stop on EVIDENCE of
plateau rather than on a wall-clock guess. Every iteration:

1. fresh joint on-policy rollouts (dev seeds, filtered informative
   decisions -- see docs/plans/risk-actor-next.md caveat: this is a
   selected-experience update, not standard full-buffer PPO);
2. PPO update;
3. checkpoint (actor, critic, optimizer, RNG) so a session kill loses
   nothing;
4. append one JSON line to metrics.jsonl (never truncated);
5. periodic paired dev evaluation vs Lancer, which is the ONLY thing that
   decides continuation.

Stop conditions (first hit wins), all recorded:
- plateau: no improvement of the best dev paired mean by >= --min-gain for
  --patience consecutive evals;
- gate: dev paired mean >= --gate and its CI excludes zero -> stop
  immediately and mark PASSED (do not keep burning compute);
- --max-iters or --max-hours.

Everything is dev-seed training/evaluation. The reserved evaluation block
7809-8200 is never touched here.

Usage (train venv, repo root):
  .venv-train/bin/python -m blue.training.remote_train --out blue/results/remote1 \
      --iters 200 --eval-every 5 --patience 4 --min-gain 1.0 --max-hours 8
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blue.training import mappo_guide as mg
from blue.training import risk_actor as ra

EVAL_SEEDS = [7629, 7630, 7640, 7701, 7702, 7703, 7704, 7705]
T = 2.365  # t critical, df=7 (paired dev eval)


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())     # survive a hard kill


def evaluate(actor, seeds, steps, bonus, temp_eval=0.0):
    """Paired dev evaluation vs Lancer. Greedy (temp 0) at eval."""
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    risk = ra._risk()
    base, greedy = [], []
    for seed in seeds:
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        rb = mg.run_team_episode(pol, seed, steps, **mg.ENV_KW)
        hook = ra.RiskGreedyHook(actor, bonus, risk)
        rg = mg.run_team_episode(pol, seed, steps, hook=hook, **mg.ENV_KW)
        base.append(rb["return"])
        greedy.append(rg["return"])
    d = [g - b for g, b in zip(greedy, base)]
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1)) / (len(d) ** 0.5)
    return {"paired_mean": m, "sd": float(np.std(d, ddof=1)),
            "ci95": [m - T * se, m + T * se],
            "wins": int(sum(1 for x in d if x > 0)), "n": len(d),
            "lancer_mean": float(np.mean(base)),
            "actor_mean": float(np.mean(greedy))}


def save_ckpt(out, actor, critic, opt, gen_state, it):
    import torch as th
    tmp = os.path.join(out, ".ckpt.tmp")
    os.makedirs(tmp, exist_ok=True)
    th.save(actor.state_dict(), os.path.join(tmp, "actor.th"))
    th.save(critic.state_dict(), os.path.join(tmp, "critic.th"))
    th.save(opt.state_dict(), os.path.join(tmp, "opt.th"))
    th.save(gen_state, os.path.join(tmp, "rng.th"))
    with open(os.path.join(tmp, "state.json"), "w") as f:
        json.dump({"iter": it}, f)
    old = os.path.join(out, "latest")
    if os.path.isdir(old):
        subprocess.run(["rm", "-rf", old], check=False)
    os.replace(tmp, old)
    with open(os.path.join(out, "LATEST"), "w") as f:
        f.write(str(it))


def main():
    ap = argparse.ArgumentParser(description="unattended long trainer")
    ap.add_argument("--out", default="blue/results/remote1")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, nargs="*",
                    default=list(EVAL_SEEDS))
    ap.add_argument("--eval-subset", type=int, default=4,
                    help="dev seeds used for the mid-run gate check")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--train-min-proba", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-gain", type=float, default=1.0)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--gate-min-seeds", type=int, default=8,
                    help="refuse to declare a gate pass on fewer seeds")
    ap.add_argument("--gate-min-steps", type=int, default=400,
                    help="refuse to declare a gate pass on shorter episodes")
    ap.add_argument("--min-rows", type=int, default=8,
                    help="skip the PPO update below this many usable rows")
    ap.add_argument("--resume", action="store_true")
    cli = ap.parse_args()

    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy

    os.makedirs(cli.out, exist_ok=True)
    metrics_path = os.path.join(cli.out, "metrics.jsonl")
    actor = ra.build_actor(hidden=cli.hidden)
    critic = mg.build_central_critic(hidden=cli.hidden)
    opt = th.optim.Adam(list(actor.parameters())
                        + list(critic.parameters()), lr=cli.lr)
    start_it = 1
    if cli.resume and os.path.exists(os.path.join(cli.out, "latest")):
        d = os.path.join(cli.out, "latest")
        actor.load_state_dict(th.load(os.path.join(d, "actor.th"),
                                      map_location="cpu",
                                      weights_only=False))
        critic.load_state_dict(th.load(os.path.join(d, "critic.th"),
                                       map_location="cpu",
                                       weights_only=False))
        opt.load_state_dict(th.load(os.path.join(d, "opt.th"),
                                    map_location="cpu", weights_only=False))
        rng = th.load(os.path.join(d, "rng.th"), map_location="cpu",
                      weights_only=False)
        th.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        with open(os.path.join(d, "state.json")) as f:
            start_it = int(json.load(f)["iter"]) + 1
        print(f"resumed at iter {start_it}", flush=True)

    manifest = {"algo": "risk-as-feature residual PPO, filtered buffer",
                "git_commit": git_rev(),
                "selected_experience_caveat": "buffer filtered to picks "
                "with risk proba >= --train-min-proba; NOT standard "
                "full-buffer PPO",
                "params": vars(cli),
                "eval_seeds": list(cli.eval_seeds),
                "reserved_block": "7809-8200 never used here"}
    with open(os.path.join(cli.out, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    risk = ra._risk()
    cycle = [7706, 7707, 7708, 7709]
    t0 = time.time()
    best = -1e9
    stale = 0
    stop = None
    ev_seeds = cli.eval_seeds[:max(1, cli.eval_subset)]
    for it in range(start_it, cli.iters + 1):
        gen = th.Generator().manual_seed(cli.seed * 100003 + it)
        rec = ra.RiskRecorder(actor, cli.bonus, gen, temp=cli.temp,
                              risk=risk, min_proba=cli.train_min_proba)
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        rets = []
        for e in range(cli.eps_per_iter):
            seed = cycle[(it * cli.eps_per_iter + e) % len(cycle)]
            r = mg.run_team_episode(pol, seed, cli.steps, hook=rec,
                                    recorder=rec, **mg.ENV_KW)
            rets.append(r["return"])
        if len(rec.rows) < cli.min_rows:
            # Skip the update rather than learn from a degenerate batch:
            # a 1-row update normalised advantages to NaN and poisoned the
            # actor weights irrecoverably (observed 2026-10-06).
            st = {"kl": 0.0, "pol": 0.0, "vf": 0.0, "ent": 0.0}
            row = {"iter": it, "train_return": float(np.mean(rets)),
                   "skipped_update": True, "trained_rows": len(rec.rows),
                   "total_rows": rec.total, "elapsed_s": round(time.time()-t0, 1)}
        else:
            st = mg.ppo_central_update(actor, critic, opt, rec.rows,
                                       bonus=cli.bonus)
        row = {"iter": it, "train_return": float(np.mean(rets)),
               "trained_rows": len(rec.rows), "total_rows": rec.total,
               "kl": st["kl"], "elapsed_s": round(time.time() - t0, 1)}
        save_ckpt(cli.out, actor, critic, opt,
                  {"torch": th.get_rng_state(),
                   "numpy": np.random.get_state(),
                   "python": random.getstate()}, it)

        if it % cli.eval_every == 0 or it == cli.iters:
            actor.eval()
            ev = evaluate(actor, ev_seeds, cli.steps, cli.bonus)
            actor.train()
            row["eval"] = ev
            lo = ev["ci95"][0]
            if ev["paired_mean"] > best + cli.min_gain:
                best = max(best, ev["paired_mean"])
                stale = 0
            else:
                stale += 1
            row["best_mean"] = best
            row["stale"] = stale
            print(f"iter {it} train {row['train_return']:+.1f} "
                  f"eval {ev['paired_mean']:+.2f} "
                  f"CI[{ev['ci95'][0]:+.1f},{ev['ci95'][1]:+.1f}] "
                  f"W{ev['wins']}/{ev['n']} best {best:+.2f} "
                  f"stale {stale}/{cli.patience}", flush=True)
            append_jsonl(metrics_path, row)
            if stale >= cli.patience:
                stop = "PLATEAU"
                print(f"STOP: no improvement > {cli.min_gain} for "
                      f"{cli.patience} evals", flush=True)
                break
        else:
            append_jsonl(metrics_path, row)
        if (time.time() - t0) / 3600.0 >= cli.max_hours:
            stop = "TIME"
            print("STOP: max-hours reached", flush=True)
            break

    # Authoritative gate decision: FULL seed list at FULL horizon. The
    # mid-run evals are a cheap subset for plateau tracking only - a small
    # subset must never be able to declare a pass (our own decision rules).
    actor.eval()
    final = evaluate(actor, cli.eval_seeds, cli.steps, cli.bonus)
    actor.train()
    th.save(actor.state_dict(), os.path.join(cli.out, "actor_final.th"))
    append_jsonl(metrics_path, {"iter": it, "final_full_eval": final})
    gate_passed = bool(final["paired_mean"] >= cli.gate
                       and final["ci95"][0] > 0
                       and final["n"] >= cli.gate_min_seeds
                       and cli.steps >= cli.gate_min_steps)
    print("FINAL FULL EVAL " + json.dumps(final), flush=True)
    print("GATE " + ("PASSED" if gate_passed else "NOT PASSED"), flush=True)
    summary = {"final_full_eval": final, "gate_passed": gate_passed,
               "stop_reason": stop or "ITERS",
               "iters_done": it, "best_dev_mean": best,
               "gate": cli.gate,
               "git_commit": git_rev()}
    with open(os.path.join(cli.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()