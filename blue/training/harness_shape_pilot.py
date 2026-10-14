"""Measure the shaped signal's magnitude against native return, before scaling.

The concern this answers: a per-tick shaping penalty accumulates over hundreds
of ticks. With a cap of 10 the shaped term could reach -4000 across a 400-step
episode while native return is about -140, so PPO would be optimising debt
and ignoring the true objective entirely. This script reports both totals and
their ratio on real episodes, using the same rollout path as training.

Run before committing to the full matrix:

    .venv-train/bin/python -m blue.training.harness_shape_pilot \
        --seeds 7706 7707 --red-agent finite --state-shaping both
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# A shaped total larger than this multiple of |native| is reported as unsafe:
# the training objective would no longer be recognisably the defence task.
UNSAFE_RATIO = 3.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7706, 7707])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--red-agent", default="finite")
    ap.add_argument("--state-shaping", default="both",
                    choices=("off", "detection", "confirm", "both"))
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--cap", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if any(7809 <= s <= 8200 for s in args.seeds):
        raise SystemExit("refusing: reserved final seeds 7809-8200")

    from blue.training import mappo_guide as mg
    from blue.training import harness_rl as hr
    from blue.training.harness_rl import env_kwargs

    hook = None
    if args.state_shaping != "off":
        from blue.harness.state_reward import make_reward_hook
        hook = make_reward_hook(term=args.state_shaping, weight=args.weight,
                                cap=args.cap, horizon=args.steps)

    rows = []
    for seed in args.seeds:
        from blue.harness.policy import HarnessPolicy
        policy = HarnessPolicy(max_age=80.0)
        result = mg.run_team_episode(policy, seed, args.steps,
                                     reward_hook=hook,
                                     **env_kwargs(args.red_agent))
        native = result["return"]
        shaped = result["shaped_return"]
        added = (shaped - native) if hook is not None else 0.0
        ratio = abs(added) / abs(native) if native else float("inf")
        rows.append({"seed": seed, "native": native, "shaped": shaped,
                     "added": added, "ratio": ratio,
                     "coverage": result["coverage"]})
        print("seed=%d native=%+.1f shaped=%s added=%+.1f ratio=%.2f"
              % (seed, native, "%+.1f" % shaped if hook else "n/a",
                 added, ratio), flush=True)

    if hook is None:
        print("\nno shaping: nothing to calibrate")
        return 0

    mean_ratio = statistics.mean(r["ratio"] for r in rows)
    print("\nnative mean      : %+.1f" % statistics.mean(
        r["native"] for r in rows))
    print("shaping added    : %+.1f (mean per episode)"
          % statistics.mean(r["added"] for r in rows))
    print("mean |added|/|native| ratio: %.2f  (unsafe above %.1f)"
          % (mean_ratio, UNSAFE_RATIO))
    verdict = ("UNSAFE: the shaped term dominates native return; PPO would be "
               "optimising debt, not defence"
               if mean_ratio > UNSAFE_RATIO else
               "OK: shaped term is a modifier, not a replacement")
    print("verdict: " + verdict)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "rows": rows,
                       "mean_ratio": mean_ratio,
                       "unsafe_ratio": UNSAFE_RATIO,
                       "verdict": verdict}, f, indent=2)
        print("wrote", args.out)
    return 0 if mean_ratio <= UNSAFE_RATIO else 1


if __name__ == "__main__":
    sys.exit(main())