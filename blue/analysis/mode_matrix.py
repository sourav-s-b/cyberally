"""Mode-vs-regime matrix: do defensive modes differ by attack regime?

Arms (frozen in mode-regime-matrix-20261005.json): lancer / a4susp /
strict, each on discovery + finite, paired dev seeds. If no mode beats
Lancer in any regime, a context router has nothing to select: stop before
any training.

Usage (train venv): .venv-train/bin/python -m blue.analysis.mode_matrix
    --out blue/results/mode_regime.json
"""

import argparse
import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SEEDS = [7629, 7630, 7640, 7701, 7702, 7703, 7704, 7705]
REDS = ["discovery", "finite"]


def make_policy(mode):
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.policies.zone_priority import Agent4SuspicionHook
    if mode == "lancer":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                             guard=False), None
    if mode == "a4susp":
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        hook = Agent4SuspicionHook()
        return pol, hook
    if mode == "strict":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                             guard=True), None
    raise ValueError(mode)


def main():
    from blue.training import mappo_guide as mg
    ap = argparse.ArgumentParser(description="mode-vs-regime matrix")
    ap.add_argument("--modes", nargs="*", default=["lancer", "a4susp",
                                                   "strict"])
    ap.add_argument("--reds", nargs="*", default=list(REDS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="blue/results/mode_regime.json")
    cli = ap.parse_args()
    partial = cli.out + ".partial"
    out = {}
    if os.path.exists(partial):
        try:
            with open(partial) as f:
                out = {k: {int(s): v for s, v in cells.items()}
                       for k, cells in
                       json.load(f).get("cells", {}).items()}
            print(f"resumed {sum(len(c) for c in out.values())} cells",
                  flush=True)
        except Exception as e:
            print(f"partial unreadable ({e}); starting over", flush=True)
    for red in cli.reds:
        kw = dict(mg.ENV_KW)
        kw["red_agent"] = red
        for mode in cli.modes:
            for s in cli.seeds:
                key = f"{red}/{mode}/{s}"
                if key in out.get("cells", {}):
                    continue
                pol, hook = make_policy(mode)
                if hook is not None and hasattr(hook, "reset"):
                    hook.reset()
                r = mg.run_team_episode(pol, s, cli.steps, hook=hook,
                                        **kw)
                out.setdefault("cells", {})[key] = r["return"]
                print(f"{red:10s} {mode:8s} seed={s} {r['return']:+.0f}",
                      flush=True)
                with open(partial, "w") as f:
                    json.dump({"partial": True, "cells": out["cells"]},
                              f)
    cells = out["cells"]
    summary = {}
    for red in cli.reds:
        lr = [cells[f"{red}/lancer/{s}"] for s in cli.seeds]
        for mode in cli.modes:
            if mode == "lancer":
                continue
            d = [cells[f"{red}/{mode}/{s}"] - l
                 for s, l in zip(cli.seeds, lr)]
            m, sd = sum(d) / len(d), statistics.stdev(d)
            se = sd / (len(d) ** 0.5)
            summary[f"{red}/{mode}_minus_lancer"] = {
                "paired_mean": m, "sd": sd,
                "ci95": [m - 2.365 * se, m + 2.365 * se]}
    with open(cli.out, "w") as f:
        json.dump({"cells": cells, "summary": summary}, f)
    try:
        os.remove(partial)
    except OSError:
        pass
    for k, v in summary.items():
        print(f"{k}: {v['paired_mean']:+.2f} CI[{v['ci95'][0]:+.1f},"
              f"{v['ci95'][1]:+.1f}]")
    print("wrote", cli.out)


if __name__ == "__main__":
    main()
