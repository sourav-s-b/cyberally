"""Headroom screen: does any deployable heuristic beat Lancer anywhere?

Paired multi-seed comparison (evaluate_policies) of Lancer against
alternative investigation orderings. Goal: find one concrete slice
(seeds/episodes) where Lancer makes a poor choice and another available
action improves the outcome. If no slice shows a systematic edge, more
investigation-ordering RL has no demonstrated opportunity.

Arms: lancer (OrderedPolicy, guard off), roundrobin, stalest-first,
suspicion-sweep, uniform-investigation (lancer rules + uniform sweep
choice), zero-residual-sampling (lancer rules + softmax(base) sweep
choice, fixed generator per episode).

Usage (train venv): .venv-train/bin/python -m blue.analysis.heuristic_screen
    --seeds 7629 7630 ... --out blue/results/heuristic_screen.json
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEV_SEEDS = ([7629, 7630, 7640] + list(range(7701, 7730)))
ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}


def _uniform_hook(rng):
    def hook(env, agent, cands, scored):
        return cands[int(rng.integers(len(cands)))]
    return hook


def _zero_residual_hook(gen):
    import torch as th
    from blue.training.residual import masked_choice
    from blue.training.scorer import host_rows  # noqa (kept for symmetry)
    del host_rows

    def hook(env, agent, cands, scored):
        base = th.tensor([s for s, _ in scored], dtype=th.float32)
        mask = [1.0] * len(cands)
        idx, _, _ = masked_choice(base, mask, sample=True, seed_rng=gen)
        return cands[idx]
    return hook


def factories(seed):
    import numpy as np
    import torch as th
    from blue.core.baselines import (RoundRobinBaseline,
                                     StalestFirstBaseline,
                                     SuspicionSweepBaseline)
    from blue.policies.ordered import LancerValues, OrderedPolicy
    return {
        "lancer": lambda: OrderedPolicy(
            scorer=LancerValues(fruitless_decay=0.5), guard=False),
        "roundrobin": lambda: RoundRobinBaseline(),
        "stalest": lambda: StalestFirstBaseline(),
        "suspicion": lambda: SuspicionSweepBaseline(),
        "uniform_inv": lambda: _hooked_policy(_uniform_hook(
            np.random.default_rng(seed))),
        "zero_res": lambda: _hooked_policy(_zero_residual_hook(
            th.Generator().manual_seed(seed))),
    }


def _hooked_policy(hook):
    from blue.policies.ordered import LancerValues, OrderedPolicy
    pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                        guard=False)
    pol.hook = hook
    return pol


def main():
    import numpy as np
    from blue.core.baselines import evaluate_policies
    ap = argparse.ArgumentParser(description="heuristic headroom screen")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--red-agent", default="discovery")
    ap.add_argument("--out", default="blue/results/heuristic_screen.json")
    cli = ap.parse_args()
    env_kw = dict(ENV_KW)
    env_kw["red_agent"] = cli.red_agent
    results = {}
    for seed in cli.seeds:
        facs = factories(seed)
        res = evaluate_policies(facs, [seed], steps=cli.steps,
                                **env_kw)
        for name in facs:
            results.setdefault(name, {})[seed] = {
                "return": res[name][seed]["cumulative_return"],
                "final_compromised":
                    res[name][seed]["final"]["total"],
            }
        line = " ".join(f"{n}={results[n][seed]['return']:+.0f}"
                        for n in facs)
        print(f"seed={seed} {line}", flush=True)
    summary = {}
    lr = np.array([results["lancer"][s]["return"] for s in cli.seeds])
    for name in results:
        r = np.array([results[name][s]["return"] for s in cli.seeds])
        d = r - lr
        summary[name] = {"mean": float(r.mean()),
                         "paired_mean": float(d.mean()),
                         "paired_sd": float(d.std(ddof=1)) if len(d) > 1
                         else 0.0,
                         "wins": int((d > 0).sum()),
                         "losses": int((d < 0).sum())}
    with open(cli.out, "w") as f:
        json.dump({"red_agent": cli.red_agent, "seeds": list(cli.seeds),
                   "results": results, "summary": summary}, f)
    for n, s in summary.items():
        print(f"{n}: mean {s['mean']:+.1f} paired {s['paired_mean']:+.2f} "
              f"(sd {s['paired_sd']:.1f}) W{s['wins']}/L{s['losses']}")
    print("wrote", cli.out)


if __name__ == "__main__":
    main()
