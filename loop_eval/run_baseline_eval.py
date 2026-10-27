"""Paired multi-seed baseline evaluation with results table.

Runs Sleep, MaskedRandom, RoundRobin, StalestFirst, and SuspicionSweep
on the same fixed seeds and produces a Markdown table + JSON results file
for direct comparison. No training involved.

Usage:
    python run_baseline_eval.py --seeds 10 20 30 --steps 200 \
        --output results/baseline_eval_001
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BLUE = _ROOT / "blue"
if str(_BLUE) not in sys.path:
    sys.path.insert(0, str(_BLUE))

_SIM_LOADED = False
wrapper = None
MaskedRandomBaseline = RoundRobinBaseline = SleepBaseline = None
StalestFirstBaseline = SuspicionSweepBaseline = evaluate_policies = None


def _import_sim():
    global wrapper, MaskedRandomBaseline, RoundRobinBaseline, SleepBaseline
    global StalestFirstBaseline, SuspicionSweepBaseline, evaluate_policies, _SIM_LOADED
    if _SIM_LOADED:
        return
    import cc4_epymarl_wrapper as _w
    from blue_baselines import (
        MaskedRandomBaseline as _MR,
        RoundRobinBaseline as _RR,
        SleepBaseline as _SL,
        StalestFirstBaseline as _SF,
        SuspicionSweepBaseline as _SS,
        evaluate_policies as _ep,
    )
    wrapper = _w
    MaskedRandomBaseline = _MR
    RoundRobinBaseline = _RR
    SleepBaseline = _SL
    StalestFirstBaseline = _SF
    SuspicionSweepBaseline = _SS
    evaluate_policies = _ep
    _SIM_LOADED = True


EVAL_VERSION = "baseline-eval-v1"

# Policy key names — class refs resolve lazily after _import_sim()
POLICY_NAMES = frozenset(["sleep", "masked-random", "round-robin",
                           "stalest-first", "suspicion-sweep"])


def _get_policy_factories():
    _import_sim()
    return {
        "sleep":           lambda: SleepBaseline(),
        "masked-random":   lambda: MaskedRandomBaseline(seed=0),
        "round-robin":     lambda: RoundRobinBaseline(),
        "stalest-first":   lambda: StalestFirstBaseline(),
        "suspicion-sweep": lambda: SuspicionSweepBaseline(),
    }


def _mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def run_baseline_evaluation(
    seeds: list[int],
    steps: int = 200,
    snapshot_steps: tuple = (100, 200),
    output: Path | None = None,
    red_agent: str = "discovery",
    policies: list[str] | None = None,
) -> dict:
    """Run all baselines across seeds and return structured results."""
    _import_sim()
    POLICY_FACTORIES = _get_policy_factories()
    selected = {k: v for k, v in POLICY_FACTORIES.items()
                if policies is None or k in policies}

    print(f"Policies: {list(selected)}")
    print(f"Seeds:    {seeds}  |  Steps: {steps}  |  Red: {red_agent}\n")

    raw = evaluate_policies(
        selected,
        seeds,
        steps=steps,
        snapshot_steps=snapshot_steps,
        red_agent=red_agent,
    )

    # Compute summary stats per policy
    agg = {}
    for name, seed_results in raw.items():
        returns = [r["cumulative_return"] for r in seed_results.values()]
        finals = [r["final"]["total"] for r in seed_results.values()]
        roots = [r["final"]["root"] for r in seed_results.values()]
        snap_200 = [
            r["snapshots"].get(200, {}).get("total")
            for r in seed_results.values()
        ]
        agg[name] = {
            "mean_return": _mean(returns),
            "std_return": _std(returns),
            "mean_compromised_final": _mean(finals),
            "mean_root_final": _mean(roots),
            "mean_compromised_at_200": _mean([x for x in snap_200 if x is not None]),
            "per_seed": {
                str(s): {
                    "return": r["cumulative_return"],
                    "compromised_final": r["final"]["total"],
                    "root_final": r["final"]["root"],
                    "snapshots": r["snapshots"],
                }
                for s, r in seed_results.items()
            },
        }

    result = {
        "eval_version": EVAL_VERSION,
        "seeds": seeds,
        "steps": steps,
        "red_agent": red_agent,
        "snapshot_steps": list(snapshot_steps),
        "results": agg,
    }

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "baseline_results.json"
        meta = dict(result)
        try:
            meta["source_commit"] = subprocess.check_output(
                ["git", "-C", str(_ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            meta["source_commit"] = "unknown"
        meta["python"] = sys.version
        result_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        md_path = output / "baseline_results.md"
        md_path.write_text(_make_markdown_table(agg, seeds, steps, red_agent),
                           encoding="utf-8")
        print(f"\nResults JSON: {result_path}")
        print(f"Results MD:   {md_path}")

    return result


def _std(values):
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def _make_markdown_table(agg: dict, seeds: list, steps: int, red: str) -> str:
    lines = [
        f"# Baseline Evaluation Results",
        f"",
        f"**Seeds:** {seeds}  |  **Steps:** {steps}  |  **Red:** {red}",
        f"",
        f"| Policy | Mean Return | Std Return | Compromised@200 | Compromised Final | Root Final |",
        f"|--------|-------------|------------|-----------------|-------------------|------------|",
    ]
    for name, stats in agg.items():
        mr = f"{stats['mean_return']:.2f}" if stats["mean_return"] is not None else "N/A"
        sr = f"{stats['std_return']:.2f}" if stats["std_return"] is not None else "N/A"
        c200 = f"{stats['mean_compromised_at_200']:.1f}" if stats["mean_compromised_at_200"] is not None else "N/A"
        cf = f"{stats['mean_compromised_final']:.1f}" if stats["mean_compromised_final"] is not None else "N/A"
        rf = f"{stats['mean_root_final']:.1f}" if stats["mean_root_final"] is not None else "N/A"
        lines.append(f"| {name:<20} | {mr:>11} | {sr:>10} | {c200:>15} | {cf:>17} | {rf:>10} |")

    lines += [
        "",
        "> Compromised counts are privileged eval labels (true simulator state).",
        "> Never used as policy inputs.",
        "",
        f"*Generated by `run_baseline_eval.py` — Eval version: {EVAL_VERSION}*",
    ]
    return "\n".join(lines) + "\n"


def _print_table(result: dict) -> None:
    agg = result["results"]
    print(f"\n{'='*78}")
    print(f"Baseline Evaluation — Seeds: {result['seeds']} | Steps: {result['steps']} | Red: {result['red_agent']}")
    print(f"{'='*78}")
    header = f"{'Policy':<22} {'Return':>10} {'±':>8} {'Comp@200':>10} {'Comp Final':>12} {'Root':>6}"
    print(header)
    print("-" * 70)
    for name, stats in agg.items():
        mr = f"{stats['mean_return']:.1f}" if stats["mean_return"] is not None else "N/A"
        sr = f"{stats['std_return']:.1f}" if stats["std_return"] is not None else ""
        c200 = f"{stats['mean_compromised_at_200']:.1f}" if stats["mean_compromised_at_200"] is not None else "N/A"
        cf = f"{stats['mean_compromised_final']:.1f}" if stats["mean_compromised_final"] is not None else "N/A"
        rf = f"{stats['mean_root_final']:.1f}" if stats["mean_root_final"] is not None else "N/A"
        print(f"  {name:<20} {mr:>10} {sr:>8} {c200:>10} {cf:>12} {rf:>6}")
    print()


if __name__ == "__main__":
    _import_sim()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--red-agent", choices=sorted(wrapper.RED_AGENTS),
                        default="discovery")
    parser.add_argument("--policies", nargs="+", choices=sorted(POLICY_NAMES),
                        default=None, help="Subset of policies to run")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")

    result = run_baseline_evaluation(
        seeds=args.seeds,
        steps=args.steps,
        output=args.output,
        red_agent=args.red_agent,
        policies=args.policies,
    )
    _print_table(result)
