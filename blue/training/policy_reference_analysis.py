"""Paired analysis of PPO vs persistent Lancer-reference KL policy screen.

The two training RNGs are the independent model units. Evaluation seeds are
paired by RNG and Red, and remain descriptive observations rather than extra
training runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("ppo", "reference")
REDS = ("discovery", "finite")
POLICIES = ("learned", "untrained", "strict_lancer")


def load_reports(base: Path, seeds):
    seed_keys = tuple(str(s) for s in seeds)
    cells = {}
    for arm in ARMS:
        for model_seed in (0, 1):
            path = base / "evaluations" / f"{arm}_s{model_seed}.json"
            report = json.loads(path.read_text())
            if report.get("eval_seeds") != list(seeds):
                raise ValueError(f"evaluation seed mismatch: {path}")
            for red in REDS:
                for policy in POLICIES:
                    table = report["results"][red][policy]
                    if set(table) != set(seed_keys):
                        raise ValueError(f"incomplete {red}/{policy}: {path}")
            cells[(arm, model_seed)] = report
    return cells


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    sx = np.sort(x)
    trim = int(len(sx) * 0.25)
    return {"n_eval_pairs": int(len(x)), "mean": float(x.mean()),
            "median": float(np.median(x)),
            "iqm": float(sx[trim:len(sx)-trim].mean()) if trim else float(x.mean()),
            "win_fraction": float((x > 0).mean()),
            "p10": float(np.quantile(x, .10)),
            "p90": float(np.quantile(x, .90))}


def report_eval_metrics(cells, seeds):
    out = {}
    for (arm, model_seed), report in cells.items():
        for red in REDS:
            for policy in POLICIES:
                rows = [report["results"][red][policy][str(s)] for s in seeds]
                metrics = {}
                for key in ("return", "coverage", "n_actor", "n_forced"):
                    vals = [r[key] for r in rows if r.get(key) is not None]
                    if vals:
                        metrics[key] = summary(vals)
                out.setdefault(arm, {}).setdefault(str(model_seed), {})[
                    red + "/" + policy] = metrics
    return out


def report_train_metrics(cells):
    keys = ("policy_steps", "kl_stopped", "kl_exact", "reference_kl",
            "reference_kl_coef", "clip_fraction", "entropy",
            "mean_decision_duration")
    out = {}
    for (arm, model_seed), report in cells.items():
        manifest = report["train_manifest"]
        history = manifest.get("history", [])
        out.setdefault(arm, {})[str(model_seed)] = {
            "config": manifest.get("config", {}),
            "iterations": [{k: row.get(k) for k in keys} for row in history],
            "episode_returns_by_red": {
                red: [ep["return"] for row in history for ep in row["episodes"]
                      if ep["red_agent"] == red]
                for red in REDS},
        }
    return out


def returns(cells, arm, model_seed, red, policy, seeds):
    table = cells[(arm, model_seed)]["results"][red][policy]
    return np.asarray([table[str(s)]["return"] for s in seeds], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="blue/results/policy_reference_kl_v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap-reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        ap.error("--eval-seeds must be unique")
    cells = load_reports(Path(args.base), args.eval_seeds)
    rng = np.random.default_rng(args.seed)

    report = {
        "claim": "two-training-RNG descriptive policy screen; not confirmation",
        "evaluation_seeds": args.eval_seeds,
        "bootstrap": {"reps": args.bootstrap_reps, "seed": args.seed,
                      "unit": "training RNG clusters; paired eval seeds within Red"},
        "evaluation_diagnostics": report_eval_metrics(cells, args.eval_seeds),
        "training_diagnostics": report_train_metrics(cells),
        "comparisons": {},
    }
    contrasts = {
        "reference_minus_ppo": ("reference", "learned", "ppo", "learned"),
        "reference_minus_untrained": ("reference", "learned", "reference", "untrained"),
        "reference_minus_lancer": ("reference", "learned", "reference", "strict_lancer"),
        "ppo_minus_untrained": ("ppo", "learned", "ppo", "untrained"),
        "ppo_minus_lancer": ("ppo", "learned", "ppo", "strict_lancer"),
    }
    for name, (arm_a, policy_a, arm_b, policy_b) in contrasts.items():
        item = {"by_red": {}, "by_rng_and_red": {}, "macro_mean_by_rng": {}}
        macro_by_rng = {}
        pooled = []
        for red in REDS:
            arrays = {}
            for model_seed in (0, 1):
                x = (returns(cells, arm_a, model_seed, red, policy_a,
                             args.eval_seeds)
                     - returns(cells, arm_b, model_seed, red, policy_b,
                               args.eval_seeds))
                arrays[model_seed] = x
                item["by_rng_and_red"].setdefault(str(model_seed), {})[red] = summary(x)
                macro_by_rng.setdefault(model_seed, []).append(float(x.mean()))
                pooled.extend(x.tolist())
            boot_by_rng = {}
            for model_seed, x in arrays.items():
                ix = rng.integers(0, len(x), size=(args.bootstrap_reps, len(x)))
                boot_by_rng[str(model_seed)] = [float(v) for v in np.quantile(
                    x[ix].mean(axis=1), [.025, .975])]
            hierarchical = np.empty(args.bootstrap_reps)
            for b in range(args.bootstrap_reps):
                chosen = rng.choice([0, 1], size=2, replace=True)
                vals = [arrays[int(m)][rng.integers(0, len(arrays[int(m)]),
                                                     len(arrays[int(m)]))].mean()
                        for m in chosen]
                hierarchical[b] = np.mean(vals)
            item["by_red"][red] = {
                "pooled": summary(np.concatenate(list(arrays.values()))),
                "bootstrap_95_by_rng": boot_by_rng,
                "hierarchical_macro_95": [float(v) for v in np.quantile(
                    hierarchical, [.025, .975])],
            }
        item["pooled_rows"] = summary(pooled)
        item["macro_mean_by_rng"] = {
            str(k): float(np.mean(v)) for k, v in macro_by_rng.items()}
        item["macro_mean"] = float(np.mean(list(item["macro_mean_by_rng"].values())))
        report["comparisons"][name] = item

    out = Path(args.out) if args.out else Path(args.base) / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
