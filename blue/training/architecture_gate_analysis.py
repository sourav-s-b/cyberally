"""Paired analysis for the SetActor/fixed-blend/learned-gate screen.

The two training RNGs are retained as separate clusters. The bootstrap samples
evaluation seeds within each Red regime and training RNG; it does not treat the
64 seed/RNG/regime rows as 64 independent trained models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ("set", "fixed", "hybrid")
REDS = ("discovery", "finite")
POLICIES = ("learned", "untrained", "strict_lancer")
SEEDS = tuple(str(s) for s in range(8417, 8433))


def load_reports(base: Path):
    cells = {}
    for arm in ARMS:
        for rng_seed in (0, 1):
            path = base / "evaluations" / f"{arm}_s{rng_seed}.json"
            if not path.exists():
                raise FileNotFoundError(f"missing completed eval: {path}")
            report = json.loads(path.read_text())
            if report.get("eval_seeds") != list(range(8417, 8433)):
                raise ValueError(f"evaluation seed mismatch in {path}")
            for red in REDS:
                for policy in POLICIES:
                    table = report["results"][red][policy]
                    missing = set(SEEDS) - set(table)
                    if missing:
                        raise ValueError(f"{path}: {red}/{policy} missing {sorted(missing)}")
            cells[(arm, rng_seed)] = report
    return cells


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    sx = np.sort(x)
    trim = int(len(sx) * 0.25)
    iqm = sx[trim:len(sx) - trim].mean() if trim else sx.mean()
    return {"n": int(len(x)), "mean": float(x.mean()),
            "median": float(np.median(x)), "iqm": float(iqm),
            "win_fraction": float((x > 0).mean()),
            "p10": float(np.quantile(x, 0.10)),
            "p90": float(np.quantile(x, 0.90))}


def deltas(cells, arm_a, source_a, arm_b, source_b, red):
    out = {}
    for rng_seed in (0, 1):
        ra = cells[(arm_a, rng_seed)]["results"][red][source_a]
        rb = cells[(arm_b, rng_seed)]["results"][red][source_b]
        out[rng_seed] = np.asarray([
            ra[s]["return"] - rb[s]["return"] for s in SEEDS
        ], dtype=np.float64)
    return out


def paired_bootstrap_by_rng(arrays, rng, reps=10000):
    result = {}
    for model_rng, x in arrays.items():
        n = len(x)
        ix = rng.integers(0, n, size=(reps, n))
        means = x[ix].mean(axis=1)
        result[str(model_rng)] = [float(v) for v in
                                  np.quantile(means, [0.025, 0.975])]
    # Hierarchical descriptive interval: resample model RNG clusters, then
    # paired evaluation seeds within selected clusters.
    model_rngs = sorted(arrays)
    boot = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        selected = rng.choice(model_rngs, size=len(model_rngs), replace=True)
        sampled_model_means = []
        for model_rng in selected:
            x = arrays[int(model_rng)]
            sampled_model_means.append(x[rng.integers(0, len(x), len(x))].mean())
        boot[b] = np.mean(sampled_model_means)
    result["hierarchical_macro_95"] = [float(v) for v in
                                       np.quantile(boot, [0.025, 0.975])]
    return result


def bootstrap_model_clusters(values, rng, reps=10000):
    model_rngs = sorted(values)
    vals = np.asarray([values[k] for k in model_rngs], dtype=np.float64)
    ix = rng.integers(0, len(vals), size=(reps, len(vals)))
    boot = vals[ix].mean(axis=1)
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="blue/results/architecture_gate_ablation_v1")
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    base = Path(args.base)
    cells = load_reports(base)
    rng = np.random.default_rng(args.seed)

    comparisons = {
        "hybrid_minus_fixed": ("hybrid", "learned", "fixed", "learned"),
        "hybrid_minus_set": ("hybrid", "learned", "set", "learned"),
        "set_minus_fixed": ("set", "learned", "fixed", "learned"),
        "hybrid_minus_untrained": ("hybrid", "learned", "hybrid", "untrained"),
        "fixed_minus_untrained": ("fixed", "learned", "fixed", "untrained"),
        "set_minus_untrained": ("set", "learned", "set", "untrained"),
        "hybrid_minus_lancer": ("hybrid", "learned", "hybrid", "strict_lancer"),
        "fixed_minus_lancer": ("fixed", "learned", "fixed", "strict_lancer"),
        "set_minus_lancer": ("set", "learned", "set", "strict_lancer"),
    }
    report = {"claim": "two-training-RNG descriptive screen; not confirmation",
              "evaluation_seeds": [int(s) for s in SEEDS],
              "bootstrap": {"reps": args.bootstrap_reps, "seed": args.seed,
                            "cluster": "training RNG; paired seed resampling within Red"},
              "comparisons": {}}
    for name, (aa, pa, ab, pb) in comparisons.items():
        item = {"by_red": {}, "by_rng_and_red": {}, "pooled_rows": {}}
        pooled = []
        macro_by_rng = {}
        for red in REDS:
            d = deltas(cells, aa, pa, ab, pb, red)
            all_red = np.concatenate([d[0], d[1]])
            item["by_red"][red] = {
                "pooled": summarize(all_red),
                "bootstrap_95": paired_bootstrap_by_rng(
                    d, rng, args.bootstrap_reps),
            }
            pooled.extend(all_red.tolist())
            for model_rng, x in d.items():
                item["by_rng_and_red"].setdefault(str(model_rng), {})[red] = summarize(x)
                macro_by_rng.setdefault(model_rng, []).append(float(x.mean()))
        item["pooled_rows"] = summarize(pooled)
        macro = {k: float(np.mean(v)) for k, v in macro_by_rng.items()}
        item["macro_mean_by_rng"] = {str(k): v for k, v in macro.items()}
        item["macro_mean"] = float(np.mean(list(macro.values())))
        item["macro_bootstrap_95"] = bootstrap_model_clusters(
            macro, rng, args.bootstrap_reps)
        report["comparisons"][name] = item

    out = Path(args.out) if args.out else base / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
