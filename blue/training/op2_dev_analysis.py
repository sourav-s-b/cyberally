"""OP2 frozen dev-window analysis (paired per-seed percentile bootstrap).

This IS the fixed analysis script the OP2 prereg names. It reads:

1. The G1 re-measure report (guard control cells) —
   blue/results/op2_jsrl_guide_v1/guard_remeasure/report.json
2. One or more mappo_guide cmd_eval output JSONs (mode eval, each already
   run over the same dev seed list with --red-agent matching the G1
   re-measure red), keyed as: --label jsrl --eval-json <path> (repeatable).

Primary contrast (prereg TAU = 0.0): per-seed JSRL score is the MEAN of the
learned-arm returns across the rng_seeds supplied for that label; the
contrast is JSRL per-seed mean minus the G1 guard per-seed return on the
seed intersection. Secondary: any other label (e.g. noguide ablation)
against guard, and jsrl versus any other label.

Bootstrap: 10000 paired resamples, numpy default_rng(20261022), percentile
2.5/97.5 interval. No performance claims are made here beyond intervals;
the gate decision belongs to whoever runs this against real data.

Usage (train venv, repo root):
  .venv-train/bin/python -m blue.training.op2_dev_analysis \
      --g1-report blue/results/op2_jsrl_guide_v1/guard_remeasure/report.json \
      --label jsrl --eval-json blue/results/op2_jsrl_guide_v1/jsrl_s0/eval_dev.json \
      --label jsrl --eval-json blue/results/op2_jsrl_guide_v1/jsrl_s1/eval_dev.json \
      --label jsrl --eval-json blue/results/op2_jsrl_guide_v1/jsrl_s2/eval_dev.json \
      --label noguide --eval-json blue/results/op2_cuda_guide_v1/noguide_s*/eval_dev.json \
      --out blue/results/op2_jsrl_guide_v1/dev_analysis.json
"""
import argparse
import json
import os

import numpy as np

RNG_SEED = 20261022
N_BOOT = 10000


def mean_across_evals(cells):
    """cells: list of {seed(str): return} dicts -> {seed: mean}."""
    out = {}
    keys = set.intersection(*(set(c) for c in cells))
    for k in keys:
        out[k] = float(np.mean([c[k] for c in cells]))
    return out


def paired_boot(a, b, rng):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boots[i] = d[idx].mean()
    return {
        "paired_mean": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)),
                 float(np.percentile(boots, 97.5))],
        "n": n,
        "resamples": N_BOOT,
    }


def main():
    ap = argparse.ArgumentParser(description="OP2 frozen dev analysis")
    ap.add_argument("--g1-report", required=True)
    ap.add_argument("--label", action="append", required=True,
                    help="arm label; repeat with matching --eval-json")
    ap.add_argument("--eval-json", action="append", required=True)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    assert len(cli.label) == len(cli.eval_json), (
        "--label and --eval-json must pair up")

    with open(cli.g1_report) as f:
        g1 = json.load(f)
    guard_cells = {str(s): float(c["return"])
                   for s, c in g1["cells"]["guard"].items()}

    labels = list(dict.fromkeys(cli.label))
    arm_cells = {}
    for lab in labels:
        cells = [json.load(open(p)) for p, l in zip(cli.eval_json, cli.label)
                 if l == lab]
        learned = []
        for blob in cells:
            learned.append({str(s): float(r["return"])
                            for s, r in blob["results"]["learned"].items()})
        assert learned, f"label {lab}: no learned-arm cells found"
        arm_cells[lab] = mean_across_evals(learned)

    rng = np.random.default_rng(RNG_SEED)
    out = {"rng_seed": RNG_SEED, "g1_report": cli.g1_report,
           "eval_jsons": cli.eval_json, "labels": labels}

    order = ["jsrl"] if "jsrl" in labels else [labels[0]]
    others = [l for l in labels if l not in order]
    flat = []
    for other in order + others:
        shared = sorted(set(arm_cells[other]) & set(guard_cells))
        assert shared, (f"no seed overlap between {other} eval "
                        f"({len(arm_cells[other])} seeds) and G1 guard "
                        f"({len(guard_cells)} seeds)")
        c = paired_boot([arm_cells[other][s] for s in shared],
                        [guard_cells[s] for s in shared], rng)
        out[f"{other}_minus_guard"] = c
        flat.append(f"{other} minus guard: {c['paired_mean']:+.2f} "
                    f"CI95 [{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}] "
                    f"(n={c['n']})")
    if len(labels) >= 2:
        shared = sorted(set(arm_cells["jsrl"]) & set(arm_cells[others[0]]))
        c = paired_boot([arm_cells["jsrl"][s] for s in shared],
                        [arm_cells[others[0]][s] for s in shared], rng)
        out[f"jsrl_minus_{others[0]}"] = c
        flat.append(f"jsrl minus {others[0]}: {c['paired_mean']:+.2f} "
                    f"CI95 [{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}]")
    gate = out[f"{order[0]}_minus_guard"]
    out["gate"] = {
        "tau_prereg": 0.0,
        "point": gate["paired_mean"],
        "passes_tau": bool(gate["paired_mean"] >= 0.0),
        "ci_lower": gate["ci95"][0],
    }
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    with open(cli.out, "w") as f:
        json.dump(out, f, indent=1)
    for line in flat:
        print(line)
    print("gate(point >= tau=0.0):", out["gate"]["passes_tau"],
          "| CI cross-zero:", gate["ci95"][0] < 0.0 <= gate["ci95"][1])
    print("wrote", cli.out)


if __name__ == "__main__":
    main()
