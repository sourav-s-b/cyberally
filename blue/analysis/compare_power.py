"""Powered comparison on fresh seeds (review#2 steps 4-5).

Reads *-fresh pool manifests (16 seeds, 8201-8216, paired across all
policies), and reports per algorithm: mean, median, IQM with 95%
algorithm: mean, median, IQM with 95% stratified-bootstrap CIs, and
P(improvement) vs each teacher. Algorithms: BC arm (10 train seeds),
IQL-auto arm (10 train seeds), round_robin, lancer_v2 (1 run x 16 tasks).

Also computes selector regret: for FQE rank and for random choice, the
normalized regret of the best true-mean policy among the top-k ranked
(regret = (best_true - best_of_topk) / range_true). FQE values come from
--fqe-jsonl (blue_fqe_select.py --out on the 20 fresh ckpts); without it,
only the random baseline is reported.

Usage:
  ../.venv-train/bin/python blue/blue_powered_compare.py [--fqe-jsonl FILE]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FRESH_SEEDS = list(range(8201, 8217))
N_BOOT = 2000


def load_fresh(pattern="docs/proposals/manifests/*-fresh-20261003.json",
               root="/home/sourav/Projects/cyberally"):
    arms = {"bc": [], "iql": []}
    teachers = {}
    names = {}
    for f in sorted(glob.glob(os.path.join(root, pattern))):
        m = json.load(open(f))
        for pol, per in m["cells"].items():
            rets = np.array([per[str(s)]["return"] for s in FRESH_SEEDS],
                            dtype=float)
            base = os.path.basename(f)
            if base.startswith("bc-v1mix"):
                k = int(base.split("-s")[1].split("-")[0])
                arms["bc"].append((k, rets))  # raw returns: higher = better
                names[("bc", k)] = base
            elif base.startswith("iql-v2auto"):
                k = int(base.split("-s")[1].split("-")[0])
                arms["iql"].append((k, rets))
                names[("iql", k)] = base
            elif base.startswith("teachers"):
                teachers[pol] = rets
    for a in arms:
        arms[a].sort()
    return arms, teachers, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fqe-jsonl", default=None)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    pass  # estimators reimplemented below; see note in out["note"]

    arms, teachers, names = load_fresh()
    scores = {
        "bc": np.stack([r for _, r in arms["bc"]]),
        "iql_auto": np.stack([r for _, r in arms["iql"]]),
        "round_robin": teachers["round_robin"][None, :],
        "lancer_v2": teachers["hybrid_lancer_v2"][None, :],
    }
    print("shapes (runs x tasks):", {k: v.shape for k, v in scores.items()})

    def iqm(x):
        x = np.sort(np.asarray(x, dtype=float).ravel())
        n = len(x)
        lo, hi = int(np.ceil(0.25 * n)), int(np.floor(0.75 * n))
        return float(x[lo:hi].mean())

    def boot_ci(x, fn, reps=N_BOOT, seed=0):
        """Stratified bootstrap CI: resample RUNS with replacement, keep
        all tasks (== rliable's stratified procedure, implemented here
        because the installed rliable/arch/matplotlib trio is API-broken
        in both directions)."""
        rng = np.random.RandomState(seed)
        x = np.asarray(x, dtype=float)
        ests = [fn(x[rng.randint(0, x.shape[0], x.shape[0])])
                for _ in range(reps)]
        return float(fn(x)), [float(np.percentile(ests, 2.5)),
                              float(np.percentile(ests, 97.5))]

    agg_fns = {"mean": np.mean, "median": np.median, "iqm": iqm}
    out = {"n_boot": N_BOOT, "seeds": FRESH_SEEDS, "algorithms": {},
           "note": "CIs are manual stratified bootstraps (resample runs, "
                   "all tasks); rliable installed but its arch-dependent "
                   "bootstrap path is version-broken, so estimators are "
                   "reimplemented per Agarwal et al. 2021"}
    for algo in scores:
        row = {}
        for fname, fn in agg_fns.items():
            est, ci = boot_ci(scores[algo], fn)
            row[fname] = est
            row[fname + "_ci"] = ci
        n, t = scores[algo].shape
        row["n_runs"] = n
        out["algorithms"][algo] = row
        print(f"{algo:12s} mean {row['mean']:+8.1f} "
              f"[{row['mean_ci'][0]:+.1f},{row['mean_ci'][1]:+.1f}] "
              f"median {row['median']:+8.1f} "
              f"IQM {row['iqm']:+8.1f} "
              f"[{row['iqm_ci'][0]:+.1f},{row['iqm_ci'][1]:+.1f}]")

    # P(improvement): stratified bootstrap (resample runs within each
    # task), fraction of (algo, base) pairs with algo > base -- the
    # Agarwal et al. estimator; this rliable version ships only the
    # point estimate, so the interval is done explicitly here.
    rng = np.random.RandomState(0)
    for base in ("round_robin", "lancer_v2"):
        xb = scores[base]
        for algo in ("bc", "iql_auto"):
            xa = scores[algo]
            ps = []
            for _ in range(N_BOOT):
                ia = rng.randint(0, xa.shape[0], xa.shape[0])
                ib = rng.randint(0, xb.shape[0], xb.shape[0])
                ps.append(float((xa[ia][:, None, :] > xb[ib][None, :, :])
                                .mean()))
            p = float(np.mean(ps))
            ci = [float(np.percentile(ps, 2.5)),
                  float(np.percentile(ps, 97.5))]
            out.setdefault("p_improve", {})[f"{algo}_vs_{base}"] = {
                "p": p, "ci": ci}
            print(f"P({algo} > {base}): {p:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")

    # Selector regret: need true fresh-means per ckpt + FQE rank.
    true_means, labels = {}, {}
    for arm, tag in (("bc", "bc"), ("iql", "iql")):
        for k, r in arms[arm]:
            true_means[(tag, k)] = float(r.mean())
            labels[(tag, k)] = f"{tag}_s{k}"
    keys = sorted(true_means)
    vals = np.array([true_means[k] for k in keys])
    best, worst = vals.max(), vals.min()
    span = (best - worst) or 1.0
    regret = {"true_means": {labels[k]: true_means[k] for k in keys}}
    rng = np.random.RandomState(0)
    for k in (1, 3, 5):
        # Random-choice expected regret: sample subsets.
        draws = np.array([max(rng.choice(vals, size=k, replace=False))
                          for _ in range(5000)])
        regret[f"random_top{k}"] = float(((best - draws) / span).mean())
    if cli.fqe_jsonl:
        fqe = {}
        for line in open(cli.fqe_jsonl):
            r = json.loads(line)
            fqe[r["label"]] = r["v_s0_mean"]
        order = sorted(keys, key=lambda k: -fqe[labels[k]])
        for k in (1, 3, 5):
            top = [true_means[c] for c in order[:k]]
            regret[f"fqe_top{k}"] = float((best - max(top)) / span)
        regret["fqe_rank"] = [labels[c] for c in order]
    for k in (1, 3, 5):
        msg = f"top-{k} regret: random {regret[f'random_top{k}']:.3f}"
        if f"fqe_top{k}" in regret:
            msg += f"  FQE {regret[f'fqe_top{k}']:.3f}"
        print(msg)
    out["regret"] = regret

    if cli.out:
        with open(cli.out, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", cli.out)


if __name__ == "__main__":
    main()
