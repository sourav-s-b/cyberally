"""Phase 5 development comparison matrix with corrected metrics.

Replaces the ad-hoc per-script loops used for the 2026-10-04 ordering and
guard measurements. One harness, one cell definition, one metric
implementation (:mod:`blue.analysis.metrics` via
:mod:`blue.analysis.recorder`), so a difference between arms can only come
from the arm.

Design commitments
------------------
- **Identical everything else.** Every arm is an ``OrderedPolicy`` with the
  same rules 1-3, the same env kwargs, the same episode horizon, the same
  native reward and the same validity masks. The only difference between
  arms is the sweep ordering/guard. (Unguarded ``OrderedPolicy`` is
  bit-exact with ``HybridBluePolicy``, proven in
  ``tests_blue/test_ordered_parity.py``.)
- **No silent overwrites.** The expected cell set is the cartesian product
  of arms x seeds; a duplicate seed in the request, a missing result or an
  unexpected result key is a hard error.
- **Interruption-safe.** Cells stream to a JSONL file as they complete, so
  a kill mid-matrix keeps every finished cell.
- **One interval method, fixed in advance.** Paired-seed percentile
  bootstrap over seed indices (``BOOTSTRAP_N`` resamples, fixed RNG seed,
  ``ddof=1`` sample sd). Each resample draws whole seed indices, so pairs
  are never broken.
- These are DEVELOPMENT seeds already consumed by earlier phases. The
  intervals are **exploratory**: they support engineering decisions, not a
  final claim.

Usage (repo root):
  .venv/bin/python -m blue.analysis.compare --run-id matrix-a \\
      --arms unguarded_lancer strict_lancer strict_oracle \\
             maxage_lancer maxage_oracle unguarded_oracle \\
      --seeds-file docs/proposals/manifests/ordering-phase2-20261004.json \\
      --max-age 24 --workers 6 --out blue/results/matrix-a
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import warnings
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MANIFEST_DIR = os.path.join(REPO_ROOT, "docs", "proposals", "manifests")

# The pristine simulator imports legacy gym, which prints an unmaintained-
# library notice once per worker process. It is not actionable here and it
# buries the actual cell lines in the streamed log.
warnings.filterwarnings("ignore")

ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}

# Fallback threshold only; ``maxage<N>_*`` arms pin their own. The value
# used for the headline arms is decided from the unguarded baseline's measured
# age profile (see docs/proposals/manifests/guard-maxage-20261004.json) BEFORE
# any guarded return is computed.
DEFAULT_MAX_AGE = 48.0
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261004
CI_LEVEL = 0.95
ADVANCE_THRESHOLD = 5.0

# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------
def arm_threshold(arm, default):
    """``maxage48_lancer`` pins its own threshold; plain ``maxage_*`` uses
    ``--max-age``. Threshold names keep the threshold in the cell key, so two
    thresholds can never overwrite each other's results."""
    if arm.startswith("maxage") and arm != "maxage":
        rest = arm[len("maxage"):]
        head, sep, tail = rest.partition("_")
        if sep and head.replace(".", "", 1).isdigit():
            return float(head)
    return default


def build_arm(name, max_age, steps):
    """Return a zero-arg policy factory for ``name`` (fork-picklable)."""
    default = max_age if max_age is not None else DEFAULT_MAX_AGE
    a = float(arm_threshold(name, default))
    kinds = {"unguarded_lancer": ("lancer", "off"),
             "unguarded_oracle": ("oracle", "off"),
             "unguarded_oldest": ("oldest", "off"),
             "unguarded_obsbump": ("obsbump", "off"),
             "strict_lancer": ("lancer", "strict"),
             "strict_oracle": ("oracle", "strict"),
             "strict_oldest": ("oldest", "strict")}
    if name in kinds:
        kind, mode = kinds[name]
    elif name.startswith("maxage") and "_" in name:
        kind = name.split("_", 1)[1]
        mode = "max_age"
    else:
        raise KeyError(name)
    return lambda: _ordered(kind, guard=(mode == "strict"),
                            max_age=(a if mode == "max_age" else None))


def _ordered(kind, guard=False, max_age=None):
    from blue.analysis.ordering import ObsBump, OldestFirst, OracleOnset
    from blue.policies.ordered import CursorSweep, LancerValues, OrderedPolicy
    if kind == "lancer":
        scorer = LancerValues(fruitless_decay=0.5)
        return OrderedPolicy(scorer=scorer, guard=guard, max_age=max_age)
    if kind == "oracle":
        return OrderedPolicy(scorer=OracleOnset(), guard=guard, max_age=max_age)
    if kind == "oldest":
        return OrderedPolicy(scorer=OldestFirst(), guard=guard,
                             max_age=max_age)
    if kind == "obsbump":
        return OrderedPolicy(scorer=ObsBump(), guard=guard, max_age=max_age)
    if kind == "parity":
        return OrderedPolicy(order_fn=CursorSweep(), guard=guard)
    raise KeyError(kind)


# --------------------------------------------------------------------------
# Cell execution (module level so it is picklable)
# --------------------------------------------------------------------------
def run_cell(cell):
    """One (arm, seed) cell. No torch, so it is fork-picklable."""
    from blue.analysis.metrics import summarize
    from blue.analysis.recorder import run_measured_episode
    from blue.policies.eval_parallel import trace_sha
    from blue.policies.ordered import completed_investigation_age_summary
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    factory = build_arm(cell["arm"], cell.get("max_age"), cell["steps"])
    policy = factory()
    box = {}
    res = run_measured_episode(policy, seed=cell["seed"],
                               steps=cell["steps"], env_kwargs=ENV_KW,
                               on_decision=lambda rec, env, actions:
                               box.__setitem__("env", env))
    m = summarize(res["timeline"], extra={"arm": cell["arm"],
                                          "seed": cell["seed"]})
    env = box.get("env")
    ages = ({} if env is None
            else {a: completed_investigation_age_summary(env, a)
                  for a in env.hostnames})
    return {"arm": cell["arm"], "seed": cell["seed"], "steps": cell["steps"],
            "return": res["return"], "end_tick": res["steps"],
            "trace_sha256": trace_sha(res["trace"]),
            "privileged_calls": res["privileged_calls"],
            "guard_stats": (policy.guard_stats()
                            if hasattr(policy, "guard_stats") else None),
            "guard_log": (None if policy.age_guard is None
                          else policy.age_guard.log[:50]),
            "completed_investigation_ages": ages,
            "metrics": m}


# --------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------
def paired_bootstrap_ci(diffs, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED,
                        level=CI_LEVEL):
    """Percentile bootstrap over SEED INDICES.

    ``diffs`` is already the per-seed paired difference, so resampling
    indices keeps every pair intact. Method fixed in advance.
    """
    n = len(diffs)
    if n == 0:
        return (None, None)
    if n == 1:
        return (float(diffs[0]), float(diffs[0]))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha := (1.0 - level) / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha) * n_boot))]
    return (lo, hi)


def paired_stats(base, arm, level=CI_LEVEL):
    """``base``/``arm``: {seed: value}. Only complete pairs are used, and
    the dropped seeds are reported, never silently ignored."""
    b, a = base, arm
    shared = sorted(set(b) & set(a))
    missing = {"in_base_not_arm": sorted(set(b) - set(a)),
               "in_arm_not_base": sorted(set(a) - set(b))}
    diffs = [a[s] - b[s] for s in shared]
    if not diffs:
        return {"n_matched": 0, "missing": missing}
    sd = statistics.stdev(diffs) if len(diffs) > 1 else None
    lo, hi = paired_bootstrap_ci(diffs, level=level)
    ordered = sorted(zip(shared, diffs), key=lambda sd_: sd_[1])
    return {
        "n_matched": len(shared),
        "missing": missing,
        "mean_diff": statistics.fmean(diffs),
        "sd_diff_ddof1": sd,
        "median_diff": statistics.median(diffs),
        "ci95_low": lo, "ci95_high": hi,
        "ci_method": (f"paired-seed percentile bootstrap, n={BOOTSTRAP_N}, "
                      f"rng_seed={BOOTSTRAP_SEED}, level={level}"),
        "wins": sum(1 for d in diffs if d > 0),
        "ties": sum(1 for d in diffs if d == 0),
        "losses": sum(1 for d in diffs if d < 0),
        "worst_3": [{"seed": s, "diff": d} for s, d in ordered[:3]],
        "best_3": [{"seed": s, "diff": d} for s, d in ordered[-3:][::-1]],
        "per_seed_diff": {str(s): d for s, d in zip(shared, diffs)},
    }


def verdict(diag, threshold=ADVANCE_THRESHOLD):
    """Three-outcome advancement rule for the CURRENT experiment."""
    lo, hi = diag.get("ci95_low"), diag.get("ci95_high")
    if lo is None:
        return "inconclusive", "no interval"
    if lo > threshold:
        return "advance", f"lower bound {lo:.1f} > +{threshold}"
    if hi < threshold:
        return "stop", f"upper bound {hi:.1f} < +{threshold}"
    return "inconclusive", f"interval [{lo:.1f},{hi:.1f}] spans +{threshold}"


def metric_aggregate(rows, path):
    """Mean over cells of a numeric metric addressed by a dotted path."""
    vals = []
    for r in rows:
        node = r
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, (int, float)):
            vals.append(float(node))
    if not vals:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(vals), "mean": statistics.fmean(vals),
            "median": statistics.median(vals)}


# --------------------------------------------------------------------------
# Matrix driver
# --------------------------------------------------------------------------
def load_seeds(path, count=None):
    """Verify the historical seed list rather than reconstructing it."""
    with open(path) as f:
        man = json.load(f)
    seeds = [int(s) for s in man["seeds"]]
    if len(set(seeds)) != len(seeds):
        dupes = sorted({s for s in seeds if seeds.count(s) > 1})
        raise ValueError(f"duplicate seeds in {path}: {dupes}")
    if count is not None and len(seeds) != count:
        raise ValueError(f"{path} has {len(seeds)} seeds, expected {count}")
    return seeds, man


def run_matrix(arms, seeds, max_age, steps, out_dir, workers,
               start_method="fork"):
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "cells.jsonl")
    done = {}
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[(rec["arm"], rec["seed"])] = rec
    cells = [{"arm": a, "seed": s, "steps": steps,
              "max_age": arm_threshold(a, max_age)}
             for a in arms for s in seeds]
    expected = {(c["arm"], c["seed"]) for c in cells}
    if len(expected) != len(cells):
        raise ValueError("duplicate (arm, seed) cells requested")
    todo = [c for c in cells if (c["arm"], c["seed"]) not in done]
    print(f"matrix: {len(cells)} cells, {len(done)} cached, {len(todo)} to run",
          flush=True)
    t0 = time.time()
    if todo:
        ctx = mp.get_context(start_method)
        with cf.ProcessPoolExecutor(max_workers=workers,
                                    mp_context=ctx) as pool:
            futs = [pool.submit(run_cell, c) for c in todo]
            with open(jsonl, "a") as sink:
                for n, fut in enumerate(cf.as_completed(futs), 1):
                    rec = fut.result()
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                    done[(rec["arm"], rec["seed"])] = rec
                    el = time.time() - t0
                    eta = el / n * (len(todo) - n)
                    print(f"cell {n}/{len(todo)} {rec['arm']} "
                          f"seed={rec['seed']} return={rec['return']:+.1f} "
                          f"cov={rec['metrics']['coverage']:.2f} "
                          f"t={el:.0f}s eta={eta:.0f}s", flush=True)
    missing = sorted(expected - set(done))
    extra = sorted(set(done) - expected)
    if missing or extra:
        raise RuntimeError(f"incomplete matrix: missing={missing[:5]} "
                           f"extra={extra[:5]}")
    print(f"matrix complete: {len(done)} cells in {time.time()-t0:.0f}s",
          flush=True)
    return done


def report(arms, done, max_age, steps, seeds, out_dir, run_id,
           baseline="unguarded_lancer", threshold=ADVANCE_THRESHOLD):
    per_arm = {}
    for a in arms:
        rows = [done[(a, s)] for s in seeds]
        returns = {r["seed"]: r["return"] for r in rows}
        per_arm[a] = {
            "n_seeds": len(rows),
            "mean_return": statistics.fmean(returns.values()),
            "sd_return_ddof1": statistics.stdev(returns.values()),
            "median_return": statistics.median(returns.values()),
            "min_return": min(returns.values()),
            "max_return": max(returns.values()),
            "returns": {str(k): v for k, v in sorted(returns.items())},
            "coverage": metric_aggregate(rows, "metrics.coverage"),
            "n_never_investigated": metric_aggregate(
                rows, "metrics.n_never_investigated_hosts"),
            "post_compromise_investigation_delay_median":
                metric_aggregate(
                    rows, "metrics.post_compromise_investigation_delay.median"),
            "post_compromise_investigation_delay_n": metric_aggregate(
                rows, "metrics.post_compromise_investigation_delay.n"),
            "post_compromise_investigation_delay_censored":
                metric_aggregate(
                    rows, "metrics.post_compromise_investigation_delay.n_censored"),
            "detection_delay_median": metric_aggregate(
                rows, "metrics.detection_delay.median"),
            "detection_delay_n": metric_aggregate(
                rows, "metrics.detection_delay.n"),
            "detection_delay_censored": metric_aggregate(
                rows, "metrics.detection_delay.n_censored"),
            "end_of_episode_age_median": metric_aggregate(
                rows, "metrics.end_of_episode_age.median"),
            "end_of_episode_age_max": metric_aggregate(
                rows, "metrics.end_of_episode_age.max"),
            "max_age_during_episode_median": metric_aggregate(
                rows, "metrics.max_age_during_episode.median"),
            "max_age_during_episode_max": metric_aggregate(
                rows, "metrics.max_age_during_episode.max"),
            "n_compromise_episodes": metric_aggregate(
                rows, "metrics.n_compromise_episodes"),
            "detections_observed": metric_aggregate(
                rows, "metrics.detections_observed"),
            "investigations_completed": metric_aggregate(
                rows, "metrics.investigations_completed"),
            "investigations_failed": metric_aggregate(
                rows, "metrics.investigations_failed"),
            "guard": _guard_summary(rows),
        }
    base = {r["seed"]: r["return"] for r in (done[(baseline, s)]
                                             for s in seeds)}
    comparisons = {}
    for a in arms:
        if a == baseline:
            continue
        arm = {r["seed"]: r["return"] for r in (done[(a, s)] for s in seeds)}
        st = paired_stats(base, arm)
        st["verdict_vs_unguarded_lancer"] = dict(
            zip(("outcome", "reason"), verdict(st, threshold)))
        comparisons[f"{a}_vs_{baseline}"] = st
    # oracle vs lancer INSIDE each guard regime (separates guard from scorer).
    # Derived from the arm names actually run: ``<guardmode>_<kind>`` with the
    # guard mode being unguarded / strict / maxage<N>.
    within = {}
    modes = {}
    for a in arms:
        mode, _, kind = a.partition("_")
        modes.setdefault((mode, kind), a)
    for (mode, kind), arm_a in sorted(modes.items()):
        if kind != "lancer":
            continue
        peer = modes.get((mode, "oracle"))
        if peer is None or peer not in per_arm:
            continue
        base_ret = {s: done[(arm_a, s)]["return"] for s in seeds}
        peer_ret = {s: done[(peer, s)]["return"] for s in seeds}
        st = paired_stats(base_ret, peer_ret)
        st["arms"] = [arm_a, peer]
        within[f"oracle_vs_lancer__{mode}"] = st
    doc = {
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "dirty": _git_dirty(),
        "steps": steps, "env": {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in ENV_KW.items()},
        "mask_mode": "validity (wrapper default)",
        "reward": "native simulator reward, common_reward normalization "
                  "(rewards[0]); no shaping",
        "arms": arms, "seeds": seeds, "n_seeds": len(seeds),
        "max_age_threshold_by_arm": {a: arm_threshold(a, max_age)
                                      for a in arms},
        "baseline_arm": baseline,
        "advance_threshold": threshold,
        "interval_method": f"paired-seed percentile bootstrap n={BOOTSTRAP_N} "
                           f"rng_seed={BOOTSTRAP_SEED}",
        "exploratory": True,
        "seed_block": "development seeds already consumed by earlier phases; "
                      "no final/held-out block touched",
        "per_arm": per_arm,
        "comparisons_vs_unguarded_lancer": comparisons,
        "within_regime_oracle_vs_lancer": within,
    }
    path = os.path.join(out_dir, "report.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return doc, path


def _guard_summary(rows):
    keys = ("guard_decisions", "interventions", "deferrals_urgent",
            "overdue_host_decisions", "max_overdue_hosts",
            "max_threshold_overshoot", "repeated_selection",
            "revisit_selections", "sweep_selections")
    out = {}
    for k in keys:
        vals = [r["guard_stats"][k] for r in rows
                if r.get("guard_stats")]
        out[k] = ({"mean": statistics.fmean(vals),
                   "max": max(vals), "n_cells": len(vals)} if vals
                  else None)
    ages = []
    for r in rows:
        for a in (r.get("completed_investigation_ages") or {}).values():
            if a.get("median") is not None:
                ages.append(float(a["median"]))
    out["completed_investigation_age_median_mean"] = (
        statistics.fmean(ages) if ages else None)
    return out


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse",
                                        "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty():
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "-C", REPO_ROOT, "status", "--porcelain"],
            text=True).strip()
        import hashlib
        return {"dirty": bool(out), "diff_sha256":
                hashlib.sha256(out.encode()).hexdigest()[:16],
                "files": out.splitlines()[:40]}
    except Exception:
        return {"dirty": None}


def main():
    ap = argparse.ArgumentParser(description="Phase 5 comparison matrix")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--seeds-file",
                    default=os.path.join(MANIFEST_DIR,
                                         "ordering-phase2-20261004.json"),
                    help="manifest whose exact seed list must be reused")
    ap.add_argument("--expect-seeds", type=int, default=32)
    ap.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default="unguarded_lancer")
    ap.add_argument("--threshold", type=float, default=ADVANCE_THRESHOLD)
    cli = ap.parse_args()

    if cli.seeds:
        seeds = list(cli.seeds)
        if len(set(seeds)) != len(seeds):
            raise SystemExit("duplicate seeds requested")
    else:
        seeds, _ = load_seeds(cli.seeds_file, count=cli.expect_seeds)
    out_dir = cli.out or os.path.join(REPO_ROOT, "blue", "results", cli.run_id)
    done = run_matrix(cli.arms, seeds, cli.max_age, cli.steps, out_dir,
                      cli.workers)
    doc, path = report(cli.arms, done, cli.max_age, cli.steps, seeds,
                       out_dir, cli.run_id, baseline=cli.baseline,
                       threshold=cli.threshold)
    for arm, s in doc["per_arm"].items():
        print(f"{arm:20s} mean={s['mean_return']:+8.2f} "
              f"sd={s['sd_return_ddof1']:6.2f} "
              f"cov={s['coverage']['mean']:.3f} "
              f"age_end_med={s['end_of_episode_age_median']['mean']}")
    print("--- vs unguarded_lancer ---")
    for name, st in doc["comparisons_vs_unguarded_lancer"].items():
        print(f"{name:34s} n={st['n_matched']} "
              f"mean={st['mean_diff']:+7.2f} "
              f"sd={st['sd_diff_ddof1']:6.2f} "
              f"CI[{st['ci95_low']:+7.2f},{st['ci95_high']:+7.2f}] "
              f"W/T/L={st['wins']}/{st['ties']}/{st['losses']} "
              f"-> {st['verdict_vs_unguarded_lancer']['outcome']}")
    print("wrote", path)


if __name__ == "__main__":
    main()