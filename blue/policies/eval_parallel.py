"""Parallel (policy x seed) evaluation with committed manifests.

Each cell runs ``run_episode`` in its own process: the env is constructed
in-child (never shared), with an explicit seed, so concurrent episodes are
independent (simulator verified process-safe; randomness is seed-derived).
Heuristic cells run under the current interpreter with fork; any torch
cell forces spawn and requires the train venv (torch + EPyMARL link).

Serial ``evaluate_policies`` stays the reference implementation; see
tests_blue/test_eval_parallel.py for the equivalence check.

Manifests go to docs/proposals/manifests/<run-id>.json (committable;
results/ is gitignored). Manifests store returns + trace hashes, not full
traces; full-trace equality stays in pytest.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MANIFEST_DIR = os.path.join(_REPO_ROOT, "docs", "proposals", "manifests")

# Persisted cell-result cache (Phase C): results keyed by everything that
# affects the number — policy spec, checkpoint bytes, code files, env
# config, seed. Re-running teacher baselines across experiments then costs
# zero cells. Cache lives in blue/.cache/ (gitignored); any code/ckpt
# change alters the key, so stale hits require identical bytes.
_CACHE_DIR = os.path.join(_REPO_ROOT, "blue", ".cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "pool_cells.json")
# Source files whose bytes affect cell results (policies, harness, wrapper).
_CACHE_SOURCES = ("blue/policies/eval_parallel.py",
                  "blue/core/baselines.py",
                  "blue/policies/registry.py", "blue/core/wrapper.py",
                  "blue/policies/hybrid.py", "blue/policies/eval_mappo.py",
                  "blue/policies/eval_rvs.py")


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cell_key(cell):
    """Content hash identifying a cell's result. Any change that could
    alter the number (code, ckpt, config, seed) changes the key."""
    parts = [cell["policy"], str(cell["seed"]), str(cell["steps"]),
             cell.get("mask_mode", "validity"),
             json.dumps(cell.get("env_kwargs", {}), sort_keys=True,
                        default=str),
             json.dumps(cell.get("policy_kwargs", {}), sort_keys=True,
                        default=str)]
    srcs = list(_CACHE_SOURCES)
    kwargs = cell.get("policy_kwargs", {})
    if "ckpt_dir" in kwargs:
        ckpt = os.path.join(kwargs["ckpt_dir"], "agent.th")
        if os.path.exists(ckpt):
            parts.append("ckpt:" + _file_sha(ckpt))
        else:
            parts.append("ckpt:missing")
    else:
        parts.append("ckpt:heuristic")
        try:
            from blue.policies.registry import REGISTRY
            if cell["policy"] in REGISTRY:
                srcs.append(REGISTRY[cell["policy"]][0].replace(".", "/")
                            + ".py")
        except Exception:
            pass
    for src in sorted(set(srcs)):
        p = os.path.join(_REPO_ROOT, src)
        if os.path.exists(p):
            parts.append(src + ":" + _file_sha(p))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _load_cache():
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, _CACHE_PATH)


def _canonical_trace(trace):
    return json.dumps([[t["step"], t["agent"], t["action"], t["host"]]
                       for t in trace], separators=(",", ":"))


def trace_sha(trace):
    return hashlib.sha256(_canonical_trace(trace).encode()).hexdigest()


# Per-worker policy cache (Phase C): ProcessPoolExecutor reuses worker
# processes across cells, so a policy built once serves all its cells in
# that worker. Safe because run_episode() calls policy.reset() at the
# start of every episode, restoring deterministic RNG/hidden state —
# cached reuse is bit-identical to fresh builds (covered by
# test_eval_parallel equivalence + test_pool_policy_cache below).
_POLICY_CACHE = {}


def _cached_build(name, kwargs):
    key = name + "|" + json.dumps(kwargs, sort_keys=True, default=str)
    policy = _POLICY_CACHE.get(key)
    if policy is None:
        from blue.policies.registry import build
        policy = build(name, **kwargs)
        _POLICY_CACHE[key] = policy
    return policy


def run_cell(cell):
    """Execute one (policy, seed) cell. Must stay module-level (picklable)."""
    if _REPO_ROOT not in sys.path:  # spawn re-import: fix path in child
        sys.path.insert(0, _REPO_ROOT)
    from blue.core.baselines import run_episode
    from blue.policies.registry import needs_torch
    kwargs = dict(cell.get("policy_kwargs", {}))
    if needs_torch(cell["policy"]):
        # Size the checkpoint probe (and geometry checks) from the same env
        # flags as the rollout env; _build_ckpt forwards them to the probe.
        # mask_mode included so validity checks run under the same config.
        for key in ("temporal_features", "include_root_session",
                    "mask_mode"):
            val = cell.get("env_kwargs", {}).get(key)
            if val is not None:
                kwargs.setdefault(key, val)
        kwargs.setdefault("steps", cell["steps"])
    policy = _cached_build(cell["policy"], kwargs)
    res = run_episode(policy, seed=cell["seed"], steps=cell["steps"],
                      snapshot_steps=tuple(cell.get("snapshot_steps",
                                                   (cell["steps"],))),
                      mask_mode=cell.get("mask_mode", "validity"),
                      **cell.get("env_kwargs", {}))
    return {"policy": cell["policy"], "seed": cell["seed"],
            "return": res["cumulative_return"], "steps": res["steps"],
            "trace_sha256": trace_sha(res["trace"]),
            "final": res["final"], "snapshots": res["snapshots"]}


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def build_cells(policies, policy_kwargs, seeds, steps, mask_mode, env_kwargs):
    cells = []
    for name in policies:
        for seed in seeds:
            cells.append({"policy": name,
                          "policy_kwargs": dict(policy_kwargs.get(name, {})),
                          "seed": seed, "steps": steps,
                          "snapshot_steps": (min(200, steps), steps),
                          "mask_mode": mask_mode,
                          "env_kwargs": dict(env_kwargs)})
    return cells


def run_pool(cells, workers, start_method, use_cache=True):
    """Run cells, serving content-hash cache hits without a worker.

    Returns results aligned with ``cells``; each carries "cached": bool.
    The persisted cache (blue/.cache/, gitignored) is keyed by code + ckpt
    + config + seed, so hits are bit-identical reruns, not approximations.
    """
    cache = _load_cache() if use_cache else {}
    pending, order, results = [], [], [None] * len(cells)
    hits = 0
    for i, cell in enumerate(cells):
        key = _cell_key(cell)
        if use_cache and key in cache:
            r = dict(cache[key])
            r["cached"] = True
            results[i] = r
            hits += 1
        else:
            order.append(i)
            pending.append(cell)
    if pending:
        from blue.common import logutil
        ctx = mp.get_context(start_method)
        cprog = logutil.Progress(len(pending))
        with cf.ProcessPoolExecutor(max_workers=workers,
                                    mp_context=ctx) as pool:
            futs = {pool.submit(run_cell, cell): pos
                    for pos, cell in enumerate(pending)}
            fresh = [None] * len(pending)
            n_done = 0
            for fut in cf.as_completed(futs):
                pos = futs[fut]
                fresh[pos] = fut.result()
                n_done += 1
                r = fresh[pos]
                print(f"cell {cprog.line(n_done)} "
                      f"{r['policy']} seed={r['seed']} "
                      f"return={r['return']:.1f}", flush=True)
        for pos, (i, r) in enumerate(zip(order, fresh)):
            r["cached"] = False
            results[i] = r
            if use_cache:
                cache[_cell_key(pending[pos])] = {
                    k: r[k] for k in ("policy", "seed", "return", "steps",
                                      "trace_sha256", "final", "snapshots")}
        if use_cache:
            _save_cache(cache)
    print(f"cache: {hits}/{len(cells)} hits", flush=True)
    return results


def write_manifest(run_id, cells, cell_results, steps, mask_mode, env_kwargs,
                   workers, start_method):
    per_policy = {}
    for r in cell_results:
        per_policy.setdefault(r["policy"], {})[str(r["seed"])] = {
            "return": r["return"], "steps": r["steps"],
            "trace_sha256": r["trace_sha256"], "final": r["final"],
            "snapshots": r["snapshots"]}
    manifest = {
        "run_id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(), "steps": steps, "mask_mode": mask_mode,
        "env_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in env_kwargs.items()},
        "seeds": sorted({c["seed"] for c in cells}),
        "policies": sorted({c["policy"] for c in cells}),
        "policy_kwargs": {c["policy"]: c["policy_kwargs"] for c in cells},
        "workers": workers, "start_method": start_method,
        "interpreter": sys.executable, "cells": per_policy}
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    path = os.path.join(MANIFEST_DIR, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return path


def main():
    from blue.policies.registry import names, needs_torch
    ap = argparse.ArgumentParser(description="Parallel policy x seed eval")
    ap.add_argument("--policies", nargs="+", required=True,
                    help=f"registry names {names()}")
    ap.add_argument("--mappo-ckpt", default=None,
                    help="ckpt dir for the mappo_ckpt policy")
    ap.add_argument("--target-return", type=float, default=-50.0,
                    help="RvS conditioning target (rvs_ckpt only, proposal 14)")
    ap.add_argument("--attn-layers", type=int, default=0,
                    help="cross-slot attention layers of the ckpt's head")
    ap.add_argument("--reset-interval", type=int, default=0,
                    help="zero carried GRU hidden every K per-agent steps "
                         "(V6 drift ablation; 1 = memoryless; ckpt policies "
                         "except rvs_ckpt)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[7629, 7630, 7640, 7701, 7702, 7703, 7704, 7705])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mask-mode", default="validity",
                    choices=["validity", "evidence"])
    ap.add_argument("--red-agent", default="discovery",
                    choices=["discovery", "finite", "verbose", "random",
                             "sleep"],
                    help="red behavior variant (proposal 15)")
    ap.add_argument("--temporal-groups", nargs="*",
                    default=["ages", "belief", "freshness", "mission"])
    ap.add_argument("--drop-root-session", action="store_true")
    ap.add_argument("--run-id", default=None)
    cli = ap.parse_args()

    unknown = [p for p in cli.policies if p not in names()]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}; known: {names()}")
    policy_kwargs = {}
    ckpt_policies = [p for p in cli.policies if needs_torch(p)]
    if ckpt_policies:
        if not cli.mappo_ckpt:
            raise SystemExit(f"{ckpt_policies} need --mappo-ckpt DIR")
        for p in ckpt_policies:
            policy_kwargs[p] = {"ckpt_dir": cli.mappo_ckpt,
                                "attn_layers": cli.attn_layers,
                                "reset_interval": cli.reset_interval}
            if p == "rvs_ckpt":
                policy_kwargs[p]["target_return"] = cli.target_return
    use_torch = any(needs_torch(p) for p in cli.policies)
    if use_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise SystemExit(
                "torch cells need the train venv: rerun with "
                "../.venv-train/bin/python -m blue.policies.eval_parallel ...")
        start_method = "spawn"
    else:
        start_method = "fork"
    env_kwargs = {"temporal_features": tuple(cli.temporal_groups),
                  "include_root_session": not cli.drop_root_session,
                  "red_agent": cli.red_agent}
    run_id = cli.run_id or (time.strftime("%Y%m%dT%H%M%S") + "-eval")
    cells = build_cells(cli.policies, policy_kwargs, cli.seeds, cli.steps,
                        cli.mask_mode, env_kwargs)
    workers = max(1, min(cli.workers, len(cells)))
    t0 = time.time()
    results = run_pool(cells, workers, start_method)
    path = write_manifest(run_id, cells, results, cli.steps, cli.mask_mode,
                          env_kwargs, workers, start_method)
    for r in sorted(results, key=lambda r: (r["policy"], r["seed"])):
        print(f"{r['policy']} seed={r['seed']} return={r['return']:.1f} "
              f"trace={r['trace_sha256'][:12]}")
    print(f"{len(results)} cells in {time.time()-t0:.1f}s "
          f"({workers} workers, {start_method}) -> {path}")


if __name__ == "__main__":
    main()
