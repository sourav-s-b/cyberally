"""Parallel harness correctness: pool == serial, cells deterministic.

Uses short horizons for speed; the full >=8-seed parity manifest is produced
by blue_eval_parallel.py and recorded in docs/proposals/.
The torch-spec test never loads a checkpoint: under the sim venv the build
must fail loudly at the torch import; under the train venv it must fail on
the bogus ckpt path instead of silently mis-scoring.
"""
import pytest

from blue_eval_parallel import build_cells, run_pool, trace_sha
from blue_baselines import RoundRobinBaseline, evaluate_policies
from blue_hybrid import HybridBluePolicy
from blue_policy_registry import build, needs_torch

SEEDS = (7629, 7640)
STEPS = 25


def _serial_results():
    return evaluate_policies(
        {"round_robin": RoundRobinBaseline,
         "hybrid_none": lambda: HybridBluePolicy(priority_fn=None)},
        seeds=SEEDS, steps=STEPS, snapshot_steps=(STEPS,))


def test_pool_matches_serial():
    serial = _serial_results()
    cells = build_cells(["round_robin", "hybrid_none"], {}, list(SEEDS),
                        STEPS, "validity", {})
    pooled = run_pool(cells, 2, "fork")
    by_policy = {}
    for r in pooled:
        by_policy.setdefault(r["policy"], {})[r["seed"]] = r
    assert set(by_policy) == {"round_robin", "hybrid_none"}
    for seed in SEEDS:
        s = serial["round_robin"][seed]
        rr, hy = by_policy["round_robin"][seed], by_policy["hybrid_none"][seed]
        assert rr["return"] == s["cumulative_return"] == hy["return"] == \
            serial["hybrid_none"][seed]["cumulative_return"]
        assert rr["trace_sha256"] == hy["trace_sha256"]


def test_cross_worker_determinism():
    cells = build_cells(["round_robin"], {}, [7629], STEPS, "validity", {})
    first = run_pool(cells, 1, "fork")[0]
    second = run_pool(cells, 1, "fork")[0]
    assert first["trace_sha256"] == second["trace_sha256"]
    assert first["return"] == second["return"]


def test_registry_builds_heuristics():
    assert needs_torch("mappo_ckpt") and not needs_torch("round_robin")
    assert isinstance(build("hybrid_none"), HybridBluePolicy)


def test_torch_spec_fails_loudly_without_checkpoint():
    with pytest.raises((ImportError, FileNotFoundError, AssertionError,
                        OSError)):
        build("mappo_ckpt", ckpt_dir="/nonexistent-ckpt-dir")
