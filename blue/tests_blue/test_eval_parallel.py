"""Parallel harness correctness: pool == serial, cells deterministic.

Uses short horizons for speed; the full >=8-seed parity manifest is produced
by blue.policies.eval_parallel and recorded in docs/proposals/.
The torch-spec test never loads a checkpoint: under the sim venv the build
must fail loudly at the torch import; under the train venv it must fail on
the bogus ckpt path instead of silently mis-scoring.
"""
import pytest

from blue.policies.eval_parallel import build_cells, run_pool, trace_sha, _cached_build
from blue.core.baselines import RoundRobinBaseline, evaluate_policies, run_episode
from blue.policies.hybrid import HybridBluePolicy
from blue.policies.registry import build, needs_torch

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


def test_spawn_matches_serial():
    # Spawn path (used for torch cells): same numbers as the reference.
    serial = _serial_results()
    cells = build_cells(["round_robin", "hybrid_none"], {}, list(SEEDS),
                        STEPS, "validity", {})
    by_policy = {}
    for r in run_pool(cells, 2, "spawn"):
        by_policy.setdefault(r["policy"], {})[r["seed"]] = r
    for seed in SEEDS:
        assert (by_policy["round_robin"][seed]["return"]
                == serial["round_robin"][seed]["cumulative_return"])
        assert (by_policy["round_robin"][seed]["trace_sha256"]
                == by_policy["hybrid_none"][seed]["trace_sha256"])


def test_registry_builds_heuristics():
    assert needs_torch("mappo_ckpt") and not needs_torch("round_robin")
    assert isinstance(build("hybrid_none"), HybridBluePolicy)


def test_torch_spec_fails_loudly_without_checkpoint():
    with pytest.raises((ImportError, FileNotFoundError, AssertionError,
                        OSError)):
        build("mappo_ckpt", ckpt_dir="/nonexistent-ckpt-dir")


def test_pool_policy_cache_reuse_is_identical():
    # Same worker reusing one built policy across episodes must match fresh
    # builds exactly (run_episode resets RNG/hidden state per episode).
    p1 = _cached_build("round_robin", {})
    p2 = _cached_build("round_robin", {})
    assert p1 is p2
    r1 = run_episode(p1, seed=7629, steps=STEPS, snapshot_steps=(STEPS,))
    r2 = run_episode(p2, seed=7629, steps=STEPS, snapshot_steps=(STEPS,))
    assert r1["cumulative_return"] == r2["cumulative_return"]
    p3 = _cached_build("hybrid_none", {})
    assert p3 is not p1  # different specs do not collide


def test_pool_result_cache_serves_identical(tmp_path, monkeypatch):
    # Second identical pool run is served from the persisted cache with
    # bit-identical numbers; disabling the cache reruns everything.
    # Cache path is isolated so repo state never affects this test.
    import blue.policies.eval_parallel as par
    monkeypatch.setattr(par, "_CACHE_PATH",
                        str(tmp_path / "pool_cells.json"))
    cells = build_cells(["round_robin"], {}, [7701], STEPS, "validity", {})
    first = run_pool(cells, 1, "fork")
    assert first[0]["cached"] is False
    second = run_pool(cells, 1, "fork")
    assert second[0]["cached"] is True
    assert second[0]["return"] == first[0]["return"]
    assert second[0]["trace_sha256"] == first[0]["trace_sha256"]
    third = run_pool(cells, 1, "fork", use_cache=False)
    assert third[0]["cached"] is False
    assert third[0]["return"] == first[0]["return"]
