"""Parity: HybridBluePolicy(None) must equal RoundRobinBaseline exactly.

Proposal 01 requires the hybrid's fixed rules + default sweep to reproduce
the teacher, so any future learned priority_fn is measurable in isolation.
Runs both policies on fixed seeds and asserts identical native returns and
identical per-tick action traces (agent, action, host).

Seeds here are a fast regression subset; the full >=8-seed parity manifest
is recorded in docs/proposals/01-hybrid-priority.md.
"""
from blue.core.baselines import RoundRobinBaseline, evaluate_policies
from blue.policies.hybrid import HybridBluePolicy

PARITY_SEEDS = (7629, 7640)
STEPS = 400


def _trace_key(trace):
    return [(t["step"], t["agent"], t["action"], t["host"]) for t in trace]


def test_hybrid_none_matches_round_robin():
    results = evaluate_policies(
        {"round_robin": RoundRobinBaseline,
         "hybrid_none": lambda: HybridBluePolicy(priority_fn=None)},
        seeds=PARITY_SEEDS,
        steps=STEPS,
    )
    for seed in PARITY_SEEDS:
        rr = results["round_robin"][seed]
        hy = results["hybrid_none"][seed]
        assert hy["cumulative_return"] == rr["cumulative_return"], seed
        assert _trace_key(hy["trace"]) == _trace_key(rr["trace"]), seed
