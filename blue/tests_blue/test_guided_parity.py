"""Bit-exact parity: guided.py reproduces hybrid_lancer_v2 (Phase 1 gate).

Runs both policies through run_episode on fixed seeds and compares
canonical trace hashes. Any divergence is a bug in guided.py. Uses the
sim venv (no torch).
"""

from blue.core.baselines import run_episode
from blue.policies.eval_parallel import trace_sha
from blue.policies.guided import GuidedBluePolicy, GuidedPriority
from blue.policies.hybrid import HybridBluePolicy

ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}


def _trace(factory, seed, steps=400):
    run = run_episode(factory(), seed=seed, steps=steps,
                      snapshot_steps=(), **ENV_KW)
    return trace_sha(run["trace"]), run["cumulative_return"]


def _factories():
    ref = lambda: HybridBluePolicy(priority_fn="lancer",  # noqa: E731
                                   priority_kwargs={"fruitless_decay": 0.5})
    new = lambda: GuidedBluePolicy(GuidedPriority())  # noqa: E731
    none_ref = lambda: HybridBluePolicy(priority_fn=None)  # noqa: E731
    none_new = lambda: GuidedBluePolicy(priority_fn=None)  # noqa: E731
    return [(ref, new), (none_ref, none_new)]


def test_guided_reproduces_lancer_v2_traces():
    for ref, new in _factories():
        for seed in (7629, 7630, 7702):
            h_ref, r_ref = _trace(ref, seed)
            h_new, r_new = _trace(new, seed)
            assert h_new == h_ref, f"trace diverged seed={seed}"
            assert r_new == r_ref, f"return diverged seed={seed}"


def test_guided_scorer_matches_lancer_values():
    """Per-call scorer equality on a live episode prefix (catches drift
    in dynamics even when traces happen to agree)."""
    import blue.core.wrapper as wrapper
    env = wrapper.CC4MARLEnv(seed=7629, steps=400, **ENV_KW)
    env.reset(seed=7629)
    from blue.policies.hybrid import LancerPriority
    a = LancerPriority(fruitless_decay=0.5)
    b = GuidedPriority()
    for agent in wrapper.BLUE_AGENTS:
        for host in env.hostnames[agent][:6]:
            assert a(env, agent, host) == b(env, agent, host)
    env.close()
