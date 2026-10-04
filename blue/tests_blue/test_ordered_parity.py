"""Phase 1 acceptance: clean OrderedPolicy reproduces the reference.

Bit-exact trace equality (step, agent, action, host) against
HybridBluePolicy on full episodes, for both the parity and the
lancer-values configurations. Time-box note: if these fail past the
budget, the fallback is wrapping the reference rule code.
"""

from blue.core.baselines import run_episode
from blue.policies.eval_parallel import trace_sha
from blue.policies.hybrid import HybridBluePolicy
from blue.policies.ordered import CursorSweep, LancerValues, OrderedPolicy

ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}
SEEDS = [7629, 7630]


def _traces(policy, seed):
    return run_episode(policy, seed=seed, steps=400, **ENV_KW)["trace"]


def test_parity_reproduces_reference():
    for seed in SEEDS:
        ref = _traces(HybridBluePolicy(priority_fn=None), seed)
        new = _traces(OrderedPolicy(order_fn=CursorSweep()), seed)
        assert trace_sha(new) == trace_sha(ref)


def test_lancer_values_reproduce_reference():
    for seed in SEEDS:
        ref = _traces(HybridBluePolicy(priority_fn="lancer",
                                       priority_kwargs={
                                           "fruitless_decay": 0.5}),
                      seed)
        new = _traces(OrderedPolicy(scorer=LancerValues(
            fruitless_decay=0.5)), seed)
        assert trace_sha(new) == trace_sha(ref)


def test_zero_residual_hook_reproduces_reference():
    """A zero-head residual in the hook path (guard off) must reproduce
    the lancer trace hash exactly. Catches tie-break divergence between
    reference max-by-(score, host) and first-candidate argmax on
    early-episode init ties. Train venv only (torch)."""
    torch = __import__("pytest").importorskip(
        "torch", reason="residual needs torch")
    from blue.training.residual import build_nets
    from blue.training.residual_pilot import GreedyHook
    residual, _ = build_nets()
    for p in residual.parameters():
        p.detach().zero_()
    hook = GreedyHook(residual, bonus=1.0)
    pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                        guard=False)
    pol.hook = hook
    for seed in SEEDS:
        ref = _traces(HybridBluePolicy(priority_fn="lancer",
                                       priority_kwargs={
                                           "fruitless_decay": 0.5}),
                      seed)
        # Fresh policy per seed (hook holds no state, but reset anyway).
        pol.reset()
        pol.hook = hook
        new = _traces(pol, seed)
        assert trace_sha(new) == trace_sha(ref), f"seed={seed}"
