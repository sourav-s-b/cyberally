"""Agent-4 suspicion hook: Lancer elsewhere, valid picks for agent 4."""

from blue.policies.ordered import LancerValues, OrderedPolicy
from blue.policies.zone_priority import Agent4SuspicionHook


def _run(policy, seed, steps=60, **env_kw):
    from blue.training import mappo_guide as mg
    kw = dict(mg.ENV_KW)
    kw.update(env_kw)
    return mg.run_team_episode(policy, seed, steps, **kw)


def _policies():
    base = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                         guard=False)
    hook = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                         guard=False)
    hook.hook = Agent4SuspicionHook()
    return base, hook


def test_other_agents_unaffected_by_hook():
    """Hook returns Lancer argmax off agent_4: spot-check the delegation."""
    from blue.policies.ordered import argmax_pick
    hook = Agent4SuspicionHook()
    scored = [(3.0, "h1"), (1.0, "h2")]
    assert hook(None, "blue_agent_0", ["h1", "h2"], scored) == \
        argmax_pick(scored)


def test_skill_executes_and_covers():
    base, hook = _policies()
    for seed in (7629, 7701):
        rb = _run(base, seed)
        rh = _run(hook, seed)
        assert rh["ticks"] == rb["ticks"] > 0
        assert 0.0 <= rh["coverage"] <= 1.0
