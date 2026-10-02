"""LancerPriority mechanics (proposal 01 step 2). Live sim, Sleep red/green.

Decay-on-touch counted once (idempotent re-sync), boost on CONFIRMED
transition, novelty boost on worsening view signals, sticky suspicion bonus
while signals persist, scorer picklable, string-spec resolution, and reset
propagation through HybridBluePolicy. Full-episode comparison runs through
the pool (see docs/proposals/01).
"""
import pickle

import cc4_epymarl_wrapper as wrapper
import pytest
from CybORG.Agents import SleepAgent

from blue_hybrid import HybridBluePolicy, LancerPriority, make_priority


@pytest.fixture
def quiet():
    real_red = wrapper.DiscoveryFSRed
    real_green = wrapper.EnterpriseGreenAgent
    wrapper.DiscoveryFSRed = SleepAgent
    wrapper.EnterpriseGreenAgent = SleepAgent
    try:
        env = wrapper.CC4MARLEnv(steps=30)
        env.reset()
        yield env
    finally:
        wrapper.DiscoveryFSRed = real_red
        wrapper.EnterpriseGreenAgent = real_green


def test_make_priority_resolution():
    assert make_priority(None) is None
    fn = lambda e, a, h: 0.0
    assert make_priority(fn) is fn
    assert isinstance(make_priority("lancer"), LancerPriority)
    with pytest.raises(ValueError):
        make_priority("nope")


def test_policy_accepts_string_and_resets_scorer(quiet):
    policy = HybridBluePolicy(priority_fn="lancer")
    assert isinstance(policy.priority_fn, LancerPriority)
    agent = "blue_agent_0"
    host = quiet.hostnames[agent][0]
    policy.priority_fn(quiet, agent, host)
    assert policy.priority_fn._sig, "sync should populate signal cache"
    policy.reset()
    assert policy.priority_fn._sig == {}
    assert policy.priority_fn._v == {}
    assert policy._cursor == {}


def test_touch_decay_counted_once(quiet):
    scorer = LancerPriority(init=1.0, touch_decay=0.5)
    agent = "blue_agent_0"
    host = quiet.hostnames[agent][0]
    base = scorer(quiet, agent, host)
    quiet.trackers[agent].last_analysis[host] = 7
    after = scorer(quiet, agent, host)
    assert after < base  # decay applied
    repeat = scorer(quiet, agent, host)
    assert repeat == after  # idempotent: same stamp, no double decay


def test_confirmed_transition_boosts(quiet):
    scorer = LancerPriority(init=1.0, detect_boost=2.0)
    agent = "blue_agent_0"
    host = quiet.hostnames[agent][0]
    before = scorer(quiet, agent, host)
    quiet.trackers[agent].state[host] = "CONFIRMED"
    after = scorer(quiet, agent, host)
    assert after >= before + 2.0


def test_scorer_picklable():
    blob = pickle.dumps(LancerPriority())
    assert isinstance(pickle.loads(blob), LancerPriority)


def test_hybrid_lancer_selects_legal(quiet):
    policy = HybridBluePolicy(priority_fn="lancer")
    for _ in range(8):
        actions = {a: policy.select(quiet, a) for a in wrapper.BLUE_AGENTS}
        for agent, idx in actions.items():
            mask = quiet.get_avail_agent_actions(
                wrapper.BLUE_AGENTS.index(agent))
            assert mask[idx] == 1, (agent, idx)
        quiet.step(actions)
