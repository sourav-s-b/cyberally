"""Strict-guard integration checks for the event-time learner."""

import hashlib
import json

from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
from blue.harness.policy import StrictHarnessPolicy
from blue.policies.ordered import LancerValues, OrderedPolicy, argmax_pick
from blue.training import mappo_guide as mg


KW = {"red_agent": "finite", "temporal_features": ("ages", "belief"),
      "include_root_session": True}


def _trace(policy, hook=None):
    env = CC4MARLEnv(seed=7629, steps=40, **KW)
    env.reset(seed=7629)
    policy.reset()
    policy.hook = hook
    digest = hashlib.sha256()
    rewards = []
    for _ in range(40):
        acts = {a: int(policy.select(env, a)) for a in BLUE_AGENTS}
        digest.update(json.dumps(sorted(acts.items()),
                                 separators=(",", ":")).encode())
        digest.update(b"\n")
        _, rew, term, trunc, _ = env.step(acts)
        rewards.append(float(rew[0]))
        if term or trunc:
            break
    return digest.hexdigest(), sum(rewards)


def test_strict_harness_matches_same_guard_lancer_and_noop_hook():
    baseline = _trace(OrderedPolicy(
        scorer=LancerValues(fruitless_decay=.5), guard=True))
    harness = _trace(StrictHarnessPolicy())
    noop = _trace(StrictHarnessPolicy(),
                  lambda env, agent, cands, scored: argmax_pick(scored))
    assert baseline == harness == noop


def test_rollout_factory_preserves_finite_behavior():
    def factory(seed, steps):
        return CC4MARLEnv(seed=seed, steps=steps, **KW)

    a = mg.run_team_episode(StrictHarnessPolicy(), 7629, 40, **KW)
    b = mg.run_team_episode(StrictHarnessPolicy(), 7629, 40,
                            env_factory=factory)
    assert a["return"] == b["return"]
    assert a["coverage"] == b["coverage"]
    assert a["ticks"] == b["ticks"]
