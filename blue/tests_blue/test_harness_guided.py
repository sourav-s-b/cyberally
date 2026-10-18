"""Guide-prefix routing and schedule contract."""
from types import SimpleNamespace

import pytest

from blue.harness.guided import GuidePrefixHook
from blue.harness.learning import decision_credit
from blue.training.harness_redesign import env_kwargs, resolve_guide_prefix_schedule


class Learner:
    def __init__(self):
        self.calls = []
        self.observations = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def observe(self, env, agent):
        self.observations.append((env._tick, agent))

    def __call__(self, env, agent, cands, scored):
        self.calls.append((env._tick, agent))
        return cands[-1]


def test_guide_prefix_routes_only_suffix_decisions_to_learner():
    learner = Learner()
    hook = GuidePrefixHook(learner, prefix_ticks=2)
    cands = ["lancer", "learner"]
    scored = [(1.0, "lancer"), (0.0, "learner")]
    for tick in (0, 1):
        env = SimpleNamespace(_tick=tick)
        hook.observe(env, "blue_agent_1")
        assert hook(env, "blue_agent_1", cands, scored) == "lancer"
    env = SimpleNamespace(_tick=2)
    hook.observe(env, "blue_agent_1")
    assert hook(env, "blue_agent_1", cands, scored) == "learner"
    assert learner.calls == [(2, "blue_agent_1")]
    assert hook.guide_decisions == 2
    assert hook.learner_decisions == 1
    assert learner.observations == [(0, "blue_agent_1"), (1, "blue_agent_1"),
                                    (2, "blue_agent_1")]
    hook.reset()
    assert learner.resets == 1
    assert hook.guide_decisions == hook.learner_decisions == 0


def test_guide_reward_prefix_is_not_attributed_to_suffix_actor_decisions():
    # The simulator still scores every tick; a learned decision's policy
    # gradient target begins at that decision and includes its full suffix.
    _, targets, durations = decision_credit(
        [2], [100.0, 100.0, 3.0, 4.0], [0.0], gamma=1.0, gae_lambda=1.0)
    assert targets.tolist() == [7.0]
    assert durations.tolist() == [2]


def test_guide_schedule_repeats_single_value_and_requires_nonincreasing_ticks():
    assert resolve_guide_prefix_schedule([0], 4, 400) == (0, 0, 0, 0)
    assert resolve_guide_prefix_schedule([267, 134, 0], 3, 400) == (267, 134, 0)
    with pytest.raises(ValueError, match="one value or one per iteration"):
        resolve_guide_prefix_schedule([267, 0], 3, 400)
    with pytest.raises(ValueError, match="nonincreasing"):
        resolve_guide_prefix_schedule([0, 134, 0], 3, 400)
    with pytest.raises(ValueError, match="within the episode horizon"):
        resolve_guide_prefix_schedule([401], 3, 400)


def test_training_environment_can_pin_finite_red_without_other_config_drift():
    assert env_kwargs("finite")["red_agent"] == "finite"
    assert env_kwargs("finite")["temporal_features"] == ("ages", "belief")
    with pytest.raises(ValueError, match="unknown red agent"):
        env_kwargs("unregistered")
