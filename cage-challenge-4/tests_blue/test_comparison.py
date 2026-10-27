"""Reject misleading pairings and verify native-return comparison direction."""
import pytest

from blue_compare import paired_summary


def test_paired_summary_uses_seeds_not_episode_order():
    policy = [{"seed": 1, "joint_steps": 399, "native_return": -50},
              {"seed": 2, "joint_steps": 399, "native_return": -100}]
    baseline = [{"seed": 2, "joint_steps": 399, "native_return": -80},
                {"seed": 1, "joint_steps": 399, "native_return": -90}]
    result = paired_summary(policy, baseline)
    assert result["paired_return_differences"] == [40, -20]
    assert result["mean_paired_return_difference"] == 10
    assert result["mappo_wins"] == 1


@pytest.mark.parametrize("baseline", [[],
    [{"seed": 2, "joint_steps": 399, "native_return": 0}],
    [{"seed": 1, "joint_steps": 398, "native_return": 0}],
    [{"seed": 1, "joint_steps": 399, "native_return": 0}] * 2])
def test_mismatched_comparisons_rejected(baseline):
    with pytest.raises(ValueError):
        paired_summary([{"seed": 1, "joint_steps": 399, "native_return": 0}], baseline)
