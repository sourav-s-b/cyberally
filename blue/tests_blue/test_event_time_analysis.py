import pytest

from blue.training.event_time_analysis import analyze
from blue.training.event_time_rl import FROZEN_QWEN


def _blob():
    seeds = list(range(7641, 7665))
    def rows(value):
        return {str(s): {"return": value, "red_plan_requests": 12,
                         "remediation": 10, "confirmed_hosts": 5}
                for s in seeds}
    return {"model": dict(FROZEN_QWEN), "seeds": seeds, "steps": 400,
            "arms": {"lancer": rows(-20), "lancer_guard": rows(-15),
                     "actor_guard": rows(-14)}}


def test_same_guard_paired_contrast_and_seed_units():
    out = analyze(_blob(), n_boot=100)
    assert out["n_seeds"] == 24
    same = out["contrasts"]["actor_guard_minus_lancer_guard"]
    assert same["mean_paired_delta"] == 1
    assert same["ci95_episode_bootstrap"] == [1, 1]
    assert same["seeds_improved"] == 24
    assert out["contrasts"]["lancer_guard_minus_lancer"][
        "mean_paired_delta"] == 5


def test_incomplete_or_wrong_model_is_rejected():
    blob = _blob()
    del blob["arms"]["actor_guard"]["7641"]
    with pytest.raises(ValueError, match="incomplete"):
        analyze(blob, n_boot=10)
    blob = _blob()
    blob["model"]["digest"] = "wrong"
    with pytest.raises(ValueError, match="drift"):
        analyze(blob, n_boot=10)
