"""Inspection-age input never treats a recent clean scan as certainty."""

from types import SimpleNamespace

import numpy as np
import pytest

from blue.harness.inspection_value import inspection_value_rows


def test_never_recent_and_stale_inspections_have_distinct_inputs():
    env = SimpleNamespace(_tick=100, trackers={"agent": SimpleNamespace(
        last_analysis={"never": None, "recent": 100, "stale": 20})})
    x = inspection_value_rows(env, "agent", ["never", "recent", "stale"],
                              [.8, .8, .8], 80)
    np.testing.assert_allclose(x, [[1., .8], [0., 0.], [1., .8]])
    assert x.dtype == np.float32


def test_bad_timestamps_and_scores_fail_closed():
    env = SimpleNamespace(_tick=10, trackers={"agent": SimpleNamespace(
        last_analysis={"host": 11})})
    with pytest.raises(ValueError, match="timestamp"):
        inspection_value_rows(env, "agent", ["host"], [.5], 80)
    with pytest.raises(ValueError, match="risk vector"):
        inspection_value_rows(env, "agent", ["host"], [1.5], 80)
