"""Consequence proxies use public objectives and Blue evidence only."""

import numpy as np
import pytest

from blue.harness.consequence_context import consequence_context
from blue.harness.features import NAMES


def row(phase, analysis_age=1., analysis_available=0.):
    x = np.zeros(len(NAMES), dtype=float)
    x[NAMES.index(f"phase_{phase}")] = 1.
    x[NAMES.index("analysis_age")] = analysis_age
    x[NAMES.index("analysis_available")] = analysis_available
    return x


def test_public_cost_and_unreviewed_evidence_define_exposure_not_penalty():
    host = "operational_zone_a_subnet_user_host_0"
    memory = [1., 0., 0., 0., .2, 1., 1., 1.]
    early = consequence_context(host, row(0), memory)
    critical = consequence_context(host, row(1), memory)
    assert early[0] == pytest.approx(.1)
    assert critical[0] == pytest.approx(1.)
    assert critical[1] == 1.
    assert critical[2] == pytest.approx(.5)
    recent = consequence_context(host, row(1, 0., 1.), [0., 0., 0., 0., 1., 1., 1., 1.])
    assert recent[1:] == (0., 0.)


def test_unknown_zone_and_invalid_phase_fail_closed():
    with pytest.raises(ValueError, match="zone"):
        consequence_context("invented_host", row(0), [0.]*8)
    with pytest.raises(ValueError, match="phase"):
        consequence_context("operational_zone_a_subnet_user_host_0",
                            np.zeros(len(NAMES)), [0.]*8)
