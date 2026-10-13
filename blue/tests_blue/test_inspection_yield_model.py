"""Outcome labels must not influence the model's pre-decision features."""

import json

import numpy as np
import pytest

from blue.harness.features import NAMES
from blue.training.inspection_yield_model import feature_vector, load_partition


def test_selected_completed_actions_only_and_no_outcome_feature_leak(tmp_path):
    values = [0.] * len(NAMES)
    values[0] = None
    values[NAMES.index("phase_0")] = 1.
    values[NAMES.index("analysis_age")] = 1.
    pre = {"feature_values": values,
           "event_memory_values": [0., 0., 0., 0., 1., 1., 1., 1.],
           "risk": .2, "novelty_percentile": .3, "base_score": 1.,
           "decision_branch": "sweep_scorer"}
    rows = [
        {"agent": "blue_agent_0", "host": "operational_zone_a_subnet_user_host_0",
         "pre_input": pre,
         "outcome": {"status": "success", "new_confirmation": False}},
        {"agent": "blue_agent_0", "host": "operational_zone_a_subnet_user_host_0",
         "pre_input": pre,
         "outcome": {"status": "success", "new_confirmation": True}},
        {"agent": "blue_agent_0", "host": "operational_zone_a_subnet_user_host_0",
         "pre_input": pre,
         "outcome": {"status": "censored_at_episode_end", "new_confirmation": None}},
        {"agent": "blue_agent_0", "host": "operational_zone_a_subnet_user_host_0",
         "pre_input": {**pre, "decision_branch": "verification"},
         "outcome": {"status": "success", "new_confirmation": True}},
    ]
    path = tmp_path / "episode_9_requests.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    agents = ["blue_agent_0", "blue_agent_1"]
    X, y, group, risk = load_partition(tmp_path, [9], agents)
    assert X.shape == (2, len(NAMES) + 8 + 3 + 3 + 2)
    np.testing.assert_array_equal(X[0], X[1])
    assert np.isnan(X[0, 0])
    assert y.tolist() == [0, 1]
    assert group.tolist() == [9, 9]
    assert risk.tolist() == pytest.approx([.2, .2])
