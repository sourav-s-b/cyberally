import random

import numpy as np
import torch

from blue.harness.redesign import FEATURE_NAMES
from blue.training.harness_dagger import (
    DaggerActor, DaggerHook, collate, pack_rows, unpack_rows,
)


def _row(label, n=3, episode_seed=1):
    return {
        "features": np.arange(n * len(FEATURE_NAMES), dtype=np.float32).reshape(
            n, len(FEATURE_NAMES)) / 100.0,
        "label": label,
        "agent": "blue_agent_1",
        "candidates": [f"host_{i}" for i in range(n)],
        "episode_meta": {"episode_seed": episode_seed, "round": 0, "beta": 1.0},
    }


def test_dagger_actor_is_set_equivariant_and_has_no_base_argument():
    torch.manual_seed(4)
    actor = DaggerActor(hidden=16).eval()
    x = torch.randn(5, len(FEATURE_NAMES))
    mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)
    logits = actor(x, mask)
    perm = torch.tensor([3, 0, 4, 1, 2])
    permuted = actor(x[perm], mask[perm])
    assert torch.allclose(permuted, logits[perm], atol=1e-6)
    try:
        actor(x, torch.zeros(5), mask)
    except TypeError:
        pass
    else:
        raise AssertionError("DaggerActor unexpectedly accepted Lancer base scores")


def test_collate_and_dataset_round_trip_preserve_candidate_labels(tmp_path):
    rows = [_row(label=1, n=3, episode_seed=7), _row(label=2, n=4, episode_seed=8)]
    x, mask, labels = collate(rows)
    assert tuple(x.shape) == (2, 4, len(FEATURE_NAMES))
    assert mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert labels.tolist() == [1, 2]
    path = tmp_path / "dataset.npz"
    pack_rows(path, rows)
    restored = unpack_rows(path)
    assert [r["label"] for r in restored] == [1, 2]
    assert [r["episode_meta"]["episode_seed"] for r in restored] == [7, 8]
    assert np.array_equal(restored[1]["features"], rows[1]["features"])


def test_dagger_hook_uses_lancer_label_but_student_features_only():
    class Provider:
        def features(self, env, agent, cands):
            return np.ones((len(cands), len(FEATURE_NAMES)), dtype=np.float32)

    class SpyActor:
        def __init__(self):
            self.called = False

        def __call__(self, features, mask):
            self.called = True
            assert features.shape == (2, len(FEATURE_NAMES))
            assert mask.tolist() == [True, True]
            return torch.tensor([2.0, 1.0])

    hook = DaggerHook.__new__(DaggerHook)
    hook.feature_provider = Provider()
    hook.actor = SpyActor()
    hook.beta = 0.0
    hook.rng = random.Random(3)
    hook.collect = True
    hook.episode_meta = {"episode_seed": 9}
    hook.rows = []
    hook.per_agent = {}
    hook.decisions = 0
    hook.agreement = 0

    chosen = hook(None, "blue_agent_1", ["router", "user"],
                  [(0.1, "router"), (0.9, "user")])
    assert hook.actor.called
    assert chosen == "router"  # learner executes its own choice at beta=0
    assert hook.rows[0]["label"] == 1  # Lancer label is retained separately
    assert hook.rows[0]["candidates"] == ["router", "user"]
    assert hook.agreement == 0
