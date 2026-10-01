"""BC warm-start pipeline tests: demo schema + train-driver flag passthrough."""

import numpy as np
import pytest

import cc4_epymarl_wrapper as wrapper
from blue_collect_bc import collect_episode


def test_collect_episode_schema():
    ep = collect_episode(7629, steps=6)
    n_agents = len(wrapper.BLUE_AGENTS)
    ticks = ep["obs"].shape[1]
    assert 1 <= ticks <= 6  # early termination is legal; shapes must agree
    assert ep["actions"].shape == (n_agents, ticks)
    assert ep["masks"].shape[:2] == (n_agents, ticks)
    # Teacher actions are always legal under the recorded mask.
    for i in range(n_agents):
        for t in range(ticks):
            assert ep["masks"][i, t, ep["actions"][i, t]] == 1


def test_build_config_warmstart_flags():
    pytest.importorskip("torch", reason="train driver imports torch")
    from blue_train_mappo import build_config
    cfg = build_config(mask_mode="evidence",
                       init_ckpt="results/models/bc_rr_X")
    assert cfg["env_args"]["mask_mode"] == "evidence"
    assert cfg["checkpoint_path"] == "results/models/bc_rr_X"
    default = build_config()
    assert default["env_args"]["mask_mode"] == "validity"
    assert default["checkpoint_path"] == ""
