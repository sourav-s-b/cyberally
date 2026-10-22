"""JSRL shrinking guide-window tests.

The guide mechanism must satisfy four contracts:
1. the schedule is the pre-registered monotone max-floor curriculum;
2. with the guide disabled, run_team_episode is byte-identical to the
   historical call (no new fields changing behavior);
3. during the guide window, no decision becomes a PPO row and every
   agent-tick is guide-attributed (n_guide / forced_cats["guide"]);
4. after the window, recorded decisions exist only past the window's last
   tick, and cmd_train's manifest pins the guide block and red agent.
"""
import argparse
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="jsrl guide needs torch")

import blue.training.mappo_guide as mg  # noqa: E402
from blue.policies.ordered import LancerValues, OrderedPolicy  # noqa: E402


def test_guide_steps_schedule_monotone_with_floor():
    assert mg.guide_steps_for_iter(1, None, 0.6, 10) == 0
    assert mg.guide_steps_for_iter(1, 0, 0.6, 10) == 0
    assert mg.guide_steps_for_iter(1, 200, 0.6, 10) == 200
    assert mg.guide_steps_for_iter(5, 200, 0.6, 10) == 25
    assert mg.guide_steps_for_iter(9, 200, 0.6, 10) == 10
    seq = [mg.guide_steps_for_iter(i, 200, 0.6, 10) for i in range(1, 12)]
    assert all(b <= a for a, b in zip(seq, seq[1:]))
    assert all(s >= 10 for s in seq)


def _policy_and_rec(hidden=64, seed=0):
    from blue.training.residual import build_nets
    actor, _ = build_nets(hidden=hidden)
    gen = torch.Generator().manual_seed(seed)
    rec = mg.JointRecorder(actor, 1.0, gen)
    pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                        guard=False)
    return pol, rec


def test_guide_disabled_is_unchanged():
    pol, rec = _policy_and_rec()
    old = mg.run_team_episode(pol, 7630, 8, hook=rec, recorder=rec)
    pol2, rec2 = _policy_and_rec()
    new = mg.run_team_episode(pol2, 7630, 8, hook=rec2, recorder=rec2,
                              guide_policy=None, guide_steps=0)
    assert old["return"] == new["return"]
    assert old["n_actor"] == new["n_actor"]
    assert old["forced_cats"] == new["forced_cats"]
    assert "n_guide" in new and new["n_guide"] == 0


EPS_KW = {"red_agent": "discovery"}


def test_full_guide_window_emits_no_ppo_rows():
    pol, rec = _policy_and_rec()
    guide = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                          max_age=80.0)
    r = mg.run_team_episode(pol, 7630, 6, hook=rec, recorder=rec,
                            guide_policy=guide, guide_steps=10, **EPS_KW)
    assert r["n_guide"] == r["ticks"] * 5
    assert r["forced_cats"]["guide"] == r["ticks"] * 5
    assert r["n_actor"] == 0
    assert len(rec.rows) == 0


def test_window_boundary_rows_only_after_window():
    pol, rec = _policy_and_rec()
    guide = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                          max_age=80.0)
    r = mg.run_team_episode(pol, 7630, 14, hook=rec, recorder=rec,
                            guide_policy=guide, guide_steps=2, **EPS_KW)
    assert r["n_guide"] == 2 * 5
    assert r["n_actor"] > 0
    assert len(rec.rows) > 0
    # row["tick"] is env._tick (0-based at decision time): loop ticks 1-2
    # are the guide window and correspond to env._tick 0-1, so every
    # learner row must carry env._tick >= 2.
    assert all(row["tick"] >= 2 for row in rec.rows)


def test_cmd_train_pins_guide_block_and_red_agent(tmp_path):
    a = argparse.Namespace(
        train_seeds=[7706], steps=8, iters=2, eps_per_iter=1, hidden=64,
        lr=3e-4, bonus=1.0, temperature=1.0, temp_end=None, seed=0,
        min_coverage=0.0, min_agree=0.0, max_kl=10.0,
        phase3_model="blue/results/scorer_mlp.npz",
        resume_from=None, out=str(tmp_path / "a"),
        guide_h0=5, guide_decay=0.6, guide_floor=1, guide_max_age=80.0,
        red_agent="finite")
    if not __import__("os").path.exists(a.phase3_model):
        pytest.skip(f"gitignored artifact absent: {a.phase3_model}")
    mg.cmd_train(a)
    with open(tmp_path / "a" / "pilot_manifest.json") as f:
        m = json.load(f)
    assert m["guide"] == {"enabled": True, "h0": 5, "decay": 0.6,
                          "floor": 1, "max_age": 80.0,
                          "policy": "OrderedPolicy(LancerValues("
                                    "fruitless_decay=0.5), "
                                    "max_age=...) hook=None"}
    assert m["red_agent"] == "finite"
    with open(tmp_path / "a" / "train_log.jsonl") as f:
        rows = [json.loads(line) for line in f]
    assert rows[0]["guide_steps"] == 5
    assert rows[1]["guide_steps"] == 3
    assert all(r["n_guide"] > 0 for r in rows)
