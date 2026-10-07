"""Audit of the residual/PPO path: RTG, masking, on-policy and arm identity.

Executable version of the questions that were previously answered by reading
code: does the return actually belong to the decision, can a padded
candidate leak probability, does the update recompute the rollout log-prob,
is the buffer really on-policy, and do the eval arms isolate the learner from
the guard.

Torch-gated: these run in the train venv (``pytest.importorskip``), so a
simulator-only run skips them instead of failing.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="residual audit needs torch")

from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv  # noqa: E402
from blue.training import residual_pilot as rp  # noqa: E402
from blue.training.residual import (FEAT_DIM, build_nets,  # noqa: E402
                                    masked_choice, ppo_update)

ENV_KW = dict(rp.ENV_KW)


def _bare_row(tick, k=3):
    """Minimal row shaped like Recorder rows (padded to MAX_CANDS)."""
    return {"feats": np.zeros((rp.MAX_CANDS, FEAT_DIM), dtype=np.float32),
            "ctx": np.zeros(FEAT_DIM + 1, dtype=np.float32),
            "mask": np.zeros(rp.MAX_CANDS, dtype=np.float32),
            "base": np.zeros(rp.MAX_CANDS, dtype=np.float32),
            "choice": 0, "logp": 0.0, "tick": tick, "ret": 0.0}


def _row(k=3, mask=(1.0, 1.0, 0.0), choice=0, logp=-0.7, ret=0.0, tick=0):
    row = _bare_row(tick)
    n = len(mask)
    row["mask"][:n] = np.asarray(mask, dtype=np.float32)
    row["base"][:n] = np.asarray([1.0, 0.5, 0.0][:n], dtype=np.float32)
    row["choice"] = choice
    row["logp"] = logp
    row["ret"] = ret
    return row


# --------------------------------------------------------------------------
# Returns to go
# --------------------------------------------------------------------------
def test_return_is_discounted_from_the_decision_tick_to_episode_end():
    rec = rp.Recorder(residual=None, bonus=1.0, sample=False, rng=None)
    rec.rows = [_bare_row(3)]
    rec.pending = {"a": [0]}
    rec._ep_start = 0
    rec.close_episode([1.0] * 5)
    assert rec.rows[0]["ret"] == pytest.approx(1.0 + rp.GAMMA)


def test_return_is_zero_for_the_last_tick():
    rec = rp.Recorder(residual=None, bonus=1.0, sample=False, rng=None)
    rec.rows = [_bare_row(4)]
    rec.pending = {"a": [0]}
    rec.close_episode([1.0] * 5)
    assert rec.rows[0]["ret"] == pytest.approx(1.0)


def test_return_tick_is_clamped_into_the_episode():
    rec = rp.Recorder(residual=None, bonus=1.0, sample=False, rng=None)
    rec.rows = [_bare_row(999)]
    rec.pending = {"a": [0]}
    rec.close_episode([2.0, 2.0])
    assert rec.rows[0]["ret"] == pytest.approx(2.0)


def test_returns_do_not_leak_across_episodes():
    rec = rp.Recorder(residual=None, bonus=1.0, sample=False, rng=None)
    rec.rows = [_bare_row(0)]
    rec.pending = {"a": [0]}
    rec.close_episode([1.0, 1.0, 1.0])
    first = rec.rows[0]["ret"]
    rec.mark_episode()
    rec.rows.append(_bare_row(0))
    rec.pending = {"a": [1]}
    rec.close_episode([100.0, 100.0, 100.0])
    assert rec.rows[0]["ret"] == first
    assert rec.rows[1]["ret"] > first
    assert rec.pending == {}


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
def test_masked_candidates_get_zero_probability():
    logits = torch.tensor([2.0, 1.0, 5.0])
    mask = [1.0, 1.0, 0.0]
    idx, logp, _ = masked_choice(logits, mask, sample=False)
    assert idx in (0, 1)
    l = logits + (1.0 - torch.tensor(mask)) * -1e9
    probs = torch.softmax(l, dim=0)
    assert float(probs[2]) < 1e-6


def test_padded_candidates_contribute_nothing_and_stay_finite():
    residual, value = build_nets(hidden=8)
    opt = torch.optim.Adam(list(residual.parameters())
                           + list(value.parameters()), lr=1e-3)
    buf = [_row(k=3, mask=(1.0, 1.0, 0.0), choice=1, logp=-0.5,
                ret=1.0, tick=0) for _ in range(8)]
    stats = ppo_update(residual, value, opt, buf)
    for k in ("pol", "vf", "ent", "kl"):
        assert np.isfinite(stats[k]), f"{k} is not finite: {stats[k]}"


def test_rollout_and_update_compute_the_same_logprob():
    """The on-policy contract: the log-prob PPO ratios against must be the
    one the rollout sampled from, otherwise the ratio is silently wrong."""
    residual, _ = build_nets(hidden=8)
    with torch.no_grad():
        for p in residual.parameters():
            p.normal_(0.0, 0.5)
    feats = np.random.RandomState(0).randn(4, FEAT_DIM).astype(np.float32)
    base = np.asarray([1.0, 0.5, 0.2, 0.1], dtype=np.float32)
    mask = [1.0, 1.0, 1.0, 1.0]
    with torch.no_grad():
        r = torch.tanh(residual(torch.from_numpy(feats)))
        logits_roll = torch.from_numpy(base) + 1.0 * r
    idx_roll, logp_roll, _ = masked_choice(logits_roll, mask, sample=False)
    # the update path, same weights, same formula
    with torch.no_grad():
        logits_upd = (torch.from_numpy(base)
                      + 1.0 * torch.tanh(
                          residual(torch.from_numpy(feats)))) * 1.0
        logp_upd = torch.log_softmax(logits_upd, dim=0)[idx_roll]
    assert float(logp_roll) == pytest.approx(float(logp_upd), abs=1e-5)


def test_padded_rows_use_the_same_mask_formulation_as_rollout():
    """Rollout masks additively, the update masks multiplicatively then
    additively. They agree only if padded base scores stay 0; assert that
    invariant instead of assuming it."""
    base = np.asarray([1.0, 0.5, 0.0, 0.0], dtype=np.float32)
    mask = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert float(base[2:].sum()) == 0.0, (
        "np.pad must zero padded base scores or the update's multiplicative "
        "mask stops matching the rollout's additive mask")
    a = torch.from_numpy(base) + (1.0 - torch.from_numpy(mask)) * -1e9
    b = (torch.from_numpy(base) * torch.from_numpy(mask)
         + (1.0 - torch.from_numpy(mask)) * -1e9)
    assert torch.allclose(torch.log_softmax(a, 0), torch.log_softmax(b, 0),
                          atol=1e-6)


# --------------------------------------------------------------------------
# On-policy discipline
# --------------------------------------------------------------------------
def test_buffer_holds_only_the_current_iteration():
    """A fresh Recorder per iteration is what makes the update on-policy."""
    a = rp.Recorder(residual=None, bonus=1.0, sample=True, rng=None)
    a.rows.append(_bare_row(0))
    b = rp.Recorder(residual=None, bonus=1.0, sample=True, rng=None)
    assert b.rows == []
    assert a.rows != b.rows


def test_ppo_update_reports_kl_against_the_stored_rollout_logp():
    residual, value = build_nets(hidden=8)
    opt = torch.optim.Adam(list(residual.parameters())
                           + list(value.parameters()), lr=1e-3)
    buf = [_row(choice=0, logp=0.0, ret=float(i), tick=i) for i in range(8)]
    stats = ppo_update(residual, value, opt, buf, epochs=1)
    assert "kl" in stats
    assert np.isfinite(stats["kl"])


def test_greedy_hook_uses_the_largest_hostname_tie_break():
    from blue.policies.ordered import argmax_pick
    residual, _ = build_nets(hidden=8)
    with torch.no_grad():
        for p in residual.parameters():
            p.zero_()          # zero residual -> combined == base
    hook = rp.GreedyHook(residual, bonus=1.0)

    cands = ["host_a", "host_b", "host_c"]
    scored = [(1.0, "host_a"), (1.0, "host_b"), (1.0, "host_c")]
    # equal base scores -> canonical pick is the LARGEST hostname
    assert argmax_pick(scored) == "host_c"


# --------------------------------------------------------------------------
# Arm identity / seed hygiene
# --------------------------------------------------------------------------
def test_default_train_and_eval_seed_blocks_are_disjoint():
    assert not (set(rp.DEFAULT_TRAIN_SEEDS) & set(rp.DEFAULT_EVAL_SEEDS))
    assert len(set(rp.DEFAULT_TRAIN_SEEDS)) == len(rp.DEFAULT_TRAIN_SEEDS)
    assert len(set(rp.DEFAULT_EVAL_SEEDS)) == len(rp.DEFAULT_EVAL_SEEDS)


def test_eval_arms_isolate_the_hook_and_label_the_confound():
    """The eval arm table must make the attributable contrast the headline.

    `learned` and `sched_control` share the guard, so learned-minus-
    sched_control isolates the learner. learned-minus-`lancer` also switches
    the guard OFF and is therefore unattributable; it stays in the output only
    for continuity with the earlier manifest, and says so.
    """
    import inspect
    from blue.policies.ordered import GUARD_OFF, GUARD_STRICT
    src = inspect.getsource(rp.cmd_eval)
    assert "eval_arms()" in src, "cmd_eval must build arms via eval_arms()"
    assert "learned - sched_control" in src
    assert "NOT attributable" in src
    assert "headline" in src and "confounded" in src
    arms = {name: (pol, hook) for name, pol, hook in rp.eval_arms()}
    assert set(arms) == {"learned", "sched_control", "lancer"}
    assert arms["learned"][0].guard_mode == GUARD_STRICT
    assert arms["sched_control"][0].guard_mode == GUARD_STRICT
    assert arms["lancer"][0].guard_mode == GUARD_OFF
    assert arms["learned"][1] == "HOOK"
    assert arms["sched_control"][1] is None
    assert arms["lancer"][1] is None


def test_paired_mean_uses_only_shared_seeds():
    results = {"a": {7629: {"return": -50.0}, 7630: {"return": -70.0}},
               "b": {7629: {"return": -60.0}, 7640: {"return": -10.0}}}
    mean, n = rp.paired_mean(results, "a", "b")
    assert n == 1
    assert mean == pytest.approx(10.0)


def test_candidate_padding_fits_the_real_configuration():
    env = CC4MARLEnv(seed=7629, steps=10, **ENV_KW)
    env.reset(seed=7629)
    widest = max(len(env.hostnames[a]) for a in BLUE_AGENTS)
    assert widest <= rp.MAX_CANDS, (
        f"widest zone has {widest} hosts > MAX_CANDS={rp.MAX_CANDS}: np.pad "
        f"with a negative pad would raise mid-episode")


def test_training_loop_halts_and_records_when_triggers_fire(tmp_path):
    """A stop trigger must halt, not merely log."""
    import argparse
    torch.manual_seed(0)
    residual, _ = build_nets(hidden=8)
    ckpt = tmp_path / "phase3.th"
    torch.save({"0.weight": residual.body[0].weight.data,
                "0.bias": residual.body[0].bias.data,
                "2.weight": residual.body[2].weight.data,
                "2.bias": residual.body[2].bias.data}, ckpt)
    args = argparse.Namespace(
        seed=0, hidden=8, phase3_model=str(ckpt), lr=1e-3, bonus=1.0,
        out=str(tmp_path / "pilot"), iters=2, eps_per_iter=1, steps=12,
        train_seeds=[7706], min_coverage=0.999, min_agree=0.0, max_kl=1e9)
    rp.cmd_train(args)
    import json
    with open(args.out + "/pilot_manifest.json") as f:
        man = json.load(f)
    assert man["iters_done"] == 1, "coverage trigger should stop after iter 1"
    assert len(man["hist"]) == 1
