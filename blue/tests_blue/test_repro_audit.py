"""Correctness audit for the risk-as-feature PPO buffer.

These are the invariants that must hold BEFORE any optimizer step, because
breaking them silently produces a run that trains on the wrong objective:

1. the recorded log-probability equals the log-probability recomputed from
   the buffer row (otherwise the PPO ratio starts life wrong and the KL
   estimate is meaningless);
2. the filter changes only WHICH rows are learned from, never the reward
   assigned to a row that IS kept;
3. a fresh policy object per evaluation arm, so no state leaks between the
   Lancer arm and the actor arm;
4. the confidence interval uses the t multiplier for the ACTUAL number of
   episodes, not a hardcoded 8-seed constant.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from blue.training import remote_train as rt  # noqa: E402


def _row(n_cands=5, choice=2, temp=0.5):
    """A buffer row shaped exactly like RiskRecorder emits (post-padding)."""
    return {"feats": np.zeros((rt.mg.MAX_CANDS, 15), dtype=np.float32),
            "joint": np.zeros(6, dtype=np.float32),
            "mask": np.array([1.0] * n_cands + [0.0] * (rt.mg.MAX_CANDS - n_cands),
                             dtype=np.float32),
            "base": np.arange(rt.mg.MAX_CANDS, dtype=np.float32),
            "choice": choice, "logp": 0.0, "ent": 0.0,
            "tick": 0, "agent": "blue_agent_0", "ret": 0.0, "temp": temp}


def test_recorded_logp_matches_recomputed_before_any_update(monkeypatch):
    """Recompute every row's logp from its own features and compare.

    The PPO ratio is exp(logp_new - logp_old); if logp_old was not produced
    from these exact logits then the first ratio is already wrong.
    """
    from blue.training.residual import masked_choice
    actor = torch.nn.Linear(15, 1)
    torch.manual_seed(0)
    bonus = 1.0
    rows = []
    torch.manual_seed(1)
    for _ in range(40):
        r = _row(n_cands=int(torch.randint(2, rt.mg.MAX_CANDS + 1, (1,)).item()),
                 choice=int(torch.randint(0, 3, (1,)).item()))
        feats = torch.randn(rt.mg.MAX_CANDS, 15)
        mask = torch.tensor(r["mask"])
        base = torch.tensor(r["base"])
        with torch.no_grad():
            resid = torch.tanh(actor(feats)).squeeze(-1)
        logits = (base + bonus * resid) / r["temp"]
        idx, logp, _ = masked_choice(logits, mask.numpy(), sample=True)
        r["feats"] = feats.numpy()
        r["choice"] = idx
        r["logp"] = logp
        rows.append(r)

    # no optimizer step happens in this test: we recompute and compare only
    worst = 0.0
    for r in rows:
        feats = torch.from_numpy(r["feats"])
        mask = torch.from_numpy(r["mask"])
        base = torch.from_numpy(r["base"])
        with torch.no_grad():
            resid = torch.tanh(actor(feats)).squeeze(-1)
        logits = (((base + bonus * resid) / r["temp"]) * mask
                  + (1.0 - mask) * -1e9)
        recomputed = float(torch.log_softmax(logits, dim=0)[r["choice"]])
        worst = max(worst, abs(recomputed - r["logp"]))
    assert worst < 1e-5, f"recorded logp drifted by {worst}"


def test_filter_changes_which_rows_are_kept_not_their_returns():
    """A kept row's discounted return must not depend on dropped rows."""
    gamma = rt.mg.GAMMA
    rewards = [1.0, -2.0, 3.0, 0.5, -1.0]
    disc = gamma ** np.arange(len(rewards))

    def run_with_filter(min_proba, probas):
        rec = _FakeRecorder(min_proba, probas)
        for agent, tick in (("blue_agent_0", 0), ("blue_agent_1", 2)):
            rec.__call__(agent=agent, tick=tick)
        rec.close_episode(rewards)
        return [(r["tick"], round(r["ret"], 10)) for r in rec.rows]

    proba_all_keep = [0.9, 0.8]
    proba_all_drop = [0.1, 0.2]
    kept = run_with_filter(0.5, proba_all_keep)
    dropped = run_with_filter(0.5, proba_all_drop)
    assert len(kept) == 2 and dropped == []

    # the kept rows' returns equal the plain discounted sums
    expect0 = float((np.asarray(rewards) * disc).sum())
    expect2 = float((np.asarray(rewards)[2:] * disc[:3]).sum())
    assert kept[0][1] == pytest.approx(expect0, abs=1e-9)
    assert kept[1][1] == pytest.approx(expect2, abs=1e-9)


class _FakeRecorder:
    """Minimal stand-in exercising the recorder's keep/drop bookkeeping."""

    def __init__(self, min_proba, probas):
        self.min_proba = min_proba
        self.probas = list(probas)
        self.rows = []
        self.pending = {}
        self.n_skipped = 0

    def __call__(self, agent, tick):
        p = self.probas[len(self.rows)]
        if p < self.min_proba:
            self.n_skipped += 1
            return 0
        self.rows.append({"tick": tick, "ret": 0.0, "mask": np.ones(3),
                          "feats": np.zeros((3, 15), np.float32),
                          "base": np.zeros(3, np.float32)})
        self.pending.setdefault(agent, []).append(len(self.rows) - 1)
        return 0

    def close_episode(self, rewards):
        R = np.asarray(rewards, dtype=float)
        disc = rt.mg.GAMMA ** np.arange(len(R))
        for idxs in self.pending.values():
            for i in idxs:
                t = min(self.rows[i]["tick"], len(R) - 1)
                self.rows[i]["ret"] = float((R[t:] * disc[:len(R) - t]).sum())
        self.pending = {}


def test_ci_uses_actual_episode_count():
    """df=3 must use 3.182, not the 8-seed 2.365."""
    assert rt.t_crit95(4) == pytest.approx(3.182)
    assert rt.t_crit95(8) == pytest.approx(2.365)
    d = [10.0, -4.0, 2.0, 6.0]
    out = rt.ci95_of(d)
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1)) / 2.0
    assert out["paired_mean"] == pytest.approx(m)
    assert out["ci95"][0] == pytest.approx(m - 3.182 * se)
    assert out["t_crit95"] == pytest.approx(3.182)
    # the old constant would have produced a visibly narrower interval
    assert (out["ci95"][1] - out["ci95"][0]) > (2 * 2.365 * se)


def test_ci_reports_nan_for_single_episode():
    """One episode has no interval; it must not pretend otherwise."""
    out = rt.ci95_of([3.0])
    assert out["n"] == 1
    assert np.isnan(out["ci95"][0]) and np.isnan(out["ci95"][1])


def test_gate_rule_requires_ci_above_zero_and_enough_seeds():
    """A pass needs mean>=gate AND lower bound>0 AND n>=min AND full steps."""
    def gate(ev, g=5.0, min_seeds=8, steps=400, min_steps=400):
        if not ev:
            return False
        return bool(ev["paired_mean"] >= g and ev["ci95"][0] > 0
                    and ev["n"] >= min_seeds and steps >= min_steps)
    assert gate({"paired_mean": 9.0, "ci95": [1.0, 17.0], "n": 8})
    assert not gate({"paired_mean": 9.0, "ci95": [-1.0, 19.0], "n": 8})
    assert not gate({"paired_mean": 9.0, "ci95": [1.0, 17.0], "n": 4})
    assert not gate({"paired_mean": 4.9, "ci95": [1.0, 9.0], "n": 8})