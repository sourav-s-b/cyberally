"""Expert-guided MAPPO: central critic, returns, updates, reload.

Torch-gated: these run in the train venv (``pytest.importorskip``), so a
simulator-only run skips them instead of failing.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="mappo guide needs torch")

from blue.training import mappo_guide as mg  # noqa: E402
from blue.training.residual import build_nets, masked_choice  # noqa: E402

ENV_KW = dict(mg.ENV_KW)


def _row(feat_val=0.0, k=3, tick=5, ret=1.0, choice=0, temp=1.0):
    feats = np.full((mg.MAX_CANDS, mg.FEAT_DIM), feat_val,
                    dtype=np.float32)
    joint = np.full(mg.JOINT_CTX_DIM, 0.1, dtype=np.float32)
    mask = np.zeros(mg.MAX_CANDS, dtype=np.float32)
    mask[:k] = 1.0
    base = np.zeros(mg.MAX_CANDS, dtype=np.float32)
    base[:k] = np.arange(k, dtype=np.float32)
    return {"feats": feats, "joint": joint, "k": k, "mask": mask,
            "base": base, "choice": choice, "logp": -1.0, "tick": tick,
            "agent": "a", "ret": ret, "temp": temp}


def test_joint_context_dim_and_blue_visible_only():
    from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
    from blue.policies.ordered import LancerValues, OrderedPolicy
    env = CC4MARLEnv(seed=7629, steps=10, **ENV_KW)
    env.reset(seed=7629)
    pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                        guard=False)
    for _ in range(3):
        actions = {a: int(pol.select(env, a)) for a in BLUE_AGENTS}
        env.step(actions)
    j = mg.joint_context(env, pol)
    assert j.shape == (mg.JOINT_CTX_DIM,)
    assert np.all(np.isfinite(j))
    assert len(BLUE_AGENTS) == mg.N_AGENTS


def test_generator_repeatability_and_global_untouched():
    logits = torch.tensor([1.0, 2.0, 3.0])
    mask = [1.0, 1.0, 1.0]
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    seq1 = [masked_choice(logits, mask, sample=True, seed_rng=g1)[0]
            for _ in range(20)]
    seq2 = [masked_choice(logits, mask, sample=True, seed_rng=g2)[0]
            for _ in range(20)]
    assert seq1 == seq2
    before = torch.get_rng_state().clone()
    g3 = torch.Generator().manual_seed(999)
    [masked_choice(logits, mask, sample=True, seed_rng=g3)
     for _ in range(10)]
    assert torch.equal(before, torch.get_rng_state())


def test_close_episode_returns_math():
    import torch as th
    actor, _ = build_nets(hidden=8)
    rec = mg.JointRecorder(actor, bonus=1.0,
                           gen=th.Generator().manual_seed(0))
    rec._policy_ref = None
    r1 = _row(tick=0)
    r2 = _row(tick=2)
    rec.rows = [r1, r2]
    rec._ep_start = 0
    rec.pending = {"a": [0, 1]}
    rec.close_episode([1.0, 2.0, 3.0])
    g = mg.GAMMA
    assert rec.rows[0]["ret"] == pytest.approx(1.0 + 2.0 * g + 3.0 * g**2)
    assert rec.rows[1]["ret"] == pytest.approx(3.0)


def test_ppo_central_updates_both_nets():
    import torch as th
    torch.manual_seed(0)
    actor, _ = build_nets(hidden=8)
    critic = mg.build_central_critic(hidden=8)
    opt = th.optim.Adam(list(actor.parameters())
                        + list(critic.parameters()), lr=3e-4)
    # Recompute true logps for the synthetic rows so the buffer is
    # on-policy by construction.
    buf = [_row(feat_val=float(i % 3), k=3, tick=i, ret=float(i % 5 - 2),
                choice=i % 3) for i in range(64)]
    for b in buf:
        F = th.from_numpy(b["feats"][:3])
        with th.no_grad():
            r = th.tanh(actor(F))
        logits = th.from_numpy(b["base"][:3]) + r
        b["logp"] = float(th.log_softmax(logits, dim=0)[b["choice"]])
    a_before = [p.detach().clone() for p in actor.parameters()]
    c_before = [p.detach().clone() for p in critic.parameters()]
    stats = mg.ppo_central_update(actor, critic, opt, buf, epochs=2)
    assert any(not torch.equal(a, b)
               for a, b in zip(a_before, actor.parameters()))
    assert any(not torch.equal(a, b)
               for a, b in zip(c_before, critic.parameters()))
    for k in ("pol", "vf", "ent", "kl"):
        assert np.isfinite(stats[k]), k


def test_rollout_logprob_rebuild_identity():
    """One short sampling episode: rebuilding logprobs from saved rows
    with the same nets must reproduce the recorded logps (on-policy
    contract for the central-critic path)."""
    import torch as th
    from blue.core.wrapper import BLUE_AGENTS
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.residual import build_nets
    actor, _ = build_nets(hidden=8)
    gen = th.Generator().manual_seed(7)
    rec = mg.JointRecorder(actor, bonus=1.0, gen=gen)
    pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                        guard=False)
    out = mg.run_team_episode(pol, 7630, 30, hook=rec, recorder=rec,
                              **ENV_KW)
    assert out["n_decisions"] > 0
    # Bookkeeping consistency (not behavioral proof): per-agent hook
    # counts sum to the actor total, and the forced categories partition
    # the non-actor steps. The split itself is validated by lockout_sleep
    # genuinely occurring (pending lockout exists) and by the row-level
    # invariants below.
    assert (sum(rec.per_agent.values()) == rec.total == out["n_actor"])
    assert set(out["forced_cats"]) == {"urgent", "verification",
                                       "lockout_sleep", "idle"}
    assert sum(out["forced_cats"].values()) == out["n_forced"]
    assert out["forced_cats"]["lockout_sleep"] > 0
    assert out["n_forced"] > 0
    for b in rec.rows:
        assert 0 <= b["choice"] < b["k"]
        assert 0 <= b["tick"] < out["ticks"]
        assert b["agent"] in list(BLUE_AGENTS)
        assert b["joint"].shape == (mg.JOINT_CTX_DIM,)
        assert np.isfinite(b["ent"]) and b["ent"] >= 0.0
    worst = 0.0
    for b in rec.rows:
        k = b["k"]
        F = th.from_numpy(np.asarray(b["feats"][:k], dtype=np.float32))
        with th.no_grad():
            r = th.tanh(actor(F))
        logits = ((th.from_numpy(np.asarray(b["base"][:k],
                                                  dtype=np.float32)) + r)
                  / b.get("temp", 1.0))
        lp = float(th.log_softmax(logits, dim=0)[b["choice"]])
        worst = max(worst, abs(lp - b["logp"]))
    assert worst < 1e-5


def test_save_reload_greedy_identical(tmp_path):
    import torch as th
    from blue.training.residual import build_nets
    actor, _ = build_nets(hidden=8)
    p = tmp_path / "actor.th"
    th.save(actor.state_dict(), str(p))
    actor2, _ = build_nets(hidden=8)
    actor2.load_state_dict(th.load(str(p), map_location="cpu"))
    actor2.eval()
    F = th.randn(4, mg.FEAT_DIM)
    with th.no_grad():
        assert torch.equal(actor(F), actor2(F))


def test_temperature_flows_into_update():
    """Same buffer at temp 0.5 vs 1.0 must give different policy losses:
    the temperature is really used, not a dead knob."""
    import torch as th
    from blue.training.residual import build_nets
    torch.manual_seed(1)
    buf1 = [_row(feat_val=float(i % 3), k=3, tick=i,
                 ret=float(i % 5 - 2), choice=i % 3, temp=1.0)
            for i in range(64)]
    buf2 = [dict(b, temp=0.5) for b in buf1]
    for buf in (buf1, buf2):
        actor, _ = build_nets(hidden=8)
        torch.manual_seed(1)
        for b in buf:
            F = th.from_numpy(b["feats"][:3])
            with th.no_grad():
                r = th.tanh(actor(F))
            logits = ((th.from_numpy(b["base"][:3]) + r)
                      / b["temp"])
            b["logp"] = float(th.log_softmax(logits, dim=0)[b["choice"]])
    s1, s2 = [], []
    for buf, out in ((buf1, s1), (buf2, s2)):
        actor, _ = build_nets(hidden=8)
        torch.manual_seed(1)
        critic = mg.build_central_critic(hidden=8)
        opt = th.optim.Adam(list(actor.parameters())
                            + list(critic.parameters()))
        out.append(mg.ppo_central_update(actor, critic, opt, buf,
                                         epochs=1))
    assert s1[0]["pol"] != pytest.approx(s2[0]["pol"])


def test_resume_continues_from_saved_state(tmp_path):
    """Train 1 iter, resume to iter 2: manifest continues the history and
    the reloaded optimizer state keeps training (params keep moving)."""
    import argparse
    import json
    import torch as th
    a = argparse.Namespace(
        train_seeds=[7706], steps=40, iters=1, eps_per_iter=1, hidden=64,
        lr=3e-4, bonus=1.0, temperature=1.0, temp_end=None, seed=0,
        min_coverage=0.0, min_agree=0.0, max_kl=10.0,
        phase3_model="blue/results/scorer_mlp.npz",
        resume_from=None, out=str(tmp_path / "a"))
    mg.cmd_train(a)
    with open(tmp_path / "a" / "pilot_manifest.json") as f:
        m1 = json.load(f)
    assert m1["iters_done"] == 1
    b = argparse.Namespace(**{**vars(a), "iters": 2,
                               "resume_from": str(tmp_path / "a"),
                               "out": str(tmp_path / "b")})
    mg.cmd_train(b)
    with open(tmp_path / "b" / "pilot_manifest.json") as f:
        m2 = json.load(f)
    assert m2["iters_done"] == 2
    assert len(m2["hist"]) == 2
    assert m2["resumed"]["from"] == str(tmp_path / "a")
    sa = th.load(str(tmp_path / "a" / "actor.th"), map_location="cpu",
                 weights_only=False)
    sb = th.load(str(tmp_path / "b" / "actor.th"), map_location="cpu",
                 weights_only=False)
    assert any(not torch.equal(sa[k], sb[k]) for k in sa)


def test_shield_margin_inf_reproduces_lancer():
    """Infinite margin shields every decision: return must equal Lancer
    exactly on the same seed."""
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.mappo_guide import ShieldHook
    from blue.training.residual import build_nets
    from blue.training.residual_pilot import GreedyHook
    actor, _ = build_nets(hidden=8)
    base = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                         guard=False)
    sh = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                       guard=False)
    rb = mg.run_team_episode(base, 7630, 30, **ENV_KW)
    hook = ShieldHook(GreedyHook(actor, 1.0), float("inf"))
    rs = mg.run_team_episode(sh, 7630, 30, hook=hook, **ENV_KW)
    assert hook.n_actor == 0
    assert hook.n_shield > 0
    assert rs["return"] == rb["return"]


def test_shield_margin_zero_equals_greedy():
    """Zero margin shields nothing: return must equal the greedy hook,
    and counters must partition the hook calls."""
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.mappo_guide import ShieldHook
    from blue.training.residual import build_nets
    from blue.training.residual_pilot import GreedyHook
    actor, _ = build_nets(hidden=8)
    mk = lambda: OrderedPolicy(  # noqa: E731
        scorer=LancerValues(fruitless_decay=0.5), guard=False)
    rg = mg.run_team_episode(mk(), 7630, 30,
                             hook=GreedyHook(actor, 1.0), **ENV_KW)
    hook = ShieldHook(GreedyHook(actor, 1.0), 0.0)
    rs = mg.run_team_episode(mk(), 7630, 30, hook=hook, **ENV_KW)
    assert hook.n_shield == 0
    assert hook.n_actor > 0
    assert rs["return"] == rg["return"]
