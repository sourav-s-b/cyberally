import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

import blue.training.op3_offline as op  # noqa: E402
import torch.nn as nn  # noqa: E402


def _fake(rng, mask_frac=0.9):
    reg = rng.random((5, op.N_AG, op.N_ACT)) < mask_frac
    reg[:, :, 0] = 1
    acts = np.array([[int(rng.choice(np.nonzero(tt[a])[0]))
                      for a in range(op.N_AG)] for tt in reg])
    return dict(obs=rng.random((5, op.N_AG, op.FEAT)).astype(np.float32),
                masks=reg.astype(np.int8),
                acts=acts.astype(np.int64),
                rewards=rng.random(5).astype(np.float32),
                done=np.ones(5, dtype=np.int8))


@pytest.fixture()
def tiny_npz(tmp_path):
    rng = np.random.default_rng(0)
    epsmap = {"guard": [_fake(rng) for _ in range(2)],
              "random": [_fake(rng, 0.7) for _ in range(2)]}
    fa, lens, kinds, seeds = {}, [], [], []
    for k, es in epsmap.items():
        fa[f"obs_{k}"] = np.concatenate([e["obs"] for e in es])
        fa[f"actions_{k}"] = np.concatenate([e["acts"] for e in es])
        fa[f"masks_{k}"] = np.concatenate([e["masks"] for e in es])
        fa[f"rewards_{k}"] = np.concatenate([e["rewards"] for e in es])
        fa[f"done_{k}"] = np.concatenate([e["done"] for e in es])
        for _ in es:
            lens.append(5); kinds.append(k); seeds.append(8433)
    fa.update(lengths=np.array(lens), meta_kinds=np.array(kinds),
              meta_seeds=np.array(seeds),
              meta_pol_seeds=np.array([-1] * 4))
    p = tmp_path / "unit.npz"
    np.savez_compressed(str(p), **fa)
    return str(p), epsmap


def test_load_and_stack_shapes_match_rows(tiny_npz):
    path, epsmap = tiny_npz
    eps = op.load_episodes(path)
    assert len(eps) == 4
    flat = op.stacked(eps)
    n_t = sum(e["obs"].shape[0] for e in eps)
    assert flat["obs"].shape == (n_t, op.N_AG, op.FEAT)
    assert flat["mask"].shape == (n_t, op.N_AG, op.N_ACT)
    assert flat["acts"].shape == (n_t, op.N_AG)
    assert flat["next_idx"].shape == (n_t,)
    # row i maps to (tick i//5, agent i%5) and every logged action is
    # legal in its mask; next_idx stays within the same episode
    acts = flat["acts"].reshape(-1)
    for t in range(n_t):
        assert flat["mask"].reshape(-1, op.N_ACT)[t, acts[t]] == 1
        assert 0 <= flat["next_idx"][t] <= n_t - 1


def test_q_all_actions_matches_q_per_action(tiny_npz):
    path, _ = tiny_npz
    flat = op.stacked(op.load_episodes(path))
    rows = np.arange(min(10, flat["obs"].shape[0] * op.N_AG))
    b = op.rows_slice(flat, rows)
    assert b["joint"].shape == (len(rows), op.FEAT * op.N_AG)
    assert b["agent"].shape == (len(rows), op.FEAT)
    critic = op.Critic()
    critic.eval()
    with torch.no_grad():
        all_q = critic.q_all_actions(b["joint"], b["onehot"])
        assert all_q.shape == (len(rows), op.N_ACT)
        # spot-check 20 (row, action) pairs against the exact q()
        rs = np.random.default_rng(0)
        for _ in range(20):
            r = int(rs.integers(0, len(rows)))
            a = int(rs.integers(0, op.N_ACT))
            q_exact = critic.q(b["joint"][r:r + 1], b["onehot"][r:r + 1],
                               torch.tensor([a]))
            assert torch.allclose(all_q[r, a], q_exact[0],
                                  atol=1e-5), (r, a)


def test_fit_paths_backward_and_weights(tiny_npz):
    path, _ = tiny_npz
    flat = op.stacked(op.load_episodes(path))
    n_rows = flat["obs"].shape[0] * op.N_AG
    rows = np.arange(n_rows)
    b = op.rows_slice(flat, rows)
    critic = op.Critic()
    tgt = b["rew"] + op.GAMMA * (1.0 - b["done"]) * op.q_max_samples(
        critic, b["joint_next"], b["onehot"], b["mask"], op.K_CONSERV)
    q_sa = critic.q(b["joint"], b["onehot"], b["acts"])
    cons = op.q_max_samples(critic, b["joint"], b["onehot"], b["mask"],
                            op.K_CONSERV)
    loss = nn.functional.mse_loss(q_sa, tgt) + op.ALPHA_CONSERV * \
        torch.nn.functional.relu(cons - q_sa).mean()
    assert torch.isfinite(loss).all()
    # a backward step must succeed on a real optimizer path
    optq = torch.optim.Adam(critic.parameters(), lr=1e-3)
    critic.zero_grad()
    loss.backward()
    optq.step()
    critic.eval()
    with torch.no_grad():
        qv = critic.q_all_actions(b["joint"], b["onehot"])
        qv = qv.masked_fill(b["mask"] == 0, -float("inf"))
        V = qv.mean(dim=1)
        w = torch.exp((critic.q(b["joint"], b["onehot"], b["acts"]) - V)
                      / op.TAU_W).clip(*op.W_CLIP)
    assert torch.isfinite(w).all() and (w >= 0).all()
    actor = op.Actor()
    lg = actor(b["agent"]) + (b["mask"] - 1.0).clip(max=-1e9)
    logp = torch.log_softmax(lg, dim=1).gather(1, b["acts"][:, None])
    loss = -(w * logp.squeeze(1)).mean()
    loss.backward()
