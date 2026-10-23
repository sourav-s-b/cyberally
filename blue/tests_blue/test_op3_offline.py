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
    assert flat["s"].shape == (n_t, op.FEAT * op.N_AG)
    assert flat["c_x"].shape == (n_t * op.N_AG, op.FEAT)
    assert flat["mask"].shape == (n_t * op.N_AG, op.N_ACT)
    assert flat["acts"].shape == (n_t, op.N_AG)
    # every logged action legal in its mask
    acts = flat["acts"].reshape(-1)
    for t in range(acts.shape[0]):
        assert flat["mask"][t, acts[t]] == 1


def test_fit_paths_backward_and_weights(tiny_npz):
    path, _ = tiny_npz
    flat = op.stacked(op.load_episodes(path))
    n_t = flat["s"].shape[0]
    n_rows = n_t * op.N_AG
    dev_m = torch.from_numpy(flat["mask"])
    dev_a = torch.from_numpy(flat["acts"].reshape(-1))
    dev_x = torch.from_numpy(flat["c_x"])
    ag = torch.from_numpy(np.tile(np.eye(op.N_AG,
                                          dtype=np.float32), (n_t, 1)))
    oc = torch.from_numpy(np.repeat(flat["s"], op.N_AG, axis=0))
    nextx = torch.from_numpy(np.repeat(flat["s_next"], op.N_AG, axis=0))
    rj = torch.from_numpy(np.repeat(flat["rew"], op.N_AG).astype(np.float32))
    dj = torch.from_numpy(np.repeat(flat["dones"], op.N_AG).astype(np.float32))
    b = torch.arange(n_rows)
    critic = op.Critic()
    tgt = rj[b] + op.GAMMA * (1.0 - dj[b]) * op.q_max_samples(
        critic, nextx[b], ag[b], dev_m[b], op.K_CONSERV)
    q_sa = critic.q(oc[b], ag[b], dev_a[b])
    cons = op.q_max_samples(critic, oc[b], ag[b], dev_m[b], op.K_CONSERV)
    loss = nn.functional.mse_loss(q_sa, tgt) + op.ALPHA_CONSERV * \
        torch.nn.functional.relu(cons - q_sa).mean()
    assert torch.isfinite(loss).all()
    critic.eval()
    with torch.no_grad():
        q_all = critic.q(oc[b], ag[b], dev_a[b])
        qv = critic.q(oc[b].repeat_interleave(op.N_ACT, dim=0),
                      ag[b].repeat_interleave(op.N_ACT, dim=0),
                      torch.arange(op.N_ACT).repeat(n_rows))
        qv = qv.view(n_rows, op.N_ACT).masked_fill(dev_m[b] == 0,
                                                   -float("inf"))
        V = qv.mean(dim=1)
        w = torch.exp((q_all - V) / op.TAU_W).clip(*op.W_CLIP)
    assert torch.isfinite(w).all() and (w >= 0).all()
    actor = op.Actor()
    lg = actor(dev_x[b]) + (dev_m[b] - 1.0).clip(max=-1e9)
    logp = torch.log_softmax(lg, dim=1).gather(1, dev_a[b][:, None])
    loss = -(w * logp.squeeze(1)).mean()
    loss.backward()
