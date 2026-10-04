"""Narrow PPO residual for the sweep scorer (Phase 4).

Policy: OrderedPolicy rules 1-3 fixed + ResidualAnchor(base lancer
scorer, bonus M) + learned residual MLP over the 14-dim Phase-3 host
rows. Only sweep Analyse decisions are learner actions; rules/emergency
actions never enter the buffer (no masquerading).

PPO-clip on masked categoricals over eligible hosts, Monte-Carlo
advantage with a learned value baseline (pooled context), fresh
on-policy rollouts per iteration. Zero-initialized final layer ->
exact teacher at start.
"""

import numpy as np

FEAT_DIM = 14


def build_nets(feat_dim=FEAT_DIM, hidden=64):
    import torch.nn as nn

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU())
            self.head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x):
            return self.head(self.body(x)).squeeze(-1)

    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feat_dim + 1, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Residual(), Value()


def load_phase3_init(residual, path, hidden=64):
    """Copy Phase-3 imitation body into the residual; zero the head so
    argmax still reproduces the teacher exactly at pilot start."""
    import torch as th
    state = th.load(path, map_location="cpu")
    body = residual.body
    body[0].weight.data.copy_(state["0.weight"])
    body[0].bias.data.copy_(state["0.bias"])
    body[2].weight.data.copy_(state["2.weight"])
    body[2].bias.data.copy_(state["2.bias"])
    n_copied = 4
    return n_copied


def masked_choice(logits, mask, sample=True, seed_rng=None):
    """Sample (or argmax) from a masked categorical. Returns
    (index_into_cands, logprob, entropy)."""
    import torch as th
    m = th.from_numpy(np.asarray(mask, dtype=np.float32))
    l = logits + (1.0 - m) * -1e9
    probs = th.softmax(l, dim=0)
    if sample:
        idx = int(th.multinomial(probs, 1).item())
    else:
        idx = int(probs.argmax().item())
    logp = float(th.log_softmax(l, dim=0)[idx])
    ent = float(-(probs * th.log_softmax(l, dim=0)).sum())
    return idx, logp, ent


def ppo_update(residual, value, opt, buf, bonus=1.0, epochs=4,
               minibatch=256, clip=0.2, vf_coef=0.5, ent_coef=0.01):
    """One PPO pass over a fresh on-policy buffer. buf rows carry the
    frozen base (lancer) scores; combined logits = base + M*tanh(r) are
    rebuilt from the current residual so logprobs match rollout
    behavior exactly. Advantages normalized per buffer."""
    import torch as th
    F = th.from_numpy(np.stack([b["feats"] for b in buf]).astype(np.float32))
    C = th.from_numpy(np.stack([b["ctx"] for b in buf]).astype(np.float32))
    M = th.from_numpy(np.stack([b["mask"] for b in buf]).astype(np.float32))
    B = th.from_numpy(np.stack([b["base"] for b in buf]).astype(np.float32))
    CH = th.from_numpy(np.array([b["choice"] for b in buf]).astype(np.int64))
    OLD = th.from_numpy(np.array([b["logp"] for b in buf]).astype(np.float32))
    RET = th.from_numpy(np.array([b["ret"] for b in buf]).astype(np.float32))
    with th.no_grad():
        adv = RET - value(C)
        adv = (adv - adv.mean()) / (adv.std().clamp(min=1e-6))
    n = len(buf)
    tot = {"pol": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "nb": 0}
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, minibatch):
            bi = idx[s:s + minibatch]
            r = th.tanh(residual(F[bi]))
            logits = (B[bi] + bonus * r) * M[bi] + (1.0 - M[bi]) * -1e9
            logp_all = th.log_softmax(logits, dim=1)
            logp = logp_all.gather(1, CH[bi].unsqueeze(1)).squeeze(1)
            ratio = th.exp(logp - OLD[bi])
            pg1 = ratio * adv[bi]
            pg2 = th.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[bi]
            pol = -th.minimum(pg1, pg2).mean()
            vpred = value(C[bi])
            vf = th.nn.functional.mse_loss(vpred, RET[bi])
            probs = th.softmax(logits, dim=1)
            ent = -(probs * logp_all).sum(1).mean()
            loss = pol + vf_coef * vf - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                list(residual.parameters()) + list(value.parameters()), 1.0)
            opt.step()
            with th.no_grad():
                kl = float((OLD[bi] - logp).mean())
            tot["pol"] += float(pol.detach())
            tot["vf"] += float(vf.detach())
            tot["ent"] += float(ent.detach())
            tot["kl"] += kl
            tot["nb"] += 1
    return {k: (v / max(tot["nb"], 1)) for k, v in tot.items() if k != "nb"}
