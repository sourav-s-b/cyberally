"""IQL-discrete on heuristic logs (proposal 14, rung 2).

In-sample offline RL: expectile value regression (upper envelope, never
queries OOD actions) + SARSA-style Q backup + advantage-weighted
extraction into the factorized GRU actor. The extracted policy is a plain
FactorizedRNNAgent(872) — keys match the EPyMARL layout, so pool eval
reuses the mappo_ckpt_factorized path.

Q/V are feedforward MLPs on single-step transitions (obs already carries
temporal ages/belief features); only the extracted policy is recurrent,
trained with per-step AWR weights from the frozen Q/V.
"""

import argparse
import json
import os
from types import SimpleNamespace as SN

import numpy as np


def expectile_loss(diff, tau):
    """diff = Q - V. Asymmetric L2: tau weight above, (1-tau) below."""
    import torch as th
    w = th.where(diff < 0, 1.0 - tau, tau)
    return (w * diff.pow(2)).mean()


def awr_weights(adv, beta, max_w=100.0):
    """Clipped exp weights. Returns (weights, frac_clipped)."""
    import torch as th
    w = th.exp(beta * adv)
    clipped = (w > max_w).float().mean()
    return w.clamp(max=max_w), clipped


def ess_frac(w):
    """Effective-sample-size fraction (Martino et al. 2017): (Sw)^2/Sw^2 / N.
    1.0 = uniform (degenerate, no selection); ->0 = single-point collapse."""
    import torch as th
    w = w.float()
    n = w.numel()
    return float((w.sum().pow(2) / w.pow(2).sum().clamp(min=1e-12)) / max(n, 1))


def auto_beta(adv, target=0.3, lo=0.0, hi=8.0, iters=25):
    """Bisection on beta so ESS frac of exp(beta*adv) hits target (V2/MPO:
    dual-solved temperature replaces the static-alpha hack). Monotone
    decreasing in beta, so bisection is exact. Returns (beta, achieved)."""
    import torch as th
    assert adv.numel() > 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        with th.no_grad():
            w = th.exp(mid * adv).clamp(max=100.0)
        if ess_frac(w) > target:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    with th.no_grad():
        achieved = ess_frac(th.exp(beta * adv).clamp(max=100.0))
    return float(beta), float(achieved)


class QNet:
    pass


def build_mlp(in_dim, out_dim, hidden=256, layernorm=False):
    import torch.nn as nn
    layers = [nn.Linear(in_dim, hidden)]
    if layernorm:
        layers.append(nn.LayerNorm(hidden))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden, hidden))
    if layernorm:
        layers.append(nn.LayerNorm(hidden))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def main():
    parser = argparse.ArgumentParser(description="IQL-discrete on offline logs")
    parser.add_argument("--demos", default="blue/results/offline_logs_14.npz")
    parser.add_argument("--tau", type=float, default=0.7,
                        help="expectile for V (upper envelope)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="AWR inverse temperature (fixed mode, or the "
                             "upper bound for auto mode)")
    parser.add_argument("--awr-mode", default="fixed",
                        choices=("fixed", "auto", "binary"),
                        help="fixed: exp(beta*adv) as before; auto: bisect "
                             "beta for --ess-target (V2/MPO dual-style); "
                             "binary: CRR-style 1[adv>0] filter (spike-proof)")
    parser.add_argument("--ess-target", type=float, default=0.3,
                        help="target ESS fraction for auto mode")
    parser.add_argument("--awr-alpha", type=float, default=1.0,
                        help="blend AWR weights with uniform BC (1.0 = pure "
                             "IQL, 0.0 = BC). Coverage guardrail: sweeping "
                             "actions have ~zero advantage and are dropped "
                             "by pure AWR.")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="divide team rewards by this before Q/V "
                             "training (100.0 matches the RvS RTG scale; "
                             "tames bootstrap magnitudes)")
    parser.add_argument("--layernorm", action="store_true",
                        help="LayerNorm in Q/V MLPs (value-stabilization "
                             "variant, rung-3 groundwork)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q-iters", type=int, default=None,
                        help="Q/V update iters. Default: scaled to data "
                             "(Q_EPOCHS passes over transitions; Phase C "
                             "rule — fixed iters undertrain big logs and "
                             "over-updating diverges, see proposal 14 "
                             "Phase A). Set explicitly to override.")
    parser.add_argument("--q-epochs", type=int, default=210,
                        help="target passes over transitions when --q-iters "
                             "is unset (24-ep @20k iters ~= 210 epochs, the "
                             "stable rung-2 regime)")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pol-epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    cli = parser.parse_args()

    import torch as th
    th.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    import sys
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from blue.common import logutil
    sys.path.insert(0, os.path.join(_REPO_ROOT, "third_party", "epymarl", "src"))
    from modules.agents import REGISTRY as agent_REGISTRY
    from blue.policies import factorized
    agent_REGISTRY["rnn_factorized"] = factorized.FactorizedRNNAgent

    data = np.load(cli.demos)
    obs = data["obs"]        # (E, A, T, D)
    actions = data["actions"]
    masks = data["masks"]
    rewards = data["rewards"]  # (E, T) team reward
    lengths = data["lengths"]
    n_eps, n_agents, max_len, obs_dim = obs.shape
    n_actions = int(data["masks"].shape[3])

    # Flatten to transitions. s' = next tick obs (same agent seq); done at
    # last valid tick. Free-tick mask (mask sum > 1) marks real decisions.
    S, A, R, NS, D, F = [], [], [], [], [], []
    for e in range(n_eps):
        t = int(lengths[e])
        for i in range(n_agents):
            ep_o = obs[e, i, :t]
            ep_a = actions[e, i, :t]
            ep_m = masks[e, i, :t]
            ep_r = rewards[e, :t]
            free = (ep_m.sum(-1) > 1).astype(np.float32)
            S.append(ep_o[:-1])
            A.append(ep_a[:-1])
            R.append(ep_r[:-1])
            NS.append(ep_o[1:])
            D.append(np.zeros(t - 1, dtype=np.float32))
            D[-1][-1] = 1.0  # episode ends after last tick... (see below)
            F.append(free[:-1])
    # Fix dones: only the final tick of each episode is terminal.
    S = np.concatenate(S).astype(np.float32)
    A = np.concatenate(A)
    R = np.concatenate(R).astype(np.float32)
    NS = np.concatenate(NS).astype(np.float32)
    F = np.concatenate(F).astype(np.float32)
    n_tr = S.shape[0]
    # Rebuild dones exactly: last transition of each (ep, agent) sequence.
    D = np.zeros(n_tr, dtype=np.float32)
    off = 0
    for e in range(n_eps):
        t = int(lengths[e]) - 1
        for _ in range(n_agents):
            D[off + t - 1] = 1.0
            off += t
    q_iters = cli.q_iters
    if q_iters is None:
        q_iters = max(1, int(cli.q_epochs * n_tr / cli.batch))
        print(f"q-iters auto: {q_iters} ({cli.q_epochs} epochs x "
              f"{n_tr} transitions / batch {cli.batch})")
    print(f"transitions: {n_tr} (from {n_eps} eps x {n_agents} agents), "
          f"free frac {F.mean():.3f}, reward range [{R.min():.1f}, {R.max():.1f}]")

    tS = th.from_numpy(S)
    tA = th.from_numpy(A)
    tR = th.from_numpy(R / cli.reward_scale)
    print(f"reward scale 1/{cli.reward_scale}, scaled range "
          f"[{float(tR.min()):.3f}, {float(tR.max()):.3f}]")
    tNS = th.from_numpy(NS)
    tD = th.from_numpy(D)

    q = build_mlp(obs_dim, n_actions, layernorm=cli.layernorm)
    qt = build_mlp(obs_dim, n_actions, layernorm=cli.layernorm)
    qt.load_state_dict(q.state_dict())
    v = build_mlp(obs_dim, 1, layernorm=cli.layernorm)
    opt_q = th.optim.Adam(q.parameters(), lr=cli.lr)
    opt_v = th.optim.Adam(v.parameters(), lr=cli.lr)
    qprog = logutil.Progress(q_iters)
    qhist = []

    for it in range(1, q_iters + 1):
        idx = th.from_numpy(np.random.choice(n_tr, cli.batch, replace=False))
        s, a, r, ns, d = tS[idx], tA[idx], tR[idx], tNS[idx], tD[idx]
        with th.no_grad():
            q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            target_q = r + cli.gamma * (1.0 - d) * v(ns).squeeze(1)
        v_pred = v(s).squeeze(1)
        # Expectile on (Q - V): V chases the upper envelope of Q.
        loss_v = expectile_loss(q_sa.detach() - v_pred, cli.tau)
        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()
        with th.no_grad():
            v_s = v(s).squeeze(1)
            v_ns = v(ns).squeeze(1)
            tq = r + cli.gamma * (1.0 - d) * v_ns
        q_all = q(s)
        q_pred = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        loss_q = th.nn.functional.mse_loss(q_pred, tq)
        opt_q.zero_grad()
        loss_q.backward()
        opt_q.step()
        with th.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(0.995).add_(p, alpha=0.005)
        if it % 5000 == 0 or it == 1:
            el = qprog.elapsed()
            rate = it / max(el, 1e-9)
            qhist.append({"iter": it, "elapsed_s": round(el, 1),
                          "iters_per_s": round(rate, 1),
                          "loss_v": float(loss_v), "loss_q": float(loss_q),
                          "v_mean": float(v_s.mean())})
            print(f"iter {it:6d} {rate:6.1f}it/s loss_v {float(loss_v):.4f} "
                  f"loss_q {float(loss_q):.4f} v_mean {float(v_s.mean()):.2f} "
                  f"{qprog.line(it)}",
                  flush=True)

    # Advantages on ALL transitions (frozen nets), then AWR weights.
    with th.no_grad():
        Q_all = q(tS).gather(1, tA.unsqueeze(1)).squeeze(1)
        V_all = v(tS).squeeze(1)
        adv = Q_all - V_all
    free_t = th.from_numpy(F)  # busy ticks contribute nothing (as in BC)
    adv_free = adv[free_t.bool()]
    beta_used = cli.beta
    awr_note = cli.awr_mode
    if cli.awr_mode == "binary":
        # CRR-style: keep positives at full weight, floor the rest (never
        # 0: zero-advantage sweeps stay legal under the alpha blend).
        w = th.where(adv > 0, th.ones_like(adv), th.full_like(adv, 0.05))
        frac_clip = th.tensor(0.0)
    else:
        if cli.awr_mode == "auto":
            beta_used, ess_hit = auto_beta(adv_free, target=cli.ess_target,
                                           hi=cli.beta)
            awr_note = f"auto(beta={beta_used:.3f},ess={ess_hit:.3f})"
        w, frac_clip = awr_weights(adv, beta_used)
    ess0 = ess_frac((w * free_t)[free_t.bool()])
    if ess0 > 0.9 and cli.awr_mode == "auto":
        # Uniformity escape (RLPD-style): advantages collapsed (~0 std);
        # standardize over free ticks and re-solve so selection exists.
        with th.no_grad():
            mu, sd = adv_free.mean(), adv_free.std().clamp(min=1e-6)
            adv_std = (adv - mu) / sd
        beta_used, ess_hit = auto_beta(
            adv_std[free_t.bool()], target=cli.ess_target, hi=cli.beta)
        w, frac_clip = awr_weights(adv_std, beta_used)
        ess0 = ess_frac((w * free_t)[free_t.bool()])
        awr_note = (f"auto-standardized(beta={beta_used:.3f},ess={ess_hit:.3f})")
    w = w * free_t
    if cli.awr_alpha < 1.0:
        w = cli.awr_alpha * w + (1.0 - cli.awr_alpha) * free_t
    print(f"adv: mean {float(adv.mean()):.3f} std {float(adv.std()):.3f} "
          f"max {float(adv.max()):.2f} | mode {awr_note} "
          f"ess_frac {ess0:.3f} | w mean {float(w.mean()):.3f} "
          f"frac_clipped {float(frac_clip):.4f} "
          f"frac_zero(free) {(float((w == 0).float().mean())):.3f}")

    # Extraction: factorized GRU, full-episode weighted CE.
    seq_obs = th.from_numpy(obs.reshape(-1, max_len, obs_dim))
    seq_act = th.from_numpy(actions.reshape(-1, max_len))
    seq_mask = th.from_numpy(masks.reshape(-1, max_len, n_actions))
    seq_len = th.from_numpy(np.repeat(lengths, n_agents))
    # Per-step weights aligned to (seq, tick): rebuild from transition order.
    w_seq = w.reshape(-1, max_len - 1)
    pad = th.zeros(w_seq.shape[0], 1)
    w_seq = th.cat([w_seq, pad], dim=1)  # last tick has no transition
    n_seqs = seq_obs.shape[0]
    args = SN(hidden_dim=cli.hidden_dim, n_actions=n_actions, use_rnn=True,
              n_agents=n_agents, attn_layers=0)
    policy = factorized.FactorizedRNNAgent(obs_dim, args)
    opt_p = th.optim.Adam(policy.parameters(), lr=1e-3)
    order = np.arange(n_seqs)
    pprog = logutil.Progress(cli.pol_epochs)
    phist = []
    for epoch in range(1, cli.pol_epochs + 1):
        np.random.shuffle(order)
        tot_loss, tot_w = 0.0, 0.0
        for start in range(0, n_seqs, 16):
            idx = order[start:start + 16]
            b_obs = seq_obs[idx]
            b_act = seq_act[idx]
            b_len = seq_len[idx]
            b_w = w_seq[idx]
            b, t, _ = b_obs.shape
            h = th.zeros(b, cli.hidden_dim)
            logits = []
            hh = h
            for step in range(t):
                qq, hh = policy(b_obs[:, step], hh)
                logits.append(qq)
            logits = th.stack(logits, dim=1)
            free = (seq_mask[idx].sum(-1) > 1).float()
            valid = ((th.arange(t).unsqueeze(0) < b_len.unsqueeze(1)).float()
                     * free)
            ce = th.nn.functional.cross_entropy(
                logits.reshape(-1, n_actions), b_act.reshape(-1),
                reduction="none").reshape(b, t)
            denom = (b_w * valid).sum().clamp(min=1e-6)
            loss = ((ce * b_w * valid).sum() / denom)
            opt_p.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt_p.step()
            tot_loss += float(loss) * float(denom)
            tot_w += float(denom)
        phist.append({"epoch": epoch, "elapsed_s": round(pprog.elapsed(), 1),
                      "wloss": tot_loss / max(tot_w, 1e-9)})
        print(f"pol epoch {epoch:3d} wloss {tot_loss/max(tot_w,1e-9):.4f} "
              f"{pprog.line(epoch)}",
              flush=True)

    import datetime
    out = cli.out or ("blue/results/models/iql_"
                      + datetime.datetime.now(datetime.timezone.utc)
                      .strftime("%Y%m%dT%H%M%SZ"))
    ckpt_dir = os.path.join(out, "0")
    os.makedirs(ckpt_dir, exist_ok=True)
    th.save(policy.state_dict(), os.path.join(ckpt_dir, "agent.th"))
    with open(os.path.join(out, "iql_manifest.json"), "w") as f:
        json.dump({"demos": cli.demos, "tau": cli.tau, "beta": cli.beta,
                   "awr_mode": cli.awr_mode, "ess_target": cli.ess_target,
                   "beta_used": beta_used, "ess_frac": ess0,
                   "awr_note": awr_note,
                   "awr_alpha": cli.awr_alpha,
                   "reward_scale": cli.reward_scale,
                   "layernorm": cli.layernorm,
                   "gamma": cli.gamma, "q_iters": q_iters,
                   "q_epochs": cli.q_epochs,
                   "pol_epochs": cli.pol_epochs,
                   "hidden_dim": cli.hidden_dim, "seed": cli.seed,
                   "final_loss_v": float(loss_v), "final_loss_q": float(loss_q),
                   "adv_mean": float(adv.mean()), "adv_std": float(adv.std()),
                   "adv_max": float(adv.max()),
                   "w_mean": float(w.mean()),
                   "frac_clipped": float(frac_clip)}, f, indent=1)
    with open(os.path.join(out, "train_log.jsonl"), "w") as f:
        for row in qhist:
            f.write(json.dumps({"phase": "q", **row}) + "\n")
        for row in phist:
            f.write(json.dumps({"phase": "pol", **row}) + "\n")
    print("wrote", ckpt_dir + "/agent.th")
    return out


if __name__ == "__main__":
    main()
