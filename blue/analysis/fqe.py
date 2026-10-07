"""Rollout-free checkpoint ranking via TRUE FQE-V(s0) (Stream C redo / V3).

v3 was invalid (see feasibility audit 2026-10-03): it regressed Q(s,pi(s))
on rewards earned by logged actions. This version implements Le et al.,
ICML 2019, Algorithm 3 faithfully:

  - regression on LOGGED actions:  Q(s_t, a_t) <- r_t + gamma*Q(s_{t+1}, pi(s_{t+1}))
  - bootstrap actions from the candidate with CARRIED GRU hidden state
    (zero-init at episode start, stepped through each sequence) and the
    DEPLOYMENT validity mask (masked argmax), matching rollout behavior.
  - discounted Monte-Carlo anchor for the logging policy (same gamma).

Usage:
  python -m blue.analysis.fqe --demos blue/results/offline_logs_14.npz \
      --ckpt blue/results/models/iql_14a05 --label iql_14a05 --out /tmp/fqe.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from blue.common import logutil


def flatten_log(demos, gamma):
    """Transitions (s, a_logged, r, s', done) + s0 + masks + sequences."""
    data = np.load(demos)
    obs = data["obs"]        # (E, A, T, D)
    actions = data["actions"]
    masks = data["masks"]
    rewards = data["rewards"]  # (E, T) team reward
    lengths = data["lengths"]
    n_eps, n_agents, max_len, obs_dim = obs.shape
    n_actions = int(masks.shape[3])
    S, A, R, NS = [], [], [], []
    seq_s, seq_m, seq_len = [], [], []  # full sequences for carried-hidden
    for e in range(n_eps):
        t = int(lengths[e])
        for i in range(n_agents):
            ep_o = obs[e, i, :t]
            ep_a = actions[e, i, :t]
            ep_m = masks[e, i, :t]
            ep_r = rewards[e, :t]
            S.append(ep_o[:-1])
            A.append(ep_a[:-1])
            R.append(ep_r[:-1])
            NS.append(ep_o[1:])
            seq_s.append(ep_o)
            seq_m.append(ep_m)
            seq_len.append(t)
    S = np.concatenate(S).astype(np.float32)
    A = np.concatenate(A)
    R = np.concatenate(R).astype(np.float32)
    NS = np.concatenate(NS).astype(np.float32)
    S0 = np.stack([obs[e, i, 0] for e in range(n_eps)
                   for i in range(n_agents)]).astype(np.float32)
    n_tr = S.shape[0]
    D = np.zeros(n_tr, dtype=np.float32)
    off = 0
    for e in range(n_eps):
        t = int(lengths[e]) - 1
        for _ in range(n_agents):
            D[off + t - 1] = 1.0
            off += t
    disc = gamma ** np.arange(max_len, dtype=np.float32)
    mc_anchor = float(((rewards[:, :max_len] * disc).sum(axis=1)).mean())
    mc_undisc = float(rewards.sum(axis=1).mean())
    return {"S": S, "A": A, "R": R, "NS": NS, "D": D, "S0": S0,
            "seq_s": seq_s, "seq_m": seq_m, "seq_len": seq_len,
            "obs_dim": obs_dim, "n_actions": n_actions,
            "n_agents": n_agents, "max_len": max_len,
            "mc_anchor": mc_anchor, "mc_undisc": mc_undisc}


def load_policy(ckpt_dir, obs_dim, n_actions, n_agents, hidden_dim,
                attn_layers, rtg_dim):
    import torch as th
    from types import SimpleNamespace as SN
    from blue.policies import factorized
    args = SN(hidden_dim=hidden_dim, n_actions=n_actions, use_rnn=True,
              n_agents=n_agents, attn_layers=attn_layers, rtg_dim=rtg_dim)
    policy = factorized.FactorizedRNNAgent(obs_dim, args)
    agent_path = os.path.join(ckpt_dir, "0", "agent.th")
    policy.load_state_dict(th.load(agent_path, map_location="cpu"))
    policy.eval()
    return policy


def carried_greedy_actions(policy, flat, hidden_dim):
    """Masked argmax per tick with carried hidden state (zero-init at each
    sequence start). Returns pi_S, pi_NS aligned to transitions + pi_S0."""
    import torch as th
    seq_s, seq_m, seq_len = flat["seq_s"], flat["seq_m"], flat["seq_len"]
    n_seq = len(seq_s)
    max_len = flat["max_len"]
    n_actions = flat["n_actions"]
    # Pad to batch.
    b_obs = np.zeros((n_seq, max_len, flat["obs_dim"]), dtype=np.float32)
    b_msk = np.zeros((n_seq, max_len, n_actions), dtype=np.float32)
    for k in range(n_seq):
        t = seq_len[k]
        b_obs[k, :t] = seq_s[k]
        b_msk[k, :t] = seq_m[k]
    t_obs = th.from_numpy(b_obs)
    t_msk = th.from_numpy(b_msk)
    lens = th.tensor(seq_len)
    pi = np.zeros((n_seq, max_len), dtype=np.int64)
    with th.no_grad():
        h = th.zeros(n_seq, hidden_dim)
        for step in range(max_len):
            logits, h_new = policy(t_obs[:, step], h)
            masked = logits + (1.0 - t_msk[:, step]) * -1e9
            pi[:, step] = masked.argmax(-1).numpy()
            # Freeze hidden past sequence end (keep last valid).
            alive = (step + 1 < lens).float().unsqueeze(1)
            h = h_new * alive + h * (1.0 - alive)
    pi_S, pi_NS, pi_S0 = [], [], []
    for k in range(n_seq):
        t = seq_len[k]
        pi_S.append(pi[k, :t - 1])
        pi_NS.append(pi[k, 1:t])
        pi_S0.append(pi[k, 0])
    return (np.concatenate(pi_S), np.concatenate(pi_NS),
            np.array(pi_S0, dtype=np.int64))


def fit_fqe(flat, pi_NS, obs_dim, n_actions, gamma,
            q_iters, batch, lr, seed):
    """TRUE FQE: regress Q(s, a_logged); bootstrap Q(s', pi(s'))."""
    import torch as th
    from blue.training.iql import build_mlp
    th.manual_seed(seed)
    rng = np.random.RandomState(seed)
    S, A, R, NS, D = flat["S"], flat["A"], flat["R"], flat["NS"], flat["D"]
    n_tr = S.shape[0]
    tS = th.from_numpy(S)
    tA = th.from_numpy(A)
    tR = th.from_numpy(R)
    tNS = th.from_numpy(NS)
    tD = th.from_numpy(D)
    tPiNS = th.from_numpy(pi_NS)
    q = build_mlp(obs_dim, n_actions)
    qt = build_mlp(obs_dim, n_actions)
    qt.load_state_dict(q.state_dict())
    opt = th.optim.Adam(q.parameters(), lr=lr)
    prog = logutil.Progress(q_iters)
    for it in range(1, q_iters + 1):
        idx = th.from_numpy(rng.choice(n_tr, batch, replace=False))
        s, a, r, ns, d = (tS[idx], tA[idx], tR[idx], tNS[idx], tD[idx])
        with th.no_grad():
            tq = r + gamma * (1.0 - d) * qt(ns).gather(
                1, tPiNS[idx].unsqueeze(1)).squeeze(1)
        pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)  # LOGGED action
        loss = th.nn.functional.mse_loss(pred, tq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with th.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(0.995).add_(p, alpha=0.005)
        if it % 5000 == 0 or it == 1:
            print(f"  fqe iter {it:6d} loss {float(loss.detach()):.4f} "
                  f"{prog.line(it)}", flush=True)
    with th.no_grad():
        pred_all = q(tS).gather(1, tA.unsqueeze(1)).squeeze(1)
        tq_all = tR + gamma * (1.0 - tD) * qt(tNS).gather(
            1, tPiNS.unsqueeze(1)).squeeze(1)
        bellman_resid = float(th.nn.functional.mse_loss(pred_all, tq_all))
    return q, float(loss.detach()), bellman_resid


def score_candidate(flat, ckpt_dir, hidden_dim, attn_layers, rtg_dim,
                    gamma, q_iters, batch, lr, fit_seeds):
    import torch as th
    obs_dim, n_actions = flat["obs_dim"], flat["n_actions"]
    policy = load_policy(ckpt_dir, obs_dim, n_actions, flat["n_agents"],
                         hidden_dim, attn_layers, rtg_dim)
    pi_S, pi_NS, pi_S0 = carried_greedy_actions(policy, flat, hidden_dim)
    assert pi_S.shape[0] == flat["S"].shape[0]
    v_s0, resids = [], []
    with th.no_grad():
        tS0 = th.from_numpy(flat["S0"])
        tPiS0 = th.from_numpy(pi_S0)
    for fs in fit_seeds:
        q, final_loss, resid = fit_fqe(flat, pi_NS, obs_dim, n_actions,
                                      gamma, q_iters, batch, lr, fs)
        with th.no_grad():
            v = float(q(tS0).gather(1, tPiS0.unsqueeze(1)).squeeze(1)
                      .mean())
        v_s0.append(v)
        resids.append(resid)
    return v_s0, resids


def main():
    ap = argparse.ArgumentParser(description="true-FQE V(s0) ckpt ranking")
    ap.add_argument("--demos", default="blue/results/offline_logs_14.npz")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--q-iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--attn-layers", type=int, default=0)
    ap.add_argument("--rtg-dim", type=int, default=0)
    ap.add_argument("--fit-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    assert cli.ckpt, "pass at least one --ckpt"
    labels = cli.label or [os.path.basename(c.rstrip("/"))
                           for c in cli.ckpt]
    assert len(labels) == len(cli.ckpt), "--label count must match --ckpt"

    flat = flatten_log(cli.demos, cli.gamma)
    print(f"demos {cli.demos}: {flat['S'].shape[0]} transitions, "
          f"MC log-policy anchor disc {flat['mc_anchor']:.1f} "
          f"(undisc {flat['mc_undisc']:.1f})")

    rows = []
    cprog = logutil.Progress(len(cli.ckpt))
    for n, (ckpt_dir, label) in enumerate(zip(cli.ckpt, labels), 1):
        print(f"candidate {label} {cprog.line(n - 1)}", flush=True)
        v_s0, resids = score_candidate(
            flat, ckpt_dir, cli.hidden_dim, cli.attn_layers,
            cli.rtg_dim, cli.gamma, cli.q_iters, cli.batch, cli.lr,
            cli.fit_seeds)
        row = {"label": label, "ckpt": ckpt_dir,
               "v_s0_mean": float(np.mean(v_s0)),
               "v_s0_seeds": [float(v) for v in v_s0],
               "v_s0_spread": float(max(v_s0) - min(v_s0)),
               "bellman_resid": [float(r) for r in resids],
               "mc_anchor_disc": flat["mc_anchor"],
               "mc_anchor_undisc": flat["mc_undisc"],
               "gamma": cli.gamma, "q_iters": cli.q_iters,
               "fit_seeds": cli.fit_seeds}
        rows.append(row)
        print(f"{label:16s} V(s0) {row['v_s0_mean']:+8.1f} "
              f"seeds {[f'{v:+.1f}' for v in v_s0]} "
              f"spread {row['v_s0_spread']:.1f} resid {resids[0]:.3f}")

    rows.sort(key=lambda r: -r["v_s0_mean"])
    print("\nrank (true-FQE V(s0), higher is better): "
          + " > ".join(r["label"] for r in rows))
    if len(rows) > 1 and len(cli.fit_seeds) > 1:
        total_range = abs(rows[0]["v_s0_mean"] - rows[-1]["v_s0_mean"])
        agree = all(r["v_s0_spread"] < total_range / 2 for r in rows)
        print(f"fit-seed rank stability: {'STABLE' if agree else 'UNSTABLE'} "
              f"(spreads vs total range {total_range:.1f})")
    else:
        print("fit-seed rank stability: n/a (need >1 ckpt and >1 fit seed)")
    if cli.out:
        with open(cli.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", cli.out)


if __name__ == "__main__":
    main()
