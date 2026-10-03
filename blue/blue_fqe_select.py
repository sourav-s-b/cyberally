"""Rollout-free checkpoint ranking via FQE-V(s0) (Stream C / V3).

For each candidate factorized-GRU checkpoint: fit a fresh feedforward Q
with plain MSE TD backups *under the candidate's own greedy actions*
(Fitted Q Evaluation — never queries the sim), then score
V(s0) = mean_e,i Q(s0_ep,i, pi(s0_ep,i)) over log-episode initial ticks.
Rank candidates by seed-averaged V(s0).

Cross-checks (BVFT-lite, honest version):
  - every candidate is fit TWICE (two fit seeds); rank agreement across
    fit seeds is reported. A ranking that flips with the fit seed is not
    a ranking (this is the F1 lottery applied to the selector itself).
  - Monte-Carlo logging-policy anchor: mean per-episode log return. A
    sane FQE value sits near/above it, not 10x away (blowup detector).

Usage:
  python blue_fqe_select.py --demos results/offline_logs_14.npz \
      --ckpt results/models/iql_14 --ckpt results/models/iql_14a05 \
      --label iql_14 --label iql_14a05 --out /tmp/fqe.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_AGENTS_FALLBACK = 5


def flatten_log(demos):
    """Same transition layout as blue_iql (s, a, r, s', done, free) + s0."""
    data = np.load(demos)
    obs = data["obs"]        # (E, A, T, D)
    actions = data["actions"]
    masks = data["masks"]
    rewards = data["rewards"]  # (E, T) team reward
    lengths = data["lengths"]
    n_eps, n_agents, max_len, obs_dim = obs.shape
    n_actions = int(masks.shape[3])
    S, A, R, NS, S0 = [], [], [], [], []
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
            S0.append(ep_o[0])
    S = np.concatenate(S).astype(np.float32)
    A = np.concatenate(A)
    R = np.concatenate(R).astype(np.float32)
    NS = np.concatenate(NS).astype(np.float32)
    S0 = np.stack(S0).astype(np.float32)
    n_tr = S.shape[0]
    D = np.zeros(n_tr, dtype=np.float32)
    off = 0
    for e in range(n_eps):
        t = int(lengths[e]) - 1
        for _ in range(n_agents):
            D[off + t - 1] = 1.0
            off += t
    mc_anchor = float(rewards[:, :].sum(axis=1).mean())  # per-ep return
    return S, A, R, NS, D, S0, obs_dim, n_actions, n_agents, max_len, mc_anchor


def load_policy(ckpt_dir, obs_dim, n_actions, n_agents, hidden_dim,
                attn_layers, rtg_dim):
    import torch as th
    from types import SimpleNamespace as SN
    import blue_factorized_agent as factorized
    args = SN(hidden_dim=hidden_dim, n_actions=n_actions, use_rnn=True,
              n_agents=n_agents, attn_layers=attn_layers, rtg_dim=rtg_dim)
    policy = factorized.FactorizedRNNAgent(obs_dim, args)
    agent_path = os.path.join(ckpt_dir, "0", "agent.th")
    policy.load_state_dict(th.load(agent_path, map_location="cpu"))
    policy.eval()
    return policy


def greedy_actions(policy, S, hidden_dim, batch=4096):
    """Greedy actions with zero-init hidden per transition (memoryless
    scoring; matches how Q/V MLPs see the data, not full recurrence)."""
    import torch as th
    out = np.empty(S.shape[0], dtype=np.int64)
    with th.no_grad():
        for s in range(0, S.shape[0], batch):
            b = th.from_numpy(S[s:s + batch])
            h = th.zeros(b.shape[0], hidden_dim)
            logits, _ = policy(b, h)
            out[s:s + batch] = logits.argmax(-1).numpy()
    return out


def fit_fqe(S, A, R, NS, D, pi_S, pi_NS, obs_dim, n_actions, gamma,
            q_iters, batch, lr, seed):
    """Plain MSE FQE under fixed greedy actions pi(*). Returns Q net."""
    import torch as th
    from blue_iql import build_mlp
    th.manual_seed(seed)
    rng = np.random.RandomState(seed)
    n_tr = S.shape[0]
    tS = th.from_numpy(S)
    tR = th.from_numpy(R)
    tNS = th.from_numpy(NS)
    tD = th.from_numpy(D)
    tPiS = th.from_numpy(pi_S)
    tPiNS = th.from_numpy(pi_NS)
    q = build_mlp(obs_dim, n_actions)
    qt = build_mlp(obs_dim, n_actions)
    qt.load_state_dict(q.state_dict())
    opt = th.optim.Adam(q.parameters(), lr=lr)
    for it in range(1, q_iters + 1):
        idx = th.from_numpy(rng.choice(n_tr, batch, replace=False))
        s, r, ns, d = tS[idx], tR[idx], tNS[idx], tD[idx]
        with th.no_grad():
            tq = r + gamma * (1.0 - d) * qt(ns).gather(
                1, tPiNS[idx].unsqueeze(1)).squeeze(1)
        pred = q(s).gather(1, tPiS[idx].unsqueeze(1)).squeeze(1)
        loss = th.nn.functional.mse_loss(pred, tq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with th.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(0.995).add_(p, alpha=0.005)
    with th.no_grad():
        pred_all = q(tS).gather(1, tPiS.unsqueeze(1)).squeeze(1)
        with th.no_grad():
            tq_all = tR + gamma * (1.0 - tD) * qt(tNS).gather(
                1, tPiNS.unsqueeze(1)).squeeze(1)
        bellman_resid = float(th.nn.functional.mse_loss(
            pred_all, tq_all))
    return q, float(loss.detach()), bellman_resid


def score_candidate(demos_flat, ckpt_dir, hidden_dim, attn_layers, rtg_dim,
                    gamma, q_iters, batch, lr, fit_seeds):
    import torch as th
    S, A, R, NS, D, S0, obs_dim, n_actions, n_agents, _, _ = demos_flat
    policy = load_policy(ckpt_dir, obs_dim, n_actions, n_agents,
                         hidden_dim, attn_layers, rtg_dim)
    pi_S = greedy_actions(policy, S, hidden_dim)
    pi_S0 = greedy_actions(policy, S0, hidden_dim)
    pi_NS = greedy_actions(policy, NS, hidden_dim)
    v_s0, resids = [], []
    with th.no_grad():
        tS0 = th.from_numpy(S0)
        tPiS0 = th.from_numpy(pi_S0)
    for fs in fit_seeds:
        q, final_loss, resid = fit_fqe(S, A, R, NS, D, pi_S, pi_NS,
                                      obs_dim, n_actions, gamma,
                                      q_iters, batch, lr, fs)
        with th.no_grad():
            v = float(q(tS0).gather(1, tPiS0.unsqueeze(1)).squeeze(1)
                      .mean())
        v_s0.append(v)
        resids.append(resid)
    return v_s0, resids


def main():
    ap = argparse.ArgumentParser(description="FQE-V(s0) ckpt ranking")
    ap.add_argument("--demos", default="results/offline_logs_14.npz")
    ap.add_argument("--ckpt", action="append", default=[],
                    help="candidate ckpt dir (repeatable)")
    ap.add_argument("--label", action="append", default=[],
                    help="label per --ckpt (repeatable, same order)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--q-iters", type=int, default=20000,
                    help="fixed FQE budget per candidate per fit seed "
                         "(ranking needs consistency, not convergence)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
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

    flat = flatten_log(cli.demos)
    mc_anchor = flat[-1]
    print(f"demos {cli.demos}: {flat[0].shape[0]} transitions, "
          f"MC log-policy anchor {mc_anchor:.1f}")

    rows = []
    for ckpt_dir, label in zip(cli.ckpt, labels):
        v_s0, resids = score_candidate(
            flat, ckpt_dir, cli.hidden_dim, cli.attn_layers,
            cli.rtg_dim, cli.gamma, cli.q_iters, cli.batch, cli.lr,
            cli.fit_seeds)
        row = {"label": label, "ckpt": ckpt_dir,
               "v_s0_mean": float(np.mean(v_s0)),
               "v_s0_seeds": [float(v) for v in v_s0],
               "v_s0_spread": float(max(v_s0) - min(v_s0)),
               "bellman_resid": [float(r) for r in resids],
               "mc_anchor": mc_anchor, "gamma": cli.gamma,
               "q_iters": cli.q_iters, "fit_seeds": cli.fit_seeds}
        rows.append(row)
        print(f"{label:16s} V(s0) {row['v_s0_mean']:+8.1f} "
              f"seeds {[f'{v:+.1f}' for v in v_s0]} "
              f"spread {row['v_s0_spread']:.1f} resid {resids[0]:.3f}")

    rows.sort(key=lambda r: -r["v_s0_mean"])
    print("\nrank (FQE-V(s0), higher is better): "
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
