"""Return-conditioned supervised training (RvS rung 1, proposal 14).

Same loop as blue_bc_pretrain.py on the factorized head, but the GRU also
sees returns-to-go (team RTG / RTG_SCALE appended to obs), trained with
cross-entropy against the logged teacher action. At eval (blue_eval_rvs.py)
the policy is conditioned on a HIGH target return instead of the true RTG.

Data must carry per-tick team rewards (blue_collect_bc.py --mix logs).
"""

import argparse
import json
import os
from types import SimpleNamespace as SN

import numpy as np

RTG_SCALE = 100.0


def compute_rtg(rewards, lengths):
    """Per-episode returns-to-go, normalized. rewards (E, T) team reward."""
    E, T = rewards.shape
    rtg = np.zeros_like(rewards)
    for e in range(E):
        t = int(lengths[e])
        rtg[e, :t] = np.cumsum(rewards[e, :t][::-1])[::-1] / RTG_SCALE
    return rtg.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="RvS pre-train factorized actor")
    parser.add_argument("--demos", default="results/offline_logs_14.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-seqs", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-layers", type=int, default=0)
    parser.add_argument("--agent", default="rnn_factorized",
                        choices=("rnn", "rnn_factorized"))
    parser.add_argument("--aux-host-weight", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    cli = parser.parse_args()

    import torch as th
    th.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "/home/sourav/Projects/cyberally/third_party/epymarl/src")
    from modules.agents import REGISTRY as agent_REGISTRY
    import blue_factorized_agent as factorized
    agent_REGISTRY["rnn_factorized"] = factorized.FactorizedRNNAgent
    data = np.load(cli.demos)
    obs = data["obs"]        # (E, A, T, D)
    actions = data["actions"]
    masks = data["masks"]
    rewards = data["rewards"]  # (E, T) team reward; REQUIRED for RvS
    lengths = data["lengths"]
    n_eps, n_agents, max_len, obs_dim = obs.shape
    n_actions = int(data["masks"].shape[3])
    rtg = compute_rtg(rewards, lengths)  # (E, T)
    print(f"demos: {n_eps} eps x {n_agents} agents x {max_len} ticks, "
          f"obs {obs_dim}, actions {n_actions}")
    print(f"episode returns: min {rewards.sum(1).min():.1f} "
          f"max {rewards.sum(1).max():.1f} mean {rewards.sum(1).mean():.1f}")

    # Flatten episodes x agents; RTG is team-shared so repeat per agent.
    seq_obs = th.from_numpy(obs.reshape(-1, max_len, obs_dim))
    seq_rtg = th.from_numpy(np.repeat(rtg, n_agents, axis=0)
                            .reshape(-1, max_len, 1))
    seq_in = th.cat([seq_obs, seq_rtg], dim=-1)
    seq_act = th.from_numpy(actions.reshape(-1, max_len))
    seq_mask = th.from_numpy(masks.reshape(-1, max_len, n_actions))
    seq_len = th.from_numpy(np.repeat(lengths, n_agents))
    n_seqs = seq_in.shape[0]
    in_dim = obs_dim + 1

    freq = np.bincount(actions.reshape(-1), minlength=n_actions).astype(np.float64)
    weight = th.from_numpy((1.0 / np.sqrt(freq + 1.0)).astype(np.float32))
    weight = weight * (n_actions / weight.sum())
    loss_fn = th.nn.CrossEntropyLoss(weight=weight, reduction="none")

    args = SN(hidden_dim=cli.hidden_dim, n_actions=n_actions, use_rnn=True,
              n_agents=n_agents, attn_layers=cli.attn_layers,
              rtg_dim=1)  # trailing RTG scalar; heads see obs+id as before
    if cli.agent != "rnn_factorized":
        raise SystemExit("RvS needs rnn_factorized (flat head has no RTG split)")
    agent = agent_REGISTRY[cli.agent](in_dim, args)
    opt = th.optim.Adam(agent.parameters(), lr=cli.lr)

    order = np.arange(n_seqs)
    tot_nonsleep_correct = tot_nonsleep = 0
    host_correct = host_toks = 0
    use_aux = cli.aux_host_weight > 0
    if use_aux and cli.agent != "rnn_factorized":
        raise SystemExit("--aux-host-weight needs --agent rnn_factorized")
    aux_fn = th.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    for epoch in range(1, cli.epochs + 1):
        np.random.shuffle(order)
        tot_loss, tot_correct, tot_toks = 0.0, 0, 0
        for start in range(0, n_seqs, cli.batch_seqs):
            idx = order[start:start + cli.batch_seqs]
            b_in = seq_in[idx]
            b_act = seq_act[idx]
            b_len = seq_len[idx]
            b, t, _ = b_in.shape
            h = th.zeros(b, cli.hidden_dim)
            logits = []
            hosts = [] if use_aux else None
            hh = h
            for step in range(t):
                if use_aux:
                    q, hh, hs, _, _ = agent.decompose(b_in[:, step], hh)
                    hosts.append(hs)
                else:
                    q, hh = agent(b_in[:, step], hh)
                logits.append(q)
            logits = th.stack(logits, dim=1)
            free = (seq_mask[idx].sum(-1) > 1).float()
            mask = ((th.arange(t).unsqueeze(0) < b_len.unsqueeze(1)).float()
                    * free)
            denom = mask.sum().clamp(min=1)
            loss = (loss_fn(logits.reshape(-1, n_actions),
                            b_act.reshape(-1)).reshape(b, t) * mask).sum() / denom
            host_acc_term = ""
            if use_aux:
                host_logits = th.stack(hosts, dim=1)
                host_tgt = ((b_act - 2) // 3).clone()
                host_tgt[(b_act < 2)] = -100
                aux = (aux_fn(host_logits.reshape(-1, host_logits.shape[-1]),
                              host_tgt.reshape(-1)).reshape(b, t)
                       * mask).sum() / denom
                loss = loss + cli.aux_host_weight * aux
                with th.no_grad():
                    hpred = host_logits.argmax(-1)
                    hmask = (host_tgt != -100) & mask.bool()
                    host_correct += int(((hpred == host_tgt) * hmask).sum())
                    host_toks += int(hmask.sum())
                    host_acc_term = (f" host-acc "
                                     f"{host_correct / max(host_toks, 1):.4f}")
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(agent.parameters(), 10.0)
            opt.step()
            with th.no_grad():
                pred = logits.argmax(-1)
                correct = (pred == b_act) * mask.bool()
                non_sleep = (b_act != 0) * mask.bool()
                tot_correct += int(correct.sum())
                tot_nonsleep_correct += int((correct * non_sleep).sum())
                tot_nonsleep += int(non_sleep.sum())
                tot_toks += int(mask.sum())
                tot_loss += float(loss) * int(mask.sum())
            print(f"epoch {epoch:3d} loss {tot_loss/tot_toks:.4f} "
                  f"acc {tot_correct/tot_toks:.4f} "
                  f"nonsleep-acc {tot_nonsleep_correct/max(tot_nonsleep,1):.4f}"
                  f"{host_acc_term if use_aux else ''}",
                  flush=True)

    import datetime
    out = cli.out or ("results/models/rvs_"
                      + datetime.datetime.now(datetime.timezone.utc)
                      .strftime("%Y%m%dT%H%M%SZ"))
    ckpt_dir = os.path.join(out, "0")
    os.makedirs(ckpt_dir, exist_ok=True)
    th.save(agent.state_dict(), os.path.join(ckpt_dir, "agent.th"))
    # NOTE: no critic quad — the (obs+RTG) input dim matches no EPyMARL
    # actor layout, so there is no fine-tune resume path yet (proposal 16
    # would need an RvS-aware updater). Eval only via blue_eval_rvs.py.
    with open(os.path.join(out, "rvs_manifest.json"), "w") as f:
        json.dump({"demos": cli.demos, "epochs": cli.epochs, "lr": cli.lr,
                   "agent": cli.agent,
                   "aux_host_weight": cli.aux_host_weight,
                   "attn_layers": cli.attn_layers,
                   "hidden_dim": cli.hidden_dim, "seed": cli.seed,
                   "rtg_scale": RTG_SCALE, "weighted": True,
                   "final_acc": tot_correct / tot_toks,
                   "final_nonsleep_acc": (tot_nonsleep_correct
                                          / max(tot_nonsleep, 1))}, f, indent=1)
    print("wrote", ckpt_dir + "/agent.th")


if __name__ == "__main__":
    main()
