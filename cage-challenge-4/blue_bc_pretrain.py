"""Supervised behavior-cloning pre-training of the MAPPO RNN actor.

Trains RNNAgent (same geometry as training/eval: obs+id input, 155 actions)
with cross-entropy against round-robin teacher trajectories, carrying GRU
state across each full episode. Writes a standard checkpoint layout
(<out>/0/agent.th) loadable via EPyMARL --checkpoint-path for PPO fine-tune.
"""

import argparse
import json
import os
from types import SimpleNamespace as SN

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="BC pre-train MAPPO actor")
    parser.add_argument("--demos", default="results/bc_demos_rr.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--opt-lr", type=float, default=5e-5,
                        help="lr stamped into the saved optimiser states. "
                             "EPyMARL resume restores optimiser state "
                             "wholesale INCLUDING lr, so fine-tune runs "
                             "inherit this, not --lr. Set it to the "
                             "intended PPO lr.")
    parser.add_argument("--batch-seqs", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", default="rnn",
                        choices=("rnn", "rnn_factorized"),
                        help="actor head to distill into (proposal 02)")
    parser.add_argument("--out", default=None)
    cli = parser.parse_args()

    import torch as th
    from torch.utils.data import TensorDataset
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
    masks = data["masks"]    # (E, A, T, N)
    lengths = data["lengths"]
    n_eps, n_agents, max_len, obs_dim = obs.shape
    n_actions = int(data["masks"].shape[3])
    print(f"demos: {n_eps} eps x {n_agents} agents x {max_len} ticks, "
          f"obs {obs_dim}, actions {n_actions}")

    # Flatten episodes x agents into independent sequences.
    seq_obs = th.from_numpy(obs.reshape(-1, max_len, obs_dim))
    seq_act = th.from_numpy(actions.reshape(-1, max_len))
    seq_mask = th.from_numpy(masks.reshape(-1, max_len, n_actions))
    seq_len = th.from_numpy(np.repeat(lengths, n_agents))
    n_seqs = seq_obs.shape[0]

    args = SN(hidden_dim=cli.hidden_dim, n_actions=n_actions, use_rnn=True,
              n_agents=n_agents)
    agent = agent_REGISTRY[cli.agent](obs_dim, args)
    opt = th.optim.Adam(agent.parameters(), lr=cli.lr)
    # Inverse-sqrt class weights: the teacher is ~55% Sleep and exact
    # teacher-action prediction is hard (cross-slot argmax through a flat
    # head; the sweep cursor itself IS recoverable from obs, see proposal
    # 12), so unweighted CE collapses to majority-class prediction and
    # never learns rare remediation.
    freq = np.bincount(actions.reshape(-1), minlength=n_actions).astype(np.float64)
    weight = th.from_numpy((1.0 / np.sqrt(freq + 1.0)).astype(np.float32))
    weight = weight * (n_actions / weight.sum())
    loss_fn = th.nn.CrossEntropyLoss(weight=weight, reduction="none")

    order = np.arange(n_seqs)
    tot_nonsleep_correct = tot_nonsleep = 0
    for epoch in range(1, cli.epochs + 1):
        np.random.shuffle(order)
        tot_loss, tot_correct, tot_toks = 0.0, 0, 0
        for start in range(0, n_seqs, cli.batch_seqs):
            idx = order[start:start + cli.batch_seqs]
            b_obs = seq_obs[idx]
            b_act = seq_act[idx]
            b_len = seq_len[idx]
            b, t, _ = b_obs.shape
            h = th.zeros(b, cli.hidden_dim)
            logits = []
            hh = h
            for step in range(t):
                q, hh = agent(b_obs[:, step], hh)
                logits.append(q)
            logits = th.stack(logits, dim=1)  # (b, t, n_actions)
            # Zero the loss on busy ticks (mask sum<=1: teacher always Sleeps
            # there and obs shows pending) so Sleep-noise gradients cannot
            # drown the rare free-decision signal.
            free = (seq_mask[idx].sum(-1) > 1).float()
            mask = ((th.arange(t).unsqueeze(0) < b_len.unsqueeze(1)).float()
                    * free)
            # Guard the all-busy degenerate batch (never observed; would NaN).
            denom = mask.sum().clamp(min=1)
            loss = (loss_fn(logits.reshape(-1, n_actions),
                            b_act.reshape(-1)).reshape(b, t) * mask).sum() / denom
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
              f"nonsleep-acc {tot_nonsleep_correct/max(tot_nonsleep,1):.4f}",
              flush=True)

    stamp = th.__version__  # noqa: keep linters quiet about unused import shape
    del stamp
    import datetime
    out = cli.out or ("results/models/bc_rr_"
                      + datetime.datetime.now(datetime.timezone.utc)
                      .strftime("%Y%m%dT%H%M%SZ"))
    ckpt_dir = os.path.join(out, "0")
    os.makedirs(ckpt_dir, exist_ok=True)
    th.save(agent.state_dict(), os.path.join(ckpt_dir, "agent.th"))
    # Full resume quad: EPyMARL load_models also restores critic.th and
    # both Adam states, so save a fresh critic (matching training geometry)
    # plus initial optimiser states. Fine-tune re-learns the value head.
    sys.path.insert(0, "/home/sourav/Projects/cyberally/third_party/epymarl/src")
    from modules.critics import REGISTRY as critic_REGISTRY
    import cc4_epymarl_wrapper as wrapper
    probe = wrapper.CC4MARLEnv(steps=10, temporal_features=("ages", "belief"),
                               include_root_session=True)
    state_shape = probe.get_env_info()["state_shape"]
    probe.close()
    critic_args = SN(n_actions=n_actions, n_agents=5, hidden_dim=cli.hidden_dim,
                     obs_individual_obs=False, obs_last_action=False,
                     obs_agent_id=True)
    critic = critic_REGISTRY["cv_critic"]({"state": {"vshape": state_shape}},
                                          critic_args)
    # Fresh optimiser states at --opt-lr (NOT the BC-training opt, whose
    # 1e-3 lr would otherwise override the fine-tune --lr on resume).
    save_agent_opt = th.optim.Adam(agent.parameters(), lr=cli.opt_lr)
    save_critic_opt = th.optim.Adam(critic.parameters(), lr=cli.opt_lr)
    th.save(critic.state_dict(), os.path.join(ckpt_dir, "critic.th"))
    th.save(save_agent_opt.state_dict(),
            os.path.join(ckpt_dir, "agent_opt.th"))
    th.save(save_critic_opt.state_dict(),
            os.path.join(ckpt_dir, "critic_opt.th"))
    with open(os.path.join(out, "bc_manifest.json"), "w") as f:
        json.dump({"demos": cli.demos, "epochs": cli.epochs, "lr": cli.lr,
                   "opt_lr": cli.opt_lr, "agent": cli.agent,
                   "hidden_dim": cli.hidden_dim, "seed": cli.seed,
                   "weighted": True,
                   "final_acc": tot_correct / tot_toks,
                   "final_nonsleep_acc": (tot_nonsleep_correct
                                          / max(tot_nonsleep, 1))}, f, indent=1)
    print("wrote", ckpt_dir + "/agent.th")


if __name__ == "__main__":
    main()
