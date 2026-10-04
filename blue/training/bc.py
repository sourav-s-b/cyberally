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
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="episode-blocked validation fraction (V1; "
                             "agents of an episode stay together)")
    parser.add_argument("--val-seed", type=int, default=0,
                        help="seed for the blocked split (inner/nested "
                             "splits vary this)")
    parser.add_argument("--early-stop-patience", type=int, default=8,
                        help="Prechelt strips of rising val-loss before "
                             "stop + best restore; 0 = legacy fixed "
                             "--epochs (reproducible)")
    parser.add_argument("--attn-layers", type=int, default=0,
                        help="cross-slot self-attention layers in the "
                             "factorized head (proposal 03); 0 = off")
    parser.add_argument("--agent", default="rnn",
                        choices=("rnn", "rnn_factorized"),
                        help="actor head to distill into (proposal 02)")
    parser.add_argument("--aux-host-weight", type=float, default=0.0,
                        help="auxiliary CE on the factorized host selector "
                             "(host index of non-global teacher actions; "
                             "needs --agent rnn_factorized; proposal 02 "
                             "phase 2: distill WHAT host, not just WHAT id)")
    parser.add_argument("--out", default=None)
    cli = parser.parse_args()

    import torch as th
    from torch.utils.data import TensorDataset
    th.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import blue_val_split as valsplit
    import blue_logutil as logutil
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
    # V1: episode-blocked split BEFORE any fitting (class weights included).
    seq_episode = np.repeat(np.arange(n_eps), n_agents)
    train_eps, val_eps = valsplit.blocked_split(n_eps, cli.val_frac,
                                                cli.val_seed)
    train_idx, val_idx = valsplit.seq_split(seq_episode, train_eps)
    print(f"split: {len(train_eps)} train eps / {len(val_eps)} val eps "
          f"(seed {cli.val_seed})")
    stopper = valsplit.EarlyStopper(patience=cli.early_stop_patience)

    args = SN(hidden_dim=cli.hidden_dim, n_actions=n_actions, use_rnn=True,
              n_agents=n_agents, attn_layers=cli.attn_layers)
    agent = agent_REGISTRY[cli.agent](obs_dim, args)
    opt = th.optim.Adam(agent.parameters(), lr=cli.lr)
    # Inverse-sqrt class weights: the teacher is ~55% Sleep and exact
    # teacher-action prediction is hard (cross-slot argmax through a flat
    # head; the sweep cursor itself IS recoverable from obs, see proposal
    # 12), so unweighted CE collapses to majority-class prediction and
    # never learns rare remediation.
    freq = np.bincount(actions[train_eps].reshape(-1),
                       minlength=n_actions).astype(np.float64)
    weight = th.from_numpy((1.0 / np.sqrt(freq + 1.0)).astype(np.float32))
    weight = weight * (n_actions / weight.sum())
    loss_fn = th.nn.CrossEntropyLoss(weight=weight, reduction="none")

    torder = train_idx.copy()
    prog = logutil.Progress(cli.epochs)
    hist = []  # per-epoch curve rows -> train_log.jsonl
    tot_nonsleep_correct = tot_nonsleep = 0
    host_correct = host_toks = 0
    use_aux = cli.aux_host_weight > 0
    if use_aux and cli.agent != "rnn_factorized":
        raise SystemExit("--aux-host-weight needs --agent rnn_factorized")
    aux_fn = th.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    for epoch in range(1, cli.epochs + 1):
        np.random.shuffle(torder)
        tot_loss, tot_correct, tot_toks = 0.0, 0, 0
        for start in range(0, len(torder), cli.batch_seqs):
            idx = torder[start:start + cli.batch_seqs]
            b_obs = seq_obs[idx]
            b_act = seq_act[idx]
            b_len = seq_len[idx]
            b, t, _ = b_obs.shape
            h = th.zeros(b, cli.hidden_dim)
            logits = []
            hosts = [] if use_aux else None
            hh = h
            for step in range(t):
                if use_aux:
                    q, hh, hs, _, _ = agent.decompose(b_obs[:, step], hh)
                    hosts.append(hs)
                else:
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
            host_acc_term = ""
            if use_aux:
                host_logits = th.stack(hosts, dim=1)  # (b, t, 51)
                host_tgt = ((b_act - 2) // 3).clone()
                host_tgt[(b_act < 2)] = -100  # Sleep/Monitor: no host
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
        # V1 validation pass (no grad, same masking/weights, hidden
        # zero-init per sequence exactly as in training).
        with th.no_grad():
            v_loss, v_toks = 0.0, 0
            for start in range(0, len(val_idx), cli.batch_seqs):
                idx = val_idx[start:start + cli.batch_seqs]
                b_obs = seq_obs[idx]
                b_act = seq_act[idx]
                b_len = seq_len[idx]
                b, t, _ = b_obs.shape
                hh = th.zeros(b, cli.hidden_dim)
                logits = []
                for step in range(t):
                    q, hh = agent(b_obs[:, step], hh)
                    logits.append(q)
                logits = th.stack(logits, dim=1)
                free = (seq_mask[idx].sum(-1) > 1).float()
                mask = ((th.arange(t).unsqueeze(0)
                         < b_len.unsqueeze(1)).float() * free)
                denom = mask.sum().clamp(min=1)
                vl = (loss_fn(logits.reshape(-1, n_actions),
                              b_act.reshape(-1)).reshape(b, t)
                      * mask).sum() / denom
                v_loss += float(vl) * int(mask.sum())
                v_toks += int(mask.sum())
            val_loss = v_loss / max(v_toks, 1)
        hist.append({"epoch": epoch, "elapsed_s": round(prog.elapsed(), 1),
                     "train_loss": tot_loss / tot_toks,
                     "train_acc": tot_correct / tot_toks,
                     "train_nonsleep_acc": (tot_nonsleep_correct
                                            / max(tot_nonsleep, 1)),
                     "val_loss": val_loss,
                     "val_best": stopper.best,
                     "val_best_epoch": stopper.best_epoch})
        print(f"epoch {epoch:3d} val-loss {val_loss:.4f} "
              f"(best {stopper.best if stopper.best is not None else float('nan'):.4f} "
              f"@{stopper.best_epoch}) "
              f"{prog.line(epoch)}", flush=True)
        if stopper.update(epoch, val_loss, agent):
            print(f"early stop at epoch {epoch} "
                  f"(best val {stopper.best:.4f} @{stopper.best_epoch})",
                  flush=True)
            break
    stopper.restore(agent)

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
                   "aux_host_weight": cli.aux_host_weight,
                   "attn_layers": cli.attn_layers,
                   "hidden_dim": cli.hidden_dim, "seed": cli.seed,
                   "weighted": True,
                   "val_frac": cli.val_frac, "val_seed": cli.val_seed,
                   "val_eps": [int(e) for e in val_eps],
                   **{"val_" + k: v for k, v in
                      stopper.summary().items()},
                   "final_acc": tot_correct / tot_toks,
                   "final_nonsleep_acc": (tot_nonsleep_correct
                                          / max(tot_nonsleep, 1))}, f, indent=1)
    with open(os.path.join(out, "train_log.jsonl"), "w") as f:
        for row in hist:
            f.write(json.dumps(row) + "\n")
    print("wrote", ckpt_dir + "/agent.th")


if __name__ == "__main__":
    main()
