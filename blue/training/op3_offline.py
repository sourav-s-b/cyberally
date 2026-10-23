"""OP3 offline extraction (prereg op3-awdc-logs-8433, FROZEN).

Stages per freeze: BC init (masking-aware cross-entropy on logged
actions) -> conservative CQL-style Q(s, a) offline fit -> AWDC
advantage-weighted regression checkpoints (AWDC-1, AWDC-2, final).
NO online stage, NO fresh rollouts into updates. Data comes ONLY from
log_transitions' npz (current-contract per-agent obs, masks, native
rewards).

Actor: Linear(867->64) ReLU -> 64 ReLU -> 155 logits; mask enforced at
argmax eval and never defeated by the sampling path (greedy eval only).
Critic: Q(s_joint, agent, action) = MLP(concat(joint obs 4335, agent
one-hot 5, action embedding 16) -> 128 -> 128 -> 1) with a conservative
max-over-masked-random-actions hinge penalty (ALPHA_CONSERV).

AWDC weights: A = Q(s, a_log) - V(s) with V(s) = EXACT masked-mean Q
(otherwise the logits infeasibly priced: one full batched masked sweep
over all 155 candidate actions per state, computed once against the
final critic); w = exp(A/TAU_W) clipped to [0, 20] (freeze).

CLI (train venv, repo root):
  .venv-train/bin/python -m blue.training.op3_offline \
      --data blue/results/op3_awdc_logs_v1/logged.npz \
      --out blue/results/op3_awdc_logs_v1/extract_s0 --rng-seed 0

Wallclock-cap guarded (4 h). Eval JSONs use the mappo cmd_eval shape so
blue.training.op2_dev_analysis stays the single analysis script.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

GAMMA = 0.99
FEAT = 867
N_ACT = 155
N_AG = 5
H_ACTOR = 64
H_CRITIC = 128
EMB = 16
DATA_KINDS = ("guard", "lancer", "sleep", "random")
ALPHA_CONSERV = 0.1
TAU_W = 1.0
W_CLIP = (0.0, 20.0)
K_CONSERV = 8
Q_EPOCHS = 12
TD_SUB_FRAC = 0.33  # deterministic TD-row subsample once per rng seed
BC_EPOCHS = 3
AWDC_LR = 1e-4
BUDGET_S = 4 * 3600
DEV_SEEDS = list(range(8445, 8465))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode()
    except Exception:
        return "unknown"


def load_episodes(path):
    d = np.load(path)
    kinds = [str(k) for k in d["meta_kinds"]]
    lens = d["lengths"]
    eps = []
    for key in DATA_KINDS:
        if f"obs_{key}" not in d.files:
            continue
        obs = d[f"obs_{key}"].astype(np.float32)
        acts = d[f"actions_{key}"].astype(np.int64)
        mask = d[f"masks_{key}"].astype(np.int8)
        rew = d[f"rewards_{key}"].astype(np.float32)
        done = d[f"done_{key}"].astype(np.int8)
        c = 0
        for i, k in enumerate(kinds):
            if k != key:
                continue
            t = int(lens[i])
            eps.append({"obs": obs[c:c + t], "mask": mask[c:c + t],
                        "acts": acts[c:c + t], "rew": rew[c:c + t],
                        "done": done[c:c + t], "kind": key,
                        "seed": int(d["meta_seeds"][i])})
            c += t
    assert sum(e["obs"].shape[0] for e in eps) == sum(int(l) for l in lens), \
        "episode split mismatch"
    return eps


def stacked(eps):
    """Flatten rows: s (T,4335), s_next, dones, rew, acts (T,5), masks
    (T,5,155), plus per-agent actor rows c_x (T*5,867)."""
    s, s2, dones, rs, acts, masks, c_x = [], [], [], [], [], [], []
    for e in eps:
        T = e["obs"].shape[0]
        s.append(e["obs"].reshape(T, -1))
        s2.append(np.vstack([e["obs"][1:], e["obs"][-1:]]).reshape(T, -1))
        dones.append(e["done"].astype(np.float32))
        rs.append(e["rew"])
        acts.append(e["acts"])
        masks.append(e["mask"])
        c_x.append(e["obs"].reshape(T * N_AG, FEAT))
    return {
        "s": np.concatenate(s).astype(np.float32),
        "s_next": np.concatenate(s2).astype(np.float32),
        "dones": np.concatenate(dones).astype(np.float32),
        "rew": np.concatenate(rs).astype(np.float32),
        "acts": np.concatenate(acts).astype(np.int64),
        # (T_total,5,155) -> (T_total*5,155), row t*5+agent, same order
        # as the obs reshape used for c_x.
        "mask": np.concatenate(masks).reshape(-1, N_ACT).astype(
            np.float32),
        "c_x": np.concatenate(c_x).astype(np.float32),
    }


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(FEAT, H_ACTOR), nn.ReLU(),
                                  nn.Linear(H_ACTOR, H_ACTOR), nn.ReLU())
        self.head = nn.Linear(H_ACTOR, N_ACT)

    def forward(self, x):
        return self.head(self.body(x))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_ACT, EMB)
        d = FEAT * N_AG + N_AG + EMB
        self.net = nn.Sequential(nn.Linear(d, H_CRITIC), nn.ReLU(),
                                 nn.Linear(H_CRITIC, H_CRITIC), nn.ReLU(),
                                 nn.Linear(H_CRITIC, 1))

    def q(self, s, ag, a):
        z = torch.cat([s, ag, self.emb(a)], dim=1)
        return self.net(z).squeeze(-1)


def eval_dev(actor, out_path):
    from blue.training.mappo_guide import ENV_KW, run_team_episode
    env_kw = dict(ENV_KW)
    env_kw["red_agent"] = "finite"
    actor.eval()
    res = {}
    for seed in DEV_SEEDS:
        pol = _Greedy(actor)
        r = run_team_episode(pol, int(seed), 400, **env_kw)
        res[str(seed)] = {"return": r["return"],
                          "coverage": r["coverage"]}
        print(f"  seed={seed} ret={r['return']:+7.1f}", flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": {"learned": res}}, f, indent=1)


class _Greedy:
    """Greedy masked policy for eval; matches the harness call form."""

    def __init__(self, actor):
        self.actor = actor
        self.actor.eval()

    def reset(self):
        return

    def select(self, env, agent):
        from blue.core.wrapper import BLUE_AGENTS
        idx = BLUE_AGENTS.index(agent)
        obs = np.asarray(env.get_obs_agent(idx), dtype=np.float32)
        mask = np.asarray(env.get_avail_agent_actions(idx)).astype(
            np.float32)
        with torch.no_grad():
            lg = self.actor(torch.from_numpy(obs[None]).float())[0]
            lg = lg + torch.from_numpy((mask - 1.0).clip(max=-1e9)
                                       ).float()
        return int(lg.argmax().item())


def q_max_samples(critic, state, ag_row, mask_row, k):
    """For each row: max over k random masked actions of Q(state, a).

    state (B, 4335), ag_row (B, 5), mask_row (B, 155); returns (B,)."""
    with torch.no_grad():
        a = mask_row.multinomial(k, replacement=True)          # (B,k)
        s = state.repeat_interleave(k, dim=0)                  # (B*k,:)
        g = ag_row.repeat_interleave(k, dim=0)
        q = critic.q(s, g, a.reshape(-1)).view(-1, k)
        return q.max(dim=1).values


def main():
    ap = argparse.ArgumentParser(description="OP3 offline extraction")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--reuse-bc-eval", default=None,
                    help="path of a completed deterministic BC eval JSON "
                         "to copy into the out dir instead of re-running "
                         "the 20 dev episodes (only valid when the BC "
                         "train recipe and rng seed are unchanged)")
    cli = ap.parse_args()
    torch.set_num_threads(cli.cpu_threads)
    torch.manual_seed(cli.rng_seed)
    rng_np = np.random.default_rng(cli.rng_seed)
    os.makedirs(cli.out, exist_ok=True)
    t0 = time.time()

    eps = load_episodes(cli.data)
    flat = stacked(eps)
    n_t = flat["s"].shape[0]          # ticks
    n_rows = n_t * N_AG               # agent-rows
    print(f"data: {len(eps)} episodes, {n_t} ticks, {n_rows} agent-rows",
          flush=True)

    dev_x = torch.from_numpy(flat["c_x"])                     # (rows,867)
    dev_m = torch.from_numpy(flat["mask"])                    # (rows,155)
    dev_a = torch.from_numpy(flat["acts"].reshape(-1))        # (rows,)

    actor = Actor()
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    for ep in range(BC_EPOCHS):  # freeze: 3 epochs
        perm = rng_np.permutation(n_rows)
        for b0 in range(0, n_rows, 2048):
            b = torch.from_numpy(perm[b0:b0 + 2048])
            lg = actor(dev_x[b]) + (dev_m[b] - 1.0).clip(max=-1e9) \
                .float()
            loss = nn.functional.cross_entropy(lg, dev_a[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"BC epoch {ep + 1}: loss {float(loss):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    if cli.reuse_bc_eval:
        import shutil
        os.makedirs(cli.out, exist_ok=True)
        shutil.copyfile(cli.reuse_bc_eval,
                        os.path.join(cli.out, "eval_bc.json"))
        print("BC eval reused from", cli.reuse_bc_eval, flush=True)
    else:
        eval_dev(actor, os.path.join(cli.out, "eval_bc.json"))

    # ---- conservative critic fit -------------------------------------
    s = torch.from_numpy(flat["s"])                            # (T,4335)
    s2 = torch.from_numpy(flat["s_next"])
    dones = torch.from_numpy(flat["dones"])                    # (T,)
    rew_t = torch.from_numpy(flat["rew"])                      # (T,)
    ag = torch.from_numpy(np.tile(np.eye(N_AG, dtype=np.float32),
                                  (n_t, 1)))                   # (T*5,5)
    nextx = torch.from_numpy(np.repeat(flat["s_next"], N_AG,
                                       axis=0))                 # (T*5,4335)
    nextag = ag
    oc = torch.from_numpy(np.repeat(flat["s"], N_AG, axis=0))  # (rows,4335)
    ac = dev_a
    rj = torch.from_numpy(np.repeat(flat["rew"], N_AG).astype(np.float32))
    dj = torch.from_numpy(np.repeat(flat["dones"], N_AG).astype(np.float32))
    critic = Critic()
    optq = torch.optim.Adam(critic.parameters(), lr=1e-3)
    row_idx = rng_np.permutation(n_rows)[: int(n_rows * TD_SUB_FRAC)]
    for ep in range(Q_EPOCHS):  # freeze: fitted-Q sweeps, freeze params
        perm = rng_np.permutation(len(row_idx))
        for b0 in range(0, len(row_idx), 2048):
            b = torch.from_numpy(row_idx[perm[b0:b0 + 2048]])
            tgt = rj[b] + GAMMA * (1.0 - dj[b]) * \
                q_max_samples(critic, nextx[b], nextag[b], dev_m[b],
                              K_CONSERV)
            q_sa = critic.q(oc[b], ag[b], ac[b])
            # conservative hinge on DATA states: a random masked action's
            # q must not exceed the logged action's q
            cons = q_max_samples(critic, oc[b], ag[b], dev_m[b],
                                 K_CONSERV)
            loss = nn.functional.mse_loss(q_sa, tgt) + ALPHA_CONSERV * \
                nn.functional.relu(cons - q_sa).mean()
            critic.zero_grad()
            loss.backward()
            optq.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"critic epoch {ep + 1}/{Q_EPOCHS} "
                  f"loss {float(loss):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
        if time.time() - t0 > BUDGET_S:
            print("wallclock budget reached mid-critic; aborting per "
                  "prereg stop rule (partial report)", flush=True)
            break

    # ---- AWDC weights from the FROZEN critic --------------------------
    critic.eval()
    with torch.no_grad():
        q_sa_all = critic.q(oc, ag, ac)                          # (rows,)
        # V(s): exact masked-mean via one full batched candidate sweep
        vx = oc.repeat_interleave(N_ACT, dim=0)
        vag = ag.repeat_interleave(N_ACT, dim=0)
        vac = torch.arange(N_ACT).repeat(n_rows)                  # (rows*155,)
        qv = critic.q(vx, vag, vac).view(n_rows, N_ACT)
        qv = qv.masked_fill(dev_m == 0, -float("inf"))
        V = qv.mean(dim=1)                                       # (rows,)
        A = q_sa_all - V
        w_t = torch.exp(A / TAU_W).clip(*W_CLIP)
        w_mean = float(w_t.mean())
        print(f"AWDC weights: mean {w_mean:.4f}, max {float(w_t.max()):.2f}",
              flush=True)

    optw = torch.optim.Adam(actor.parameters(), lr=AWDC_LR)
    n_running_rows = n_rows
    for ck, epochs in ((1, 4), (2, 4), (3, 8)):  # freeze checkpoint plan
        if time.time() - t0 > BUDGET_S:
            break
        for _ in range(epochs):
            perm = rng_np.permutation(n_running_rows)
            for b0 in range(0, n_running_rows, 2048):
                b = torch.from_numpy(perm[b0:b0 + 2048])
                lg = actor(dev_x[b]) + (dev_m[b] - 1.0).float() \
                    .clip(max=-1e9)
                logp = torch.log_softmax(lg, dim=1).gather(
                    1, dev_a[b][:, None]).squeeze(1)
                loss = -(w_t[b] * logp).mean()
                optw.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(actor.parameters(), 5.0)
                optw.step()
        name = "final" if ck == 3 else f"awdc_{ck}"
        print(f"checkpoint {name} ({time.time() - t0:.0f}s)", flush=True)
        eval_dev(actor, os.path.join(cli.out, f"eval_{name}.json"))

    manifest = {
        "purpose": "OP3 offline extraction per frozen prereg; no online "
                   "stage; evals run through op2_dev_analysis vs the G1 "
                   "guard report",
        "git_commit": git_commit(),
        "rng_seed": cli.rng_seed,
        "data": os.path.abspath(cli.data),
        "config": {"bc_epochs": BC_EPOCHS, "q_epochs": Q_EPOCHS,
                   "tau_w": TAU_W, "alpha_conserv": ALPHA_CONSERV,
                   "k_cons": K_CONSERV, "w_clip": list(W_CLIP),
                   "awdc_lr": AWDC_LR, "h_actor": H_ACTOR,
                   "h_critic": H_CRITIC, "emb": EMB},
        "rows": {"ticks": int(n_t), "agent_rows": int(n_rows)},
        "episodes": len(eps),
        "wallclock_s": round(time.time() - t0),
    }
    with open(os.path.join(cli.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", os.path.join(cli.out, "manifest.json"))


if __name__ == "__main__":
    main()
