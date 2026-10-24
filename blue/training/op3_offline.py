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
import hashlib
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
GUARD_DEV_MEAN = -127.80
RUN_ID = "op3-awdc-logs-8433-v5"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
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
    """Lazy row store (v4). Keeps ONLY the (T*episodes, 5, FEAT) base
    arrays on RAM; row i maps to (tick t = i // N_AG, agent a = i % N_AG).
    The v1-v3 materializations (row-replicated 5x duplicates of the joint
    states) exceeded the 7.5 GB box and OOM-killed silently, so the fit
    loops now stream per-batch rows via rows_slice instead."""
    obs = np.concatenate([e["obs"] for e in eps]).astype(np.float32)
    acts = np.concatenate([e["acts"] for e in eps]).astype(np.int64)
    mask = np.concatenate([e["mask"] for e in eps]).astype(np.int8)
    rew = np.concatenate([e["rew"] for e in eps]).astype(np.float32)
    dones = np.concatenate([e["done"] for e in eps]).astype(np.float32)
    next_idx = np.empty(obs.shape[0], dtype=np.int64)
    c = 0
    for e in eps:
        T = e["obs"].shape[0]
        for t in range(T):
            # terminal bootstrap is zeroed by done anyway; self-loop at
            # the episode end keeps the index in bounds
            next_idx[c + t] = c + t + 1 if t + 1 < T else c + t
        c += T
    return {"obs": obs, "acts": acts, "mask": mask, "rew": rew,
            "dones": dones, "next_idx": next_idx}


def rows_slice(flat, rows):
    """Batch tensors for row indices (t = row // N_AG, a = row % N_AG):
    joint (B,4335), agent row obs (B,867), agent one-hot (B,5), mask
    (B,155), actions (B,), label reward/done (B,), and the next-joint
    state for TD bootstrapping."""
    t = np.asarray(rows) // N_AG
    a = np.asarray(rows) % N_AG
    B = len(t)
    joint = torch.from_numpy(
        flat["obs"][t].reshape(B, -1).astype(np.float32))
    agent = torch.from_numpy(
        flat["obs"][t, a].astype(np.float32))
    return {
        "joint": joint,
        "agent": agent,
        "onehot": torch.from_numpy(np.eye(N_AG, dtype=np.float32)[a]),
        "mask": torch.from_numpy(
            flat["mask"][t, a].astype(np.float32)),
        "mask_next": torch.from_numpy(
            flat["mask"][flat["next_idx"][t], a].astype(np.float32)),
        "acts": torch.from_numpy(flat["acts"][t, a]),
        "rew": torch.from_numpy(flat["rew"][t]),
        "done": torch.from_numpy(flat["dones"][t]),
        "joint_next": torch.from_numpy(
            flat["obs"][flat["next_idx"][t]].reshape(B, -1).astype(
                np.float32)),
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
    """Q(s_joint, agent, action) = rest(relul(gt_state(s, ag) +
    emb_lin(emb(a)))). v3 memory decomposition: the first Linear is split
    into a state-only base and an action-embedding contribution so the
    all-actions sweep (needed for the exact masked-mean V) runs as small
    chunked matmuls instead of a rows*155 materialization that OOM-killed
    extraction v1 mid-run. q() outputs are numerically IDENTICAL to the
    v1 concatenation form."""

    def __init__(self):
        super().__init__()
        d_in = FEAT * N_AG + N_AG
        self.lin_state = nn.Linear(d_in, H_CRITIC)
        self.lin_emb = nn.Linear(EMB, H_CRITIC, bias=False)
        self.emb = nn.Embedding(N_ACT, EMB)
        self.rest = nn.Sequential(nn.Linear(H_CRITIC, H_CRITIC), nn.ReLU(),
                                  nn.Linear(H_CRITIC, 1))

    def _base(self, s, ag):
        return self.lin_state(torch.cat([s, ag], dim=1))

    def q(self, s, ag, a):
        h = nn.functional.relu(self._base(s, ag)
                               + self.lin_emb(self.emb(a)))
        return self.rest(h).squeeze(-1)

    def q_all_actions(self, s, ag, chunk=8192):
        """(rows, N_ACT) Q for every candidate action, chunked so the
        (rows*155, H_CRITIC) rest-input never materializes in full.
        base_raw stays PRE-activation; q() is relu(base_raw + emb_lin)."""
        base_raw = self._base(s, ag)                   # (rows,H)
        e_all = self.lin_emb(self.emb.weight)          # (155,H)
        out = torch.empty(base_raw.shape[0], N_ACT)
        for c0 in range(0, base_raw.shape[0], chunk):
            h = nn.functional.relu(base_raw[c0:c0 + chunk].unsqueeze(1)
                                   + e_all.unsqueeze(0))
            out[c0:c0 + chunk] = self.rest(h).squeeze(-1)
        return out


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_actor(actor, out_dir, name):
    path = os.path.join(out_dir, f"actor_{name}.pt")
    torch.save(actor.state_dict(), path)
    return {"path": os.path.abspath(path), "sha256": sha256_file(path)}


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
    return res


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
            lg = mask_logits(lg, torch.from_numpy(mask))
        return int(lg.argmax().item())


def mask_logits(logits, mask):
    """Exclude illegal actions while preserving the legal logits."""
    return logits.masked_fill(mask == 0, -1e9)


def normalized_awdc_weights(q_logged, q_actions, mask):
    """Per-state normalized exp advantage, clipped to the frozen range."""
    valid = mask != 0
    n_valid = valid.sum(dim=1).clamp_min(1)
    v = (q_actions * valid).sum(dim=1) / n_valid
    z_all = (q_actions - v[:, None]) / TAU_W
    z_log = (q_logged - v) / TAU_W
    z_masked = z_all.masked_fill(~valid, -float("inf"))
    z_max = z_masked.max(dim=1).values
    denom = torch.exp(z_masked - z_max[:, None]).sum(dim=1) / n_valid
    return (torch.exp(z_log - z_max) / denom).clamp(*W_CLIP)


def q_max_samples(critic, state, ag_row, mask_row, k, *, grad=False):
    """For each row: max over k random masked actions of Q(state, a).

    state (B, 4335), ag_row (B, 5), mask_row (B, 155); returns (B,)."""
    a = mask_row.multinomial(k, replacement=True)          # (B,k)
    s = state.repeat_interleave(k, dim=0)                  # (B*k,:)
    g = ag_row.repeat_interleave(k, dim=0)
    with torch.enable_grad() if grad else torch.no_grad():
        q = critic.q(s, g, a.reshape(-1)).view(-1, k)
        return q.max(dim=1).values


def q_logsumexp_samples(critic, state, ag_row, mask_row, k):
    """Conservative masked-sample logsumexp with gradients to sampled Q."""
    a = mask_row.multinomial(k, replacement=True)
    s = state.repeat_interleave(k, dim=0)
    g = ag_row.repeat_interleave(k, dim=0)
    q = critic.q(s, g, a.reshape(-1)).view(-1, k)
    return torch.logsumexp(q, dim=1)


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
    n_t = flat["obs"].shape[0]         # ticks
    n_rows = n_t * N_AG                # agent-rows
    print(f"data: {len(eps)} episodes, {n_t} ticks, {n_rows} agent-rows",
          flush=True)

    actor = Actor()
    checkpoints = {}
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    for ep in range(BC_EPOCHS):  # freeze: 3 epochs
        perm = rng_np.permutation(n_rows)
        for b0 in range(0, n_rows, 2048):
            sl = rows_slice(flat, perm[b0:b0 + 2048])
            lg = mask_logits(actor(sl["agent"]), sl["mask"])
            loss = nn.functional.cross_entropy(lg, sl["acts"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"BC epoch {ep + 1}: loss {float(loss):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    checkpoints["bc"] = save_actor(actor, cli.out, "bc")
    if cli.reuse_bc_eval:
        import shutil
        os.makedirs(cli.out, exist_ok=True)
        shutil.copyfile(cli.reuse_bc_eval,
                        os.path.join(cli.out, "eval_bc.json"))
        print("BC eval reused from", cli.reuse_bc_eval, flush=True)
    else:
        eval_dev(actor, os.path.join(cli.out, "eval_bc.json"))
    with open(os.path.join(cli.out, "eval_bc.json")) as f:
        bc_results = json.load(f)["results"]["learned"]
    bc_mean = float(np.mean([float(v["return"]) for v in bc_results.values()]))
    print(f"G2 BC-init mean {bc_mean:.3f}; guard {GUARD_DEV_MEAN:.3f}",
          flush=True)
    if bc_mean > GUARD_DEV_MEAN + 1.0:
        raise RuntimeError("G2 leak gate failed: BC-init ahead of guard by >1")

    # ---- conservative critic fit (streamed rows, config unchanged) ----
    critic = Critic()
    optq = torch.optim.Adam(critic.parameters(), lr=1e-3)
    row_idx = rng_np.permutation(n_rows)[: int(n_rows * TD_SUB_FRAC)]
    for ep in range(Q_EPOCHS):  # freeze: fitted-Q sweeps, freeze params
        perm = rng_np.permutation(len(row_idx))
        for b0 in range(0, len(row_idx), 2048):
            sl = rows_slice(flat, row_idx[perm[b0:b0 + 2048]])
            tgt = sl["rew"] + GAMMA * (1.0 - sl["done"]) * \
                q_max_samples(critic, sl["joint_next"], sl["onehot"],
                              sl["mask_next"], K_CONSERV)
            q_sa = critic.q(sl["joint"], sl["onehot"], sl["acts"])
            # conservative hinge on the SAME batch (data states)
            cons = q_logsumexp_samples(critic, sl["joint"], sl["onehot"],
                                       sl["mask"], K_CONSERV)
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

    # ---- AWDC weights from the FROZEN critic (streamed, exact) --------
    critic.eval()
    w_t = torch.empty(n_rows)
    with torch.no_grad():
        for r0 in range(0, n_rows, 4096):
            rows = np.arange(r0, min(r0 + 4096, n_rows))
            sl = rows_slice(flat, rows)
            q_sa_c = critic.q(sl["joint"], sl["onehot"], sl["acts"])
            qv = critic.q_all_actions(sl["joint"], sl["onehot"])
            w_t[r0:r0 + len(rows)] = normalized_awdc_weights(
                q_sa_c, qv, sl["mask"])
    w_mean = float(w_t.mean())
    w_clip_frac = float((w_t >= W_CLIP[1]).float().mean())
    print(f"AWDC weights: mean {w_mean:.4f}, max {float(w_t.max()):.2f}, "
          f"clip_frac {w_clip_frac:.4f}", flush=True)

    optw = torch.optim.Adam(actor.parameters(), lr=AWDC_LR)
    for ck, epochs in ((1, 4), (2, 4), (3, 8)):  # freeze checkpoint plan
        if time.time() - t0 > BUDGET_S:
            break
        for _ in range(epochs):
            perm = rng_np.permutation(n_rows)
            for b0 in range(0, n_rows, 2048):
                sl = rows_slice(flat, perm[b0:b0 + 2048])
                lg = mask_logits(actor(sl["agent"]), sl["mask"])
                logp = torch.log_softmax(lg, dim=1).gather(
                    1, sl["acts"][:, None]).squeeze(1)
                loss = -(w_t[perm[b0:b0 + 2048]] * logp).mean()
                optw.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(actor.parameters(), 5.0)
                optw.step()
        name = "final" if ck == 3 else f"awdc_{ck}"
        print(f"checkpoint {name} ({time.time() - t0:.0f}s)", flush=True)
        checkpoints[name] = save_actor(actor, cli.out, name)
        eval_dev(actor, os.path.join(cli.out, f"eval_{name}.json"))

    manifest = {
        "run_id": RUN_ID,
        "purpose": "OP3 offline extraction per frozen prereg; no online "
                   "stage; evals run through op2_dev_analysis vs the G1 "
                   "guard report",
        "git_commit": git_commit(),
        "rng_seed": cli.rng_seed,
        "data": os.path.abspath(cli.data),
        "data_sha256": sha256_file(cli.data),
        "dev_seeds": DEV_SEEDS,
        "bc_dev_mean": bc_mean,
        "g2_guard_dev_mean": GUARD_DEV_MEAN,
        "bc_eval_reused_from": cli.reuse_bc_eval,
        "checkpoint_artifacts": checkpoints,
        "awdc_weight_mean": w_mean,
        "awdc_weight_clip_fraction": w_clip_frac,
        "config": {"bc_epochs": BC_EPOCHS, "q_epochs": Q_EPOCHS,
                   "td_subsample_frac": TD_SUB_FRAC,
                   "tau_w": TAU_W, "alpha_conserv": ALPHA_CONSERV,
                   "k_cons": K_CONSERV, "w_clip": list(W_CLIP),
                   "awdc_lr": AWDC_LR, "h_actor": H_ACTOR,
                   "h_critic": H_CRITIC, "emb": EMB},
        "rows": {"ticks": int(n_t), "agent_rows": int(n_rows)},
        "episodes": len(eps),
        "wallclock_s": round(time.time() - t0),
        "complete": len(checkpoints) == 4,
    }
    with open(os.path.join(cli.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", os.path.join(cli.out, "manifest.json"))


if __name__ == "__main__":
    main()
