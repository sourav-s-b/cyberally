"""Expert-guided MAPPO Blue team for investigation scheduling (CTDE).

Fixed guides: :class:`OrderedPolicy` rules 1-3 (CONFIRMED remediation /
escalation, VERIFY re-analysis) plus the frozen ``LancerValues`` base
scorer. Guard OFF. Only the sweep-stage Analyse host choice is a learned
decision; decoy deployment, urgent responses and verification are never
learner data (no masquerading).

Learner (shared actor): the residual MLP over per-agent 14-dim host rows
(10 Blue-visible view features + 4 observation-derived history features:
analysis/remediation age, CONFIRMED/VERIFY flags). Zero-initialized head
-> exact teacher at start.

Critic (centralized, training only): an MLP over the JOINT Blue-visible
context -- concatenated per-agent pooled host-row means plus normalized
tick -- estimating the team return. Advantages are computed against this
central critic, not a pooled local baseline. That central critic is the
concrete difference from the narrow residual trainer.

Training: fresh joint on-policy rollouts per iteration, Monte-Carlo team
returns (gamma .99, intervening rewards included), PPO-clip. Execution is
decentralized: per-agent argmax over the combined scorer, critic unused.

What this is, truthfully: shared-actor MAPPO (CTDE) restricted to
investigation scheduling. It does NOT learn decoy/urgent/verification
actions and it is NOT the full 155-way EPyMARL MAPPO (which diverged on
fine-tune). Sampling consumes an explicit ``torch.Generator`` threaded
through ``masked_choice``; the global torch RNG is never touched.

Usage (train venv, repo root):
  .venv-train/bin/python -m blue.training.mappo_guide --mode train \
      --train-seeds 7706 7707 7708 7709 --iters 6 --eps-per-iter 4 \
      --out blue/results/mappo_guide1
  .venv-train/bin/python -m blue.training.mappo_guide --mode eval \
      --model-dir blue/results/mappo_guide1 --eval-seeds 7629 7630 \
      --out blue/results/mappo_guide1_eval.json
"""

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blue.common import logutil

GAMMA = 0.99
MAX_CANDS = 51
N_AGENTS = 5
FEAT_DIM = 14
JOINT_CTX_DIM = N_AGENTS * FEAT_DIM + 1  # 5 pooled host-row means + tick
DEFAULT_TRAIN_SEEDS = [7706, 7707, 7708, 7709]
DEFAULT_EVAL_SEEDS = [7629, 7630, 7640, 7701, 7702, 7703, 7704, 7705]
ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}
ALGO = ("heuristic-guided MAPPO for investigation scheduling: shared "
        "residual actor over local host rows + central-V critic over "
        "joint Blue-visible context, fresh joint on-policy rollouts, "
        "PPO-clip, decentralized greedy execution")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def dirty_tree():
    """Working-tree provenance: short status + sha of the full diff, as in
    the earlier matrix manifests. Best effort; never fails the run."""
    try:
        st = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
        diff = subprocess.check_output(
            ["git", "diff"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL)
        import hashlib
        return {"status": st.splitlines(),
                "diff_sha256": hashlib.sha256(diff).hexdigest()[:16]}
    except Exception:
        return {"status": [], "diff_sha256": "unknown"}


def build_central_critic(joint_dim=JOINT_CTX_DIM, hidden=64):
    """Central-V critic: joint Blue-visible context -> team return."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(joint_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1))


def joint_context(env, policy, _cache=None):
    """Joint Blue-visible context at the current tick: per-agent pooled
    host-row means (zeros where the agent has no legal Analyse candidate)
    plus normalized tick. No privileged state. Results are cached per
    tick in ``_cache`` (cleared each episode) because host-row
    computation dominates rollout cost and repeats ~5x per tick."""
    tick = int(getattr(env, "_tick", 0))
    if _cache is not None and tick in _cache:
        return _cache[tick]
    from blue.core.wrapper import BLUE_AGENTS
    from blue.training.scorer import host_rows
    parts = []
    T = float(getattr(env, "episode_limit", 400) or 400)
    for a in BLUE_AGENTS:
        idx = BLUE_AGENTS.index(a)
        mask = env.get_avail_agent_actions(idx)
        cands = policy._sweep_cands(env, a, mask)
        if cands:
            F = np.asarray(host_rows(env, a, cands), dtype=np.float32)
            parts.append(F.mean(axis=0))
        else:
            parts.append(np.zeros(FEAT_DIM, dtype=np.float32))
    parts.append(np.asarray([tick / max(T, 1.0)], dtype=np.float32))
    out = np.concatenate(parts)
    if _cache is not None:
        _cache[tick] = out
    return out


class JointRecorder:
    """Sampler hook recording sweep decisions with joint context.

    ``gen`` is an explicit ``torch.Generator``: sampling consumes it, the
    global torch RNG is untouched. Rules/emergency actions never reach the
    hook, so every recorded row is a sampled actor decision. ``temp``
    sharpens (temp<1) or flattens (temp>1) the sampling softmax toward /
    away from the greedy deployment policy; temp=1 reproduces the earlier
    pilot's sampling distribution.
    """

    def __init__(self, actor, bonus, gen, temp=1.0):
        import torch as th
        self.th = th
        self.actor = actor
        self.bonus = bonus
        self.temp = float(temp)
        self.gen = gen
        self.rows = []
        self.pending = {}
        self.per_agent = {}
        self.agree = 0
        self.total = 0
        self._ep_start = 0
        self._jcache = {}

    def mark_episode(self):
        self._ep_start = len(self.rows)
        self._jcache = {}

    def __call__(self, env, agent, cands, scored):
        from blue.policies.ordered import argmax_pick
        from blue.training.residual import masked_choice
        from blue.training.scorer import host_rows
        th = self.th
        rows = host_rows(env, agent, cands)
        F = th.from_numpy(np.asarray(rows, dtype=np.float32))
        with th.no_grad():
            r = th.tanh(self.actor(F))
        base = th.tensor([s for s, _ in scored], dtype=th.float32)
        combined = (base + self.bonus * r) / self.temp
        mask = [1.0] * len(cands)
        idx, logp, ent = masked_choice(combined, mask, sample=True,
                                       seed_rng=self.gen)
        pairs = [(float(combined[i]), cands[i]) for i in range(len(cands))]
        arg_comb = argmax_pick(pairs)
        arg_base = argmax_pick([(s, h) for s, h in scored])
        self.agree += int(arg_comb == arg_base)
        self.total += 1
        tick = env._tick
        jctx = joint_context(env, self._policy_ref, self._jcache)
        row = {"feats": F.numpy(), "joint": jctx, "k": len(cands),
               "mask": mask, "base": base.numpy(), "choice": idx,
               "logp": logp, "ent": ent, "tick": tick, "agent": agent,
               "ret": 0.0, "temp": self.temp}
        self.rows.append(row)
        self.pending.setdefault(agent, []).append(len(self.rows) - 1)
        self.per_agent[agent] = self.per_agent.get(agent, 0) + 1
        return cands[idx]

    def close_episode(self, rewards):
        """Discounted Monte-Carlo team returns (to episode end) per
        recorded decision; pads variable candidate lists to MAX_CANDS."""
        R = np.asarray(rewards, dtype=float)
        T = len(R)
        disc = GAMMA ** np.arange(T)
        for row in self.rows[self._ep_start:]:
            k = len(row["mask"])
            pad = MAX_CANDS - k
            row["feats"] = np.pad(row["feats"], ((0, pad), (0, 0)))
            row["mask"] = np.pad(row["mask"], (0, pad))
            row["base"] = np.pad(row["base"], (0, pad))
        for idxs in self.pending.values():
            for i in idxs:
                t = self.rows[i]["tick"]
                t = max(0, min(t, T - 1))
                self.rows[i]["ret"] = float(
                    (R[t:] * disc[:T - t]).sum())
        self.pending = {}


def run_team_episode(policy, seed, steps, hook=None, recorder=None,
                     **env_kw):
    """Joint rollout. Returns per-tick team rewards plus the forced-vs-
    sampled split. Hook calls are sampled actor decisions; everything else
    is forced heuristic behavior, split into:
    - lockout_sleep: Sleep while the agent has an outstanding action
      (one-pending-action rule; historically 54-69% of agent-ticks);
    - urgent: Remove/Restore response rules;
    - verification: rule-2 VERIFY re-analysis (non-hook Analyse);
    - idle: Sleep/Monitor with nothing pending and nothing legal.
    ``agent in env._awaiting`` is the agent's own outstanding action and
    is Blue-visible."""
    from blue.core.baselines import decode_index
    from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
    env = CC4MARLEnv(seed=seed, steps=steps, **env_kw)
    env.reset(seed=seed)
    policy.reset()
    policy.hook = hook  # None = reference behavior (never contaminated)
    if recorder is not None:
        recorder._policy_ref = policy
        recorder.mark_episode()
    rewards = []
    analysed = set()
    n_hosts = sum(len(env.hostnames[a]) for a in BLUE_AGENTS)
    cats = {"urgent": 0, "verification": 0, "lockout_sleep": 0, "idle": 0}
    n_actor, n_ticks = 0, 0
    for tick in range(1, steps + 1):
        actions = {}
        for a in BLUE_AGENTS:
            before = (recorder.per_agent.get(a, 0)
                      if recorder is not None else 0)
            act = int(policy.select(env, a))
            actions[a] = act
            if (recorder is not None
                    and recorder.per_agent.get(a, 0) > before):
                n_actor += 1
                continue
            name, _host = decode_index(env, a, act)
            if name in ("Remove", "Restore"):
                cats["urgent"] += 1
            elif name == "Analyse":
                cats["verification"] += 1
            elif a in env._awaiting:
                cats["lockout_sleep"] += 1
            else:
                cats["idle"] += 1
        _, rew, terminated, truncated, _ = env.step(actions)
        rewards.append(float(rew[0]))
        n_ticks += 1
        for agent in BLUE_AGENTS:
            name, host = decode_index(env, agent, actions[agent])
            if name == "Analyse" and host is not None:
                analysed.add((agent, host))
        if terminated or truncated:
            break
    if recorder is not None:
        recorder.close_episode(rewards)
    coverage = len(analysed) / max(n_hosts, 1)
    n_forced = sum(cats.values())
    return {"seed": seed, "return": float(sum(rewards)),
            "coverage": coverage, "ticks": n_ticks,
            "n_actor": n_actor, "n_forced": n_forced,
            "forced_cats": dict(cats),
            "n_decisions": (len(recorder.rows) if recorder is not None
                            else 0)}


def ppo_central_update(actor, critic, opt, buf, bonus=1.0, epochs=4,
                       minibatch=256, clip=0.2, vf_coef=0.5,
                       ent_coef=0.01):
    """One PPO pass over a fresh on-policy buffer. Combined logits =
    base + bonus*tanh(actor) rebuilt from the current actor so logprobs
    match rollout behavior exactly. Advantages are normalized per buffer
    and computed against the CENTRAL critic on joint context."""
    import torch as th
    F = th.from_numpy(np.stack([b["feats"] for b in buf]).astype(np.float32))
    J = th.from_numpy(np.stack([b["joint"] for b in buf]).astype(np.float32))
    M = th.from_numpy(np.stack([b["mask"] for b in buf]).astype(np.float32))
    B = th.from_numpy(np.stack([b["base"] for b in buf]).astype(np.float32))
    CH = th.from_numpy(np.array([b["choice"] for b in buf]).astype(np.int64))
    OLD = th.from_numpy(np.array([b["logp"] for b in buf]).astype(np.float32))
    RET = th.from_numpy(np.array([b["ret"] for b in buf]).astype(np.float32))
    TEMP = th.from_numpy(np.array([b.get("temp", 1.0) for b in buf])
                         .astype(np.float32))
    with th.no_grad():
        adv = RET - critic(J).squeeze(-1)
        adv = (adv - adv.mean()) / (adv.std().clamp(min=1e-6))
    n = len(buf)
    tot = {"pol": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "nb": 0}
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, minibatch):
            bi = idx[s:s + minibatch]
            r = th.tanh(actor(F[bi]))
            logits = (((B[bi] + bonus * r)
                       / TEMP[bi].unsqueeze(1)) * M[bi]
                      + (1.0 - M[bi]) * -1e9)
            logp_all = th.log_softmax(logits, dim=1)
            logp = logp_all.gather(1, CH[bi].unsqueeze(1)).squeeze(1)
            ratio = th.exp(logp - OLD[bi])
            pg1 = ratio * adv[bi]
            pg2 = th.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[bi]
            pol = -th.minimum(pg1, pg2).mean()
            vpred = critic(J[bi]).squeeze(-1)
            vf = th.nn.functional.mse_loss(vpred, RET[bi])
            probs = th.softmax(logits, dim=1)
            ent = -(probs * logp_all).sum(1).mean()
            loss = pol + vf_coef * vf - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 1.0)
            opt.step()
            with th.no_grad():
                kl = float((OLD[bi] - logp).mean())
            tot["pol"] += float(pol.detach())
            tot["vf"] += float(vf.detach())
            tot["ent"] += float(ent.detach())
            tot["kl"] += kl
            tot["nb"] += 1
    return {k: (v / max(tot["nb"], 1)) for k, v in tot.items() if k != "nb"}


class ShieldHook:
    """Post-shield fallback (Alshiekh et al. post-posed style): the actor
    pick is used only when its combined-score margin (top1 - top2) reaches
    ``min_margin``; otherwise Lancer's argmax pick is used. Lancer is the
    default safe policy; the actor must be decisive to override it.
    Counts actor vs shielded decisions for the override-rate metric."""

    def __init__(self, greedy, min_margin):
        self.greedy = greedy
        self.min_margin = float(min_margin)
        self.n_actor = 0
        self.n_shield = 0

    def __call__(self, env, agent, cands, scored):
        import torch as th
        from blue.policies.ordered import argmax_pick
        from blue.training.scorer import host_rows
        F = th.from_numpy(np.asarray(
            host_rows(env, agent, cands), dtype=np.float32))
        with th.no_grad():
            combined = (th.tensor([s for s, _ in scored])
                        + self.greedy.bonus
                        * th.tanh(self.greedy.residual(F)))
        vals = sorted((float(combined[i]) for i in range(len(cands))),
                      reverse=True)
        margin = vals[0] - vals[1] if len(vals) > 1 else float("inf")
        if margin >= self.min_margin:
            self.n_actor += 1
            return self.greedy(env, agent, cands, scored)
        self.n_shield += 1
        return argmax_pick([(s, h) for s, h in scored])


def cmd_train(args):
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.residual import build_nets, load_phase3_init
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    actor, _pooled_value = build_nets(hidden=args.hidden)
    critic = build_central_critic(hidden=args.hidden)
    opt = th.optim.Adam(list(actor.parameters())
                        + list(critic.parameters()), lr=args.lr)
    start_it, hist = 1, []
    resumed = None
    if args.resume_from:
        src = args.resume_from
        actor.load_state_dict(th.load(os.path.join(src, "actor.th"),
                                      map_location="cpu",
                                      weights_only=False))
        critic.load_state_dict(th.load(os.path.join(src, "critic.th"),
                                       map_location="cpu",
                                       weights_only=False))
        opt.load_state_dict(th.load(os.path.join(src, "opt.th"),
                                    map_location="cpu",
                                    weights_only=False))
        rng = th.load(os.path.join(src, "rng.th"), map_location="cpu",
                      weights_only=False)
        th.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        with open(os.path.join(src, "pilot_manifest.json")) as f:
            old = json.load(f)
        start_it = int(old["iters_done"]) + 1
        hist = list(old.get("hist", []))
        resumed = {"from": src,
                   "commit": old.get("git_commit", "unknown")}
        print(f"resumed from {src} at iter {start_it} "
              f"(actor/critic/opt/RNG state reloaded)")
    else:
        n_copied = load_phase3_init(actor, args.phase3_model,
                                    hidden=args.hidden)
        print(f"phase3 body init ({n_copied} tensors), head zeroed "
              f"-> exact teacher at start")
    out = args.out
    os.makedirs(out, exist_ok=True)
    prog = logutil.Progress(args.iters)
    kl_bad = 0
    seed_cycle = list(args.train_seeds)
    temp_end = args.temperature if args.temp_end is None else args.temp_end
    for it in range(start_it, args.iters + 1):
        frac = (it - 1) / max(args.iters - 1, 1)
        temp = args.temperature + (temp_end - args.temperature) * frac
        gen = th.Generator().manual_seed(args.seed * 100003 + it)
        rec = JointRecorder(actor, args.bonus, gen, temp=temp)
        policy = OrderedPolicy(
            scorer=LancerValues(fruitless_decay=0.5), guard=False)
        ep_rets, ep_covs = [], []
        cat_sum = {"urgent": 0, "verification": 0, "lockout_sleep": 0,
                   "idle": 0}
        n_actor_tot = 0
        for e in range(args.eps_per_iter):
            seed = seed_cycle[(it * args.eps_per_iter + e)
                              % len(seed_cycle)]
            r = run_team_episode(policy, seed, args.steps, hook=rec,
                                 recorder=rec, **ENV_KW)
            ep_rets.append(r["return"])
            ep_covs.append(r["coverage"])
            n_actor_tot += r["n_actor"]
            for k in cat_sum:
                cat_sum[k] += r["forced_cats"][k]
        stats = ppo_central_update(actor, critic, opt, rec.rows,
                                   bonus=args.bonus)
        agree = rec.agree / max(rec.total, 1)
        n_forced_tot = sum(cat_sum.values())
        forced_frac = n_forced_tot / max(n_forced_tot + n_actor_tot, 1)
        mean_ent = float(np.mean([b["ent"] for b in rec.rows])) \
            if rec.rows else 0.0
        row = {"iter": it, "elapsed_s": round(prog.elapsed(), 1),
               "mean_return": float(np.mean(ep_rets)),
               "mean_coverage": float(np.mean(ep_covs)),
               "agree": agree, "n_decisions": len(rec.rows),
               "n_actor": n_actor_tot,
               "forced_cats": cat_sum,
               "forced_frac": round(forced_frac, 4),
               "mean_ent": round(mean_ent, 4), "temp": round(temp, 4),
               **stats}
        hist.append(row)
        print(f"iter {it:3d} ret {row['mean_return']:+7.1f} "
              f"cov {row['mean_coverage']:.2f} agree {agree:.2f} "
              f"forced {forced_frac:.2f} (lock {cat_sum['lockout_sleep']} "
              f"urg {cat_sum['urgent']} ver {cat_sum['verification']} "
              f"idle {cat_sum['idle']}) ent {mean_ent:.3f} "
              f"kl {stats['kl']:.4f} {prog.line(it)}", flush=True)
        if row["mean_coverage"] < args.min_coverage:
            print(f"HALT: coverage {row['mean_coverage']:.2f} < "
                  f"{args.min_coverage} (stop trigger, not a log line)")
            break
        if agree < args.min_agree:
            print(f"HALT: agreement {agree:.2f} < {args.min_agree}")
            break
        kl_bad = kl_bad + 1 if stats["kl"] > args.max_kl else 0
        if kl_bad >= 2:
            print(f"HALT: KL > {args.max_kl} twice running")
            break
    th.save(actor.state_dict(), os.path.join(out, "actor.th"))
    th.save(critic.state_dict(), os.path.join(out, "critic.th"))
    th.save(opt.state_dict(), os.path.join(out, "opt.th"))
    th.save({"torch": th.get_rng_state(),
             "numpy": np.random.get_state(),
             "python": random.getstate()},
            os.path.join(out, "rng.th"))
    manifest = {
        "algo": ALGO,
        "git_commit": git_commit(),
        "dirty": dirty_tree(),
        "guard": False,
        "mode": "train",
        "iters_done": len(hist),
        "train_seeds": list(args.train_seeds),
        "bonus": args.bonus, "hidden": args.hidden,
        "lr": args.lr, "seed": args.seed,
        "temperature": args.temperature,
        "temp_end": args.temp_end,
        "resumed": resumed,
        "phase3_model": args.phase3_model,
        "env_kw": {k: list(v) if isinstance(v, tuple) else v
                   for k, v in ENV_KW.items()},
        "steps": args.steps,
        "central_critic": {"joint_dim": JOINT_CTX_DIM,
                           "inputs": "concat(5 pooled host-row means) + tick",
                           "privileged_state": False},
        "stop": {"min_coverage": args.min_coverage,
                 "min_agree": args.min_agree,
                 "max_kl": args.max_kl},
        "checkpoints": ["actor.th", "critic.th", "opt.th", "rng.th"],
        "hist": hist,
    }
    with open(os.path.join(out, "pilot_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(out, "train_log.jsonl"), "w") as f:
        for row in hist:
            f.write(json.dumps(row) + "\n")
    print("wrote", out)
    return out


def eval_arms(model_dir, hidden, bonus):
    """Learned (unguarded + trained hook) vs lancer / sleep / random.

    Learned-vs-lancer is the PRIMARY deployment comparison here: both run
    guard OFF, so the only difference is the trained hook. Sleep and
    masked-random are the noise-floor controls."""
    import torch as th
    from blue.core.baselines import MaskedRandomBaseline, SleepBaseline
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.residual import build_nets
    from blue.training.residual_pilot import GreedyHook
    actor, _ = build_nets(hidden=hidden)
    actor.load_state_dict(th.load(os.path.join(model_dir, "actor.th"),
                                 map_location="cpu", weights_only=False))
    actor.eval()
    lancer = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                           guard=False)
    learned = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
    return (
        ("learned", learned, GreedyHook(actor, bonus)),
        ("lancer", lancer, None),
        ("sleep", SleepBaseline(), None),
        ("random", MaskedRandomBaseline(seed=0), None),
    )


def cmd_eval(args):
    from blue.core.baselines import MaskedRandomBaseline
    results = {}
    arms = eval_arms(args.model_dir, args.hidden, args.bonus)
    for name, pol, hook in arms:
        for seed in args.eval_seeds:
            if name == "random":
                # Fresh RNG per episode: the random arm must not depend on
                # seed order. Seeded by the episode seed for repeatability.
                pol = MaskedRandomBaseline(seed=int(seed))
            r = run_team_episode(pol, seed, args.steps, hook=hook,
                                 **ENV_KW)
            results.setdefault(name, {})[seed] = r
            print(f"{name:8s} seed={seed} return={r['return']:+7.1f} "
                  f"cov={r['coverage']:.2f}", flush=True)
    contrasts = {}
    for other in ("lancer", "sleep", "random"):
        sa, sb = set(results["learned"]), set(results[other])
        shared = sorted(sa & sb)
        d = [results["learned"][s]["return"]
             - results[other][s]["return"] for s in shared]
        contrasts[f"learned_minus_{other}"] = {
            "paired_mean": float(sum(d) / len(d)), "n_matched": len(shared)}
    summary = {"contrasts": contrasts, "eval_seeds": list(args.eval_seeds),
               "primary": "learned_minus_lancer (both guard OFF; only "
                          "the trained hook differs)",
               "note": "point estimates only, no intervals; use "
                       "blue.analysis.compare for CIs."}
    with open(args.out, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=1)
    for k, v in contrasts.items():
        print(f"{k}: {v['paired_mean']:+.2f} over {v['n_matched']} seeds")
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser(description="Expert-guided MAPPO pilot")
    ap.add_argument("--mode", required=True, choices=("train", "eval"))
    ap.add_argument("--train-seeds", type=int, nargs="*",
                    default=list(DEFAULT_TRAIN_SEEDS))
    ap.add_argument("--eval-seeds", type=int, nargs="*",
                    default=list(DEFAULT_EVAL_SEEDS))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling softmax temperature; <1 sharpens toward "
                         "the greedy deployment policy")
    ap.add_argument("--temp-end", type=float, default=None,
                    help="linear anneal target at the last iter "
                         "(default: no anneal)")
    ap.add_argument("--resume-from", default=None,
                    help="pilot out dir to resume: reloads actor/critic/ "
                         "opt/RNG state and continues after its iters_done")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--min-agree", type=float, default=0.25)
    ap.add_argument("--max-kl", type=float, default=0.2)
    ap.add_argument("--phase3-model",
                    default="blue/results/scorer_mlp.npz")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default="blue/results/mappo_guide1")
    cli = ap.parse_args()
    if cli.mode == "train":
        cmd_train(cli)
    else:
        assert cli.model_dir, "--model-dir required for eval"
        cmd_eval(cli)


if __name__ == "__main__":
    main()
