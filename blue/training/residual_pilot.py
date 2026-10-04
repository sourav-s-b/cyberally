"""Phase 4: budgeted narrow PPO pilot on the sweep residual (dev seeds).

Policy: OrderedPolicy rules 1-3 fixed + ResidualAnchor(frozen lancer
scorer, bonus M) + learned residual MLP (Phase-3 init, zeroed head).
Guard ON (learner regime). Only sweep Analyse decisions enter the
buffer, sampled from the combined masked softmax; rules/emergency
actions are never learner data. Fresh on-policy rollouts per PPO
iteration; Monte-Carlo returns to episode end (gamma .99) with learned
value baseline.

Stop triggers (halt, not log): rolling coverage < --min-coverage,
heuristic agreement < --min-agree, approx KL > --max-kl twice running.
Budget: --iters x --eps-per-iter episodes on --train-seeds (dev only).

Eval mode: greedy learned scorer on --eval-seeds vs lancer +
scheduler-control, paired means.

Usage (repo root):
  .../python -m blue.training.residual_pilot --mode train \
      --train-seeds 7706 ... --iters 20 --eps-per-iter 8 \
      --out blue/results/resid_pilot1
  .../python -m blue.training.residual_pilot --mode eval \
      --model blue/results/resid_pilot1/residual.th --eval-seeds 7629 ...
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blue.common import logutil

GAMMA = 0.99
MAX_CANDS = 51
ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}


class GreedyHook:
    """Non-recording argmax hook for eval: combined = base + M*tanh(r)
    with the TRAINED residual. Controls run with hook=None (pure
    reference behavior, no residual contamination)."""

    def __init__(self, residual, bonus):
        import torch as th
        self.th = th
        self.residual = residual
        self.bonus = bonus

    def __call__(self, env, agent, cands, scored):
        import numpy as np
        from blue.policies.ordered import argmax_pick
        from blue.training.scorer import host_rows
        th = self.th
        F = th.from_numpy(np.asarray(
            host_rows(env, agent, cands), dtype=np.float32))
        with th.no_grad():
            combined = (th.tensor([s for s, _ in scored])
                        + self.bonus * th.tanh(self.residual(F)))
        return argmax_pick([(float(combined[i]), cands[i])
                            for i in range(len(cands))])


class Recorder:
    """Sampler hook: records sweep decisions + samples combined softmax."""

    def __init__(self, residual, bonus, sample, rng):
        import torch as th
        self.th = th
        self.residual = residual
        self.bonus = bonus
        self.sample = sample
        self.rng = rng
        self.rows = []
        self.pending = {}  # (agent) -> list of row idxs awaiting returns
        self.tick_rew = []  # per-tick team rewards this episode
        self.agree = 0
        self.total = 0
        self._ep_start = 0

    def mark_episode(self):
        self._ep_start = len(self.rows)

    def __call__(self, env, agent, cands, scored):
        from blue.training.scorer import host_rows
        from blue.training.residual import masked_choice
        th = self.th
        rows = host_rows(env, agent, cands)
        F = th.from_numpy(np.asarray(rows, dtype=np.float32))
        with th.no_grad():
            r = th.tanh(self.residual(F))
        base = th.tensor([s for s, _ in scored], dtype=th.float32)
        combined = base + self.bonus * r
        mask = [1.0] * len(cands)
        idx, logp, _ = masked_choice(combined, mask, sample=self.sample)
        from blue.policies.ordered import argmax_pick
        pairs = [(float(combined[i]), cands[i]) for i in range(len(cands))]
        arg_comb = argmax_pick(pairs)
        arg_base = argmax_pick([(s, h) for s, h in scored])
        # Deploy-time agreement (argmax vs argmax): sampling exploration
        # must not trip the divergence stop trigger.
        self.agree += int(arg_comb == arg_base)
        self.total += 1
        tick = env._tick
        ctx = F.mean(0).tolist() + [tick / 400.0]
        row = {"feats": F.numpy(), "ctx": ctx, "mask": mask,
               "base": base.numpy(), "choice": idx, "logp": logp,
               "tick": tick, "ret": 0.0}
        self.rows.append(row)
        self.pending.setdefault(agent, []).append(len(self.rows) - 1)
        return cands[idx]

    def close_episode(self, rewards):
        """Assign discounted Monte-Carlo returns (to episode end) to every
        recorded decision, padding variable candidate lists to MAX_CANDS
        for batched PPO. rewards: list of per-tick team rewards."""
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


def run_episode(policy, seed, steps, hook=None, recorder=None, **env_kw):
    from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
    env = CC4MARLEnv(seed=seed, steps=steps, **env_kw)
    env.reset(seed=seed)
    policy.reset()
    policy.hook = hook  # None = reference behavior (never contaminated)
    if recorder is not None:
        recorder.mark_episode()
    rewards = []
    analysed = set()
    n_hosts = sum(len(env.hostnames[a]) for a in BLUE_AGENTS)
    for tick in range(1, steps + 1):
        actions = {a: int(policy.select(env, a)) for a in BLUE_AGENTS}
        _, rew, terminated, truncated, _ = env.step(actions)
        rewards.append(float(rew[0]))
        for agent in BLUE_AGENTS:
            from blue.core.baselines import decode_index
            name, host = decode_index(env, agent, actions[agent])
            if name == "Analyse" and host is not None:
                analysed.add((agent, host))
        if terminated or truncated:
            break
    if recorder is not None:
        recorder.close_episode(rewards)
    coverage = len(analysed) / max(n_hosts, 1)
    return {"seed": seed, "return": float(sum(rewards)),
            "coverage": coverage,
            "n_decisions": (len(recorder.rows) if recorder is not None
                            else 0)}


def cmd_train(args):
    import numpy as np
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.residual import (build_nets, load_phase3_init,
                                        ppo_update)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    residual, value = build_nets(hidden=args.hidden)
    n_copied = load_phase3_init(residual, args.phase3_model,
                                hidden=args.hidden)
    print(f"phase3 body init ({n_copied} tensors), head zeroed "
          f"-> exact teacher at start")
    opt = th.optim.Adam(list(residual.parameters())
                        + list(value.parameters()), lr=args.lr)
    out = args.out
    os.makedirs(os.path.join(out, "0"), exist_ok=True)
    prog = logutil.Progress(args.iters)
    hist = []
    kl_bad = 0
    cov_roll, ret_roll = [], []
    seed_cycle = list(args.train_seeds)
    for it in range(1, args.iters + 1):
        rec = Recorder(residual, args.bonus, True, np.random)
        policy = OrderedPolicy(
            scorer=LancerValues(fruitless_decay=0.5), guard=True)
        ep_rets, ep_covs = [], []
        for e in range(args.eps_per_iter):
            seed = seed_cycle[(it * args.eps_per_iter + e)
                              % len(seed_cycle)]
            r = run_episode(policy, seed, args.steps, hook=rec,
                            recorder=rec, **ENV_KW)
            ep_rets.append(r["return"])
            ep_covs.append(r["coverage"])
        stats = ppo_update(residual, value, opt, rec.rows,
                           bonus=args.bonus)
        agree = rec.agree / max(rec.total, 1)
        cov_roll.append(float(np.mean(ep_covs)))
        ret_roll.append(float(np.mean(ep_rets)))
        row = {"iter": it, "elapsed_s": round(prog.elapsed(), 1),
               "mean_return": float(np.mean(ep_rets)),
               "mean_coverage": float(np.mean(ep_covs)),
               "agree": agree, "n_decisions": len(rec.rows), **stats}
        hist.append(row)
        print(f"iter {it:3d} ret {row['mean_return']:+7.1f} "
              f"cov {row['mean_coverage']:.2f} agree {agree:.2f} "
              f"kl {stats['kl']:.4f} ent {stats['ent']:.3f} "
              f"{prog.line(it)}", flush=True)
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
    th.save(residual.state_dict(), os.path.join(out, "residual.th"))
    th.save(value.state_dict(), os.path.join(out, "value.th"))
    with open(os.path.join(out, "pilot_manifest.json"), "w") as f:
        json.dump({"mode": "train", "iters_done": len(hist),
                   "train_seeds": args.train_seeds,
                   "bonus": args.bonus, "hidden": args.hidden,
                   "lr": args.lr, "seed": args.seed,
                   "stop": {"min_coverage": args.min_coverage,
                            "min_agree": args.min_agree,
                            "max_kl": args.max_kl},
                   "hist": hist}, f, indent=1)
    with open(os.path.join(out, "train_log.jsonl"), "w") as f:
        for row in hist:
            f.write(json.dumps(row) + "\n")
    print("wrote", out)


def cmd_eval(args):
    import numpy as np
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.residual import build_nets
    residual, _ = build_nets(hidden=args.hidden)
    residual.load_state_dict(th.load(args.model, map_location="cpu"))
    residual.eval()

    results = {}
    greedy = GreedyHook(residual, args.bonus)
    for name, pol, hook in (
            ("learned", OrderedPolicy(scorer=LancerValues(
                fruitless_decay=0.5), guard=True), greedy),
            ("lancer", OrderedPolicy(
                scorer=LancerValues(fruitless_decay=0.5),
                guard=False), None),
            ("sched_control", OrderedPolicy(
                scorer=LancerValues(fruitless_decay=0.5),
                guard=True), None)):
        for seed in args.eval_seeds:
            r = run_episode(pol, seed, args.steps, hook=hook, **ENV_KW)
            results.setdefault(name, {})[seed] = r
            print(f"{name:13s} seed={seed} return={r['return']:+7.1f} "
                  f"cov={r['coverage']:.2f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser(description="Phase 4 residual PPO pilot")
    ap.add_argument("--mode", required=True, choices=("train", "eval"))
    ap.add_argument("--train-seeds", type=int, nargs="*",
                    default=[7706, 7707, 7708, 7709])
    ap.add_argument("--eval-seeds", type=int, nargs="*",
                    default=[7629, 7630, 7640, 7701, 7702, 7703, 7704, 7705])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--eps-per-iter", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--min-agree", type=float, default=0.25)
    ap.add_argument("--max-kl", type=float, default=0.2)
    ap.add_argument("--phase3-model",
                    default="blue/results/scorer_mlp.npz")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="blue/results/resid_pilot1")
    cli = ap.parse_args()
    if cli.mode == "train":
        cmd_train(cli)
    else:
        assert cli.model, "--model required for eval"
        cmd_eval(cli)


if __name__ == "__main__":
    main()
