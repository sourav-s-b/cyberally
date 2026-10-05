"""Fast test: learned risk score as an INPUT FEATURE to the guided actor.

Deck Stage 2 -> Stage 3 wiring. The 2026-10-02 evidence killed adding the
risk score to the ORDERING (coverage collapse, -186/-251). This variant
keeps Lancer's ordering as the base score and appends P(compromised) as a
15th input column to the residual actor, evaluated under the margin-0.5
fallback shield so risky-chasing cannot collapse coverage. That
combination is untested.

Teacher-at-init is preserved: the Phase-3 body is copied into the widened
first layer with the new column zeroed, and the residual head stays zero,
so argmax reproduces Lancer exactly at iteration 0.

Usage (train venv, repo root):
  ... -m blue.training.risk_actor --mode train --iters 4 --eps-per-iter 4
  ... -m blue.training.risk_actor --mode eval --model-dir blue/results/risk_actor1
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

from blue.training import mappo_guide as mg

FEAT_DIM = 15           # 14 host rows + 1 learned risk probability
RISK_MODEL = "results/risk_model_v2.pkl"
DEFAULT_TRAIN_SEEDS = [7706, 7707, 7708, 7709]
DEFAULT_EVAL_SEEDS = [7629, 7630, 7640, 7701, 7702, 7703, 7704, 7705]


def _risk():
    from blue.policies.hybrid import RiskPriority
    return RiskPriority(model_path=RISK_MODEL)


def host_rows_15(env, agent, cands, risk):
    """The usual 14 Blue-visible rows + learned P(compromised) per host."""
    from blue.training.scorer import host_rows
    base = np.asarray(host_rows(env, agent, cands), dtype=np.float32)
    proba = risk._predict_all(env, agent)
    extra = np.asarray([[proba.get(h, 0.0)] for h in cands],
                       dtype=np.float32)
    return np.concatenate([base, extra], axis=1)


class RiskRecorder(mg.JointRecorder):
    """JointRecorder whose features carry the learned risk column.

    ``min_proba`` changes WHAT IS LEARNED FROM, not what is executed: every
    decision still runs, but a decision is written to the PPO buffer only
    when the actor's pick has P(compromised) >= min_proba. Motivation
    (calibration measured 2026-10-05): the 0.4-0.6 proba band holds 585
    host-ticks at 30% real, so most buffer rows teach noise. Filtering
    keeps the on-policy contract (recorded logp is the taken action) while
    concentrating gradient on informative decisions."""

    def __init__(self, *a, risk=None, min_proba=0.0, **kw):
        super().__init__(*a, **kw)
        self.risk = risk
        self.min_proba = float(min_proba)
        self.n_skipped = 0

    def __call__(self, env, agent, cands, scored):
        from blue.policies.ordered import argmax_pick
        from blue.training.residual import masked_choice
        import torch as th
        F = th.from_numpy(host_rows_15(env, agent, cands, self.risk))
        with th.no_grad():
            r = th.tanh(self.actor(F))
        base = th.tensor([s for s, _ in scored], dtype=th.float32)
        combined = (base + self.bonus * r) / self.temp
        mask = [1.0] * len(cands)
        idx, logp, ent = masked_choice(combined, mask, sample=True,
                                       seed_rng=self.gen)
        arg_base = argmax_pick([(s, h) for s, h in scored])
        arg_comb = argmax_pick([(float(combined[i]), cands[i])
                                for i in range(len(cands))])
        self.agree += int(arg_comb == arg_base)
        self.total += 1
        if float(F[idx, 14]) < self.min_proba:
            self.n_skipped += 1
            return cands[idx]          # executed, not trained on
        row = {"feats": F.numpy(), "joint": mg.joint_context(
            env, self._policy_ref, self._jcache), "k": len(cands),
            "mask": mask, "base": base.numpy(), "choice": idx, "logp": logp,
            "ent": ent, "tick": env._tick, "agent": agent, "ret": 0.0,
            "temp": self.temp}
        self.rows.append(row)
        self.pending.setdefault(agent, []).append(len(self.rows) - 1)
        self.per_agent[agent] = self.per_agent.get(agent, 0) + 1
        return cands[idx]


def build_actor(hidden=64):
    """Residual body widened to FEAT_DIM; Phase-3 weights copied with the
    risk column zeroed, head zeroed -> exact Lancer at init."""
    import torch as th
    from blue.training.residual import build_nets
    actor, _ = build_nets(feat_dim=FEAT_DIM, hidden=hidden)
    # Own widened copy: load_phase3_init assumes a 14-col first layer.
    state = th.load("blue/results/scorer_mlp.npz", map_location="cpu",
                    weights_only=False)
    body = actor.body
    with th.no_grad():
        body[0].weight[:, :14].copy_(state["0.weight"])
        body[0].weight[:, 14].zero_()
        body[0].bias.copy_(state["0.bias"])
        body[2].weight.copy_(state["2.weight"])
        body[2].bias.copy_(state["2.bias"])
        actor.head.weight.zero_()
        actor.head.bias.zero_()
    return actor


def cmd_train(args):
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    actor = build_actor(hidden=args.hidden)
    # Central critic over the joint Blue-visible context (5 pooled 14-dim
    # means + tick) - unchanged dimension, so the tested
    # ppo_central_update applies directly to these rows.
    critic = mg.build_central_critic(hidden=args.hidden)
    opt = th.optim.Adam(list(actor.parameters())
                        + list(critic.parameters()), lr=args.lr)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    risk = _risk()
    os.makedirs(args.out, exist_ok=True)
    hist = []
    cycle = list(args.train_seeds)
    for it in range(1, args.iters + 1):
        gen = th.Generator().manual_seed(args.seed * 100003 + it)
        rec = RiskRecorder(actor, args.bonus, gen, temp=args.temp,
                            risk=risk, min_proba=args.train_min_proba)
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        rets = []
        for e in range(args.eps_per_iter):
            seed = cycle[(it * args.eps_per_iter + e) % len(cycle)]
            r = mg.run_team_episode(pol, seed, args.steps, hook=rec,
                                    recorder=rec, **mg.ENV_KW)
            rets.append(r["return"])
        st = mg.ppo_central_update(actor, critic, opt, rec.rows,
                                   bonus=args.bonus)
        row = {"iter": it, "mean_return": float(np.mean(rets)),
               "n_decisions": len(rec.rows),
               "n_skipped": rec.n_skipped,
               "agree": rec.agree / max(rec.total, 1), **st}
        hist.append(row)
        print(f"iter {it} ret {row['mean_return']:+.1f} "
              f"trained {len(rec.rows)}/{rec.total} "
              f"agree {row['agree']:.2f} kl {st['kl']:.4f}", flush=True)
    th.save(actor.state_dict(), os.path.join(args.out, "actor.th"))
    th.save(critic.state_dict(), os.path.join(args.out, "critic.th"))
    with open(os.path.join(args.out, "pilot_manifest.json"), "w") as f:
        json.dump({"algo": "risk-as-feature residual PPO (14+1 cols)",
                   "iters_done": len(hist), "feat_dim": FEAT_DIM,
                   "risk_model": RISK_MODEL, "temp": args.temp,
                   "train_min_proba": args.train_min_proba,
                   "train_seeds": args.train_seeds, "hist": hist}, f,
                  indent=1)
    print("wrote", args.out)


class RiskGreedyHook:
    """Greedy residual pick using the 15-column features."""

    def __init__(self, actor, bonus, risk):
        import torch as th
        self.th = th
        self.actor = actor
        self.bonus = bonus
        self.risk = risk

    def __call__(self, env, agent, cands, scored):
        import torch as th
        from blue.policies.ordered import argmax_pick
        F = th.from_numpy(host_rows_15(env, agent, cands, self.risk))
        with th.no_grad():
            comb = (th.tensor([s for s, _ in scored], dtype=th.float32)
                    + self.bonus * th.tanh(self.actor(F)))
        return argmax_pick([(float(comb[i]), cands[i])
                            for i in range(len(cands))])


class RiskShieldHook:
    """Margin-gated fallback for the 15-column actor: actor pick only when
    its combined-score margin (top1-top2) reaches min_margin, else Lancer's
    argmax. Same rule as mappo_guide.ShieldHook but on risk features."""

    def __init__(self, greedy, min_margin):
        self.greedy = greedy
        self.min_margin = float(min_margin)
        self.n_actor = 0
        self.n_shield = 0

    def __call__(self, env, agent, cands, scored):
        import torch as th
        from blue.policies.ordered import argmax_pick
        F = th.from_numpy(host_rows_15(env, agent, cands, self.greedy.risk))
        with th.no_grad():
            comb = (th.tensor([s for s, _ in scored], dtype=th.float32)
                    + self.greedy.bonus * th.tanh(self.greedy.actor(F)))
        vals = sorted((float(comb[i]) for i in range(len(cands))),
                      reverse=True)
        margin = vals[0] - vals[1] if len(vals) > 1 else float("inf")
        if margin >= self.min_margin:
            self.n_actor += 1
            return self.greedy(env, agent, cands, scored)
        self.n_shield += 1
        return argmax_pick([(s, h) for s, h in scored])


class RiskConfidentGate:
    """Trust the risk feature only when it is confident: deviate from
    Lancer's argmax only when the actor's pick has P(compromised) >= thresh.
    Diagnosis 2026-10-05: the collapse on seed 7703 is NOT a coverage
    collapse (coverage stays 1.000); the actor overrides 79% of decisions
    because risk proba outranks lancer value, so it spends effort on
    plausible-but-clean hosts (mean proba 0.45). Gating on proba
    confidence keeps the high-confidence wins and drops the churn."""

    def __init__(self, greedy, thresh):
        self.greedy = greedy
        self.thresh = float(thresh)
        self.n_actor = 0
        self.n_shield = 0

    def __call__(self, env, agent, cands, scored):
        import torch as th
        from blue.policies.ordered import argmax_pick
        F = th.from_numpy(host_rows_15(env, agent, cands, self.greedy.risk))
        with th.no_grad():
            comb = (th.tensor([s for s, _ in scored], dtype=th.float32)
                    + self.greedy.bonus * th.tanh(self.greedy.actor(F)))
        vals = [(float(comb[i]), cands[i], float(F[i, 14]))
                for i in range(len(cands))]
        pick = max(vals, key=lambda t: t[0])
        if pick[2] >= self.thresh:
            self.n_actor += 1
            return pick[1]
        self.n_shield += 1
        return argmax_pick([(s, h) for s, h in scored])


def cmd_eval(args):
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    actor = build_actor(hidden=args.hidden)
    actor.load_state_dict(th.load(os.path.join(args.model_dir, "actor.th"),
                                  map_location="cpu", weights_only=False))
    actor.eval()
    risk = _risk()
    out = {}
    gates = [("conf%s" % t, RiskConfidentGate(
        RiskGreedyHook(actor, args.bonus, risk), t))
        for t in args.gate_thresholds]
    for name, ghook in [("lancer", None), ("greedy", None)] + gates:
        for seed in args.eval_seeds:
            pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                                guard=False)
            hook = ghook
            if name == "greedy":
                hook = RiskGreedyHook(actor, args.bonus, risk)
            r = mg.run_team_episode(pol, seed, args.steps, hook=hook,
                                    **mg.ENV_KW)
            out.setdefault(name, {})[seed] = r["return"]
            print(f"{name:6s} seed={seed} {r['return']:+.0f}", flush=True)
    base = out["lancer"]
    summ = {}
    for name in ["greedy"] + ["conf%s" % t
                              for t in args.gate_thresholds]:
        d = [out[name][s] - base[s] for s in args.eval_seeds]
        m = sum(d) / len(d)
        se = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5
        summ[f"{name}_minus_lancer"] = {"mean": m,
                                        "ci95": [m - 2.365 * se,
                                                 m + 2.365 * se]}
    with open(args.out, "w") as f:
        json.dump({"cells": out, "summary": summ}, f, indent=1)
    for k, v in summ.items():
        print(f"{k}: {v['mean']:+.2f} CI[{v['ci95'][0]:+.1f},"
              f"{v['ci95'][1]:+.1f}]")
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser(description="risk-as-feature fast test")
    ap.add_argument("--mode", required=True, choices=("train", "eval"))
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--train-min-proba", type=float, default=0.0,
                    help="record a decision for PPO only if the pick's "
                         "risk proba >= this (0 = train on all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seeds", type=int, nargs="*",
                    default=list(DEFAULT_TRAIN_SEEDS))
    ap.add_argument("--eval-seeds", type=int, nargs="*",
                    default=list(DEFAULT_EVAL_SEEDS))
    ap.add_argument("--shield-margin", type=float, default=0.5)
    ap.add_argument("--gate-thresholds", type=float, nargs="*",
                    default=[0.5, 0.7])
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default="blue/results/risk_actor1_eval.json")
    cli = ap.parse_args()
    if cli.mode == "train":
        cmd_train(cli)
    else:
        cli.model_dir = cli.model_dir or "blue/results/risk_actor1"
        cmd_eval(cli)


if __name__ == "__main__":
    main()