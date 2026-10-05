"""Phase A: decision attribution for the guided-MAPPO actor.

Compares greedy learned decisions against Lancer's argmax at the same
decision points (same candidates, same scores) and pairs episode returns
per seed. A deviation is hook_pick != lancer_pick. Per-decision causal
helped/hurt labels are out of scope (credit assignment); helped/hurt is
reported at episode level via the paired return difference.

Usage (train venv, repo root):
  .venv-train/bin/python -m blue.analysis.deviation --model-dir
      blue/results/mappo_guide2 --seeds 7629 7630 7640 7701 7702 7703
      7704 7705 --out blue/results/deviation_discovery.json
  .venv-train/bin/python -m blue.analysis.deviation --model-dir
      blue/results/mappo_guide2 --red-agent finite --seeds ... \
      --out blue/results/deviation_finite.json
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

N_FEAT = 14


class TracingHook:
    """Wraps GreedyHook: same picks, plus a per-decision trace."""

    def __init__(self, greedy):
        self.greedy = greedy
        self.trace = []

    def __call__(self, env, agent, cands, scored):
        import torch as th
        from blue.policies.ordered import argmax_pick
        from blue.training.scorer import host_rows
        pick = self.greedy(env, agent, cands, scored)
        rows = np.asarray(host_rows(env, agent, cands),
                          dtype=np.float32)
        F = th.from_numpy(rows)
        with th.no_grad():
            combined = (th.tensor([s for s, _ in scored])
                        + self.greedy.bonus
                        * th.tanh(self.greedy.residual(F)))
        base_scores = [float(s) for s, _ in scored]
        comb_scores = [float(combined[i]) for i in range(len(cands))]
        lancer_pick = argmax_pick([(s, h) for s, h in scored])
        self.trace.append({
            "tick": int(env._tick), "agent": agent, "k": len(cands),
            "pick": pick, "lancer_pick": lancer_pick,
            "deviation": bool(pick != lancer_pick),
            "base_scores": base_scores, "combined_scores": comb_scores,
            "rows": rows.tolist(),
        })
        return pick


def run_arm(model_dir, seeds, steps, red_agent, hidden, bonus):
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.residual import build_nets
    from blue.training.residual_pilot import GreedyHook
    actor, _ = build_nets(hidden=hidden)
    actor.load_state_dict(th.load(os.path.join(model_dir, "actor.th"),
                                 map_location="cpu", weights_only=False))
    actor.eval()
    env_kw = dict(mg.ENV_KW)
    env_kw["red_agent"] = red_agent
    out = {}
    for seed in seeds:
        lancer = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                               guard=False)
        rl = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                           guard=False)
        hook = TracingHook(GreedyHook(actor, bonus))
        r_lancer = mg.run_team_episode(lancer, seed, steps, **env_kw)
        r_learned = mg.run_team_episode(rl, seed, steps, hook=hook,
                                        **env_kw)
        out[seed] = {"lancer_return": r_lancer["return"],
                     "learned_return": r_learned["return"],
                     "paired_diff": (r_learned["return"]
                                     - r_lancer["return"]),
                     "trace": hook.trace}
        print(f"seed={seed} lancer={r_lancer['return']:+7.1f} "
              f"learned={r_learned['return']:+7.1f} "
              f"deviations={sum(t['deviation'] for t in hook.trace)}/"
              f"{len(hook.trace)}", flush=True)
    return out


def summarize(per_seed):
    devs = [t for s in per_seed.values() for t in s["trace"]]
    n = len(devs)
    d = [t for t in devs if t["deviation"]]
    summary = {"n_decisions": n, "n_deviations": len(d),
               "deviation_rate": (len(d) / n if n else 0.0)}
    by_agent, by_q, margins, feat_delta = {}, {}, [], []
    for t in d:
        by_agent[t["agent"]] = by_agent.get(t["agent"], 0) + 1
        by_q[f"q{1 + min(t['tick'], 399) * 4 // 400}"] = \
            by_q.get(f"q{1 + min(t['tick'], 399) * 4 // 400}", 0) + 1
        i_pick = t["combined_scores"].index(
            max(t["combined_scores"]))
        # margin: combined score of hook pick minus combined score of
        # lancer's pick (how decisive the override was under the actor)
        ci = t["base_scores"].index(max(t["base_scores"]))
        margins.append(abs(t["combined_scores"][i_pick]
                           - t["combined_scores"][ci]))
        rp = np.asarray(t["rows"][i_pick])
        rl = np.asarray(t["rows"][ci])
        feat_delta.append(np.abs(rp - rl).tolist())
    summary["deviations_by_agent"] = by_agent
    summary["deviations_by_tick_quartile"] = by_q
    summary["override_margin_mean"] = (float(np.mean(margins))
                                       if margins else 0.0)
    summary["mean_abs_feat_delta"] = (
        np.mean(feat_delta, axis=0).tolist() if feat_delta else [])
    pd = [s["paired_diff"] for s in per_seed.values()]
    summary["paired_mean"] = float(sum(pd) / len(pd))
    summary["paired_per_seed"] = {str(s): per_seed[s]["paired_diff"]
                                  for s in per_seed}
    return summary


def main():
    ap = argparse.ArgumentParser(description="MAPPO deviation attribution")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=list(mg.DEFAULT_EVAL_SEEDS))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--red-agent", default="discovery")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--out",
                    default="blue/results/deviation_discovery.json")
    cli = ap.parse_args()
    per_seed = run_arm(cli.model_dir, cli.seeds, cli.steps,
                       cli.red_agent, cli.hidden, cli.bonus)
    report = {"model_dir": cli.model_dir, "red_agent": cli.red_agent,
              "seeds": list(cli.seeds), "steps": cli.steps,
              "per_seed": per_seed, "summary": summarize(per_seed)}
    with open(cli.out, "w") as f:
        json.dump(report, f)
    s = report["summary"]
    print(f"deviation_rate={s['deviation_rate']:.3f} "
          f"paired_mean={s['paired_mean']:+.2f}")
    print("wrote", cli.out)


if __name__ == "__main__":
    main()
