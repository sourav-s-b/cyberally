"""Larger paired confirmation: agent-4 suspicion skill vs Lancer.

Finite Red, all 32 dev seeds (eval block 7809-8200 untouched), measured
episodes (corrected onset-aware metrics). Per-seed: returns, compromise
burden (n episodes), detection outcomes (miss rates, median delays).
Mechanical gate report: paired mean >= +5 AND 95% CI excludes zero
(t_31 = 2.04). Dev-seed confirmation only — NOT the frozen evaluation.

Usage (train venv): .venv-train/bin/python -m blue.analysis.a4susp_confirm
    --out blue/results/a4susp_confirm32.json
"""

import argparse
import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEV_SEEDS = [7629, 7630, 7640] + list(range(7701, 7730))
T31 = 2.04  # two-sided 95% t critical, df=31


def zone(h):
    for z in ('admin_network', 'office_network', 'public_access'):
        if h.startswith(z):
            return 'a4zones'
    return 'other'


def run_arm(make_policy, seeds, steps, red_agent, partial_out=None):
    from blue.analysis import metrics as MX
    from blue.analysis.recorder import run_measured_episode
    out = {}
    if partial_out and os.path.exists(partial_out):
        try:
            with open(partial_out) as f:
                out = {int(s): v for s, v in
                       json.load(f).get("per_seed", {}).items()}
            print(f"resumed {len(out)} seeds from {partial_out}",
                  flush=True)
        except Exception as e:
            print(f"partial unreadable ({e}); starting over", flush=True)
    for s in seeds:
        if s in out:
            continue
        pol = make_policy()
        res = run_measured_episode(pol, s, steps,
                                   env_kwargs={
                                       "temporal_features": ("ages", "belief"),
                                       "include_root_session": True,
                                       "red_agent": red_agent})
        recs = MX.episode_records(res["timeline"])
        miss = {"a4zones": [0, 0], "other": [0, 0]}
        dets = []
        for e in recs:
            z = zone(e["host"])
            miss[z][0] += 1
            if e["detection_delay"] is None:
                miss[z][1] += 1
            else:
                dets.append(e["detection_delay"])
        dets.sort()
        out[s] = {"return": res["return"], "n_episodes": len(recs),
                  "miss": miss,
                  "det_median": dets[len(dets) // 2] if dets else None}
        print(f"seed={s} return={res['return']:+.0f} "
              f"eps={len(recs)} "
              f"a4miss={miss['a4zones'][1]}/{miss['a4zones'][0]}",
              flush=True)
        if partial_out:
            with open(partial_out, "w") as f:
                json.dump({"partial": True, "per_seed": out}, f)
    return out


def main():
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.policies.zone_priority import Agent4SuspicionHook
    ap = argparse.ArgumentParser(description="a4susp 32-seed confirmation")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--red-agent", default="finite")
    ap.add_argument("--out",
                    default="blue/results/a4susp_confirm32.json")
    cli = ap.parse_args()

    def lancer():
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                             guard=False)

    def skill():
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        pol.hook = Agent4SuspicionHook()
        return pol

    partial_l = cli.out + ".partial.lancer"
    partial_s = cli.out + ".partial.skill"
    rl = run_arm(lancer, cli.seeds, cli.steps, cli.red_agent,
                 partial_out=partial_l)
    rs = run_arm(skill, cli.seeds, cli.steps, cli.red_agent,
                 partial_out=partial_s)
    diffs = [rs[s]["return"] - rl[s]["return"] for s in cli.seeds]
    mean = sum(diffs) / len(diffs)
    sd = statistics.stdev(diffs)
    se = sd / (len(diffs) ** 0.5)
    lo, hi = mean - T31 * se, mean + T31 * se
    gate = {"paired_mean": mean, "paired_sd": sd, "se": se,
            "ci95": [lo, hi],
            "mean_ge_5": bool(mean >= 5), "ci_excludes_zero": bool(lo > 0),
            "gate_pass": bool(mean >= 5 and lo > 0)}
    wins = sum(1 for d in diffs if d > 0)
    report = {"red_agent": cli.red_agent, "seeds": list(cli.seeds),
              "gate": gate, "wins": wins, "losses": sum(1 for d in diffs
                                                       if d < 0),
              "lancer": rl, "skill": rs}
    with open(cli.out, "w") as f:
        json.dump(report, f)
    for p in (partial_l, partial_s):
        try:
            os.remove(p)
        except OSError:
            pass
    print(f"paired_mean={mean:+.2f} sd={sd:.1f} "
          f"CI95=[{lo:+.1f},{hi:+.1f}] wins={wins} "
          f"GATE={'PASS' if gate['gate_pass'] else 'FAIL'}")
    print("wrote", cli.out)


if __name__ == "__main__":
    main()
