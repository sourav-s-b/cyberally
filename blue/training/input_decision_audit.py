"""Audit the frozen risk scorer at legal Blue investigation decisions.

This is a *diagnostic*, not a new policy or an off-policy return estimate.
The defended episode runs the unmodified coverage-protected heuristic.  The
scorer is read-only.  Privileged red-session labels are attached only after
all five Blue actions for the tick have been selected; no label enters the
policy, risk scorer, action mask, or candidate features.

Example (development episodes already used by this project)::

    python -m blue.training.input_decision_audit \
      --plan /path/to/architecture_study/plan.json \
      --scorer /path/to/frozen/scorer.pkl --seeds 8221 8222 \
      --out blue/results/input_audit

The JSONL contains one row per *sweep decision*, with every legal Analyse
candidate and its Blue-visible features.  Guard-forced decisions are included
and marked.  The summary compares risk and heuristic rankings at the *same*
candidate set, including never-inspected hosts; it does not pretend those
counterfactual rankings have measured defensive returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
from blue.harness.features import NAMES, VERSION
from blue.harness.policy import HarnessPolicy, MLRankingHook
from blue.harness.inspection_value import FreshnessRiskHook, inspection_value_rows
from blue.harness.redesign import EvidenceMemory
from blue.harness.scoring import MLScorer, sha256
from blue.policies.ordered import argmax_pick
from blue.training.risk_data import true_compromised


VISIBLE_FIELDS = (
    "analysis_available", "analysis_age", "Files_available", "Files_age",
    "Processes_available", "Processes_age", "belief_SUSPICIOUS",
    "empty_analysis_strikes", "delta_unknown_files", "delta_file_density",
    "delta_connections", "delta_external_connections",
)


def _rank(candidates, field):
    return argmax_pick([(float(c[field]), c["host"]) for c in candidates])


def summarize(rows):
    """Per-episode candidate ranking, never treating decisions as episodes."""
    out = {}
    for phase, tick_range in (
        ("early", (0, 100)), ("middle", (100, 200)),
        ("late", (200, math.inf)), ("full", (0, math.inf)),
    ):
        selected = [r for r in rows if tick_range[0] <= r["tick"] < tick_range[1]]
        out[phase] = {}
        for cohort in ("all", "never_inspected", "previously_inspected"):
            n_decisions = n_positive = n_candidates = n_positive_candidates = 0
            risk_hits = base_hits = 0
            random_hits = 0.0
            chosen_hits = 0
            for r in selected:
                candidates = [c for c in r["candidates"]
                              if cohort == "all" or
                              (c["visible"]["analysis_available"] == 0.0)
                              == (cohort == "never_inspected")]
                if not candidates:
                    continue
                n_decisions += 1
                n_candidates += len(candidates)
                positives = sum(int(c["compromised_label"]) for c in candidates)
                n_positive_candidates += positives
                if not positives:
                    continue
                n_positive += 1
                positive_hosts = {c["host"] for c in candidates
                                  if c["compromised_label"]}
                risk_hits += int(_rank(candidates, "risk") in positive_hosts)
                base_hits += int(_rank(candidates, "base_score") in positive_hosts)
                random_hits += positives / len(candidates)
                chosen_hits += int(r["chosen_host"] in positive_hosts)
            out[phase][cohort] = {
                "decisions_with_candidates": n_decisions,
                "decisions_with_compromised_candidate": n_positive,
                "candidate_exposures": n_candidates,
                "compromised_candidate_exposures": n_positive_candidates,
                "risk_top1_on_positive_sets": risk_hits / n_positive if n_positive else None,
                "heuristic_top1_on_positive_sets": base_hits / n_positive if n_positive else None,
                "random_top1_on_positive_sets": random_hits / n_positive if n_positive else None,
                "actual_choice_top1_on_positive_sets": chosen_hits / n_positive if n_positive else None,
            }
    return out


def collect_episode(seed, steps, scorer, env_kw, max_age, mode="guard"):
    env = CC4MARLEnv(seed=seed, steps=steps, **env_kw)
    env.reset(seed=seed)
    if mode not in ("guard", "risk_bonus", "freshness_risk"):
        raise ValueError("unknown diagnostic policy mode")
    hook = (MLRankingHook(scorer, risk_bonus=.25, novelty_bonus=0.)
            if mode == "risk_bonus" else
            FreshnessRiskHook(scorer, max_age=max_age, bonus=.25)
            if mode == "freshness_risk" else None)
    policy = HarnessPolicy(hook, max_age=max_age)
    policy.record_decisions = True
    policy.reset()
    scorer.reset()
    memory = EvidenceMemory()
    rows, rewards = [], []
    completed_analyses = 0
    confirmations = 0
    try:
        while True:
            actions, pending_rows = {}, []
            previous_analysis = {a: dict(env.trackers[a].last_analysis)
                                 for a in BLUE_AGENTS}
            for agent in BLUE_AGENTS:
                # Advance visible evidence on *every* simulator tick, including
                # ticks on which this agent is pending and cannot choose.
                raw = scorer.observe(env, agent)
                history = memory.rows(env, agent, raw)
                actions[agent] = int(policy.select(env, agent))
                decision = policy.last_decision
                if decision is None:
                    continue
                if decision["truncated"] or decision["cands"] is None:
                    raise RuntimeError("candidate recording was not enabled")
                hosts = decision["cands"]
                if not hosts:
                    raise RuntimeError("empty sweep candidate set")
                scores = scorer.score(env, agent, hosts)
                inspection = inspection_value_rows(
                    env, agent, hosts, scores[:, 0], max_age)
                lookup = {h: i for i, h in enumerate(env.hostnames[agent])}
                base = dict((h, float(s)) for s, h in decision["scores"])
                candidates = []
                for i, host in enumerate(hosts):
                    ix = lookup[host]
                    visible = {name: (float(raw[ix, NAMES.index(name)])
                                      if math.isfinite(float(raw[ix, NAMES.index(name)]))
                                      else None)
                               for name in VISIBLE_FIELDS}
                    candidates.append({
                        "host": host, "base_score": base[host],
                        "risk": float(scores[i, 0]),
                        "novelty_percentile": float(scores[i, 1]),
                        "inspection_age_fraction": float(inspection[i, 0]),
                        "freshness_adjusted_risk": float(inspection[i, 1]),
                        "visible": visible,
                        "event_memory": [float(v) for v in history[ix]],
                    })
                pending_rows.append({
                    "tick": int(env._tick), "agent": agent,
                    "branch": decision["branch"],
                    "chosen_host": decision["pick"],
                    "guard_forced": decision["branch"] == "sweep_max_age",
                    "candidates": candidates,
                })
            # Privileged state is sampled only after actions are fixed.
            compromised = true_compromised(env)
            for row in pending_rows:
                for c in row["candidates"]:
                    c["compromised_label"] = c["host"] in compromised
            rows.extend(pending_rows)
            _, reward, terminated, truncated, _ = env.step(actions)
            rewards.append(float(reward[0]))
            for agent in BLUE_AGENTS:
                tracker = env.trackers[agent]
                for host in env.hostnames[agent]:
                    if tracker.last_analysis.get(host) != previous_analysis[agent].get(host):
                        completed_analyses += 1
                        confirmations += int(tracker.state.get(host) == "CONFIRMED")
            if terminated or truncated:
                break
        return rows, {
            "seed": seed, "ticks": len(rewards), "return": sum(rewards),
            "completed_analyses": completed_analyses,
            "analysis_confirmations": confirmations,
            "ranking": summarize(rows),
        }
    finally:
        env.close()


def run(scorer_path, seeds, out, plan_path, mode="guard"):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("specify unique development seeds")
    if any(7809 <= seed <= 8200 for seed in seeds):
        raise ValueError("final evaluation seeds 7809-8200 are reserved")
    if mode not in ("guard", "risk_bonus", "freshness_risk"):
        raise ValueError("unknown diagnostic policy mode")
    frozen_plan = json.loads(Path(plan_path).read_text())
    config = frozen_plan["config"]
    if not set(seeds) <= set(config["eval_seeds"]):
        raise ValueError("audit seeds must come from the frozen development cohort")
    if sha256(scorer_path) != frozen_plan["scorer_sha256"]:
        raise ValueError("scorer differs from frozen architecture study")
    if config["env_kw"]["red_agent"] != "finite":
        raise ValueError("expected frozen finite Red configuration")
    steps, env_kw, max_age = (int(config["steps"]), config["env_kw"],
                              float(config["max_age"]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorer = MLScorer(scorer_path)
    source = Path(__file__)
    manifest = {
        "purpose": "decision-time mechanism audit, not a policy-performance claim",
        "policy": (f"HarnessPolicy(max_age={max_age}), no learning hook"
                   if mode == "guard" else
                   f"HarnessPolicy(max_age={max_age}) + MLRankingHook(risk_bonus=.25, novelty_bonus=0)"
                   if mode == "risk_bonus" else
                   f"HarnessPolicy(max_age={max_age}) + FreshnessRiskHook(bonus=.25)"),
        "frozen_plan": str(Path(plan_path).resolve()),
        "frozen_plan_sha256": sha256(plan_path),
        "scorer": str(Path(scorer_path).resolve()),
        "scorer_sha256": sha256(scorer_path),
        "feature_version": VERSION, "feature_names": list(NAMES),
        "episode_seeds": list(seeds), "steps": steps,
        "env_kw": env_kw,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "label_semantics": "privileged red-session presence; attached after actions selected",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for seed in seeds:
        rows, result = collect_episode(seed, steps, scorer, env_kw, max_age, mode)
        trace = out / f"decision_candidates_{seed}.jsonl"
        with trace.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        result["trace_file"] = trace.name
        result["trace_sha256"] = sha256(trace)
        results.append(result)
        print(f"seed={seed} return={result['return']:+.0f} "
              f"decisions={len(rows)} analyses={result['completed_analyses']}",
              flush=True)
        (out / "summary.json").write_text(json.dumps({
            "manifest": manifest, "episodes": results}, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=("guard", "risk_bonus", "freshness_risk"),
                        default="guard")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.scorer, args.seeds, args.out, args.plan, args.mode)


if __name__ == "__main__":
    main()
