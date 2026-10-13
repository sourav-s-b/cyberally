"""Collect Blue-visible Analyse decisions and their *observed* outcomes.

The training table contains only selected Analyse actions. Its labels are
completed-action observations, not simulator truth and not an estimate of the
counterfactual value of inspecting an unselected host. Privileged red-session
presence at request time is written to a separate audit file after actions
have been fixed. Episode IDs are the split unit; individual host rows are not
independent samples.

Example::

    python -m blue.training.inspection_yield_data \
      --plan /path/to/architecture_study/plan.json \
      --scorer /path/to/frozen/scorer.pkl --seeds 8221 8222 \
      --out blue/results/inspection_yield
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
from blue.harness.policy import HarnessPolicy
from blue.harness.redesign import EvidenceMemory
from blue.harness.scoring import MLScorer, sha256
from blue.training.risk_data import true_compromised

SCHEMA = "blue-inspection-yield-v1"
HISTORY_NAMES = tuple(f"unreviewed_positive_change_{i}" for i in range(4)) + \
    tuple(f"positive_change_age_{i}" for i in range(4))


def json_floats(values):
    """Preserve unavailable inputs as JSON null, never silently as zero."""
    out = []
    for value in values:
        value = float(value)
        if math.isinf(value):
            raise ValueError("infinite feature")
        out.append(value if math.isfinite(value) else None)
    return out


class AnalyseLedger:
    """Match one active request per agent to its wrapper-observed result."""

    def __init__(self):
        self.pending = {}
        self.rows = []

    def request(self, agent, host, tick, pre_input, pre_state, pre_analysis):
        if agent in self.pending:
            raise RuntimeError("agent already has an Analyse request")
        row = {
            "request_id": f"{tick}:{agent}:{host}",
            "agent": agent, "host": host, "request_tick": int(tick),
            "pre_input": pre_input,
            "pre_tracker_state": pre_state,
            "pre_last_completed_analysis": pre_analysis,
            "outcome": None,
        }
        self.pending[agent] = row
        self.rows.append(row)
        return row

    def reconcile(self, env):
        """Call immediately after the joint step and wrapper ingestion."""
        for agent, row in list(self.pending.items()):
            host = row["host"]
            waiting = env._awaiting.get(agent)
            if waiting is not None:
                if tuple(waiting) != (host, "Analyse"):
                    raise RuntimeError("pending Analyse identity changed")
                continue
            tracker = env.trackers[agent]
            result = tracker.last_result.get(host)
            if result not in (("Analyse", "TRUE"), ("Analyse", "FALSE")):
                raise RuntimeError("Analyse disappeared without observed result")
            success = result[1] == "TRUE"
            completed = tracker.last_analysis.get(host)
            if success and completed != env._tick:
                raise RuntimeError("successful Analyse has no completion stamp")
            if not success and completed != row["pre_last_completed_analysis"]:
                raise RuntimeError("failed Analyse refreshed inspection age")
            post_state = tracker.state.get(host)
            row["outcome"] = {
                "status": "success" if success else "failure",
                "completion_tick": int(env._tick),
                "duration_ticks": int(env._tick - row["request_tick"]),
                "post_tracker_state": post_state,
                # Observable new discovery, distinct from hidden truth.
                "new_confirmation": bool(success and
                                         row["pre_tracker_state"] != "CONFIRMED" and
                                         post_state == "CONFIRMED"),
            }
            del self.pending[agent]

    def finalize(self):
        for row in self.pending.values():
            row["outcome"] = {
                "status": "censored_at_episode_end",
                "completion_tick": None, "duration_ticks": None,
                "post_tracker_state": None, "new_confirmation": None,
            }
        self.pending.clear()
        return self.rows


def check_frozen_inputs(seeds, plan_path, scorer_path):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("specify unique development episodes")
    if any(7809 <= s <= 8200 for s in seeds):
        raise ValueError("final evaluation seeds 7809-8200 are reserved")
    plan = json.loads(Path(plan_path).read_text())
    config = plan["config"]
    allowed = set(config["train_seeds"]) | set(config["eval_seeds"])
    if not set(seeds) <= allowed:
        raise ValueError("seeds must be in the frozen training/development cohorts")
    if config["env_kw"].get("red_agent") != "finite":
        raise ValueError("expected frozen finite-Red configuration")
    if sha256(scorer_path) != plan["scorer_sha256"]:
        raise ValueError("frozen scorer hash mismatch")
    return config


def collect_episode(seed, config, scorer):
    from blue.core.baselines import decode_index

    env = CC4MARLEnv(seed=seed, steps=int(config["steps"]), **config["env_kw"])
    env.reset(seed=seed)
    policy = HarnessPolicy(max_age=float(config["max_age"]))
    policy.reset()
    scorer.reset()
    history = EvidenceMemory()
    ledger = AnalyseLedger()
    labels = []
    rewards = []
    try:
        while True:
            actions = {}
            requests = []
            for agent in BLUE_AGENTS:
                # The entire host table is advanced even while this agent is
                # pending. Re-reading one host in a tick is idempotent.
                raw = scorer.observe(env, agent)
                memory = history.rows(env, agent, raw)
                index = int(policy.select(env, agent))
                actions[agent] = index
                name, host = decode_index(env, agent, index)
                if name != "Analyse":
                    continue
                if agent in env._awaiting:
                    raise RuntimeError("busy agent requested Analyse")
                ix = env.hostnames[agent].index(host)
                scores = scorer.score(env, agent, [host])[0]
                decision = policy.last_decision
                base_score = (float(policy.scorer.score(env, agent, host))
                              if decision is not None else None)
                request = ledger.request(
                    agent, host, env._tick,
                    {"feature_values": json_floats(raw[ix]),
                     "event_memory_values": json_floats(memory[ix]),
                     "risk": float(scores[0]),
                     "novelty_percentile": float(scores[1]),
                     "base_score": base_score,
                     "decision_branch": (decision["branch"] if decision
                                         else "verification")},
                    env.trackers[agent].state.get(host),
                    env.trackers[agent].last_analysis.get(host))
                requests.append(request)
            # Diagnostic-only truth sampled after the policy fixed all actions.
            compromised = true_compromised(env)
            for request in requests:
                labels.append({"request_id": request["request_id"],
                               "red_session_at_request": request["host"] in compromised})
            _, reward, terminated, truncated, _ = env.step(actions)
            rewards.append(float(reward[0]))
            ledger.reconcile(env)
            if terminated or truncated:
                break
        rows = ledger.finalize()
        counts = {status: sum(row["outcome"]["status"] == status for row in rows)
                  for status in ("success", "failure", "censored_at_episode_end")}
        return rows, labels, {
            "seed": seed, "ticks": len(rewards), "return": sum(rewards),
            "analyse_requests": len(rows), "outcomes": counts,
            "new_confirmations": sum(row["outcome"]["new_confirmation"] is True
                                     for row in rows),
        }
    finally:
        env.close()


def run(seeds, plan_path, scorer_path, out):
    config = check_frozen_inputs(seeds, plan_path, scorer_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorer = MLScorer(scorer_path)
    manifest = {
        "schema": SCHEMA,
        "purpose": "selected-action inspection-yield feasibility; not policy value",
        "policy": f"HarnessPolicy(max_age={float(config['max_age'])}), no learning hook",
        "seeds": list(seeds), "feature_names": list(NAMES),
        "history_names": list(HISTORY_NAMES), "feature_version": VERSION,
        "plan_sha256": sha256(plan_path), "scorer_sha256": sha256(scorer_path),
        "env_kw": config["env_kw"], "steps": int(config["steps"]),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "truth_scope": "separate diagnostic file; never policy or model input",
        "selection_limit": "only chosen Analyse actions have observed outcomes",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for seed in seeds:
        rows, truth, summary = collect_episode(seed, config, scorer)
        stem = f"episode_{seed}"
        for filename, data in ((f"{stem}_requests.jsonl", rows),
                               (f"{stem}_truth_audit.jsonl", truth)):
            with (out / filename).open("w") as f:
                for row in data:
                    f.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            summary[filename] = sha256(out / filename)
        results.append(summary)
        (out / "summary.json").write_text(json.dumps(
            {"manifest": manifest, "episodes": results}, indent=2) + "\n")
        print(f"seed={seed} return={summary['return']:+.0f} "
              f"requests={summary['analyse_requests']} "
              f"confirmed={summary['new_confirmations']}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--scorer", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.seeds, a.plan, a.scorer, a.out)


if __name__ == "__main__":
    main()
