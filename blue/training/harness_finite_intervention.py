"""Paired one-decision interventions for finite-Red candidate diagnostics.

The miner uses existing Blue-visible candidate-audit rows to choose a frozen
set of actor/Lancer disagreements. The runner then measures both directions:

* Lancer plus one actor-selected alternative, when that host is legal;
* the actor with one Lancer-choice substitution.

Privileged traces are written by the shared evaluator for post-hoc diagnosis;
this module never reads them for selection or policy behavior.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

from blue.harness.scoring import sha256 as file_sha256
from blue.policies.ordered import LancerValues, OrderedPolicy, argmax_pick
from blue.training.harness_eval import episode, make_policy
from blue.training.harness_rl import check_seeds


def mine_targets(audit_root, count=20):
    """Select deviations from the worst source episodes using a fixed rule.

    Order: lowest episode return, then largest actor rank loss under Lancer,
    then model seed, episode seed, tick, and agent. At most one target is kept
    per model seed and source episode.
    """
    root = Path(audit_root)
    candidates = []
    for model_seed in range(6):
        model_dir = root / f"s{model_seed}"
        report_path = model_dir / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text())
        if report.get("red_agent") != "finite":
            raise ValueError(f"candidate source is not finite Red: {report_path}")
        cells = report.get("cells", {}).get("actor", {})
        for seed_text, cell in cells.items():
            seed = int(seed_text)
            trace_path = model_dir / f"actor_{seed}.jsonl"
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            for line in trace_path.read_text().splitlines():
                row = json.loads(line)
                for agent, record in row.get("agents", {}).items():
                    audit = record.get("candidate_audit") or {}
                    actor_pick = audit.get("actor_pick")
                    lancer_pick = audit.get("lancer_pick")
                    if actor_pick is None or actor_pick == lancer_pick:
                        continue
                    candidate = next((c for c in audit.get("candidates", [])
                                      if c.get("host") == actor_pick), None)
                    if candidate is None:
                        continue
                    # The audit tick is the pre-action simulator tick.
                    candidates.append({
                        "target_id": (f"m{model_seed}-e{seed}-t{audit['tick']}"
                                      f"-{agent}"),
                        "model_seed": model_seed,
                        "source_episode_seed": seed,
                        "tick": int(audit["tick"]),
                        "agent": agent,
                        "alternative_host": actor_pick,
                        "source_actor_pick": actor_pick,
                        "source_lancer_pick": lancer_pick,
                        "source_lancer_rank": int(candidate["lancer_rank"]),
                        "source_return": float(cell["return"]),
                    })
    candidates.sort(key=lambda x: (x["source_return"],
                                  -x["source_lancer_rank"],
                                  x["model_seed"], x["source_episode_seed"],
                                  x["tick"], x["agent"]))
    targets = []
    seen = set()
    for target in candidates:
        # Spread the panel across source episodes; one poor episode must not
        # contribute many near-duplicate target decisions.
        key = (target["model_seed"], target["source_episode_seed"])
        if key in seen:
            continue
        seen.add(key)
        targets.append(target)
        if len(targets) >= count:
            break
    if not targets:
        raise ValueError("no actor/Lancer disagreement targets found")
    return targets


class InterventionHook:
    """Transparent wrapper that changes at most one eligible sweep choice."""

    def __init__(self, learner, target, kind, intervene):
        if kind not in ("lancer_deviation", "actor_correction"):
            raise ValueError("unknown intervention kind")
        self.learner = learner
        self.target = target
        self.kind = kind
        self.intervene = bool(intervene)
        self.reached = False
        self.changed = False
        self.unavailable = False
        self.base_pick = None
        self.intervention_pick = None

    def reset(self):
        reset = getattr(self.learner, "reset", None)
        if callable(reset):
            reset()
        self.reached = self.changed = self.unavailable = False
        self.base_pick = self.intervention_pick = None

    def observe(self, env, agent):
        observe = getattr(self.learner, "observe", None)
        if callable(observe):
            observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        pick = (self.learner(env, agent, cands, scored)
                if self.learner is not None else argmax_pick(scored))
        if (self.reached or int(env._tick) != self.target["tick"]
                or agent != self.target["agent"]):
            return pick
        self.reached = True
        self.base_pick = pick
        if not self.intervene:
            return pick
        if self.kind == "lancer_deviation":
            replacement = self.target["alternative_host"]
            if replacement not in cands:
                self.unavailable = True
                return pick
        else:
            replacement = argmax_pick(scored)
        self.intervention_pick = replacement
        self.changed = replacement != pick
        return replacement

    def outcome(self):
        if not self.reached:
            status = "not_reached"
        elif self.unavailable:
            status = "alternative_unavailable"
        elif not self.changed:
            status = "no_change"
        else:
            status = "changed"
        return {"status": status, "reached": self.reached,
                "changed": self.changed, "base_pick": self.base_pick,
                "intervention_pick": self.intervention_pick}


def paired_prefix_check(control_trace, intervention_trace, target):
    """Require identical action history through the intervention decision."""
    left = [json.loads(line) for line in Path(control_trace).read_text().splitlines()]
    right = [json.loads(line) for line in Path(intervention_trace).read_text().splitlines()]
    # Trace rows are zero-indexed by action; their serialized tick is the
    # post-step tick, while the target records the pre-step simulator tick.
    target_row = int(target["tick"])
    changed = []
    for i, (a, b) in enumerate(zip(left, right)):
        differences = []
        for agent in a["agents"]:
            aa, bb = a["agents"][agent], b["agents"][agent]
            if (aa["action"], aa["host"]) != (bb["action"], bb["host"]):
                differences.append(agent)
        if differences:
            changed.append((i, differences))
    if not changed:
        if len(left) != len(right):
            raise ValueError("paired traces ended differently before intervention")
        return {"valid": True, "first_difference_trace_tick": None,
                "first_difference_agents": []}
    first, agents = changed[0]
    if first != target_row or agents != [target["agent"]]:
        raise ValueError("paired policies diverged before or outside the target decision")
    return {"valid": True, "first_difference_trace_tick": left[first]["tick"],
            "first_difference_agents": agents}


def _tail(values):
    ordered = sorted(float(x) for x in values)
    return {"min": ordered[0],
            "p10": ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)]}


def _coverage(episode_result):
    ages = episode_result["completed_ages"]
    total = sum(v["n"] + v["n_never_investigated"] for v in ages.values())
    seen = sum(v["n"] for v in ages.values())
    return {"investigated_hosts": seen, "total_hosts": total,
            "fraction": seen / total if total else None}


def _validate_eval_seeds(seeds):
    check_seeds(seeds)
    from blue.training.seed_audit import build_used, reserved_conflicts
    used = build_used()
    conflicts = [(s, sorted(used[s])) for s in seeds if s in used]
    reserved = [(s, reserved_conflicts(s, s)) for s in seeds
                if reserved_conflicts(s, s)]
    if conflicts:
        raise ValueError(f"evaluation seeds already used: {conflicts}")
    if reserved:
        raise ValueError(f"evaluation seeds overlap declared reservations: {reserved}")


def _new_hooked_policy(kind, target, intervene, args, model_seed):
    if kind == "lancer_deviation":
        policy = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5))
        wrapper = InterventionHook(None, target, kind, intervene)
        policy.hook = wrapper
        return policy, wrapper, "lancer"
    model_dir = Path(args.actor_root) / f"s{model_seed}"
    if not (model_dir / "actor.th").is_file():
        raise FileNotFoundError(model_dir / "actor.th")
    actor_args = argparse.Namespace(
        max_age=80.0, steps=args.steps, actor_mode="greedy",
        sample_seed_base=900000, candidate_audit_agent=None,
        scorer=args.scorer, model_dir=str(model_dir),
    )
    policy, artifact = make_policy("actor", actor_args, seed=0)
    wrapper = InterventionHook(policy.hook, target, kind, intervene)
    policy.hook = wrapper
    return policy, wrapper, artifact


def _new_base_policy(kind, args, model_seed):
    if kind == "lancer_deviation":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5)), None
    model_dir = Path(args.actor_root) / f"s{model_seed}"
    actor_args = argparse.Namespace(
        max_age=80.0, steps=args.steps, actor_mode="greedy",
        sample_seed_base=900000, candidate_audit_agent=None,
        scorer=args.scorer, model_dir=str(model_dir),
    )
    return make_policy("actor", actor_args, seed=0)


def run(args):
    seeds = list(map(int, args.seeds))
    if len(seeds) != 5:
        raise ValueError("the frozen diagnostic requires exactly five Red seeds")
    _validate_eval_seeds(seeds)
    targets = json.loads(Path(args.targets).read_text())
    if isinstance(targets, dict):
        targets = targets.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("targets file must contain a nonempty target list")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "paired one-decision finite-Red interventions",
              "red_agent": "finite", "steps": args.steps,
              "evaluation_seeds": seeds, "targets": [],
              "target_file_sha256": file_sha256(args.targets),
              "scorer_sha256": file_sha256(args.scorer),
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_files": {str(path): file_sha256(path) for path in (
                  Path(__file__), Path("blue/training/harness_eval.py"),
                  Path("blue/training/harness_rl.py"),
                  Path("blue/training/mappo_guide.py"),
                  Path("blue/harness/policy.py"),
                  Path("blue/harness/features.py"),
                  Path("blue/harness/scoring.py"),
                  Path("blue/core/baselines.py"),
                  Path("blue/policies/ordered.py"),
                  Path("blue/core/wrapper.py"))},
              "claim": "exploratory mechanism diagnostic; no held-out policy claim",
              "analysis_note": (
                  "Target-level intervals are descriptive after selecting the "
                  "worst source episodes and testing multiple deviations. "
                  "Any candidate rule requires a preregistered fresh-seed "
                  "comparison against Lancer and age-80 guard.")}
    control_cache = {}
    for target in targets:
        for kind in ("lancer_deviation", "actor_correction"):
            model_seed = int(target["model_seed"])
            per_seed = {}
            for seed in seeds:
                pair_dir = out / target["target_id"] / kind / str(seed)
                pair_dir.mkdir(parents=True, exist_ok=False)
                cache_key = (kind, model_seed if kind == "actor_correction" else None, seed)
                if cache_key not in control_cache:
                    policy, artifact = _new_base_policy(kind, args, model_seed)
                    control_path = (out / "_controls" / kind /
                                    (f"m{model_seed}" if kind == "actor_correction"
                                     else "lancer") / f"{seed}.jsonl")
                    result = episode(policy, seed, args.steps, control_path,
                                     red_agent="finite")
                    result["artifact_sha256"] = artifact
                    control_cache[cache_key] = (control_path, result)
                control_path, control_result = control_cache[cache_key]
                policy, hook, artifact = _new_hooked_policy(
                    kind, target, True, args, model_seed)
                intervention_path = pair_dir / "intervention.jsonl"
                intervention_result = episode(
                    policy, seed, args.steps, intervention_path,
                    red_agent="finite")
                intervention_result["artifact_sha256"] = artifact
                intervention_result["intervention"] = hook.outcome()
                prefix = paired_prefix_check(
                    control_path, intervention_path, target)
                per_seed[str(seed)] = {
                    "control_return": control_result["return"],
                    "intervention_return": intervention_result["return"],
                    "difference": (intervention_result["return"] -
                                   control_result["return"]),
                    "intervention_status": intervention_result["intervention"]["status"],
                    "control_coverage": _coverage(control_result),
                    "intervention_coverage": _coverage(intervention_result),
                    "prefix_check": prefix,
                }
            diffs = [row["difference"] for row in per_seed.values()]
            mean = statistics.mean(diffs)
            sd = statistics.stdev(diffs)
            # t(0.975, df=4), since this frozen diagnostic has five Red seeds.
            half = 2.7764451051977987 * sd / math.sqrt(len(diffs))
            report["targets"].append({
                "target": target, "kind": kind, "per_seed": per_seed,
                "mean_paired_difference": mean,
                "paired_95_t_interval": [mean - half, mean + half],
                "control_return_tail": _tail(
                    [r["control_return"] for r in per_seed.values()]),
                "intervention_return_tail": _tail(
                    [r["intervention_return"] for r in per_seed.values()]),
                "changed_count": sum(r["intervention_status"] == "changed"
                                     for r in per_seed.values()),
                "not_reached_count": sum(r["intervention_status"] == "not_reached"
                                          for r in per_seed.values()),
                "unavailable_count": sum(r["intervention_status"] == "alternative_unavailable"
                                          for r in per_seed.values()),
            })
            (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"targets": len(targets), "pairs": len(report["targets"]),
                      "report": str(out / "report.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mine = sub.add_parser("mine")
    mine.add_argument("--audit-root", required=True)
    mine.add_argument("--count", type=int, default=20)
    mine.add_argument("--out", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--targets", required=True)
    execute.add_argument("--seeds", nargs=5, type=int, required=True)
    execute.add_argument("--actor-root", required=True,
                         help="directory containing s0..s5 actor checkpoints")
    execute.add_argument("--scorer", required=True)
    execute.add_argument("--steps", type=int, default=400)
    execute.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "mine":
        if args.count < 1:
            raise SystemExit("--count must be positive")
        targets = mine_targets(args.audit_root, args.count)
        path = Path(args.out)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"selection_rule":
            "lowest source episode return, then greatest actor Lancer-rank loss, "
            "then stable IDs; at most one target per model/episode",
            "targets": targets}, indent=2) + "\n")
        print(json.dumps({"targets": len(targets), "out": str(path)}, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
