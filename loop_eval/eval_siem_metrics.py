"""SIEM Audit Trail and Security Metrics Evaluator — Eval role only.

Runs one or more Blue policy episodes and produces:
  - A structured NDJSON audit trail (one event per line, SIEM-ingestible).
  - A metrics summary: MTTD, MTTC, response time, containment, service impact.

This module uses ONLY Blue-visible observations and action traces.
Privileged true-state is accessed ONLY in the evaluator's count_compromised
call (clearly labelled, never fed to policy). Metrics reflect what an
external SIEM would observe from Blue action logs and telemetry.

Usage:
    python eval_siem_metrics.py --policy round-robin --seeds 10 20 30 \
        --steps 200 --output results/siem_run_001
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Blue folder is sibling of loop_eval
_ROOT = Path(__file__).resolve().parents[1]
_BLUE = _ROOT / "blue"
if str(_BLUE) not in sys.path:
    sys.path.insert(0, str(_BLUE))

from blue_telemetry import TELEMETRY_VERSION

# Simulator imports are deferred so unit tests of schema/risk logic
# can import this module without requiring CybORG/pygame.
_sim_loaded = False
wrapper = None
MaskedRandomBaseline = RoundRobinBaseline = SleepBaseline = None
StalestFirstBaseline = SuspicionSweepBaseline = None
count_compromised = decode_index = None


def _import_sim():
    global wrapper, MaskedRandomBaseline, RoundRobinBaseline, SleepBaseline
    global StalestFirstBaseline, SuspicionSweepBaseline, count_compromised
    global decode_index, _sim_loaded
    if _sim_loaded:
        return
    import cc4_epymarl_wrapper as _w
    from blue_baselines import (
        MaskedRandomBaseline as _MR,
        RoundRobinBaseline as _RR,
        SleepBaseline as _SL,
        StalestFirstBaseline as _SF,
        SuspicionSweepBaseline as _SS,
        count_compromised as _cc,
        decode_index as _di,
    )
    wrapper = _w
    MaskedRandomBaseline = _MR
    RoundRobinBaseline = _RR
    SleepBaseline = _SL
    StalestFirstBaseline = _SF
    SuspicionSweepBaseline = _SS
    count_compromised = _cc
    decode_index = _di
    _sim_loaded = True


EVAL_VERSION = "siem-eval-v1"

# Policy registry built lazily so class references resolve after _import_sim()
def _get_policies():
    _import_sim()
    return {
        "sleep": SleepBaseline,
        "masked-random": MaskedRandomBaseline,
        "round-robin": RoundRobinBaseline,
        "stalest-first": StalestFirstBaseline,
        "suspicion-sweep": SuspicionSweepBaseline,
    }

# Keep POLICIES as a module attr for CLI/tests that check membership without sim
POLICIES = frozenset(["sleep", "masked-random", "round-robin",
                      "stalest-first", "suspicion-sweep"])


# ---------------------------------------------------------------------------
# Audit event schema
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """One structured SIEM audit event. All fields are Blue-visible or derived
    from Blue action logs; no privileged Red state leaks here."""

    episode_id: str
    seed: int
    tick: int
    agent: str
    action: str
    host: str | None
    reward: float
    # Derived SIEM fields (Blue-visible tracker state)
    host_belief: str | None   # CONFIRMED / VERIFY / CLEAN / None
    alert: bool               # True when this action is a remediation response
    alert_type: str | None    # "Remove" / "Restore" / "Analyse-CONFIRMED"
    correlation_id: str | None  # links Analyse alert -> Remove/Restore for same host

    def to_ndjson(self) -> str:
        d = asdict(self)
        d["event_type"] = "blue_action"
        d["telemetry_version"] = TELEMETRY_VERSION
        d["eval_version"] = EVAL_VERSION
        return json.dumps(d, allow_nan=False)


# ---------------------------------------------------------------------------
# Episode runner with audit trail
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    seed: int
    policy: str
    steps_run: int
    cumulative_return: float

    # SIEM KPIs — all in simulator ticks (one tick = one joint step)
    first_alert_tick: int | None       # first tick Blue issued a remediation
    mttd_ticks: float | None           # mean ticks from host first seen CONFIRMED → first Remove/Restore
    mttc_ticks: float | None           # mean ticks from first Remove/Restore → host state CLEAN (verify pass)
    mean_response_ticks: float | None  # mean ticks Analyse → Remove/Restore on same host
    total_remediations: int
    total_restores: int
    total_analyses: int
    total_alerts: int                  # any remediation action
    # Service impact (proxy)
    compromised_at_200: int | None     # privileged eval label at step 200
    compromised_final: int             # privileged eval label at episode end
    root_compromised_final: int
    service_availability_pct: float    # (steps_run - ticks_any_host_compromised) / steps_run * 100
    audit_events: list[dict] = field(default_factory=list)


def _run_siem_episode(
    policy_factory,
    policy_name: str,
    seed: int,
    steps: int,
    red_agent: str = "discovery",
) -> EpisodeMetrics:
    """Run one episode and collect full SIEM audit trail + metrics."""
    env = wrapper.CC4MARLEnv(seed=seed, steps=steps, red_agent=red_agent)
    env.reset(seed=seed)
    policy = policy_factory()
    policy.reset()

    episode_id = f"{policy_name}-seed{seed}"

    # Per-host tracking for KPIs (Blue-visible only)
    host_first_confirmed: dict[str, int] = {}   # tick when CONFIRMED first seen
    host_first_remediated: dict[str, int] = {}  # tick of first Remove/Restore
    host_verify_pass: dict[str, int] = {}       # tick when Analyse returned after Remove (proxy MTTC)
    pending_correlation: dict[str, str] = {}    # host → correlation_id (Analyse alert pending removal)
    correlation_counter = 0

    audit_events: list[dict] = []

    # Compromised-tick accumulator (privileged, eval-only)
    ticks_any_compromised = 0
    compromised_at_200: int | None = None

    cumulative_return = 0.0
    total_remediations = total_restores = total_analyses = total_alerts = 0
    first_alert_tick: int | None = None
    tick = 0

    try:
        for tick in range(1, steps + 1):
            actions = {agent: int(policy.select(env, agent))
                       for agent in wrapper.BLUE_AGENTS}
            _, rewards, terminated, truncated, _ = env.step(actions)
            reward_val = float(rewards[0])
            cumulative_return += reward_val

            # --- Privileged eval (count only, never fed to policy) ---
            total_comp, root_comp = count_compromised(env)
            if total_comp > 0:
                ticks_any_compromised += 1
            if tick == 200:
                compromised_at_200 = total_comp

            # --- Per-agent audit ---
            for agent in wrapper.BLUE_AGENTS:
                act_idx = actions[agent]
                action_name, host = decode_index(env, agent, act_idx)
                tracker = env.trackers.get(agent)
                host_belief = None
                if tracker and host:
                    host_belief = tracker.state.get(host)

                # Detect CONFIRMED for MTTD
                if host and host_belief == "CONFIRMED":
                    if host not in host_first_confirmed:
                        host_first_confirmed[host] = tick

                # Classify alert
                is_alert = action_name in ("Remove", "Restore") and host is not None
                is_analyse_confirmed = (
                    action_name == "Analyse" and host is not None
                    and host_belief == "CONFIRMED"
                )
                alert_type: str | None = None
                corr_id: str | None = None

                if is_alert:
                    total_alerts += 1
                    alert_type = action_name
                    if first_alert_tick is None:
                        first_alert_tick = tick
                    if host not in host_first_remediated:
                        host_first_remediated[host] = tick
                    if action_name == "Remove":
                        total_remediations += 1
                    else:
                        total_restores += 1
                    # pop pending correlation
                    corr_id = pending_correlation.pop(host, None)
                    # After remediation issue a VERIFY probe → mark for MTTC
                    if host not in host_verify_pass:
                        host_verify_pass[host] = tick  # placeholder; refined below

                elif is_analyse_confirmed:
                    total_alerts += 1
                    alert_type = "Analyse-CONFIRMED"
                    correlation_counter += 1
                    corr_id = f"corr-{episode_id}-{correlation_counter}"
                    pending_correlation[host] = corr_id

                if action_name == "Analyse":
                    total_analyses += 1
                    # Refine MTTC: if this Analyse follows a Remove/Restore on same host
                    # and host is no longer CONFIRMED → remediation confirmed clean
                    if (host and host in host_first_remediated
                            and host_belief != "CONFIRMED"
                            and host not in host_verify_pass):
                        host_verify_pass[host] = tick

                evt = AuditEvent(
                    episode_id=episode_id,
                    seed=seed,
                    tick=tick,
                    agent=agent,
                    action=action_name,
                    host=host,
                    reward=reward_val,
                    host_belief=host_belief,
                    alert=is_alert or is_analyse_confirmed,
                    alert_type=alert_type,
                    correlation_id=corr_id,
                )
                audit_events.append(json.loads(evt.to_ndjson()))

            if terminated or truncated:
                break
    finally:
        env.close()

    # --- Compute KPIs ---
    # MTTD: mean(first_remediated[h] - first_confirmed[h]) for remediations
    mttd_samples = [
        host_first_remediated[h] - host_first_confirmed[h]
        for h in host_first_remediated
        if h in host_first_confirmed and host_first_remediated[h] >= host_first_confirmed[h]
    ]
    mttd = float(sum(mttd_samples) / len(mttd_samples)) if mttd_samples else None

    # MTTC: mean(verify_pass[h] - first_remediated[h])
    mttc_samples = [
        host_verify_pass[h] - host_first_remediated[h]
        for h in host_verify_pass
        if h in host_first_remediated and host_verify_pass[h] > host_first_remediated[h]
    ]
    mttc = float(sum(mttc_samples) / len(mttc_samples)) if mttc_samples else None

    # Response time: MTTD (Analyse-CONFIRMED → Remove/Restore) — same as MTTD here
    mean_response = mttd  # defined above as confirm-to-remediate gap

    # Service availability: ticks with no compromised host / total ticks
    svc_avail = (tick - ticks_any_compromised) / tick * 100.0 if tick > 0 else 100.0

    total_comp_final, root_comp_final = count_compromised(env) if tick > 0 else (0, 0)

    return EpisodeMetrics(
        seed=seed,
        policy=policy_name,
        steps_run=tick,
        cumulative_return=cumulative_return,
        first_alert_tick=first_alert_tick,
        mttd_ticks=mttd,
        mttc_ticks=mttc,
        mean_response_ticks=mean_response,
        total_remediations=total_remediations,
        total_restores=total_restores,
        total_analyses=total_analyses,
        total_alerts=total_alerts,
        compromised_at_200=compromised_at_200,
        compromised_final=total_comp_final,
        root_compromised_final=root_comp_final,
        service_availability_pct=svc_avail,
        audit_events=audit_events,
    )


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_siem_evaluation(
    policy_name: str,
    seeds: list[int],
    steps: int = 200,
    output: Path | None = None,
    red_agent: str = "discovery",
) -> dict:
    """Run SIEM evaluation across seeds; write audit trail + summary."""
    _import_sim()
    policies = _get_policies()
    if policy_name not in policies:
        raise ValueError(f"unknown policy {policy_name!r}; known: {sorted(policies)}")
    factory_cls = policies[policy_name]

    def make_policy():
        if policy_name == "masked-random":
            return factory_cls(seed=0)  # deterministic per seed via env RNG
        return factory_cls()

    results = []
    all_events = []
    for seed in seeds:
        print(f"  Running seed {seed}...", flush=True)
        ep = _run_siem_episode(make_policy, policy_name, seed, steps, red_agent)
        results.append(ep)
        all_events.extend(ep.audit_events)

    # Aggregate
    def _mean(vals):
        v = [x for x in vals if x is not None and math.isfinite(x)]
        return float(sum(v) / len(v)) if v else None

    summary = {
        "eval_version": EVAL_VERSION,
        "policy": policy_name,
        "red_agent": red_agent,
        "seeds": seeds,
        "steps": steps,
        "n_episodes": len(results),
        "mean_cumulative_return": _mean([r.cumulative_return for r in results]),
        "mean_mttd_ticks": _mean([r.mttd_ticks for r in results]),
        "mean_mttc_ticks": _mean([r.mttc_ticks for r in results]),
        "mean_response_time_ticks": _mean([r.mean_response_ticks for r in results]),
        "mean_total_alerts": _mean([r.total_alerts for r in results]),
        "mean_total_remediations": _mean([r.total_remediations for r in results]),
        "mean_total_restores": _mean([r.total_restores for r in results]),
        "mean_total_analyses": _mean([r.total_analyses for r in results]),
        "mean_compromised_final": _mean([r.compromised_final for r in results]),
        "mean_root_compromised_final": _mean([r.root_compromised_final for r in results]),
        "mean_service_availability_pct": _mean([r.service_availability_pct for r in results]),
        "per_seed": [
            {
                "seed": r.seed,
                "cumulative_return": r.cumulative_return,
                "steps_run": r.steps_run,
                "mttd_ticks": r.mttd_ticks,
                "mttc_ticks": r.mttc_ticks,
                "mean_response_ticks": r.mean_response_ticks,
                "first_alert_tick": r.first_alert_tick,
                "total_alerts": r.total_alerts,
                "total_remediations": r.total_remediations,
                "total_restores": r.total_restores,
                "total_analyses": r.total_analyses,
                "compromised_at_200": r.compromised_at_200,
                "compromised_final": r.compromised_final,
                "root_compromised_final": r.root_compromised_final,
                "service_availability_pct": r.service_availability_pct,
            }
            for r in results
        ],
    }

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        # Write audit trail NDJSON
        audit_path = output / "audit_trail.ndjson"
        with audit_path.open("w", encoding="utf-8") as f:
            for ev in all_events:
                f.write(json.dumps(ev, allow_nan=False) + "\n")

        # Write summary JSON
        summary_path = output / "siem_summary.json"
        summary_with_meta = dict(summary)
        summary_with_meta["audit_trail_events"] = len(all_events)
        summary_with_meta["audit_trail_sha256"] = hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest()
        try:
            summary_with_meta["source_commit"] = subprocess.check_output(
                ["git", "-C", str(_ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            summary_with_meta["source_commit"] = "unknown"
        summary_with_meta["python"] = sys.version
        summary_with_meta["dependencies"] = {
            name: importlib.metadata.version(name)
            for name in ("numpy",)
            if importlib.util.find_spec(name)
        }
        summary_path.write_text(
            json.dumps(summary_with_meta, indent=2), encoding="utf-8"
        )

        print(f"\nAudit trail: {audit_path}  ({len(all_events)} events)")
        print(f"Summary:     {summary_path}")

    return summary


def _print_summary_table(summary: dict) -> None:
    """Pretty-print the SIEM KPI table to stdout."""
    p = summary["policy"]
    r = summary["red_agent"]
    n = summary["n_episodes"]
    print(f"\n{'='*60}")
    print(f"SIEM Metrics — Policy: {p}  |  Red: {r}  |  Seeds: {n}")
    print(f"{'='*60}")
    rows = [
        ("Mean cumulative return",        summary["mean_cumulative_return"],          ""),
        ("Mean MTTD (ticks)",             summary["mean_mttd_ticks"],                 "↓ better"),
        ("Mean MTTC (ticks)",             summary["mean_mttc_ticks"],                 "↓ better"),
        ("Mean response time (ticks)",    summary["mean_response_time_ticks"],        "↓ better"),
        ("Mean alerts issued",            summary["mean_total_alerts"],               ""),
        ("Mean remediations (Remove)",    summary["mean_total_remediations"],         ""),
        ("Mean restores",                 summary["mean_total_restores"],             ""),
        ("Mean analyses",                 summary["mean_total_analyses"],             ""),
        ("Mean compromised hosts (final)",summary["mean_compromised_final"],          "↓ better"),
        ("Mean root-compromised (final)", summary["mean_root_compromised_final"],     "↓ better"),
        ("Mean service availability",     summary["mean_service_availability_pct"],   "% ↑ better"),
    ]
    for label, value, unit in rows:
        if value is None:
            val_str = "N/A"
        elif isinstance(value, float):
            val_str = f"{value:.2f}"
        else:
            val_str = str(value)
        print(f"  {label:<38} {val_str:>10}  {unit}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _import_sim()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=sorted(POLICIES),
        default="round-robin",
        help="Blue policy to evaluate",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Episode seeds (must be distinct)",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Episode length in simulator ticks",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Directory to write audit_trail.ndjson and siem_summary.json",
    )
    parser.add_argument(
        "--red-agent", choices=sorted(wrapper.RED_AGENTS), default="discovery",
        help="Red agent variant",
    )
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")

    print(f"Running SIEM evaluation: policy={args.policy} seeds={args.seeds} "
          f"steps={args.steps} red={args.red_agent}")
    summary = run_siem_evaluation(
        policy_name=args.policy,
        seeds=args.seeds,
        steps=args.steps,
        output=args.output,
        red_agent=args.red_agent,
    )
    _print_summary_table(summary)

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")

    print(f"Running SIEM evaluation: policy={args.policy} seeds={args.seeds} "
          f"steps={args.steps} red={args.red_agent}")
    summary = run_siem_evaluation(
        policy_name=args.policy,
        seeds=args.seeds,
        steps=args.steps,
        output=args.output,
        red_agent=args.red_agent,
    )
    _print_summary_table(summary)
