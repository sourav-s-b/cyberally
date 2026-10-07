"""Collect simulated Blue-visible snapshots for future anomaly experiments.

No truth labels, detector fitting, or action-execution claims. Native Red remains
active; this data must NOT be described as a verified benign training dataset.
"""
import argparse
import hashlib
import json
import importlib.metadata
from pathlib import Path
import subprocess
import sys

from blue.core.baselines import SleepBaseline, MaskedRandomBaseline, RoundRobinBaseline
from blue.core.telemetry import TELEMETRY_VERSION, record_from_dict, records_to_vector
from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv, WRAPPER_VERSION

ROOT = Path(__file__).resolve().parents[2]
FEATURE_VERSION = "host-vector-10-v1"


def collect(output, seeds, steps=30, policy_name="round-robin"):
    """Write a new dataset directory; refuse to overwrite prior artifacts."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide nonempty distinct seeds")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    policies = {"sleep": SleepBaseline, "masked-random": MaskedRandomBaseline,
                "round-robin": RoundRobinBaseline}
    policy_factory = policies[policy_name]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = CC4MARLEnv(steps=steps)
    row_count = 0
    try:
        with (output / "telemetry.jsonl").open("x", encoding="utf-8") as stream:
            for seed in seeds:
                policy = (policy_factory(seed=seed) if policy_name == "masked-random"
                          else policy_factory())
                policy.reset()
                env.reset(seed=seed)
                done = False
                while True:
                    for agent_id, agent in enumerate(BLUE_AGENTS):
                        block = env.get_telemetry(agent_id)
                        by_host = {host: [] for host in env.hostnames[agent]}
                        for row in block["telemetry"]:
                            record = record_from_dict(row)
                            by_host[record.host].append(record)
                        record = {
                            "schema_version": "blue-telemetry-collection-v1-draft",
                            "episode_id": f"seed-{seed}", "seed": seed,
                            "scenario_id": "cage4-default", "policy_version": policy_name,
                            "step": env._tick, "agent_id": agent,
                            "feature_version": FEATURE_VERSION,
                            "features_by_host": {host: records_to_vector(rows, agent)
                                                 for host, rows in by_host.items()}, **block}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        row_count += 1
                    if done:
                        break
                    actions = [policy.select(env, a) for a in BLUE_AGENTS]
                    _, _, terminated, truncated, _ = env.step(actions)
                    done = terminated or truncated
    finally:
        env.close()
    sources = [Path(__file__), ROOT / "blue/core/telemetry.py",
               ROOT / "blue/core/obs_features.py",
               ROOT / "blue/core/wrapper.py",
               ROOT / "blue/core/baselines.py",
               ROOT / "blue/core/masking.py"]
    manifest = {
        "schema_version": "blue-telemetry-collection-v1-draft",
        "telemetry_version": TELEMETRY_VERSION, "feature_version": FEATURE_VERSION,
        "action_version": WRAPPER_VERSION, "origin": "simulated",
        "time_unit": "sim_tick", "seeds": list(seeds), "steps": steps,
        "policy": policy_name, "red_policy": "DiscoveryFSRed", "records": row_count,
        "python": sys.version,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "gym", "networkx")},
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "telemetry_sha256": hashlib.sha256((output / "telemetry.jsonl").read_bytes()).hexdigest(),
        "limitations": ["Latest observed rows may repeat; not new event counts",
                        "Native Red active: no privileged clean/compromised labels",
                        "Draft schema awaiting Environment/Evaluation review"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--policy", choices=("sleep", "masked-random", "round-robin"),
                        default="round-robin")
    args = parser.parse_args()
    result = collect(args.output, args.seeds, args.steps, args.policy)
    print(f"Collected {result['records']} agent snapshots into {args.output}")
