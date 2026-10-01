"""Run the native scripted attacker inside CAGE4 and save diagnostic artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from red.strategy_adapter import StrategyConfig, create_simulator, strategy_manifest


def run_smoke(output, config, *, seeds, steps=40):
    manifest = strategy_manifest(config, seeds=seeds, steps=steps)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from CybORG.Simulator.Actions import InvalidAction, Sleep
    blue_agents = [f"blue_agent_{i}" for i in range(5)]
    episodes = []
    with (output / "red-actions.jsonl").open("x", encoding="utf-8") as stream:
        for seed in seeds:
            simulator = create_simulator(config, seed=seed, steps=steps)
            simulator.reset(seed=seed)
            tick = 0
            native_return = 0.0
            invalid_actions = 0
            while not simulator.environment_controller.done:
                simulator.parallel_step({agent: Sleep() for agent in blue_agents})
                tick += 1
                # Native team reward is duplicated per Blue agent: count once.
                native_return += float(sum(
                    simulator.environment_controller.get_reward(blue_agents[0]).values()))
                for agent in (f"red_agent_{i}" for i in range(6)):
                    actions = simulator.get_last_action(agent) or []
                    invalid_actions += sum(isinstance(a, InvalidAction) for a in actions)
                    executed = [
                        {"name": type(action).__name__,
                         "target": {key: str(getattr(action, key)) for key in
                                    ("hostname", "ip_address", "subnet")
                                    if hasattr(action, key)}} for action in actions]
                    stream.write(json.dumps({
                        "schema_version": "red-smoke-actions-v1-draft",
                        "strategy_id": config.strategy_id,
                        "episode_id": f"seed-{seed}", "seed": seed,
                        "step": tick, "time_unit": "sim_tick", "origin": "simulated",
                        "agent_id": agent, "controller_executed_actions": executed,
                        "terminated": simulator.environment_controller.done}) + "\n")
            episodes.append({"seed": seed, "joint_steps": tick,
                             "native_blue_team_return": native_return,
                             "invalid_executed_red_actions": invalid_actions})
    manifest.update({
        "episodes": episodes, "python": sys.version,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "gym", "networkx")},
        "diagnostic_sha256": hashlib.sha256((output / "red-actions.jsonl").read_bytes()).hexdigest(),
        "smoke_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostic_semantics": "Controller executed actions; no request/pending inference or success labels"})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "configs/discovery-fs-red-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    config = StrategyConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    result = run_smoke(args.output, config, seeds=args.seeds, steps=args.steps)
    print(json.dumps(result["episodes"], indent=2))


if __name__ == "__main__":
    main()
