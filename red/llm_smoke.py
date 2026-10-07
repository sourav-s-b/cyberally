"""Run a bounded local Ollama or explicitly labelled mock CAGE4 smoke episode."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cage-challenge-4"))

from red.llm_agent import (LLMConfig, OllamaClient, configured_agent_class,
                           PROMPT_VERSION, SYSTEM_PROMPT)


class MockClient:
    """Deterministic plumbing fixture, not an LLM or evaluated attack policy."""
    def __init__(self, config):
        self.config, self.requests = config, 0
        self.usage = {}

    def complete(self, messages):
        if self.requests >= self.config.max_requests:
            raise ValueError("request_budget_exhausted")
        self.requests += 1
        available = json.loads(messages[-1]["content"])["available_actions"]
        name = "DiscoverRemoteSystems" if "DiscoverRemoteSystems" in available else "Sleep"
        return json.dumps({"action": name, "parameters": {
            key: spec["values"][0] for key, spec in available[name].items() if spec["required"]}})


def create_llm_simulator(config, client, *, seed, steps):
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps < 2:
        raise ValueError("explicit nonnegative seed and horizon >= 2 required")
    generator = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=configured_agent_class(config, client), steps=steps)
    return CybORG(scenario_generator=generator, seed=seed)


def run_smoke(output, config, *, seeds, steps=20, mock=False):
    from CybORG.Simulator.Actions import Sleep, InvalidAction
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty distinct seeds required")
    for seed in seeds:
        if type(seed) is not int or seed < 0:
            raise ValueError("nonnegative integer seeds required")
    if type(steps) is not int or steps < 2:
        raise ValueError("horizon >= 2 required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    client = MockClient(config) if mock else OllamaClient(config)
    # Fail clearly before starting a run if the chosen local model is unavailable.
    model_info = {"model": "mock", "digest": None} if mock else client.check_model()
    output.mkdir(parents=True, exist_ok=False)
    episodes, reasons = [], Counter()
    with (output / "red-decisions.jsonl").open("x", encoding="utf-8") as stream:
        for seed in seeds:
            simulator = create_llm_simulator(config, client, seed=seed, steps=steps)
            simulator.reset(seed=seed)
            controller = simulator.environment_controller
            native_return, tick, invalid = 0.0, 0, 0
            seen = {}
            while not controller.done:
                simulator.parallel_step({f"blue_agent_{i}": Sleep() for i in range(5)})
                tick += 1
                native_return += float(sum(controller.get_reward("blue_agent_0").values()))
                for i in range(6):
                    name = f"red_agent_{i}"
                    agent = controller.agent_interfaces[name].agent
                    actions = simulator.get_last_action(name) or []
                    invalid += sum(isinstance(a, InvalidAction) for a in actions)
                    decision = agent.last_decision
                    new = decision is not None and seen.get(name) != decision["decision"]
                    if new:
                        seen[name] = decision["decision"]
                        reasons[decision["fallback_reason"] or "accepted"] += 1
                    stream.write(json.dumps({
                        "schema_version": "red-llm-smoke-v1-draft", "origin": "simulated",
                        "mode": "mock" if mock else "ollama", "seed": seed,
                        "step": tick, "time_unit": "sim_tick", "agent_id": name,
                        "new_decision": decision if new else None,
                        "controller_executed_actions": [type(a).__name__ for a in actions],
                        "terminated": controller.done}) + "\n")
            episodes.append({"seed": seed, "joint_steps": tick,
                             "native_blue_team_return": native_return,
                             "invalid_executed_red_actions": invalid})
    source_paths = [Path(__file__), ROOT / "red/llm_agent.py",
                    ROOT / "cage-challenge-4/CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py",
                    ROOT / "cage-challenge-4/CybORG/Simulator/SimulationController.py"]
    manifest = {
        "schema_version": "red-llm-smoke-v1-draft", "mode": "mock" if mock else "ollama",
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                  text=True).strip(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        "config": asdict(config), "model": model_info, "python": sys.version,
        "config_sha256": hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "gym", "networkx")},
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "seeds": list(seeds), "steps": steps, "episodes": episodes,
        "request_attempts": client.requests, "token_usage": client.usage,
        "decision_counts": dict(reasons),
        "diagnostic_sha256": hashlib.sha256((output / "red-decisions.jsonl").read_bytes()).hexdigest(),
        "limitations": ["Development smoke, not attacker effectiveness evaluation",
                        "Sleep Blue and native Green; full CAGE4 topology",
                        "Selection, controller execution and success are distinct",
                        "No privileged state or Blue integration; fallback is Sleep",
                        "Mock mode tests plumbing only; no model inference",
                        "Temperature zero and seed do not guarantee model reproducibility"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8123])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-requests", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    if not args.mock and not args.model:
        parser.error("--model is required for Ollama; --mock is a test fixture")
    config = LLMConfig(model=args.model or "mock", max_requests=args.max_requests,
                       timeout_seconds=args.timeout)
    result = run_smoke(args.output, config, seeds=args.seeds, steps=args.steps, mock=args.mock)
    print(json.dumps({"mode": result["mode"], "episodes": result["episodes"],
                      "decision_counts": result["decision_counts"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
