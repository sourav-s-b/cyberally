"""Main Red runner: LLM planner by default, with paired development comparisons.

Privileged active-host labels below are evaluation-only. They never enter agents.
Sleep Blue is fixed; these are development results, not a final benchmark.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cage-challenge-4"))
from red.llm_agent import LLMConfig, OllamaClient
from red.planned_agent import (configured_planner_class, PlannerConfig,
                               PROMPT_VERSION, SYSTEM_PROMPT)


class FixturePlanner:
    """Explicitly labelled fixture/static ablation; no model inference."""
    def __init__(self, adaptive=False):
        self.adaptive, self.requests, self.usage = adaptive, 0, {}

    def complete(self, messages, schema=None):
        self.requests += 1
        hosts = json.loads(messages[-1]["content"])["known_hosts"]
        states = {h["state"] for h in hosts.values()}
        goal = "spread"
        if self.adaptive:
            if states & {"R", "RD"}:
                goal = "disrupt"
            elif states & {"U", "UD"}:
                goal = "escalate"
        return json.dumps({"goal": goal, "focus_ip": None})


def create_simulator(policy_class, seed, steps):
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    return CybORG(scenario_generator=EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=policy_class, steps=steps), seed=seed)


def evaluation_host_counts(controller):
    """Draft diagnostic: unique hosts with active Red sessions, including footholds."""
    compromised, privileged = set(), set()
    for agent, sessions in controller.state.sessions.items():
        if agent.startswith("red_agent_"):
            for session in sessions.values():
                if session.active:
                    compromised.add(session.hostname)
                    if session.has_privileged_access():
                        privileged.add(session.hostname)
    return len(compromised), len(privileged)


def run_episode(policy_class, seed, steps, stream, policy, blue_policy="sleep"):
    from CybORG.Simulator.Actions import Sleep, InvalidAction
    blue_env = None
    if blue_policy == "lancer":
        from blue.core.wrapper import CC4MARLEnv
        from blue.policies.registry import build
        blue_env = CC4MARLEnv(seed=seed, steps=steps, red_agent=policy,
                             red_agent_class=policy_class)
        blue_env.reset(seed=seed)
        defender = build("hybrid_lancer_v2")
        defender.reset()
        simulator, controller = blue_env.cyborg, blue_env.env
    elif blue_policy == "sleep":
        simulator = create_simulator(policy_class, seed, steps)
        simulator.reset(seed=seed)
        controller = simulator.environment_controller
    else:
        raise ValueError("blue_policy must be sleep or lancer")
    started = time.monotonic()
    reward, tick, invalid, host_ticks, root_ticks, peak_hosts, peak_root = 0., 0, 0, 0, 0, 0, 0
    plans, actions = Counter(), Counter()
    seen = {}
    while not controller.done:
        if blue_env is None:
            simulator.parallel_step({f"blue_agent_{i}": Sleep() for i in range(5)})
        else:
            blue_env.step({f"blue_agent_{i}": defender.select(blue_env, f"blue_agent_{i}")
                           for i in range(5)})
        tick += 1
        reward += float(sum(controller.get_reward("blue_agent_0").values()))
        hosts, roots = evaluation_host_counts(controller)
        host_ticks += hosts
        root_ticks += roots
        peak_hosts, peak_root = max(peak_hosts, hosts), max(peak_root, roots)
        for i in range(6):
            name = f"red_agent_{i}"
            executed = simulator.get_last_action(name) or []
            invalid += sum(isinstance(a, InvalidAction) for a in executed)
            actions.update(type(a).__name__ for a in executed)
            decision = getattr(controller.agent_interfaces[name].agent, "last_decision", None)
            if decision and seen.get(name) != decision["decision"]:
                seen[name] = decision["decision"]
                plans[decision["plan_status"]] += 1
            else:
                decision = None
            stream.write(json.dumps({"policy": policy, "seed": seed, "tick": tick,
                "agent": name, "decision": decision,
                "executed_actions": [type(a).__name__ for a in executed]}) + "\n")
    return {"seed": seed, "joint_steps": tick, "native_blue_team_return": reward,
            "active_compromised_host_ticks": host_ticks, "active_privileged_host_ticks": root_ticks,
            "peak_compromised_hosts": peak_hosts, "peak_privileged_hosts": peak_root,
            "final_compromised_hosts": hosts, "final_privileged_hosts": roots,
            "invalid_executed_actions": invalid, "wall_seconds": time.monotonic() - started,
            "plan_status_counts": dict(plans), "executed_action_counts": dict(actions)}


def run(output, config, planner_config, seeds, steps, *, mock=False, compare=False,
        blue_policy="sleep"):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("unique nonnegative seeds required")
    if type(steps) is not int or steps < 3:
        raise ValueError("steps must be >= 3")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    client = FixturePlanner(adaptive=True) if mock else OllamaClient(config)
    model = {"model": "mock", "digest": None} if mock else client.check_model()
    policies = {"llm-planner" if not mock else "mock-planner": configured_planner_class(
        config, client, planner_config)}
    if compare:
        from CybORG.Agents import DiscoveryFSRed
        policies["scripted"] = DiscoveryFSRed
        policies["tactical-only"] = configured_planner_class(config, FixturePlanner(), planner_config)
    output.mkdir(parents=True, exist_ok=False)
    episodes = {}
    with (output / "decisions.jsonl").open("x", encoding="utf-8") as stream:
        for policy, cls in policies.items():
            episodes[policy] = []
            for seed in seeds:
                result = run_episode(cls, seed, steps, stream, policy, blue_policy)
                episodes[policy].append(result)
                print(json.dumps({"policy": policy, **result}), flush=True)
    primary = next(iter(episodes))
    paired = []
    if compare:
        for main, baseline, tactical in zip(episodes[primary], episodes["scripted"], episodes["tactical-only"]):
            paired.append({"seed": main["seed"],
                "blue_return_delta_vs_scripted": main["native_blue_team_return"] - baseline["native_blue_team_return"],
                "blue_return_delta_vs_tactical": main["native_blue_team_return"] - tactical["native_blue_team_return"],
                "compromised_host_ticks_delta_vs_scripted": main["active_compromised_host_ticks"] - baseline["active_compromised_host_ticks"],
                "privileged_host_ticks_delta_vs_scripted": main["active_privileged_host_ticks"] - baseline["active_privileged_host_ticks"]})
    paths = [Path(__file__), ROOT / "red/planned_agent.py", ROOT / "red/llm_agent.py",
             ROOT / "cage-challenge-4/CybORG/Agents/SimpleAgents/FiniteStateRedAgent.py",
             ROOT / "cage-challenge-4/CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py",
             ROOT / "cage-challenge-4/CybORG/Simulator/SimulationController.py"]
    if blue_policy == "lancer":
        paths += [ROOT / "blue/core/wrapper.py", ROOT / "blue/core/masking.py",
                  ROOT / "blue/policies/hybrid.py", ROOT / "blue/policies/registry.py"]
    report = {"schema_version": "red-planner-development-v1", "mode": "mock" if mock else "ollama",
        "main_policy": primary, "blue_policy": "SleepAgent" if blue_policy == "sleep" else
            "hybrid_lancer_v2 (local Lancer-style no-decoy variant)", "green_policy": "EnterpriseGreenAgent",
        "topology": "native-full", "seeds": seeds, "steps": steps, "config": asdict(config),
        "planner_config": asdict(planner_config), "model": model,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "prompt_version": PROMPT_VERSION, "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "trace_sha256": hashlib.sha256((output / "decisions.jsonl").read_bytes()).hexdigest(),
        "requests": client.requests, "tokens": client.usage, "episodes": episodes, "paired": paired,
        "metric_semantics": {"blue_return": "native shared reward counted once; lower favors Red",
            "host_ticks": "unique hosts with active Red sessions after each tick, includes initial footholds",
            "privileged": "active Red session with username root or SYSTEM; evaluation-only",
            "wall_seconds": "episode runtime including inference; excludes constructor/reset"},
        "limitations": ["Development seeds, not an untouched final evaluation suite",
            "One fixed defender; does not establish general attacker superiority",
            "Tactical-only ablation is fixed spread goal with the same executor",
            "Mock mode contains no LLM inference", "No general strongest-attacker claim"]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--blue", choices=["lancer", "sleep"], default="lancer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7629, 7630, 7631])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--max-requests", type=int, default=72)
    parser.add_argument("--plans-per-agent", type=int, default=4)
    parser.add_argument("--plan-interval", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    return run(args.output, LLMConfig(args.model, max_requests=args.max_requests,
        timeout_seconds=args.timeout), PlannerConfig(args.plan_interval, args.plans_per_agent),
        args.seeds, args.steps, mock=args.mock, compare=args.compare, blue_policy=args.blue)


if __name__ == "__main__":
    main()
