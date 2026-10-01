"""RED-01: replayable native DiscoveryFSRed configuration and factory.

The policy receives its native Red-local observation/action space via CybORG.
This module never reads simulator truth or supplies attacker labels to Blue.
No arbitrary import paths or executable strategy payloads are accepted.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "red-strategy-v1-draft"


@dataclass(frozen=True)
class StrategyConfig:
    schema_version: str = SCHEMA_VERSION
    strategy_id: str = "native-discovery-fs-red"
    strategy_version: str = "1"
    policy: str = "DiscoveryFSRed"

    def __post_init__(self):
        if asdict(self) != {"schema_version": SCHEMA_VERSION,
                           "strategy_id": "native-discovery-fs-red",
                           "strategy_version": "1", "policy": "DiscoveryFSRed"}:
            raise ValueError("Only the versioned native DiscoveryFSRed baseline is supported")

    @classmethod
    def from_dict(cls, data):
        # Require every versioned field, not silent defaults for external data.
        if set(data) != set(asdict(cls())):
            raise ValueError("Strategy config fields must match the versioned schema")
        return cls(**data)

    def to_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()


def red_agent_class(config):
    """Return the native class without wrapping/changing its RNG or behavior.

    EnterpriseScenarioGenerator inspects this constructor and passes its shared
    RNG and agent-local subnet list. Returning the class preserves that contract.
    """
    if not isinstance(config, StrategyConfig):
        raise TypeError("config must be a validated StrategyConfig")
    from CybORG.Agents import DiscoveryFSRed
    return DiscoveryFSRed


def create_simulator(config, *, seed, steps=400):
    """Red-owned draft smoke factory; fresh full-topology simulator per episode.

    Blue is native SleepAgent and Green is native EnterpriseGreenAgent. This
    is not wired into CC4MARLEnv; Environment/Blue review that seam separately.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be an integer of at least 2")
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    generator = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_agent_class(config), steps=steps)
    return CybORG(scenario_generator=generator, seed=seed)


def strategy_manifest(config, *, seeds, steps):
    """Provenance for a development smoke suite, not attack effectiveness proof."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    for seed in seeds:
        if type(seed) is not int or seed < 0:
            raise ValueError("seeds must be nonnegative integers")
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be an integer of at least 2")
    paths = [Path(__file__),
             ROOT / "cage-challenge-4/CybORG/Agents/SimpleAgents/FSMRedVariants.py",
             ROOT / "cage-challenge-4/CybORG/Agents/SimpleAgents/FiniteStateRedAgent.py",
             ROOT / "cage-challenge-4/CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py"]
    return {
        "schema_version": SCHEMA_VERSION, "strategy_id": config.strategy_id,
        "strategy_version": config.strategy_version,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
        "factory": "red.strategy_adapter:create_simulator",
        "config": config.to_dict(), "config_sha256": config.digest(),
        "seed_policy": "explicit simulator seed; preserve shared native RNG",
        "seeds": list(seeds), "steps": steps,
        "scenario_constraints": {"scenario": "native CAGE4 enterprise",
                                 "blue_policy": "SleepAgent",
                                 "green_policy": "EnterpriseGreenAgent",
                                 "topology": "unchanged full topology"},
        "discovered_round": 0,
        "outcome_evidence": None, "artifact_hash": None,
        "limitations": ["Draft seam; Environment/Blue/Evaluation review pending",
                        "Seeds are development smoke seeds, not training or held-out suites",
                        "No effectiveness claim; diagnostic actions are not a replayable strategy",
                        "Upstream CAGE4 behavior retained; no external execution"]}
