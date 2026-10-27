from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from red.strategy_adapter import (StrategyConfig, create_simulator, red_agent_class,
                                  strategy_manifest)


def test_versioned_config_roundtrip_and_fixed_native_class():
    from CybORG.Agents import DiscoveryFSRed
    path = Path(__file__).parents[1] / "configs/discovery-fs-red-v1.json"
    config = StrategyConfig.from_dict(json.loads(path.read_text()))
    assert StrategyConfig.from_dict(config.to_dict()) == config
    assert config.digest() == StrategyConfig().digest()
    assert red_agent_class(config) is DiscoveryFSRed
    with pytest.raises(FrozenInstanceError):
        config.policy = "other"


@pytest.mark.parametrize("change", [{"policy": "anything:eval"},
    {"schema_version": "future"}, {"strategy_version": "2"}, {"extra": "field"}])
def test_unsupported_configs_rejected(change):
    values = StrategyConfig().to_dict()
    values.update(change)
    with pytest.raises(ValueError):
        StrategyConfig.from_dict(values)


def test_missing_version_rejected():
    values = StrategyConfig().to_dict()
    del values["schema_version"]
    with pytest.raises(ValueError):
        StrategyConfig.from_dict(values)


@pytest.mark.parametrize("seed,steps", [(None, 30), (True, 30), (-1, 30),
                                       (7629, 1), (7629, 3.0)])
def test_invalid_episode_config_rejected(seed, steps):
    with pytest.raises(ValueError):
        create_simulator(StrategyConfig(), seed=seed, steps=steps)


def episode_signature(cyborg, seed):
    """Only agent-local observations/actions and native Blue rewards, no truth."""
    cyborg.reset(seed=seed)
    blue_agents = [f"blue_agent_{i}" for i in range(5)]
    initial = deepcopy({a: cyborg.get_observation(a) for a in blue_agents})
    trace = []
    from CybORG.Simulator.Actions import InvalidAction, Sleep
    while not cyborg.environment_controller.done:
        cyborg.parallel_step({a: Sleep() for a in blue_agents})
        action_rows = {}
        for agent in (f"red_agent_{i}" for i in range(6)):
            actions = cyborg.get_last_action(agent) or []
            assert all(not isinstance(action, InvalidAction) for action in actions)
            action_rows[agent] = [
                {"name": type(action).__name__,
                 "params": {key: str(getattr(action, key)) for key in
                            ("agent", "session", "hostname", "ip_address", "subnet")
                            if hasattr(action, key)}} for action in actions]
        trace.append({"actions": action_rows,
                      "blue": deepcopy({a: cyborg.get_observation(a) for a in blue_agents}),
                      "reward": cyborg.environment_controller.get_reward(blue_agents[0])})
    return json.dumps({"initial": initial, "trace": trace}, sort_keys=True, default=str)


@pytest.mark.parametrize("seed", [7629, 7630])
def test_native_baseline_parity_fresh_episodes_and_reset(seed):
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, DiscoveryFSRed
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    config = StrategyConfig()
    adapted = create_simulator(config, seed=seed, steps=40)
    native = CybORG(scenario_generator=EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=DiscoveryFSRed, steps=40), seed=seed)
    first = episode_signature(adapted, seed)
    assert first == episode_signature(native, seed)
    assert first == episode_signature(create_simulator(config, seed=seed, steps=40), seed)
    assert first == episode_signature(adapted, seed)


def test_manifest_provenance_and_no_effectiveness_labels():
    manifest = strategy_manifest(StrategyConfig(), seeds=[7629, 7630], steps=40)
    assert manifest["seeds"] == [7629, 7630]
    assert manifest["config_sha256"] == StrategyConfig().digest()
    assert len(manifest["source_commit"]) == 40
    assert all(len(digest) == 64 for digest in manifest["source_sha256"].values())
    assert manifest["outcome_evidence"] is None
    assert manifest["artifact_hash"] is None
    assert "truth" not in manifest and "labels" not in manifest


def test_smoke_artifacts_episode_boundary_and_no_overwrite(tmp_path):
    import hashlib
    from red.smoke import run_smoke
    output = tmp_path / "smoke"
    manifest = run_smoke(output, StrategyConfig(), seeds=[8111], steps=4)
    rows = [json.loads(line) for line in (output / "red-actions.jsonl").read_text().splitlines()]
    assert len(rows) == 18  # 3 native joint ticks, six Red agents
    assert manifest["episodes"][0]["joint_steps"] == 3
    assert manifest["diagnostic_sha256"] == hashlib.sha256(
        (output / "red-actions.jsonl").read_bytes()).hexdigest()
    assert all(row["time_unit"] == "sim_tick" and row["origin"] == "simulated" for row in rows)
    assert sum(row["terminated"] for row in rows) == 6
    assert all("truth" not in row and "labels" not in row for row in rows)
    with pytest.raises(FileExistsError):
        run_smoke(output, StrategyConfig(), seeds=[8111], steps=4)
