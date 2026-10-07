from io import StringIO

import pytest
from blue.core.wrapper import CC4MARLEnv
from red.blue_env import CC4LLMEnv
from red.llm_agent import LLMConfig
from red.main import FixturePlanner, run_episode
from red.planned_agent import PlannedRedAgent, PlannerConfig, configured_planner_class


def test_main_environment_uses_planner_without_mutating_baselines():
    env = CC4LLMEnv(seed=7629, steps=12, client=FixturePlanner(True))
    env.reset(seed=7629)
    assert env.get_env_info()["red_agent"] == "llm-planner"
    for name, interface in env.env.agent_interfaces.items():
        if name.startswith("red_agent_"):
            assert isinstance(interface.agent, PlannedRedAgent)
    env.step([0] * 5)
    assert env.llm_client.requests > 0
    reference = CC4MARLEnv(seed=7629, steps=12)
    assert reference.red_agent == "discovery"
    with pytest.raises(ValueError, match="selects LLM Red"):
        CC4LLMEnv(client=FixturePlanner(), red_agent="discovery")


def test_custom_injection_validated():
    with pytest.raises(TypeError, match="native agent class"):
        CC4MARLEnv(red_agent_class="bad")


def test_lancer_executes_against_new_red_with_valid_actions():
    cls = configured_planner_class(LLMConfig("mock"), FixturePlanner(True), PlannerConfig())
    result = run_episode(cls, 7629, 40, StringIO(), "mock-planner", "lancer")
    assert result["joint_steps"] == 39
    assert result["invalid_executed_actions"] == 0
    assert result["plan_status_counts"]["accepted"] > 0
    assert result["peak_compromised_hosts"] >= result["peak_privileged_hosts"]
