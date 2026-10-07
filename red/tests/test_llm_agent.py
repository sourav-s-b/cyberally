from copy import deepcopy
from ipaddress import IPv4Network
import json

import pytest
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.AbstractActions import DiscoverRemoteSystems

from red.llm_agent import (LLMConfig, LLMRedAgent, OllamaClient, catalogue,
                           validate_proposal)
from red.llm_smoke import run_smoke, create_llm_simulator, MockClient


@pytest.fixture
def space():
    return {"action": {Sleep: True, DiscoverRemoteSystems: True},
            "agent": {"red_agent_0": True}, "session": {0: True, 1: False},
            "subnet": {IPv4Network("10.0.0.0/24"): True,
                       IPv4Network("10.99.0.0/24"): False}}


def proposal(**overrides):
    result = {"action": "DiscoverRemoteSystems", "parameters": {
        "agent": "red_agent_0", "session": 0, "subnet": "10.0.0.0/24"}}
    result.update(overrides)
    return json.dumps(result)


class Reply:
    def __init__(self, response):
        self.response, self.calls, self.messages = response, 0, None

    def complete(self, messages):
        self.calls += 1
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_native_conversion_and_no_hidden_values(space):
    action = validate_proposal(proposal(), space, "red_agent_0")
    assert isinstance(action.subnet, IPv4Network)
    assert action.session == 0
    exported = json.dumps(catalogue(space))
    assert "10.99.0" not in exported
    assert catalogue(space)["DiscoverRemoteSystems"]["session"]["values"] == [0]


@pytest.mark.parametrize("params", [
    {"agent": "red_agent_0", "session": True, "subnet": "10.0.0.0/24"},
    {"agent": "red_agent_0", "session": 1, "subnet": "10.0.0.0/24"},
    {"agent": "red_agent_0", "session": 0, "subnet": "10.99.0.0/24"},
    {"agent": "red_agent_1", "session": 0, "subnet": "10.0.0.0/24"},
    {"agent": "red_agent_0", "session": 0},
    {"agent": "red_agent_0", "session": 0, "subnet": "10.0.0.0/24", "code": "ignored"},
])
def test_reject_invalid_parameters(space, params):
    with pytest.raises(ValueError):
        validate_proposal(proposal(parameters=params), space, "red_agent_0")


@pytest.mark.parametrize("raw", ["not json", "[]", '{"action":"Sleep"}',
    '{"action":"Restore","parameters":{}}',
    '{"action":"Sleep","parameters":{},"code":"x"}'])
def test_bad_response_falls_back(space, raw):
    agent = LLMRedAgent("red_agent_0", config=LLMConfig("mock"), client=Reply(raw))
    assert isinstance(agent.get_action({"success": TernaryEnum.UNKNOWN}, space), Sleep)
    assert agent.last_decision["fallback_reason"] is not None


def test_pending_and_reset_and_observation_unchanged(space):
    client = Reply(proposal())
    agent = LLMRedAgent("red_agent_0", config=LLMConfig("mock"), client=client)
    observation = {"success": TernaryEnum.IN_PROGRESS, "host": {"Files": []}}
    saved = deepcopy(observation)
    assert isinstance(agent.get_action(observation, space), Sleep)
    assert client.calls == 0
    assert observation == saved
    agent.get_action({"success": TernaryEnum.TRUE}, space)
    assert client.calls == 1
    assert "10.99.0" not in client.messages[-1]["content"]
    agent.end_episode()
    assert not agent.history and agent.last_decision is None and agent.decisions == 0


def test_timeout_and_prompt_budget(space):
    client = Reply(TimeoutError())
    agent = LLMRedAgent("red_agent_0", config=LLMConfig("mock"), client=client)
    assert isinstance(agent.get_action({}, space), Sleep)
    assert agent.last_decision["fallback_reason"] == "TimeoutError"
    short = LLMRedAgent("red_agent_0", config=LLMConfig("mock", max_prompt_chars=1), client=client)
    short.get_action({}, space)
    assert short.last_decision["fallback_reason"] == "prompt_budget_exhausted"
    assert client.calls == 1


@pytest.mark.parametrize("kwargs", [
    {"endpoint": "https://example.com"}, {"endpoint": "http://127.0.0.1:11434/api"},
    {"model": "something:cloud"}, {"timeout_seconds": 0}, {"max_requests": True}])
def test_config_rejects_bad_local_settings(kwargs):
    with pytest.raises(ValueError):
        LLMConfig(**({"model": "mock"} | kwargs))


def test_ollama_request_shape_and_budget():
    client = OllamaClient(LLMConfig("local-model", max_requests=1))
    payloads = []
    def fake(path, payload=None):
        payloads.append((path, payload))
        return {"done": True, "message": {"content": '{"action":"Sleep","parameters":{}}'},
                "prompt_eval_count": 9, "eval_count": 7}
    client._json = fake
    client.complete([{"role": "user", "content": "test"}])
    assert payloads[0][0] == "/api/chat" and payloads[0][1]["stream"] is False
    assert payloads[0][1]["format"]["type"] == "object"
    assert client.usage == {"prompt_eval_count": 9, "eval_count": 7}
    with pytest.raises(ValueError, match="request_budget"):
        client.complete([])
    assert len(payloads) == 1


def test_disabled_action_and_cross_agent_rejected(space):
    space["action"][DiscoverRemoteSystems] = False
    with pytest.raises(ValueError, match="unavailable_action"):
        validate_proposal(proposal(), space, "red_agent_0")
    space["action"][DiscoverRemoteSystems] = True
    space["agent"]["red_agent_1"] = True
    with pytest.raises(ValueError, match="wrong_agent"):
        validate_proposal(proposal(), space, "red_agent_1")


def test_local_model_preflight_rejects_cloud():
    client = OllamaClient(LLMConfig("local-model"))
    client._json = lambda path: {"models": [{"name": "local-model", "remote_host": "remote"}]}
    with pytest.raises(ValueError, match="installed local"):
        client.check_model()


def test_real_simulator_mock_and_no_overwrite(tmp_path):
    result = run_smoke(tmp_path / "smoke", LLMConfig("mock", max_requests=2),
                       seeds=[8123, 8124], steps=12, mock=True)
    assert result["mode"] == "mock" and result["request_attempts"] == 2
    assert result["decision_counts"]["accepted"] == 2
    assert result["decision_counts"]["request_budget_exhausted"] > 0
    assert all(e["joint_steps"] == 11 and e["invalid_executed_red_actions"] == 0
               for e in result["episodes"])
    rows = [json.loads(line) for line in (tmp_path / "smoke/red-decisions.jsonl").read_text().splitlines()]
    assert any(r["new_decision"] and r["new_decision"]["selected_action"] == "DiscoverRemoteSystems"
               for r in rows)
    with pytest.raises(FileExistsError):
        run_smoke(tmp_path / "smoke", LLMConfig("mock"), seeds=[8123], mock=True)


def test_native_reset_clears_llm_episode_memory():
    config = LLMConfig("mock")
    client = MockClient(config)
    simulator = create_llm_simulator(config, client, seed=8123, steps=12)
    simulator.reset(seed=8123)
    simulator.parallel_step({f"blue_agent_{i}": Sleep() for i in range(5)})
    assert any(interface.agent.history for name, interface in
               simulator.environment_controller.agent_interfaces.items() if name.startswith("red_"))
    attempted = client.requests
    simulator.reset(seed=8123)
    for name, interface in simulator.environment_controller.agent_interfaces.items():
        if name.startswith("red_"):
            assert not interface.agent.history and interface.agent.decisions == 0
    assert client.requests == attempted  # invocation-wide budget survives reset
