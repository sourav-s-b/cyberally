from copy import deepcopy
from ipaddress import IPv4Network
import json
from types import SimpleNamespace
from ipaddress import IPv4Address

import numpy as np
import pytest
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.AbstractActions import DiscoverRemoteSystems
from red.llm_agent import LLMConfig
from red.planned_agent import PlannedRedAgent, PlannerConfig, configured_planner_class, KnownDecoySelector
from red.main import FixturePlanner, create_simulator, run


class Reply:
    def __init__(self, raw):
        self.raw, self.calls, self.messages = raw, 0, None

    def complete(self, messages, schema=None):
        self.calls += 1
        self.messages = messages
        if isinstance(self.raw, Exception):
            raise self.raw
        return self.raw


def agent_fixture(reply, planner=PlannerConfig()):
    space = {"action": {Sleep: True, DiscoverRemoteSystems: True},
             "agent": {"red_agent_0": True}, "session": {0: True},
             "subnet": {IPv4Network("10.0.0.0/24"): True,
                        IPv4Network("10.99.0.0/24"): False}}
    agent = PlannedRedAgent("red_agent_0", np.random.default_rng(7),
                            config=LLMConfig("mock"), client=reply, planner_config=planner)
    agent.set_initial_values(space, {})
    agent.host_states = {"10.0.0.1": {"state": "K", "hostname": "known_server"}}
    agent.step = 1
    return agent, space


@pytest.mark.parametrize("raw", [TimeoutError(), "bad json", "[]",
    '{"goal":"disrupt","focus_ip":"10.99.0.1"}',
    '{"goal":"spread","focus_ip":null,"extra":1}',
    '{"goal":[],"focus_ip":null}', ValueError("request_budget_exhausted")])
def test_planning_failure_keeps_attacking(raw):
    agent, space = agent_fixture(Reply(raw))
    action = agent.get_action({"success": TernaryEnum.UNKNOWN}, space)
    assert isinstance(action, DiscoverRemoteSystems)
    assert agent.goal == "spread" and agent.focus_ip is None
    assert agent.last_decision["plan_status"] != "accepted"


def test_pending_skips_inference_and_preserves_inputs():
    reply = Reply('{"goal":"escalate","focus_ip":null}')
    agent, space = agent_fixture(reply)
    observation = {"success": TernaryEnum.IN_PROGRESS}
    saved = deepcopy(observation)
    assert isinstance(agent.get_action(observation, space), Sleep)
    assert reply.calls == 0 and observation == saved and agent.turns == 0


def test_plan_persists_and_local_budget_limits_attempts():
    reply = Reply('{"goal":"escalate","focus_ip":null}')
    agent, space = agent_fixture(reply, PlannerConfig(interval=1, max_plans_per_agent=2))
    for _ in range(5):
        assert isinstance(agent.get_action({"success": TernaryEnum.UNKNOWN}, space), DiscoverRemoteSystems)
    assert reply.calls == 2 and agent.goal == "escalate"
    assert "10.99.0" not in json.dumps(reply.messages)


def test_no_valid_target_returns_sleep_without_native_exception():
    agent, space = agent_fixture(Reply('{"goal":"spread","focus_ip":null}'))
    space["subnet"] = {IPv4Network("10.0.0.0/24"): False}
    assert isinstance(agent.get_action({"success": TernaryEnum.UNKNOWN}, space), Sleep)


def test_lost_access_updates_persistent_host_memory():
    agent, space = agent_fixture(Reply('{"goal":"spread","focus_ip":null}'))
    agent.host_states["10.0.0.1"]["state"] = "R"
    agent.get_action({"success": TernaryEnum.UNKNOWN}, space)
    assert agent.host_states["10.0.0.1"]["state"] == "KD"


def test_decoy_selector_accepts_native_priority_and_excludes_observed_ports():
    ip = IPv4Address("10.0.0.1")
    state = SimpleNamespace(sessions={"red_agent_0": {0: SimpleNamespace(ports={ip: [22]})}},
                            np_random=np.random.default_rng(7))
    # With the only discovered service marked as a decoy there is no exploit.
    assert KnownDecoySelector([22]).get_exploit_action(state=state, session=0,
        agent="red_agent_0", ip_address=ip, priority=99) is None
    assert KnownDecoySelector([]).get_exploit_action(state=state, session=0,
        agent="red_agent_0", ip_address=ip, priority=99) is not None


def test_native_reset_clears_all_planner_and_host_memory():
    config, client = LLMConfig("mock"), FixturePlanner(True)
    simulator = create_simulator(configured_planner_class(config, client), 92004, 20)
    simulator.reset(seed=92004)
    simulator.parallel_step({f"blue_agent_{i}": Sleep() for i in range(5)})
    assert client.requests > 0
    simulator.reset(seed=92004)
    for name, interface in simulator.environment_controller.agent_interfaces.items():
        if name.startswith("red_agent_"):
            agent = interface.agent
            assert not agent.host_states and not agent.history
            assert agent.plan_attempts == agent.turns == agent.step == 0
            assert agent.goal == "spread" and agent.focus_ip is None


def test_paired_simulator_comparison_and_output_protection(tmp_path):
    config = LLMConfig("mock")
    result = run(tmp_path / "compare", config, PlannerConfig(), [92004, 92005],
                 40, mock=True, compare=True)
    assert result["mode"] == "mock"
    assert set(result["episodes"]) == {"mock-planner", "scripted", "tactical-only"}
    assert len(result["paired"]) == 2
    for episodes in result["episodes"].values():
        assert all(e["joint_steps"] == 39 and e["invalid_executed_actions"] == 0 for e in episodes)
    with pytest.raises(FileExistsError):
        run(tmp_path / "compare", config, PlannerConfig(), [92004], 40, mock=True)
