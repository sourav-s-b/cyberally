from collections import deque
from types import SimpleNamespace

from blue.dashboard_server import Dashboard, LLM_RED_AGENT, MAX_RED_TRACE


class _Agent:
    def __init__(self, decision):
        self.last_decision = decision


class _Interfaces:
    def __init__(self):
        self.agent_interfaces = {
            "red_agent_0": SimpleNamespace(agent=_Agent({
                "agent_id": "red_agent_0",
                "decision": 4,
                "selected_action": "PrivilegeEscalate",
                "selected_parameters": {"hostname": "user_host_1"},
                "goal": "escalate",
                "focus_ip": "10.0.0.5",
                "plan_status": "accepted",
                "plan_attempts": 2,
                "prompt_sha256": "a" * 64,
                "latency_seconds": 1.25,
            })),
        }


def _dashboard():
    dashboard = Dashboard.__new__(Dashboard)
    dashboard.env = SimpleNamespace(
        env=_Interfaces(),
        llm_client=SimpleNamespace(
            config=SimpleNamespace(model="qwen2.5:3b", max_requests=12),
            requests=2,
            usage={"prompt_eval_count": 10, "eval_count": 3},
        ),
        llm_model={"digest": "digest", "ollama_version": "0.40.0"},
    )
    dashboard.red_name = LLM_RED_AGENT
    dashboard.tick = 4
    dashboard.red_trace = deque(maxlen=MAX_RED_TRACE)
    dashboard._red_marks = {}
    return dashboard


def test_llm_red_trace_is_deduplicated_and_red_only():
    dashboard = _dashboard()
    dashboard._capture_red_trace()
    dashboard._capture_red_trace()

    assert len(dashboard.red_trace) == 1
    decision = dashboard.red_trace[0]
    assert decision["agent"] == "red_agent_0"
    assert decision["goal"] == "escalate"
    assert decision["action"] == "PrivilegeEscalate"
    assert decision["parameters"] == {"hostname": "user_host_1"}
    assert decision["plan_status"] == "accepted"
    assert decision["latency_seconds"] == 1.25


def test_llm_red_state_reports_budget_and_model():
    assert _dashboard()._red_llm_state() == {
        "mode": "ollama",
        "model": "qwen2.5:3b",
        "digest": "digest",
        "ollama_version": "0.40.0",
        "requests": 2,
        "budget": 12,
        "usage": {"prompt_eval_count": 10, "eval_count": 3},
        "planner_interval": 12,
        "max_plans_per_agent": 2,
    }


class ExploitRemoteService:
    def get_params(self):
        return {
            "agent": "red_agent_0",
            "hostname": "user_host_1",
            "ip_address": "10.0.0.5",
        }


def test_native_red_trace_records_executed_action_without_planner():
    dashboard = Dashboard.__new__(Dashboard)
    dashboard.env = SimpleNamespace(
        env=SimpleNamespace(
            agent_interfaces={
                "red_agent_0": SimpleNamespace(
                    agent=SimpleNamespace(last_action=ExploitRemoteService())),
            },
            action={"red_agent_0": [ExploitRemoteService()]},
            get_last_observation=lambda name: SimpleNamespace(
                data={"success": "TRUE"}),
        ),
    )
    dashboard.red_name = "discovery"
    dashboard.tick = 7
    dashboard.red_trace = deque(maxlen=MAX_RED_TRACE)
    dashboard._red_marks = {}

    dashboard._capture_red_trace()

    assert len(dashboard.red_trace) == 1
    entry = dashboard.red_trace[0]
    assert entry["agent"] == "red_agent_0"
    assert entry["action"] == "ExploitRemoteService"
    assert entry["target"] == "user_host_1"
    assert entry["result"] == "TRUE"
    assert entry["plan_status"] == "native"
    assert entry["goal"] is None
    assert entry["parameters"]["hostname"] == "user_host_1"
