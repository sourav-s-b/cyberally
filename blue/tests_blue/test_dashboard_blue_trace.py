import threading
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace

import blue.dashboard_server as server
from blue.dashboard_server import Dashboard, MAX_BLUE_TRACE


def _dashboard():
    names = [f"blue_agent_{index}" for index in range(5)]
    host = "restricted_zone_a_subnet_server_host_0"
    tracker = SimpleNamespace(
        state={host: "CONFIRMED"},
        pending_until={host: 9},
        last_result={host: ("Remove", "TRUE")},
    )
    dashboard = Dashboard.__new__(Dashboard)
    dashboard.lock = threading.RLock()
    dashboard.env = SimpleNamespace(
        env=SimpleNamespace(),
        trackers={name: tracker for name in names},
        hostnames={name: ([host] if name == "blue_agent_0" else []) for name in names},
        subnets={name: ({host: {"10.0.0.0/24"}} if name == "blue_agent_0" else {}) for name in names},
        views={name: ({host: {"System info": {"Hostname": host}}} if name == "blue_agent_0" else {}) for name in names},
        _awaiting={"blue_agent_0": (host, "Remove")},
    )
    dashboard.policy_name = "test-policy"
    dashboard.policy_meta = None
    dashboard.red_name = "discovery"
    dashboard.status = "running"
    dashboard.seed = 7629
    dashboard.steps = 10
    dashboard.tick = 4
    dashboard.speed = 1.0
    dashboard.started_at = 0
    dashboard.cumulative_reward = 0.0
    dashboard.error = None
    dashboard.events = deque(maxlen=server.MAX_EVENTS)
    dashboard.reward_history = deque(maxlen=400)
    dashboard.red_trace = deque(maxlen=server.MAX_RED_TRACE)
    dashboard.blue_trace = deque(maxlen=MAX_BLUE_TRACE)
    dashboard.blue_trace.appendleft({
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": 4,
        "agent": "blue_agent_0",
        "action": "Remove",
        "target": host,
        "result": "IN_PROGRESS",
        "reward": 0.0,
    })
    return dashboard, host


def test_state_exposes_blue_trace_and_host_actions(monkeypatch):
    dashboard, host = _dashboard()
    monkeypatch.setattr(server, "count_compromised", lambda env: (1, 0))

    state = dashboard._state_locked()

    assert state["blue_trace"][0]["action"] == "Remove"
    exposed = next(item for item in state["hosts"] if item["name"] == host)
    assert exposed["pending"] is True
    assert exposed["pending_action"] == "Remove"
    assert exposed["last_action"] == "Remove"
    assert exposed["last_result"] == "TRUE"


def test_blue_view_exposes_blue_trace_and_host_actions():
    dashboard, host = _dashboard()

    view = dashboard.blue_view()

    assert view["blue_trace"][0]["target"] == host
    exposed = next(item for item in view["agents"][0]["hosts"] if item["name"] == host)
    assert exposed["pending_action"] == "Remove"
    assert exposed["last_action"] == "Remove"
    assert exposed["last_result"] == "TRUE"
