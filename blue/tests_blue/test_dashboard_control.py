from collections import deque
import threading

import pytest

import blue.dashboard_server as server
from blue.dashboard_server import Dashboard


def _dashboard():
    dashboard = Dashboard.__new__(Dashboard)
    dashboard.lock = threading.RLock()
    dashboard.env = object()
    dashboard.status = "running"
    dashboard.tick = 12
    dashboard.events = deque(maxlen=server.MAX_EVENTS)
    dashboard._state_locked = lambda: {
        "status": dashboard.status,
        "tick": dashboard.tick,
    }
    return dashboard


def test_stop_marks_episode_stopped_and_records_event():
    dashboard = _dashboard()

    state = dashboard.stop()

    assert state["status"] == "stopped"
    assert dashboard.events[0]["description"] == "CAGE4 episode stopped"
    assert dashboard.events[0]["step"] == 12
    with pytest.raises(ValueError, match="already ended"):
        dashboard.control(running=True)


def test_start_rejects_horizon_above_1500():
    dashboard = _dashboard()

    with pytest.raises(ValueError, match="3 to 1500"):
        dashboard.start({
            "seed": 1,
            "steps": 1501,
            "red_agent": "discovery",
            "blue_policy": "round_robin",
        })
