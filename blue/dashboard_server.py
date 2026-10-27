"""Local web dashboard backed by the project's CAGE4 simulator.

Run from the repository root with ``.venv-train/bin/python -m
blue.dashboard_server`` to enable local checkpoint discovery/loading. This
adapter uses the existing CAGE4 wrapper and policies. It is a local research
UI, not a network service or a real-host control interface.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network
import importlib.util
import mimetypes
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

try:
    from blue.core import wrapper
    from blue.core.baselines import (
        MaskedRandomBaseline,
        RoundRobinBaseline,
        SleepBaseline,
        count_compromised,
        decode_index,
    )
    from blue.model_catalog import catalog as model_catalog, get_model
except ModuleNotFoundError as exc:
    raise SystemExit(
        "CAGE4 dashboard dependencies are missing from this Python interpreter. "
        "Use the project training environment: ./.venv-train/bin/python -m blue.dashboard_server "
        "(Windows: .venv-train\\Scripts\\python.exe -m blue.dashboard_server)."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = ROOT / "frontend"
DIST_ROOT = FRONTEND_ROOT / "dist"
TICK_SECONDS = 0.6
MAX_EVENTS = 500
MAX_RED_TRACE = 90
MAX_SPEED = 8.0
MIN_SPEED = 0.1
LLM_RED_AGENT = "llm-planner"
LLM_RED_MODEL = "qwen2.5:3b"
LLM_RED_MAX_REQUESTS = 12
LLM_RED_PLAN_INTERVAL = 12
LLM_RED_MAX_PLANS = 2
LLM_RED_TIMEOUT_SECONDS = 45.0

ZONE_INFO = {
    "restricted_zone_a": ("Restricted Zone A", "restricted"),
    "operational_zone_a": ("Operational Zone A", "operational"),
    "restricted_zone_b": ("Restricted Zone B", "restricted"),
    "operational_zone_b": ("Operational Zone B", "operational"),
    "public_access_zone": ("Public Access Zone", "public"),
    "admin_network": ("Admin Network", "admin"),
    "office_network": ("Office Network", "office"),
    "contractor_network": ("Contractor Network", "contractor"),
    "internet": ("Internet", "internet"),
}


def _host_zone(host_name: str) -> tuple[str, str, str]:
    zone_id = str(host_name).split("_subnet", 1)[0]
    label, kind = ZONE_INFO.get(
        zone_id,
        (zone_id.replace("_", " ").title() or "Unknown zone", "shared"),
    )
    return zone_id, label, kind


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


class Dashboard:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.env = None
        self.policy = None
        self.policy_name = None
        self.policy_meta = None
        self.red_name = None
        self.status = "idle"
        self.seed = None
        self.steps = 400
        self.tick = 0
        self.started_at = None
        self.cumulative_reward = 0.0
        self.speed = 1.0
        self.events = deque(maxlen=MAX_EVENTS)
        self.reward_history = deque(maxlen=400)
        self.red_trace = deque(maxlen=MAX_RED_TRACE)
        self._red_marks = {}
        self.error = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @staticmethod
    def _validate_speed(value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("speed must be a number")
        value = float(value)
        if not MIN_SPEED <= value <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED:g} and {MAX_SPEED:g}")
        return value

    @staticmethod
    def _policy(name: str, seed: int):
        if name == "round_robin":
            return RoundRobinBaseline()
        if name == "masked_random":
            return MaskedRandomBaseline(seed=seed)
        if name == "sleep":
            return SleepBaseline()
        raise ValueError("unknown Blue policy")

    def start(self, payload: dict) -> dict:
        seed = payload.get("seed", 7629)
        steps = payload.get("steps", 400)
        red_name = payload.get("red_agent", "discovery")
        red_model = payload.get("red_model", LLM_RED_MODEL)
        policy_name = payload.get("blue_policy", "round_robin")
        speed = self._validate_speed(payload.get("speed", 1.0))
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if isinstance(steps, bool) or not isinstance(steps, int) or not 3 <= steps <= 400:
            raise ValueError("steps must be an integer from 3 to 400")
        red_choices = set(wrapper.RED_AGENTS) | {LLM_RED_AGENT}
        if not isinstance(red_name, str) or red_name not in red_choices:
            raise ValueError(f"unknown CAGE4 Red variant: {red_name}")
        if not isinstance(red_model, str) or not red_model.strip():
            raise ValueError("red_model must be a non-empty Ollama model tag")
        red_model = red_model.strip()
        model = None
        temporal_features = ()
        include_root_session = True
        if policy_name not in ("round_robin", "masked_random", "sleep"):
            model = get_model(policy_name)
            temporal_features = tuple(model["temporal_features"])
            include_root_session = model["include_root_session"]
        with self.lock:
            if self.env is not None:
                self.env.close()
            self.env = None
            self.status = "idle"
            self.error = None
            if red_name == LLM_RED_AGENT:
                from red.blue_env import CC4LLMEnv
                try:
                    self.env = CC4LLMEnv(
                        seed=seed, steps=steps,
                        temporal_features=temporal_features,
                        include_root_session=include_root_session,
                        llm_config={
                            "model": red_model,
                            "timeout_seconds": LLM_RED_TIMEOUT_SECONDS,
                            "max_requests": LLM_RED_MAX_REQUESTS,
                        },
                        planner_config={
                            "interval": LLM_RED_PLAN_INTERVAL,
                            "max_plans_per_agent": LLM_RED_MAX_PLANS,
                        })
                except ValueError as exc:
                    self.error = f"LLM Red unavailable: {exc}"
                    raise
                except OSError as exc:
                    self.error = (
                        "LLM Red needs local Ollama at 127.0.0.1:11434 with "
                        f"{red_model} installed: {exc}")
                    raise ValueError(self.error) from exc
            else:
                self.env = wrapper.CC4MARLEnv(
                    seed=seed, steps=steps, red_agent=red_name,
                    temporal_features=temporal_features,
                    include_root_session=include_root_session)
            self.env.reset(seed=seed)
            if model is None:
                policy = self._policy(policy_name, seed)
                policy_label = policy_name
            elif model.get("kind") == "set_actor":
                from blue.policies.setactor import SetActorCheckpointPolicy
                options = model.get("policy") or {}
                policy = SetActorCheckpointPolicy(
                    ROOT / model["actor_path"], ROOT / model["scorer_path"],
                    hidden_dim=model["hidden_dim"],
                    bonus=options.get("bonus", 0.25),
                    ml_inputs=options.get("ml_inputs") or "risk",
                    temperature=options.get("temperature", 0.5),
                    explore=options.get("explore", 0.2),
                    max_age=options.get("max_age", 80.0),
                    policy_seed=seed * 997,
                    execution="sample")
                policy_label = model["label"]
            elif model.get("kind") == "harness_residual":
                from blue.policies.harness_checkpoints import (
                    Main32CheckpointPolicy)
                policy = Main32CheckpointPolicy(
                    ROOT / model["run_dir"], ROOT / model["scorer_path"],
                    policy_seed=seed * 997)
                policy_label = model["label"]
            elif model.get("kind") == "architecture":
                from blue.policies.harness_checkpoints import (
                    ArchitectureCheckpointPolicy)
                policy = ArchitectureCheckpointPolicy(
                    ROOT / model["run_dir"], ROOT / model["scorer_path"],
                    model["arch"],
                    policy_seed=seed * 997 + int(model.get("training_seed", 0)))
                policy_label = model["label"]
            else:
                model_dir = str(ROOT / "blue" / "results" / "models" / model["id"])
                if model["return_conditioned"]:
                    from blue.policies.eval_rvs import RvSCheckpointPolicy
                    policy = RvSCheckpointPolicy(
                        model_dir, target_return=-80.0,
                        hidden_dim=model["hidden_dim"],
                        env_info=self.env.get_env_info(),
                        agent_type="rnn_factorized",
                        attn_layers=model["attention_layers"])
                else:
                    from blue.policies.eval_mappo import GreedyCheckpointPolicy
                    policy = GreedyCheckpointPolicy(
                        model_dir, hidden_dim=model["hidden_dim"],
                        env_info=self.env.get_env_info(),
                        agent_type=model["agent_type"],
                        attn_layers=model["attention_layers"])
                policy_label = model["label"]
            policy.reset()
            self.policy = policy
            self.policy_name = policy_label
            self.policy_meta = model
            self.red_name = red_name
            self.seed = seed
            self.steps = steps
            self.tick = 0
            self.cumulative_reward = 0.0
            self.speed = speed
            self.events.clear()
            self.reward_history.clear()
            self.red_trace.clear()
            self._red_marks.clear()
            self.error = None
            self.started_at = time.monotonic()
            self.status = "running"
            self._event("system", "CAGE4 episode started", {
                "seed": seed, "horizon": steps,
                "red_agent": red_name, "blue_policy": policy_name,
                **({"red_model": red_model} if red_name == LLM_RED_AGENT else {}),
            })
            return self._state_locked()

    def control(self, running: bool, speed=None) -> dict:
        with self.lock:
            if self.env is None:
                raise ValueError("launch a CAGE4 episode first")
            if self.status in ("complete", "error"):
                raise ValueError("the episode has ended; launch a new one")
            if speed is not None:
                self.speed = self._validate_speed(speed)
            self.status = "running" if running else "paused"
            self._event("system", "CAGE4 episode resumed" if running else "CAGE4 episode paused", {
                "speed": self.speed,
            })
            return self._state_locked()

    def _event(self, kind: str, description: str, details: dict) -> None:
        self.events.appendleft({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "step": self.tick,
            "type": kind,
            "description": description,
            **details,
        })

    def _run(self) -> None:
        while True:
            time.sleep(TICK_SECONDS / getattr(self, "speed", 1.0))
            with self.lock:
                if self.status != "running" or self.env is None:
                    continue
                try:
                    chosen = {
                        agent: int(self.policy.select(self.env, agent))
                        for agent in wrapper.BLUE_AGENTS
                    }
                    decoded = {
                        agent: decode_index(self.env, agent, chosen[agent])
                        for agent in wrapper.BLUE_AGENTS
                    }
                    _, rewards, terminated, truncated, _ = self.env.step(chosen)
                    if hasattr(self.policy, "observe_step"):
                        self.policy.observe_step(chosen, rewards)
                    self.tick += 1
                    reward = float(rewards[0])
                    self.cumulative_reward += reward
                    self.reward_history.append({
                        "tick": self.tick,
                        "reward": round(reward, 4),
                        "cumulative": round(self.cumulative_reward, 4),
                    })
                    for agent in wrapper.BLUE_AGENTS:
                        action, host = decoded[agent]
                        observation = self.env.env.get_last_observation(agent).data
                        success = observation.get("success")
                        if success is not None:
                            result = str(success)
                        elif action in ("Analyse", "Remove", "Restore") and host in self.env._awaiting.get(agent, (None, None)):
                            result = "pending"
                        else:
                            result = "issued"
                        self._event("defense", f"Chose {action} on {host}", {
                            "agent": agent, "action": action, "target": host,
                            "result": result, "reward": reward,
                        })
                    if self.red_name == LLM_RED_AGENT:
                        self._capture_red_trace()
                    red_total, red_root = count_compromised(self.env)
                    self._event("evaluation", "Privileged CAGE4 Red-session snapshot", {
                        "red_occupied_hosts": red_total,
                        "red_root_hosts": red_root,
                        "evaluation_only": True,
                    })
                    if terminated or truncated:
                        self.status = "complete"
                        self._event("system", "CAGE4 episode completed", {})
                except Exception as exc:  # surface simulator errors in the dashboard
                    self.status = "error"
                    self.error = f"{type(exc).__name__}: {exc}"
                    self._event("system", "CAGE4 episode stopped with an error", {"error": self.error})

    def _capture_red_trace(self) -> None:
        """Record Red planner decisions; never feed them into a Blue policy."""
        interfaces = getattr(self.env.env, "agent_interfaces", {})
        for name in sorted(interfaces):
            if not name.startswith("red_agent_"):
                continue
            decision = getattr(interfaces[name].agent, "last_decision", None)
            if not isinstance(decision, dict):
                continue
            marker = (decision.get("agent_id"), decision.get("decision"))
            if self._red_marks.get(name) == marker:
                continue
            self._red_marks[name] = marker
            latency = decision.get("latency_seconds")
            self.red_trace.appendleft({
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "step": self.tick,
                "agent": name,
                "goal": decision.get("goal"),
                "focus_ip": decision.get("focus_ip"),
                "action": decision.get("selected_action"),
                "parameters": self._json_safe(
                    decision.get("selected_parameters") or {}),
                "plan_status": (decision.get("plan_status")
                                or decision.get("fallback_reason")),
                "plan_attempts": decision.get("plan_attempts"),
                "prompt_sha256": decision.get("prompt_sha256"),
                "latency_seconds": (round(float(latency), 3)
                                    if isinstance(latency, (int, float)) else None),
            })

    def _red_llm_state(self) -> dict | None:
        if self.red_name != LLM_RED_AGENT or self.env is None:
            return None
        client = getattr(self.env, "llm_client", None)
        model = getattr(self.env, "llm_model", {}) or {}
        if client is None:
            return None
        return {
            "mode": "ollama",
            "model": client.config.model,
            "digest": model.get("digest"),
            "ollama_version": model.get("ollama_version"),
            "requests": client.requests,
            "budget": client.config.max_requests,
            "usage": dict(client.usage),
            "planner_interval": LLM_RED_PLAN_INTERVAL,
            "max_plans_per_agent": LLM_RED_MAX_PLANS,
        }

    def _state_locked(self) -> dict:
        if self.env is None:
            return {
                "status": "idle", "connected": True, "simulator": "CybORG CAGE4",
                "tick": 0, "horizon": self.steps, "events": [],
                "speed": self.speed, "reward_history": [],
                "red_llm": None, "red_trace": [],
            }
        compromised, root = count_compromised(self.env)
        hosts = []
        for agent in wrapper.BLUE_AGENTS:
            tracker = self.env.trackers[agent]
            for host in self.env.hostnames[agent]:
                zone_id, zone_label, zone_kind = _host_zone(host)
                hosts.append({
                    "name": host,
                    "agent": agent,
                    "subnet": ", ".join(sorted(map(str, self.env.subnets[agent].get(host, [])))),
                    "zone_id": zone_id,
                    "zone_label": zone_label,
                    "zone_kind": zone_kind,
                    "blue_evidence": tracker.state.get(host, "UNKNOWN"),
                    "pending": host in tracker.pending_until,
                })
        elapsed = max(0, int(time.monotonic() - self.started_at)) if self.started_at else 0
        return {
            "status": self.status,
            "connected": True,
            "simulator": "CybORG CAGE4 EnterpriseScenarioGenerator",
            "scenario": "CAGE4 Enterprise",
            "red_agent": self.red_name,
            "red_llm": self._red_llm_state(),
            "red_trace": (list(self.red_trace)
                          if self.red_name == LLM_RED_AGENT else []),
            "blue_policy": self.policy_name,
            "blue_policy_model": self.policy_meta,
            "seed": self.seed,
            "tick": self.tick,
            "horizon": self.steps,
            "elapsed_seconds": elapsed,
            "speed": self.speed,
            "tick_seconds": round(TICK_SECONDS / self.speed, 3),
            "native_team_reward_cumulative": round(self.cumulative_reward, 3),
            "reward_history": list(self.reward_history),
            "evaluation_only": {
                "red_occupied_hosts": compromised,
                "red_root_hosts": root,
                "note": "Privileged simulator truth for evaluation; never used by the Blue policy.",
            },
            "hosts": hosts,
            "events": list(self.events),
            "error": self.error,
        }

    def state(self) -> dict:
        with self.lock:
            return self._state_locked()

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(key): Dashboard._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [Dashboard._json_safe(item) for item in value]
        if isinstance(value, (IPv4Address, IPv4Network, IPv6Address, IPv6Network)):
            return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def blue_view(self) -> dict:
        """Return the per-agent Blue cache only; never include evaluation truth."""
        with self.lock:
            if self.env is None:
                return {"status": "idle", "tick": 0, "horizon": self.steps,
                        "agents": []}
            agents = []
            for agent in wrapper.BLUE_AGENTS:
                tracker = self.env.trackers[agent]
                hosts = []
                for host in self.env.hostnames[agent]:
                    zone_id, zone_label, zone_kind = _host_zone(host)
                    hosts.append({
                        "name": host,
                        "subnets": sorted(map(str, self.env.subnets[agent].get(host, []))),
                        "zone_id": zone_id,
                        "zone_label": zone_label,
                        "zone_kind": zone_kind,
                        "evidence": tracker.state.get(host, "UNKNOWN"),
                        "pending": host in tracker.pending_until,
                        "observation": self._json_safe(self.env.views[agent].get(host, {})),
                    })
                pending_action = self.env._awaiting.get(agent)
                agents.append({
                    "name": agent,
                    "hosts": hosts,
                    "pending_action": ({"host": pending_action[0],
                                        "action": pending_action[1]}
                                       if pending_action else None),
                })
            return {
                "status": self.status,
                "simulator": "CybORG CAGE4",
                "tick": self.tick,
                "horizon": self.steps,
                "blue_policy": self.policy_name,
                "agents": agents,
                "view_scope": "Blue-owned hosts, retained Blue observations and Blue evidence tracker only.",
            }


dashboard = Dashboard()


def _frontend_file(path: str) -> Path | None:
    """Map a dashboard URL to a file under frontend/dist, rejecting traversal."""
    if path in {"/", "/index.html", "/frontend", "/frontend/", "/frontend/index.html"}:
        return DIST_ROOT / "index.html"
    assets_root = (DIST_ROOT / "assets").resolve()
    for prefix in ("/assets/", "/frontend/assets/"):
        if path.startswith(prefix):
            candidate = (assets_root / path[len(prefix):]).resolve()
            if candidate.is_relative_to(assets_root) and candidate.is_file():
                return candidate
            return None
    return None


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print("[dashboard] " + fmt % args)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 16384:
            raise ValueError("request body is too large")
        if not length:
            return {}
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            return self._json(200, dashboard.state())
        if path == "/api/blue-view":
            return self._json(200, dashboard.blue_view())
        if path == "/api/models":
            try:
                return self._json(200, {
                    "models": model_catalog(),
                    "torch_available": _torch_available(),
                })
            except Exception as exc:
                print(f"[dashboard] model catalog unavailable: {exc}")
                return self._json(200, {
                    "models": [],
                    "catalog_error": f"{type(exc).__name__}: {exc}",
                    "torch_available": _torch_available(),
                })
        if path == "/api/export":
            return self._json(200, dashboard.state())
        if path.startswith("/api/"):
            return self._json(404, {"error": "unknown API endpoint"})
        # Chrome DevTools probes this path; answer quietly to keep the log clean.
        if path.startswith("/.well-known/"):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not (DIST_ROOT / "index.html").is_file():
            return self._json(
                503,
                {
                    "error": (
                        "frontend build missing; run "
                        "`cd frontend && npm install && npm run build`"
                    )
                },
            )
        file_path = _frontend_file(path)
        if file_path is None:
            return self._json(404, {"error": "file not found"})
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            payload = self._payload()
            path = urlparse(self.path).path
            if path == "/api/start":
                return self._json(200, dashboard.start(payload))
            if path == "/api/control":
                running = payload.get("running")
                if not isinstance(running, bool):
                    raise ValueError("running must be a boolean")
                return self._json(200, dashboard.control(running, payload.get("speed")))
            return self._json(404, {"error": "unknown API endpoint"})
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})


def main() -> None:
    port = int(os.environ.get("CYBERALLY_DASHBOARD_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CAGE4 dashboard: http://127.0.0.1:{port}")
    if not _torch_available():
        print("[dashboard] PyTorch not found: saved actor checkpoints will be unavailable.")
        print("[dashboard] Restart with ./.venv-train/bin/python -m blue.dashboard_server to load them.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with dashboard.lock:
            if dashboard.env is not None:
                dashboard.env.close()


if __name__ == "__main__":
    main()
