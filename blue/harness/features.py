"""One feature contract for collection, ML inference and the new RL actor.

Missing telemetry remains NaN for tree models. Neural inputs use zero plus
explicit availability/freshness flags; zero is never an implicit clean label.
No simulator truth or reward enters this module.
"""
from __future__ import annotations

import numpy as np

from blue.core.obs_features import BELIEF_STATES, FEATURE_NAMES, host_to_vector

VERSION = "blue-harness-v1"
FIELDS = ("Sessions", "Processes", "Files")
BASE_FIELDS = ("Sessions", "Processes", "Files", "Sessions", "Sessions",
               "Sessions", "Files", "Files", "Processes", "Processes")
NAMES = tuple(FEATURE_NAMES) + (
    "analysis_age", "analysis_available", "remediation_age",
    "remediation_available", *["belief_" + s for s in BELIEF_STATES],
    "empty_analysis_strikes", "busy",
    "last_action_analyse", "last_action_remove", "last_action_restore",
    "last_success", "last_failure",
    *[n for field in FIELDS for n in (field + "_available", field + "_age")],
    "delta_unknown_files", "delta_file_density", "delta_connections",
    "delta_external_connections", "episode_fraction",
    "phase_0", "phase_1", "phase_2",
    "role_server", "role_user", "role_router",
    "zone_operational_a", "zone_operational_b",
)
DELTA_IDX = (6, 7, 8, 9)
ACTOR_NAMES = NAMES + tuple("value_available_" + n for n in NAMES) + (
    "compromise_risk", "novelty_percentile")


class BlueFeatures:
    """Snapshot all owned hosts once per agent/tick, independent of query order.

    Explicit reset is required on every episode, including same-seed resets.
    Inference over candidate subsets cannot consume/change delta features.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._env = None
        self._cache = {}
        self._prev = {}

    def rows(self, env, agent, hosts=None):
        if self._env is not env:
            self.reset()
            self._env = env
        tick = int(env._tick)
        cached = self._cache.get(agent)
        if cached is not None and tick < cached[0]:
            raise ValueError("episode clock reversed; reset preprocessing")
        if cached is None or cached[0] != tick:
            tracker = env.trackers[agent]
            horizon = int(env.episode_limit)
            if horizon < 1:
                raise ValueError("invalid horizon")
            table = {}
            for host in env.hostnames[agent]:
                view = env.views[agent].get(host, {})
                stamps = env.observed_at[agent].get(host, {})
                for stamp in stamps.values():
                    if not np.isfinite(stamp) or stamp > tick or stamp < 0:
                        raise ValueError("invalid observation timestamp")
                own = env.subnets[agent]
                own = own[host] if isinstance(own, dict) else own
                base = np.asarray(host_to_vector(view, agent, own), dtype=float)
                if not np.isfinite(base).all():
                    raise ValueError("nonfinite observed telemetry")
                available = {f: f in view and f in stamps for f in FIELDS}
                for i, field in enumerate(BASE_FIELDS):
                    if not available[field]:
                        base[i] = np.nan

                def age(stamp):
                    if stamp is None:
                        return 1.0
                    if not np.isfinite(stamp) or not 0 <= stamp <= tick:
                        raise ValueError("invalid action timestamp")
                    return min((tick - stamp) / horizon, 1.0)

                analysis = tracker.last_analysis.get(host)
                remediation = tracker.last_remediation.get(host)
                state = tracker.state.get(host, "UNKNOWN")
                if state not in BELIEF_STATES:
                    raise ValueError("unknown belief state")
                action, success = tracker.last_result.get(host) or (None, None)
                history = [age(analysis), float(analysis is not None),
                           age(remediation), float(remediation is not None)]
                history += [float(state == s) for s in BELIEF_STATES]
                history += [float(tracker.empty_strikes.get(host, 0)),
                            float(agent in env._awaiting)]
                history += [float(action == a) for a in ("Analyse", "Remove", "Restore")]
                history += [float(success == "TRUE"), float(success == "FALSE")]
                freshness = [v for f in FIELDS for v in (
                    float(available[f]), age(stamps.get(f) if available[f] else None))]
                sig = base[list(DELTA_IDX)]
                key = (agent, host)
                old = self._prev.get(key)
                delta = np.zeros(4) if old is None else sig - old
                # Newly observed or missing fields have no valid previous delta.
                if old is not None:
                    delta[~(np.isfinite(sig) & np.isfinite(old))] = np.nan
                else:
                    delta[~np.isfinite(sig)] = np.nan
                self._prev[key] = sig.copy()
                # Public CC4 phase schedule: remainder assigned to earlier phases.
                q, r = divmod(horizon, 3)
                bounds = (q + (r > 0), 2 * q + (r > 0) + (r > 1))
                phase = int(tick >= bounds[0]) + int(tick >= bounds[1])
                context = [tick / horizon, *[float(phase == p) for p in range(3)],
                           float("server_host" in host), float("user_host" in host),
                           float("router" in host),
                           float(host.startswith("operational_zone_a_")),
                           float(host.startswith("operational_zone_b_"))]
                row = np.asarray([*base, *history, *freshness, *delta, *context], dtype=np.float32)
                if row.shape != (len(NAMES),) or np.isinf(row).any():
                    raise ValueError("invalid feature geometry")
                table[host] = row
            self._cache[agent] = (tick, table)
        table = self._cache[agent][1]
        hosts = env.hostnames[agent] if hosts is None else hosts
        return np.asarray([table[h] for h in hosts], dtype=np.float32).reshape(-1, len(NAMES))


def neural_rows(X, mean, scale):
    """Training-only fitted scaling; missing values become zero after scaling."""
    X = np.asarray(X, dtype=np.float32)
    mean, scale = np.asarray(mean), np.asarray(scale)
    if X.shape[-1] != len(NAMES) or mean.shape != (len(NAMES),) or scale.shape != mean.shape:
        raise ValueError("preprocessing geometry mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid fitted normalization")
    if np.isinf(X).any():
        raise ValueError("infinite input")
    return np.clip(np.nan_to_num((X - mean) / scale, nan=0.0), -10, 10).astype(np.float32)
