"""Neuro-symbolic Blue: hard-coded evidence rules + learned scan priority.

Path 2 of the pivot. The round-robin heuristic beats every learned policy
because three of its behaviours are non-negotiable rules, not judgement:

  1. remediate ONLY hosts with CONFIRMED evidence,
  2. verify after remediating (re-analyse),
  3. escalate Remove -> Restore when re-detected.

Those are fixed here and cannot be violated. What rules CANNOT judge is
WHICH host to scan next: round-robin is arbitrary, so it spends ticks on
hosts that will never be attacked. That ordering is handed to a learned
priority scorer, so learning improves detection speed without ever being
able to trigger a wrong remediation.

Blue-visible features only (host_to_vector). Privileged compromise labels
are used solely as TRAINING TARGETS for the risk model, never as policy
inputs — see contracts: only observations entering actors/masks are
Blue-restricted.
"""

import os
import pickle

import numpy as np

import cc4_epymarl_wrapper as wrapper
from blue_baselines import action_index


def _legal(mask, env, agent, host, action):
    return bool(mask[action_index(env, agent, host, action)])


class HybridBluePolicy:
    """Evidence-gated defender with a pluggable scan-priority scorer.

    ``priority_fn(env, agent, host) -> float`` orders the sweep; higher
    means scan sooner. ``None`` uses cursor round-robin (identical to
    RoundRobinBaseline), which makes the learnable part measurable in
    isolation.
    """

    def __init__(self, priority_fn=None, priority_kwargs=None):
        # String specs ("lancer") resolve to scorer builders so registry
        # entries stay picklable for the process pool (proposal 01).
        if isinstance(priority_fn, str):
            priority_fn = make_priority(priority_fn,
                                        **(priority_kwargs or {}))
        elif priority_kwargs:
            raise ValueError("priority_kwargs need a string priority_fn spec")
        self.priority_fn = priority_fn
        self._n = {}
        self._cursor = {}

    def reset(self):
        self._n.clear()
        self._cursor.clear()
        if hasattr(self.priority_fn, "reset"):
            self.priority_fn.reset()

    def select(self, env, agent):
        idx = wrapper.BLUE_AGENTS.index(agent)
        mask = env.get_avail_agent_actions(idx)
        if int(mask.sum()) <= 1:
            return 0  # busy (agent-wide pending) or no session: Sleep only
        tracker = env.trackers[agent]
        hosts = env.hostnames[agent]

        # --- Rule 1+3: CONFIRMED hosts get remediated, escalating to Restore
        # when a post-remediation analysis re-detects the attacker.
        confirmed = [h for h in hosts if tracker.state.get(h) == "CONFIRMED"]
        for host in confirmed:
            last_an = tracker.last_analysis.get(host)
            last_re = tracker.last_remediation.get(host)
            re_detected = (last_an is not None and last_re is not None
                           and last_an > last_re)
            if re_detected and _legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")
            if _legal(mask, env, agent, host, "Remove"):
                return action_index(env, agent, host, "Remove")
            if _legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")

        # --- Rule 2: VERIFY hosts get re-analysed, oldest remediation first.
        verify = [h for h in hosts if tracker.state.get(h) == "VERIFY"
                  and _legal(mask, env, agent, h, "Analyse")]
        if verify:
            host = min(verify, key=lambda h: tracker.last_remediation.get(h) or 0)
            return action_index(env, agent, host, "Analyse")

        # --- Learned part: order the remaining sweep by predicted risk.
        cands = [h for h in hosts if _legal(mask, env, agent, h, "Analyse")]
        if not cands:
            return 0
        if self.priority_fn is None:
            # Parity sweep: same cursor round-robin as RoundRobinBaseline, so
            # the learnable part is measurable in isolation (see proposal 01).
            start = self._cursor.get(agent, 0)
            for offset in range(len(hosts)):
                host = hosts[(start + offset) % len(hosts)]
                if _legal(mask, env, agent, host, "Analyse"):
                    self._cursor[agent] = (start + offset + 1) % len(hosts)
                    pick = host
                    break
            else:
                return 0
        else:
            scored = [(self.priority_fn(env, agent, h), h) for h in cands]
            pick = max(scored, key=lambda sh: (sh[0], sh[1]))[1]
        self._n[agent] = self._n.get(agent, 0) + 1
        return action_index(env, agent, pick, "Analyse")


def host_risk_features(env, agent, host):
    """Blue-visible per-host features for the risk model (no privileged data).

    Uses the same host_to_vector encoding the RL actor sees, so the risk
    model and the actor share one feature contract.
    """
    from blue_obs_features import host_to_vector
    subnets = env.subnets[agent]
    own = subnets.get(host, []) if isinstance(subnets, dict) else subnets
    return host_to_vector(env.views[agent].get(host, {}), agent, own)


# Indices into the host_to_vector 10-feature layout (blue_obs_features).
_I_UNKNOWN_FILES = 6
_I_MAX_DENSITY = 7
_I_N_EXT_CONN = 9

# Punch-style file-density tripwire (their wrapper flagged density > 0.9).
DENSITY_THRESHOLD = 0.9


class LancerPriority:
    """Lancer-style per-host scan priority (CAGE-4 Team lancer, no ML).

    Per-host float value: initialised to ``init``, multiplicatively decayed
    each time an Analyse/Remove/Restore *completes* on that host (detected
    via tracker stamps, counted once), boosted on new detections
    (state -> CONFIRMED) and on novel suspicious signals in the passively
    merged view (more unknown files / higher density / more external
    connections). The novelty boost is our mapping of lancer's
    "increased in response to Monitor observations": we never issue
    Monitor (equivalent to Sleep here), but the wrapper merges unsolicited
    Monitor observations every tick, so view deltas are the same event
    stream.

    On top of the carried value, scoring adds a sticky suspicion bonus
    (UC-style persistent flag, punch-style density tripwire): a host that
    *currently* shows unknown files, density > 0.9, external connections,
    or a CONFIRMED/VERIFY belief scores higher until remediated and
    re-observed. The bonus decays geometrically with consecutive fruitless
    re-analyses (``fruitless_decay`` per completed Analyse that yields no
    detection), which stops the v1 failure mode of re-analysing
    already-known hosts forever. Rules 1-2 still grab CONFIRMED/VERIFY
    hosts first; the bonus matters when remediation is illegal and they
    fall to the sweep.

    State is plain floats/dicts (picklable) keyed per agent; ``reset()``
    clears it (called by HybridBluePolicy.reset).
    """

    def __init__(self, init=1.0, touch_decay=0.5, detect_boost=2.0,
                 novelty_boost=1.0, suspicious_bonus=2.0,
                 fruitless_decay=1.0):
        self.init = init
        self.touch_decay = touch_decay
        self.detect_boost = detect_boost
        self.novelty_boost = novelty_boost
        self.suspicious_bonus = suspicious_bonus
        self.fruitless_decay = fruitless_decay
        self.reset()

    def reset(self):
        self._v = {}
        self._acct_an = {}
        self._acct_re = {}
        self._state = {}
        self._sig = {}
        self._fruitless = {}

    def _signals(self, env, agent, host):
        """(unknown_files, max_density, n_external) from the merged view."""
        vec = host_risk_features(env, agent, host)
        return (vec[_I_UNKNOWN_FILES], vec[_I_MAX_DENSITY],
                vec[_I_N_EXT_CONN])

    def _sync(self, env, agent):
        """Fold tracker/view deltas into carried values (idempotent)."""
        tracker = env.trackers[agent]
        for host in env.hostnames[agent]:
            key = (agent, host)
            last_an = tracker.last_analysis.get(host)
            last_re = tracker.last_remediation.get(host)
            if last_an != self._acct_an.get(key):
                # A new Analyse completed on this host: decay once, and count
                # it fruitless unless it newly detected (state transitioned
                # to CONFIRMED on this sync). Re-analysing an already-known
                # host counts fruitless too -- that is the v1 loop being
                # fixed. Remediation resets the count (see below).
                self._acct_an[key] = last_an
                self._v[key] = self._v.get(key, self.init) * self.touch_decay
                now = tracker.state.get(host)
                if (self._state.get(key) != "CONFIRMED"
                        and now == "CONFIRMED"):
                    self._fruitless[key] = 0
                else:
                    self._fruitless[key] = self._fruitless.get(key, 0) + 1
            if last_re != self._acct_re.get(key):
                # A new Remove/Restore completed: decay once, and a
                # remediation attempt resets fruitless re-analysis count.
                self._acct_re[key] = last_re
                self._v[key] = self._v.get(key, self.init) * self.touch_decay
                self._fruitless[key] = 0
            state = tracker.state.get(host)
            if state == "CONFIRMED" and self._state.get(key) != "CONFIRMED":
                self._v[key] = self._v.get(key, self.init) + self.detect_boost
            self._state[key] = state
            sig = self._signals(env, agent, host)
            old = self._sig.get(key)
            if old is not None and any(s > o for s, o in zip(sig, old)):
                self._v[key] = self._v.get(key, self.init) + self.novelty_boost
            self._sig[key] = sig

    def _suspicious_now(self, env, agent, host):
        unknown_files, density, n_ext = self._sig.get(
            (agent, host), (0, 0.0, 0))
        state = env.trackers[agent].state.get(host)
        return (unknown_files > 0 or density > DENSITY_THRESHOLD
                or n_ext > 0 or state in ("CONFIRMED", "VERIFY"))

    def __call__(self, env, agent, host):
        self._sync(env, agent)
        value = self._v.get((agent, host), self.init)
        if self._suspicious_now(env, agent, host):
            value += (self.suspicious_bonus
                      * (self.fruitless_decay
                         ** self._fruitless.get((agent, host), 0)))
        return value


PRIORITY_BUILDERS = {
    "lancer": LancerPriority,
    # "risk" handled in make_priority (needs model_path to locate weights).
}


def make_priority(spec, **kwargs):
    """Build a priority scorer from a string spec (or pass through)."""
    if spec is None or callable(spec):
        return spec
    if spec == "risk":
        return RiskPriority(**kwargs)
    try:
        return PRIORITY_BUILDERS[spec](**kwargs)
    except KeyError:
        raise ValueError(f"unknown priority {spec!r}; known: "
                         f"{sorted(PRIORITY_BUILDERS)}")


class RiskPriority:
    """Learned P(compromised | Blue-visible features) sweep scorer.

    Loads a blue_train_risk.py model (weights + standardization) and scores
    each sweep candidate with its predicted probability. Features replicate
    the training rows exactly: 10 base + ages/belief temporal + 4 view
    deltas, so the collector, trainer, and scorer share one contract
    (RISK_GROUPS, RISK_DELTAS below). Only features enter at select time;
    privileged labels were training targets only.

    Predictions are batched per (agent, tick) and cached: one model pass
    per agent per tick instead of one per candidate. ``env._tick`` keys the
    cache; without it every call rebuilds (correct, slower). Scores are
    pure probabilities -- ties break by host string in HybridBluePolicy.
    A coverage collapse would show immediately in pool eval (kept as an
    empirical guard, not extra mechanism).
    """

    GROUPS = ("ages", "belief")
    N_DELTA = 4
    DELTA_IDX = (6, 7, 8, 9)  # into the 10 base features

    def __init__(self, model_path):
        from blue_obs_features import temporal_len
        path = model_path
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"risk model not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        self.w = np.asarray(model["w"], dtype=np.float64)
        self.b = float(model["b"])
        self.mean = np.asarray(model["mean"], dtype=np.float64)
        self.std = np.asarray(model["std"], dtype=np.float64)
        expect = 10 + temporal_len(self.GROUPS) + self.N_DELTA
        if model["n_feats"] != expect or len(self.w) != expect:
            raise ValueError(
                f"risk model geometry {model['n_feats']} != scorer {expect}")
        self.model_path = path
        self.reset()

    def reset(self):
        self._prev = {}
        self._cache_key = None
        self._cache = {}

    def _row(self, env, agent, host):
        from blue_obs_features import host_to_temporal, host_to_vector
        tracker = env.trackers[agent]
        busy = agent in env._awaiting
        base = host_to_vector(env.views[agent].get(host, {}), agent,
                              env.subnets[agent][host]
                              if isinstance(env.subnets[agent], dict)
                              else env.subnets[agent])
        row = (base + host_to_temporal(
            tracker, host, env.observed_at[agent][host], env._tick,
            env.episode_limit, busy, self.GROUPS))
        sig = tuple(base[j] for j in self.DELTA_IDX)
        key = (agent, host)
        old = self._prev.get(key, sig)
        row += [s - o for s, o in zip(sig, old)]
        self._prev[key] = sig
        return row

    def _predict_all(self, env, agent):
        hosts = env.hostnames[agent]
        X = np.array([self._row(env, agent, h) for h in hosts],
                     dtype=np.float64)
        z = (X - self.mean) / self.std
        p = 0.5 * (1.0 + np.tanh(0.5 * (z @ self.w + self.b)))
        return dict(zip(hosts, (float(v) for v in p)))

    def __call__(self, env, agent, host):
        tick = getattr(env, "_tick", None)
        key = (agent, tick)
        if key != self._cache_key:
            self._cache = self._predict_all(env, agent)
            self._cache_key = key
        return self._cache[host]
