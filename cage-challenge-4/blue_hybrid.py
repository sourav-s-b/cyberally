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

    def __init__(self, priority_fn=None):
        self.priority_fn = priority_fn
        self._n = {}
        self._cursor = {}

    def reset(self):
        self._n.clear()
        self._cursor.clear()

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
    return host_to_vector(env.trackers[agent], env.views[agent][host], host,
                          env.subnets[agent], agent)
