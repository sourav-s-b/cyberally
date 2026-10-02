"""BLUE-02 baseline policies and evaluator (no training).

Policies use ONLY Blue-visible inputs: availability masks, per-agent host
lists and the BlueZoneTracker belief (itself derived from Blue observations).
Privileged Red-session state is restricted to the evaluator functions at the
bottom of this file, which must never be called by a policy.
"""

import numpy as np

import cc4_epymarl_wrapper as wrapper

ACTION_OFFSET = {"Analyse": 0, "Remove": 1, "Restore": 2}


def action_index(env, agent, host, action):
    """Discrete index for (host, action); identical under both bound modes."""
    return 2 + 3 * env.hostnames[agent].index(host) + ACTION_OFFSET[action]


def decode_index(env, agent, idx):
    """(action_name, host_or_None) for logging; mirrors wrapper._decode.

    Indices outside the agent's own width (possible for unmasked random
    policies under per-agent bounds) report as the Sleep the wrapper executes.
    """
    if idx == 0:
        return ("Sleep", None)
    if idx == 1:
        return ("Monitor", None)
    hi, action_type = divmod(idx - 2, 3)
    if hi >= len(env.hostnames[agent]) or action_type >= len(wrapper.ACTION_TEMPLATES):
        return ("Sleep", None)
    return (wrapper.ACTION_TEMPLATES[action_type], env.hostnames[agent][hi])


class SleepBaseline:
    """Always Sleep. Historical reference: ~60 compromised hosts @ step 200."""

    def reset(self):
        pass

    def select(self, env, agent):
        return 0


class MaskedRandomBaseline:
    """Uniform over the wrapper's currently legal actions (mask-respecting)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def select(self, env, agent):
        idx = wrapper.BLUE_AGENTS.index(agent)
        mask = env.get_avail_agent_actions(idx)
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))


class UnmaskedRandomBaseline:
    """Uniform over the full discrete width; the wrapper coerces illegal to Sleep.

    Reproduces the historical "unmasked random" semantics where off-mask
    exploration is impossible by construction (the fallback guard maps it to
    Sleep). Prefer MaskedRandomBaseline for a cleaner random reference.
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def select(self, env, agent):
        return int(self.rng.integers(0, env.n_actions))


class RoundRobinBaseline:
    """Strong-heuristic starting point: scan hosts round-robin, remediate on evidence.

    Priority when free (mask sum > 1):
      1. CONFIRMED host with Remove legal -> Remove, escalating to Restore
         when the host was re-detected by an Analyse issued after the last
         remediation (Remove failed to clear a privileged attacker).
      2. VERIFY host with Analyse legal -> Analyse (verify the remediation).
      3. Round-robin Analyse over hosts where Analyse is legal.
    Never issues Monitor (equivalent to Sleep here); never touches privileged state.
    """

    def __init__(self):
        self._cursor = {}

    def reset(self):
        self._cursor.clear()

    def _legal(self, mask, env, agent, host, action):
        idx = action_index(env, agent, host, action)
        return bool(mask[idx])

    def select(self, env, agent):
        idx = wrapper.BLUE_AGENTS.index(agent)
        mask = env.get_avail_agent_actions(idx)
        if int(mask.sum()) <= 1:
            return 0  # busy (agent-wide pending) or no session: only Sleep legal
        tracker = env.trackers[agent]
        hosts = env.hostnames[agent]

        confirmed = [h for h in hosts if tracker.state.get(h) == "CONFIRMED"]
        for host in confirmed:
            last_an = tracker.last_analysis.get(host)
            last_re = tracker.last_remediation.get(host)
            re_detected = (
                last_an is not None
                and last_re is not None
                and last_an > last_re
            )
            if re_detected and self._legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")
            if self._legal(mask, env, agent, host, "Remove"):
                return action_index(env, agent, host, "Remove")
            if self._legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")

        verify = [
            h
            for h in hosts
            if tracker.state.get(h) == "VERIFY"
            and self._legal(mask, env, agent, h, "Analyse")
        ]
        if verify:
            host = min(verify, key=lambda h: tracker.last_remediation.get(h) or 0)
            return action_index(env, agent, host, "Analyse")

        start = self._cursor.get(agent, 0)
        for offset in range(len(hosts)):
            host = hosts[(start + offset) % len(hosts)]
            if self._legal(mask, env, agent, host, "Analyse"):
                self._cursor[agent] = (start + offset + 1) % len(hosts)
                return action_index(env, agent, host, "Analyse")
        return 0


class StalestFirstBaseline:
    """Representation-check policy (proposal 12): same remediation core as
    RoundRobinBaseline, but the sweep serves the legal host with the
    stalest coverage -- never-analysed first, then smallest
    last_analysis -- instead of following a cursor. If the cursor carried
    information beyond Blue-visible features, this policy could not match
    round-robin; the pool manifest decides (see docs/proposals/12).
    Never issues Monitor; never touches privileged state.
    """

    def reset(self):
        pass

    def _legal(self, mask, env, agent, host, action):
        idx = action_index(env, agent, host, action)
        return bool(mask[idx])

    def select(self, env, agent):
        idx = wrapper.BLUE_AGENTS.index(agent)
        mask = env.get_avail_agent_actions(idx)
        if int(mask.sum()) <= 1:
            return 0  # busy (agent-wide pending) or no session: only Sleep legal
        tracker = env.trackers[agent]
        hosts = env.hostnames[agent]

        confirmed = [h for h in hosts if tracker.state.get(h) == "CONFIRMED"]
        for host in confirmed:
            last_an = tracker.last_analysis.get(host)
            last_re = tracker.last_remediation.get(host)
            re_detected = (
                last_an is not None
                and last_re is not None
                and last_an > last_re
            )
            if re_detected and self._legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")
            if self._legal(mask, env, agent, host, "Remove"):
                return action_index(env, agent, host, "Remove")
            if self._legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")

        verify = [
            h
            for h in hosts
            if tracker.state.get(h) == "VERIFY"
            and self._legal(mask, env, agent, h, "Analyse")
        ]
        if verify:
            host = min(verify, key=lambda h: tracker.last_remediation.get(h) or 0)
            return action_index(env, agent, host, "Analyse")

        cands = [h for h in hosts
                 if self._legal(mask, env, agent, h, "Analyse")]
        if not cands:
            return 0
        cands.sort(key=lambda h: (tracker.last_analysis.get(h) is not None,
                                  tracker.last_analysis.get(h) or 0,
                                  hosts.index(h)))
        return action_index(env, agent, cands[0], "Analyse")


class SuspicionSweepBaseline:
    """Teacher v2: round-robin remediation core, suspicion-ordered coverage.

    Same CONFIRMED->Remove / VERIFY->Analyse / Restore-escalation as
    RoundRobinBaseline, but the sweep front-loads coverage (never-analysed
    first, then stalest) and skips hosts it analysed in the last few of its
    own analyses (cooldown) so ticks go to unseen/stale hosts first.
    Never issues Monitor; never touches privileged state.
    """

    COOLDOWN = 3

    def __init__(self):
        self._recent = {}

    def reset(self):
        self._recent.clear()

    def _legal(self, mask, env, agent, host, action):
        idx = action_index(env, agent, host, action)
        return bool(mask[idx])

    def select(self, env, agent):
        idx = wrapper.BLUE_AGENTS.index(agent)
        mask = env.get_avail_agent_actions(idx)
        if int(mask.sum()) <= 1:
            return 0  # busy (agent-wide pending) or no session: only Sleep legal
        tracker = env.trackers[agent]
        hosts = env.hostnames[agent]

        confirmed = [h for h in hosts if tracker.state.get(h) == "CONFIRMED"]
        for host in confirmed:
            last_an = tracker.last_analysis.get(host)
            last_re = tracker.last_remediation.get(host)
            re_detected = (
                last_an is not None
                and last_re is not None
                and last_an > last_re
            )
            if re_detected and self._legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")
            if self._legal(mask, env, agent, host, "Remove"):
                return action_index(env, agent, host, "Remove")
            if self._legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")

        verify = [
            h
            for h in hosts
            if tracker.state.get(h) == "VERIFY"
            and self._legal(mask, env, agent, h, "Analyse")
        ]
        if verify:
            host = min(verify, key=lambda h: tracker.last_remediation.get(h) or 0)
            return action_index(env, agent, host, "Analyse")

        recent = self._recent.setdefault(agent, [])
        cands = [h for h in hosts
                 if self._legal(mask, env, agent, h, "Analyse")]
        fresh = [h for h in cands if h not in recent]
        pool = fresh or cands
        pool.sort(key=lambda h: (tracker.last_analysis.get(h) is not None,
                                 tracker.last_analysis.get(h) or 0))
        host = pool[0]
        recent.append(host)
        del recent[:-self.COOLDOWN]
        return action_index(env, agent, host, "Analyse")


# --- Privileged evaluation helpers (eval-only, never policy inputs) ---


def count_compromised(env):
    """Privileged (total, root) compromised-host counts from true state.

    A host counts when any session on it belongs to a red agent; root when
    any such session has username 'root'. Eval-only label.
    """
    controller = env.env  # SimulationController; CC4MARLEnv.env, not .environment_controller
    true_state = controller.get_true_state(controller.INFO_DICT["True"]).data
    total = root = 0
    for host_obs in true_state.values():
        if not isinstance(host_obs, dict):
            continue
        red = [s for s in (host_obs.get("Sessions", []) or []) if "red" in str(s.get("agent", ""))]
        if red:
            total += 1
            if any(s.get("username") == "root" for s in red):
                root += 1
    return total, root


def run_episode(policy, seed=7629, steps=400, snapshot_steps=(200,), mask_mode="validity",
                **env_kwargs):
    """Run one fixed-seed episode; return rewards, snapshots and action trace.

    snapshot_steps are 1-indexed env ticks (step 200 == after 200 joint steps).
    Rewards are the native common team reward (identical per agent); the
    cumulative return is summed once, not once per agent. Extra kwargs (e.g.
    ``per_agent_bounds=True``) go to the wrapper constructor.
    """
    env = wrapper.CC4MARLEnv(seed=seed, steps=steps, mask_mode=mask_mode, **env_kwargs)
    env.reset(seed=seed)
    policy.reset()
    snapshots = {}
    trace = []
    cumulative = 0.0
    for tick in range(1, steps + 1):
        actions = {agent: int(policy.select(env, agent)) for agent in wrapper.BLUE_AGENTS}
        _, rewards, terminated, truncated, _ = env.step(actions)
        cumulative += float(rewards[0])
        for agent in wrapper.BLUE_AGENTS:
            name, host = decode_index(env, agent, actions[agent])
            trace.append(
                {"step": tick, "agent": agent, "action": name, "host": host,
                 "reward": float(rewards[0])}
            )
        if tick in snapshot_steps:
            total, root = count_compromised(env)
            snapshots[tick] = {"total": total, "root": root, "return": cumulative}
        if terminated or truncated:
            break
    total, root = count_compromised(env)
    return {
        "seed": seed,
        "steps": tick,
        "cumulative_return": cumulative,
        "snapshots": snapshots,
        "final": {"total": total, "root": root},
        "trace": trace,
    }


def evaluate_policies(factories, seeds, steps=400, snapshot_steps=(200,), mask_mode="validity",
                      **env_kwargs):
    """Paired multi-seed comparison: every policy runs every seed.

    factories maps a policy name to a zero-arg callable returning a fresh
    policy (fresh RNG per episode). Returns
    ``{policy_name: {seed: run_episode(...)-result}}``. Summaries (means,
    paired diffs) are the caller's job — this helper only gathers raw results
    so training and held-out seeds stay separated by the caller.
    """
    return {
        name: {
            seed: run_episode(factory(), seed=seed, steps=steps,
                              snapshot_steps=snapshot_steps, mask_mode=mask_mode,
                              **env_kwargs)
            for seed in seeds
        }
        for name, factory in factories.items()
    }
