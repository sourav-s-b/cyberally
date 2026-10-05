"""Observe one live episode into an :class:`EpisodeTimeline`.

Everything the repaired metrics need is read from **execution outcomes**:

- investigation completions: ``BlueZoneTracker.last_analysis`` changing to
  the current post-step tick (the wrapper only stamps it on
  ``success == "TRUE"``; a failed Analyse goes to ``note_failure`` and
  leaves the stamp alone),
- investigation failures: ``last_result == ("Analyse", "FALSE")``,
- observable detections: ``tracker.state`` entering ``CONFIRMED``,
- remediations: ``last_remediation`` changing to the current tick,
- privileged compromise episodes: ``env._compromised_set()``, sampled once
  per post-step tick and used for evaluation labels only.

The privileged read is cached per tick on purpose. The pre-repair
``OracleOnset`` scorer called ``_compromised_set()`` once per host per
decision (tens of thousands of ``get_true_state`` calls per episode); this
recorder calls it at most once per step.

Tick convention: ``INVESTIGATION_START`` uses the **pre-step** ``env._tick``
(the decision tick, when the action was requested). Everything else uses
the **post-step** ``env._tick``, i.e. the tick at which the outcome was
actually consumed. Completion-based metrics never read start events.
"""

from __future__ import annotations

from blue.analysis.metrics import (COMPROMISE_CLOSE, COMPROMISE_ONSET,
                                   DETECTION, EpisodeTimeline,
                                   INVESTIGATION_COMPLETE,
                                   INVESTIGATION_FAILED,
                                   INVESTIGATION_START,
                                   REMEDIATION_COMPLETE)
from blue.core.wrapper import BLUE_AGENTS


class TimelineRecorder:
    def __init__(self, env):
        self.tl = EpisodeTimeline(
            {a: list(env.hostnames[a]) for a in BLUE_AGENTS})
        self._la = {a: dict(env.trackers[a].last_analysis) for a in BLUE_AGENTS}
        self._lr = {a: dict(env.trackers[a].last_remediation)
                    for a in BLUE_AGENTS}
        self._state = {a: dict(env.trackers[a].state) for a in BLUE_AGENTS}
        self._result = {a: dict(env.trackers[a].last_result)
                        for a in BLUE_AGENTS}
        self._prev_comp = set()
        self._comp = set()
        self._comp_tick = None
        self.privileged_calls = 0

    def compromised_set(self, env):
        """``env._compromised_set()`` cached for the current post-step tick."""
        tick = env._tick
        if self._comp_tick != tick:
            self._comp = set(env._compromised_set())
            self._comp_tick = tick
            self.privileged_calls += 1
        return self._comp

    def record_start(self, env, action, agent, name, host):
        """Pre-step request record (diagnostics / guard bookkeeping only)."""
        if name == "Analyse" and host is not None:
            self.tl.add(env._tick, INVESTIGATION_START, host, agent=agent)

    def post_step(self, env):
        """Call immediately after ``env.step``."""
        tick = env._tick
        if self.tl.first_observed_tick is None:
            self.tl.first_observed_tick = tick
        for agent in BLUE_AGENTS:
            tracker = env.trackers[agent]
            for host in env.hostnames[agent]:
                last_an = tracker.last_analysis.get(host)
                if last_an is not None and last_an != self._la[agent].get(host):
                    self._la[agent][host] = last_an
                    self.tl.add(last_an, INVESTIGATION_COMPLETE, host,
                                agent=agent)
                last_re = tracker.last_remediation.get(host)
                if (last_re is not None
                        and last_re != self._lr[agent].get(host)):
                    self._lr[agent][host] = last_re
                    self.tl.add(last_re, REMEDIATION_COMPLETE, host,
                                agent=agent)
                result = tracker.last_result.get(host)
                if result != self._result[agent].get(host):
                    self._result[agent][host] = result
                    name, success = result if result else (None, None)
                    if success == "FALSE" and name == "Analyse":
                        # The stamp was NOT refreshed; record the failure so
                        # age/coverage can prove it did not count.
                        self.tl.add(tick, INVESTIGATION_FAILED, host,
                                    agent=agent)
                state = tracker.state.get(host)
                if state != self._state[agent].get(host):
                    self._state[agent][host] = state
                    if state == "CONFIRMED":
                        self.tl.add(tick, DETECTION, host, agent=agent)
        comp = self.compromised_set(env)
        for host in comp:
            if host not in self._prev_comp:
                self.tl.add(tick, COMPROMISE_ONSET, host)
        for host in self._prev_comp:
            if host not in comp:
                self.tl.add(tick, COMPROMISE_CLOSE, host)
        self._prev_comp = set(comp)

    def seal(self, env):
        self.tl.seal(env._tick)
        return self.tl


def run_measured_episode(policy, seed, steps=400, env_kwargs=None,
                         on_decision=None, **env_kw):
    """One episode with the corrected metric timeline attached.

    ``on_decision(recorder, env, actions)`` is an optional per-tick
    observation point used by the guard decision traces. It must not
    consume randomness or change actions; tests assert both.
    """
    from blue.core.baselines import decode_index
    from blue.core.wrapper import CC4MARLEnv
    env = CC4MARLEnv(seed=seed, steps=steps, **(env_kwargs or env_kw))
    env.reset(seed=seed)
    policy.reset()
    rec = TimelineRecorder(env)
    cumulative = 0.0
    trace = []
    for _ in range(steps):
        actions = {a: int(policy.select(env, a)) for a in BLUE_AGENTS}
        for agent in BLUE_AGENTS:
            name, host = decode_index(env, agent, actions[agent])
            rec.record_start(env, actions[agent], agent, name, host)
            trace.append({"step": env._tick, "agent": agent,
                          "action": name, "host": host})
        if on_decision is not None:
            on_decision(rec, env, actions)
        _, rewards, terminated, truncated, _ = env.step(actions)
        cumulative += float(rewards[0])
        rec.post_step(env)
        if terminated or truncated:
            break
    tl = rec.seal(env)
    return {"seed": seed, "return": cumulative, "steps": env._tick,
            "trace": trace, "timeline": tl,
            "privileged_calls": rec.privileged_calls}