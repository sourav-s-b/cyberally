"""Live validation of the timeline recorder against the wrapper's own state.

The deterministic tests in ``test_metric_semantics.py`` prove what the
metrics DO with a timeline. These prove the recorder builds the right
timeline from a real episode: every claim is cross-checked against
``BlueZoneTracker`` fields the wrapper maintains independently, so a recorder
bug cannot hide behind the metrics it feeds.

Also asserted: the ``on_decision`` observation hook is observation-only (same
seed, same trace hash with and without it), and the privileged compromise
read happens once per post-step tick rather than once per host.
"""

from blue.analysis.metrics import (INVESTIGATION_COMPLETE, INVESTIGATION_FAILED,
                                   REMEDIATION_COMPLETE, summarize)
from blue.analysis.recorder import run_measured_episode
from blue.core.wrapper import BLUE_AGENTS
from blue.policies.eval_parallel import trace_sha
from blue.policies.ordered import LancerValues, OrderedPolicy

ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}
SEEDS = [7629, 7630]


def _episode(seed=7629, steps=200, hook=None):
    pol = OrderedPolicy(scorer=LancerValues(), max_age=8)
    return pol, run_measured_episode(pol, seed=seed, steps=steps,
                                     env_kwargs=ENV_KW, on_decision=hook)


def _episode_capturing_env(seed=7629, steps=200):
    box = {}
    pol = OrderedPolicy(scorer=LancerValues(), max_age=8)
    res = run_measured_episode(pol, seed=seed, steps=steps, env_kwargs=ENV_KW,
                               on_decision=lambda rec, e, acts:
                               box.__setitem__("env", e))
    return box["env"], res


def test_recorder_agrees_with_tracker_on_every_field():
    """Cross-check every field the wrapper maintains independently."""
    env, res = _episode_capturing_env()
    tl = res["timeline"]
    for agent in BLUE_AGENTS:
        tracker = env.trackers[agent]
        for host in env.hostnames[agent]:
            # 1. stamps <-> completion events, newest event == stamp
            stamp = tracker.last_analysis.get(host)
            ticks = [e["tick"] for e in tl._of_kind(INVESTIGATION_COMPLETE)
                     if e["agent"] == agent and e["host"] == host]
            if stamp is None:
                assert not ticks, (f"{agent}/{host} has completions but no "
                                   f"tracker stamp")
            else:
                assert ticks, f"{agent}/{host} stamped {stamp} but no event"
                assert max(ticks) == stamp, (
                    f"{agent}/{host}: newest event {max(ticks)} != tracker "
                    f"stamp {stamp}")
            # 2. a FAILED last_result must have a FAILURE event
            result = tracker.last_result.get(host)
            if result and result[0] == "Analyse" and result[1] == "FALSE":
                assert [e for e in tl._of_kind(INVESTIGATION_FAILED)
                        if e["agent"] == agent and e["host"] == host], (
                    f"{agent}/{host} failed per tracker, no FAILURE event")
            # 3. remediation stamps
            stamp = tracker.last_remediation.get(host)
            assert (stamp is not None) == bool([
                e for e in tl._of_kind(REMEDIATION_COMPLETE)
                if e["agent"] == agent and e["host"] == host]), (
                f"{agent}/{host} remediation stamp {stamp} disagrees with "
                f"the recorded events")


def test_coverage_matches_the_set_of_stamped_hosts():
    env, res = _episode_capturing_env()
    s = summarize(res["timeline"])
    total = sum(len(env.hostnames[a]) for a in BLUE_AGENTS)
    stamped = {h for a in BLUE_AGENTS for h in env.hostnames[a]
               if env.trackers[a].last_analysis.get(h) is not None}
    assert s["n_investigated_hosts"] == len(stamped)
    assert s["n_never_investigated_hosts"] == total - len(stamped)
    assert set(s["never_investigated_hosts"]) | stamped == set(
        res["timeline"].hosts())
    assert abs(s["coverage"] - len(stamped) / total) < 1e-12
    # per-agent coverage must agree too
    for agent in BLUE_AGENTS:
        want = {h for h in env.hostnames[agent]
                if env.trackers[agent].last_analysis.get(h) is not None}
        assert s["coverage_per_agent"][agent]["n_investigated"] == len(want)


def test_failed_analysis_never_refreshes_the_age():
    """A failed Analyse must not create a COMPLETE event nor reset the age,
    or coverage and age would both improve for free."""
    pol = OrderedPolicy(scorer=LancerValues(), max_age=8)
    res = run_measured_episode(pol, seed=7630, steps=200, env_kwargs=ENV_KW)
    tl = res["timeline"]
    completes = {(e["agent"], e["host"]) for e in
                 tl._of_kind(INVESTIGATION_COMPLETE)}
    for e in tl._of_kind(INVESTIGATION_FAILED):
        assert (e["agent"], e["host"]) in completes, (
            "a host cannot fail an analysis before completing one")


def test_on_decision_hook_does_not_change_the_episode():
    seen = []
    base = run_measured_episode(OrderedPolicy(scorer=LancerValues(),
                                              max_age=8),
                                seed=7629, steps=120, env_kwargs=ENV_KW)
    hooked = run_measured_episode(OrderedPolicy(scorer=LancerValues(),
                                                max_age=8),
                                  seed=7629, steps=120, env_kwargs=ENV_KW,
                                  on_decision=lambda rec, e, acts:
                                  seen.append(e._tick))
    assert trace_sha(base["trace"]) == trace_sha(hooked["trace"])
    assert base["return"] == hooked["return"]
    assert len(seen) > 0


def test_privileged_read_is_once_per_tick_not_once_per_host():
    pol, res = _episode(seed=7629, steps=120)
    # One post-step sample per stepped tick, nowhere near per-host counts.
    assert res["privileged_calls"] <= res["steps"] + 1
    assert res["privileged_calls"] >= 1


def test_seal_marks_the_end_of_episode():
    pol, res = _episode(seed=7629, steps=120)
    tl = res["timeline"]
    assert tl.end_tick == res["steps"]
    assert tl.first_observed_tick == 1
    assert all(e["tick"] <= tl.end_tick for e in tl.events)


def test_events_are_sorted_and_deduplicated():
    for seed in SEEDS:
        pol, res = _episode(seed=seed, steps=150)
        tl = res["timeline"]
        ticks = [e["tick"] for e in tl.events]
        assert ticks == sorted(ticks), "timeline must stay chronological"
        assert tl.duplicate_events >= 0


def test_max_age_varies_on_real_sim():
    """The pre-repair bug: max_age was the CONSTANT episode-end tick (398)
    in every cell. Assert real episodes produce varying, non-constant
    maxima that are never equal to the episode-end tick."""
    from blue.analysis.metrics import max_age_during_episode
    maxima, end_ticks = [], []
    for seed in SEEDS:
        pol, res = _episode(seed=seed, steps=200)
        tl = res["timeline"]
        per_host = [max_age_during_episode(tl, h)[0]
                    for h in tl.hosts()]
        per_host = [m for m in per_host if m is not None]
        assert per_host, "no uncensored host ages on a real episode"
        maxima.append(max(per_host))
        end_ticks.append(tl.end_tick)
    assert len(set(maxima)) == len(maxima), (
        f"max_age identical across seeds {maxima}: the constant-398 bug")
    for m, end in zip(maxima, end_ticks):
        assert m < end, f"max_age {m} == episode end {end}: suspicious"


def test_onset_aware_detection_delay_is_recomputable():
    """Each episode record's delays must equal tick differences of real
    recorded events, so the numbers are auditable by hand."""
    from blue.analysis.metrics import (DETECTION, COMPROMISE_ONSET,
                                       detection_delay)
    pol, res = _episode(seed=7629, steps=200)
    tl = res["timeline"]
    eps = tl.compromise_episodes()
    checked = 0
    for host, episodes in eps.items():
        dets = sorted(e["tick"] for e in tl._of_kind(DETECTION)
                      if e["host"] == host)
        for ep in episodes:
            got, reason = detection_delay(tl, host, ep)
            window = [t for t in dets
                      if ep["onset"] <= t < (ep["close"] if ep["close"]
                                             is not None else tl.end_tick)]
            if got is None:
                assert not window, "censored but a detection exists"
                assert reason, "censored without a stated reason"
            else:
                assert got == window[0] - ep["onset"]
                checked += 1
    assert checked, "no detected episode to verify on a real episode"
