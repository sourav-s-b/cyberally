"""Deterministic synthetic-timeline tests for the repaired metric semantics.

No simulator, no randomness: each test builds an explicit event list and
asserts one documented property. These are the regression gate for the
three pre-repair defects (issue-tick instead of completion-tick, first
analysis only via ``setdefault``, failed analyses counted as
investigations) and for the null/censoring policy.

Test numbering follows the session plan cases 1-9 plus the explicit
final-age vs maximum-age distinction.
"""

import pytest

from blue.analysis.metrics import (CENSORED_NO_DETECTION,
                                   CENSORED_NO_INVESTIGATION,
                                   CENSORED_NEVER_INVESTIGATED,
                                   CENSORED_NO_DETECTION_CHANNEL,
                                   COMPROMISE_CLOSE, COMPROMISE_ONSET,
                                   DETECTION, EpisodeTimeline,
                                   INVESTIGATION_COMPLETE,
                                   INVESTIGATION_FAILED,
                                   INVESTIGATION_START,
                                   REMEDIATION_COMPLETE, detection_delay,
                                   end_of_episode_age,
                                   episode_records, first_investigation,
                                   investigation_age_at,
                                   latest_investigation,
                                   max_age_during_episode,
                                   post_compromise_investigation_delay,
                                   summarize)

H = "zone_a_subnet_user_host_0"
H2 = "zone_a_subnet_user_host_1"


def timeline(universe=None, end_tick=None, first_tick=1):
    tl = EpisodeTimeline(universe or {"blue_agent_0": (H, H2)})
    tl.first_observed_tick = first_tick
    if end_tick is not None:
        tl.seal(end_tick)
    return tl


# --- case 1: post-compromise investigation delay ---------------------
def test_case1_post_compromise_investigation_delay_is_three():
    """Analysis completes at 2, compromise onset at 5, next completion at
    8 -> post-compromise investigation delay 3."""
    tl = timeline(end_tick=10)
    tl.add(2, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(5, COMPROMISE_ONSET, H)
    tl.add(8, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    ep = tl.compromise_episodes()[H][0]
    delay, reason = post_compromise_investigation_delay(tl, H, ep)
    assert delay == 3
    assert reason is None


def test_case1b_same_tick_onset_and_completion_is_zero_not_negative():
    """Documented convention: both events live on the post-step axis, so
    same-tick is delay 0."""
    tl = timeline(end_tick=10)
    tl.add(5, COMPROMISE_ONSET, H)
    tl.add(5, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    ep = tl.compromise_episodes()[H][0]
    assert post_compromise_investigation_delay(tl, H, ep)[0] == 0


# --- case 2: final age ------------------------------------------------
def test_case2_final_age_is_two():
    """Analyses complete at 2 and 8, episode ends at 10 -> final age 2."""
    tl = timeline(end_tick=10)
    tl.add(2, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(8, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    assert latest_investigation(tl, H) == 8
    assert end_of_episode_age(tl, H) == (2, None)


def test_final_age_differs_from_max_age_during_episode():
    """The same timeline: final age 2, but the host was 5 ticks stale at
    tick 7. The two quantities are not interchangeable."""
    tl = timeline(end_tick=10)
    tl.add(2, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(8, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    final, _ = end_of_episode_age(tl, H)
    peak, reason = max_age_during_episode(tl, H)
    assert final == 2
    assert peak == 5
    assert reason is None
    assert peak > final


# --- case 3: censored, never a zero delay -----------------------------
def test_case3_compromise_with_no_later_analysis_is_censored():
    tl = timeline(end_tick=20)
    tl.add(2, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(5, COMPROMISE_ONSET, H)
    ep = tl.compromise_episodes()[H][0]
    delay, reason = post_compromise_investigation_delay(tl, H, ep)
    assert delay is None
    assert reason == CENSORED_NO_INVESTIGATION
    rec = episode_records(tl)[0]
    assert rec["investigation_delay"] is None
    assert rec["investigation_delay_censored_reason"]
    # never coerced to zero and never dropped: n == 0, n_censored == 1
    summary = summarize(tl)
    d = summary["post_compromise_investigation_delay"]
    assert d["n"] == 0 and d["n_censored"] == 1


def test_case3b_detection_censored_is_not_replaced_by_investigation():
    """A post-compromise analysis exists but reveals nothing: the
    investigation delay is 1, the detection delay is censored. The proxy
    is never reported as detection."""
    tl = timeline(end_tick=20)
    tl.add(5, COMPROMISE_ONSET, H)
    tl.add(6, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    ep = tl.compromise_episodes()[H][0]
    assert post_compromise_investigation_delay(tl, H, ep)[0] == 1
    det, det_reason = detection_delay(tl, H, ep)
    assert det is None
    assert det_reason == CENSORED_NO_DETECTION


def test_case3c_no_detection_channel_reports_unavailable():
    """With no observable detection channel the honest answer is
    'unavailable', not the investigation proxy."""
    tl = EpisodeTimeline({"blue_agent_0": (H,)}, detection_channel=False)
    tl.first_observed_tick = 1
    tl.seal(20)
    tl.add(5, COMPROMISE_ONSET, H)
    tl.add(6, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    ep = tl.compromise_episodes()[H][0]
    delay, reason = detection_delay(tl, H, ep)
    assert delay is None
    assert reason == CENSORED_NO_DETECTION_CHANNEL


# --- case 4: failed analysis does not refresh age ---------------------
def test_case4_failed_analysis_does_not_refresh_successful_age():
    tl = timeline(end_tick=10)
    tl.add(2, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(6, INVESTIGATION_FAILED, H, agent="blue_agent_0")
    assert latest_investigation(tl, H) == 2
    assert end_of_episode_age(tl, H) == (8, None)
    assert investigation_age_at(tl, H, 10)[0] == 8


# --- case 5: started before, completed after --------------------------
def test_case5_completion_semantics_for_a_slow_analysis():
    """Requested at tick 3 (duration 2), completes at 5. Compromise onset
    at 4. The completion-based delay is 1; using the request tick would
    have produced a negative delay."""
    tl = timeline(end_tick=10)
    tl.add(3, INVESTIGATION_START, H, agent="blue_agent_0")
    tl.add(4, COMPROMISE_ONSET, H)
    tl.add(5, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    ep = tl.comprise = tl.compromise_episodes()[H][0]
    delay, reason = post_compromise_investigation_delay(tl, H, ep)
    assert delay == 1 and reason is None
    starts = [e for e in tl.events if e["kind"] == INVESTIGATION_START]
    assert starts and starts[0]["tick"] == 3
    assert investigation_age_at(tl, H, 4) == (None,
                                              CENSORED_NEVER_INVESTIGATED)


def test_case5b_completion_before_onset_does_not_count():
    """An analysis that COMPLETED before the compromise cannot be the
    post-compromise investigation."""
    tl = timeline(end_tick=10)
    tl.add(3, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(5, COMPROMISE_ONSET, H)
    ep = tl.compromise_episodes()[H][0]
    assert post_compromise_investigation_delay(tl, H, ep)[0] is None


# --- case 6: recovery then reinfection --------------------------------
def test_case6_recovery_then_reinfection_keeps_events_distinct():
    tl = timeline(end_tick=30)
    tl.add(5, COMPROMISE_ONSET, H)
    tl.add(9, DETECTION, H, agent="blue_agent_0")
    tl.add(12, REMEDIATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(14, COMPROMISE_CLOSE, H)
    tl.add(20, COMPROMISE_ONSET, H)          # reinfection
    tl.add(24, DETECTION, H, agent="blue_agent_0")
    eps = tl.compromise_episodes()[H]
    assert len(eps) == 2
    assert eps[0]["onset"] == 5 and eps[0]["close"] == 14
    assert eps[1]["onset"] == 20 and eps[1]["close"] is None
    assert post_compromise_investigation_delay(tl, H, eps[0])[0] is None
    assert detection_delay(tl, H, eps[0])[0] == 4
    # the second episode's detection is not attributed to the first
    assert detection_delay(tl, H, eps[1])[0] == 4
    records = episode_records(tl)
    assert [r["episode"] for r in records] == [0, 1]
    assert records[0]["detection_delay"] == 4
    assert records[1]["onset"] == 20


def test_case6b_initially_compromised_is_flagged_not_dropped():
    tl = timeline(end_tick=20, first_tick=1)
    tl.add(1, COMPROMISE_ONSET, H)
    eps = tl.compromise_episodes()[H]
    assert eps[0]["initially_compromised"] is True
    assert summarize(tl)["n_initially_compromised_episodes"] == 1


# --- case 7: never-investigated host ----------------------------------
def test_case7_never_investigated_host_age_and_coverage():
    tl = timeline(end_tick=10)
    tl.add(4, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    age, reason = end_of_episode_age(tl, H2)
    assert age is None and reason == CENSORED_NEVER_INVESTIGATED
    peak, reason = max_age_during_episode(tl, H2)
    assert peak is None and reason == CENSORED_NEVER_INVESTIGATED
    s = summarize(tl)
    assert s["coverage"] == 0.5
    assert s["n_investigated_hosts"] == 1
    assert s["n_never_investigated_hosts"] == 1
    assert s["never_investigated_hosts"] == [H2]
    # the censored host must not silently become the numeric maximum
    assert s["max_age_during_episode"]["n"] == 1
    assert s["max_age_during_episode"]["n_censored"] == 1
    assert s["end_of_episode_age"]["max"] == 6


# --- case 8: several observations per tick ---------------------------
def test_case8_duplicate_observations_do_not_double_count():
    tl = timeline(end_tick=10)
    for _ in range(4):
        tl.add(5, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
        tl.add(5, DETECTION, H, agent="blue_agent_0")
    tl.add(5, COMPROMISE_ONSET, H)
    assert len(tl.investigations(host=H)) == 1
    assert len(tl._of_kind(DETECTION)) == 1
    assert tl.duplicate_events == 6
    s = summarize(tl)
    assert s["investigations_completed"] == 1
    assert s["detections_observed"] == 1
    assert s["duplicate_events_suppressed"] == 6
    ep = tl.compromise_episodes()[H][0]
    assert post_compromise_investigation_delay(tl, H, ep)[0] == 0
    assert detection_delay(tl, H, ep)[0] == 0


# --- case 9: two agents referencing one host --------------------------
def test_case9_shared_host_not_double_counted_network_wide():
    tl = EpisodeTimeline({"blue_agent_0": (H,), "blue_agent_1": (H, H2)})
    tl.first_observed_tick = 1
    tl.seal(12)
    # both agents complete an investigation on H in the same tick
    tl.add(3, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(3, INVESTIGATION_COMPLETE, H, agent="blue_agent_1")
    tl.add(3, INVESTIGATION_COMPLETE, H2, agent="blue_agent_1")
    # per-(agent,host) tracking sees both; network-wide sees one tick
    assert len(tl.investigations(agent="blue_agent_0", host=H)) == 1
    assert len(tl.investigations(host=H)) == 2
    assert len(tl.network_investigations(H)) == 1
    s = summarize(tl)
    assert s["n_defendable_hosts"] == 2            # H counted once
    assert s["n_investigated_hosts"] == 2
    assert s["coverage"] == 1.0
    assert s["coverage_per_agent"]["blue_agent_0"]["coverage"] == 1.0
    assert s["coverage_per_agent"]["blue_agent_1"]["coverage"] == 1.0
    assert end_of_episode_age(tl, H) == (9, None)


def test_case9b_shared_host_ages_use_latest_completion():
    tl = EpisodeTimeline({"blue_agent_0": (H,), "blue_agent_1": (H,)})
    tl.first_observed_tick = 1
    tl.seal(20)
    tl.add(4, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(11, INVESTIGATION_COMPLETE, H, agent="blue_agent_1")
    assert latest_investigation(tl, H) == 11
    assert end_of_episode_age(tl, H) == (9, None)


# --- exclusion of non-defendable compromised hosts ---------------------
def test_non_defendable_compromised_hosts_excluded_from_delays():
    tl = EpisodeTimeline({"blue_agent_0": (H,)})
    tl.first_observed_tick = 1
    tl.seal(10)
    tl.add(2, COMPROMISE_ONSET, "contractor_network_subnet_user_host_0")
    tl.add(3, COMPROMISE_ONSET, H)
    tl.add(5, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    s = summarize(tl)
    assert s["excluded_non_defendable_compromised_hosts"] == [
        "contractor_network_subnet_user_host_0"]
    assert s["n_compromise_episodes"] == 1
    assert s["post_compromise_investigation_delay"]["n"] == 1


# --- first vs latest investigation ------------------------------------
def test_first_and_latest_investigation_are_separate_quantities():
    tl = timeline(end_tick=10)
    tl.add(2, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(8, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    assert first_investigation(tl, H) == 2
    assert latest_investigation(tl, H) == 8
    assert first_investigation(tl, H2) is None


# --- malformed event streams fail loudly ------------------------------
def test_overlapping_compromise_episodes_are_rejected():
    tl = timeline(end_tick=10)
    tl.add(2, COMPROMISE_ONSET, H)
    tl.add(5, COMPROMISE_ONSET, H)      # onset while already open
    with pytest.raises(ValueError, match="inside open episode"):
        tl.compromise_episodes()


def test_unresolved_investigation_is_reported_not_hidden():
    """A start on H is NOT resolved by a completion on a different host.

    A global started-completed-failed subtraction would report 0 here and
    hide the pending request.
    """
    tl = timeline(end_tick=10)
    tl.add(3, INVESTIGATION_START, H, agent="blue_agent_0")
    tl.add(6, INVESTIGATION_COMPLETE, H2, agent="blue_agent_0")
    s = summarize(tl)
    assert s["investigations_started"] == 1
    assert s["investigations_completed"] == 1
    assert s["investigations_unresolved_at_end"] == 1
    assert s["investigations_unresolved_detail"] == [
        {"agent": "blue_agent_0", "host": H, "start_tick": 3}]


def test_failed_request_resolves_its_own_start():
    tl = timeline(end_tick=10)
    tl.add(3, INVESTIGATION_START, H, agent="blue_agent_0")
    tl.add(6, INVESTIGATION_FAILED, H, agent="blue_agent_0")
    s = summarize(tl)
    assert s["investigations_failed"] == 1
    assert s["investigations_unresolved_at_end"] == 0


def test_repeated_requests_are_matched_greedily_not_by_count():
    """Three starts, two completions -> exactly one unresolved, and the
    unresolved one is the LATEST request."""
    tl = timeline(end_tick=20)
    tl.add(2, INVESTIGATION_START, H, agent="blue_agent_0")
    tl.add(5, INVESTIGATION_START, H, agent="blue_agent_0")
    tl.add(9, INVESTIGATION_START, H, agent="blue_agent_0")
    tl.add(4, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    tl.add(7, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    s = summarize(tl)
    assert s["investigations_started"] == 3
    assert s["investigations_completed"] == 2
    assert s["investigations_unresolved_at_end"] == 1
    assert s["investigations_unresolved_detail"] == [
        {"agent": "blue_agent_0", "host": H, "start_tick": 9}]


def test_resolution_at_same_tick_does_not_resolve_the_request():
    """Analyse latency is positive, so a same-tick 'completion' belongs to
    an earlier request, never to this one."""
    tl = timeline(end_tick=10)
    tl.add(4, INVESTIGATION_START, H, agent="blue_agent_0")
    tl.add(4, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    assert summarize(tl)["investigations_unresolved_at_end"] == 1
    tl2 = timeline(end_tick=10)
    tl2.add(1, INVESTIGATION_START, H, agent="blue_agent_0")
    tl2.add(4, INVESTIGATION_COMPLETE, H, agent="blue_agent_0")
    assert summarize(tl2)["investigations_unresolved_at_end"] == 0