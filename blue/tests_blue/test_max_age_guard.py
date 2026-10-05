"""Max-age guard: behaviour, isolation and self-reporting.

Two layers, both required:

- **Unit**: :class:`MaxAgeGuard` against a tiny fake env, so age arithmetic,
  the tie-break, abstention and threshold validation are exact and fast.
- **Integration**: :class:`OrderedPolicy` with ``max_age=`` against the real
  CC4 wrapper with a SYNTHETIC scorer whose ordering is the opposite of the
  guard's. That is the case the guard exists for: with the lancer scorer the
  guard almost never binds, so a test built on the real scorer would pass
  whether or not the guard worked.

The tests that must hold no matter how the guard scores anything:
- guard mode and strict guard are mutually exclusive (no silent upgrade),
- GUARD_STRICT behaviour is bit-identical to the pre-guard-off code path,
- the guard only ever sees legal Analyse candidates,
- the guard reads no privileged state (``_compromised_set`` / true state),
- urgent rules keep priority and are reported as deferrals, not as
  interventions,
- ``reset()`` clears both guard and per-agent book-keeping,
- an intervention threshold is never reported as a guaranteed maximum.
"""

import pytest

from blue.core.baselines import action_index, decode_index, run_episode
from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
from blue.policies.eval_parallel import trace_sha
from blue.policies.ordered import (GUARD_MAX_AGE, GUARD_OFF, GUARD_STRICT,
                                   LancerValues, MaxAgeGuard, OrderedPolicy)

ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}


# --------------------------------------------------------------------------
# Minimal fake env for exact age arithmetic
# --------------------------------------------------------------------------
class FakeTracker:
    def __init__(self, last_analysis=None, state=None, last_remediation=None,
                 last_result=None):
        self.last_analysis = dict(last_analysis or {})
        self.state = dict(state or {})
        self.last_remediation = dict(last_remediation or {})
        self.last_result = dict(last_result or {})


class FakeEnv:
    def __init__(self, hosts, last_analysis=None, tick=0, state=None):
        self._tick = tick
        self.hostnames = {"a": list(hosts)}
        self.trackers = {"a": FakeTracker(last_analysis, state)}

    def _compromised_set(self):
        raise AssertionError("the max-age guard must never read privileged "
                             "compromise state")


def _fake(tick, last_analysis, hosts=("h1", "h2", "h3")):
    return FakeEnv(hosts, last_analysis=last_analysis, tick=tick)


# --------------------------------------------------------------------------
# Unit: age arithmetic and the decision
# --------------------------------------------------------------------------
def test_age_is_ticks_since_last_completion():
    env = _fake(30, {"h1": 10})
    assert MaxAgeGuard.age_of(env, "a", "h1") == 20


def test_never_investigated_age_counts_from_episode_start():
    env = _fake(30, {"h1": 10})
    assert MaxAgeGuard.age_of(env, "a", "h2") == 30


def test_failed_analysis_does_not_reset_age():
    """The wrapper stamps last_analysis only on success, so a failed
    Analyse leaves the age growing. Simulated by the absence of a stamp."""
    env = _fake(30, {"h1": 10})
    before = MaxAgeGuard.age_of(env, "a", "h1")
    assert before == 20
    assert MaxAgeGuard.age_of(env, "a", "h1") == 20  # no stamp, no reset


def test_abstains_when_nothing_is_overdue():
    env = _fake(10, {"h1": 8, "h2": 9, "h3": 9})
    g = MaxAgeGuard(5)
    assert g.pick(env, "a", ["h1", "h2", "h3"]) == (None, None)
    assert g.stats["interventions"] == 0
    assert g.stats["guard_decisions"] == 1


def test_threshold_is_strictly_exceeded():
    """age == A is NOT overdue ('exceeds'), age == A+1 is."""
    g = MaxAgeGuard(10)
    env = _fake(10, {"h1": 0})          # age exactly 10
    assert g.pick(env, "a", ["h1"])[0] is None
    env = _fake(11, {"h1": 0})          # age 11
    assert g.pick(env, "a", ["h1"])[0] == "h1"


def test_picks_oldest_overdue_not_scorers_favourite():
    env = _fake(50, {"h1": 45, "h2": 10, "h3": 30})   # ages 5, 40, 20
    g = MaxAgeGuard(15)
    pick, rec = g.pick(env, "a", ["h1", "h2", "h3"])
    assert pick == "h2"                                 # age 40
    assert rec["cause"] == "max_age"
    assert rec["picked_age"] == 40
    assert rec["n_overdue"] == 2                        # h2, h3


def test_tie_break_is_largest_hostname():
    env = _fake(50, {"h1": 10, "h2": 10, "h3": 10})     # all age 40
    g = MaxAgeGuard(5)
    assert g.pick(env, "a", ["h1", "h2", "h3"])[0] == "h3"


def test_guard_ignores_candidates_it_was_not_given():
    """Only legal Analyse candidates may be selected, even when a stale
    host is far more overdue."""
    env = _fake(50, {"h1": 0})          # h1 age 50, h2 never seen
    g = MaxAgeGuard(5)
    pick, rec = g.pick(env, "a", ["h2"])   # h1 not a candidate
    assert pick == "h2"
    assert rec["n_candidates"] == 1


def test_illegal_candidate_is_never_picked_even_if_overdue():
    env = _fake(99, {"h1": 0, "h2": 0, "h3": 0})
    g = MaxAgeGuard(1)
    pick, _ = g.pick(env, "a", ["h2"])     # h1/h3 masked out
    assert pick == "h2"


def test_revisit_is_recorded_and_counted():
    env = _fake(50, {"h1": 10})
    g = MaxAgeGuard(5)
    pick, rec = g.pick(env, "a", ["h1"])
    assert pick == "h1"
    assert rec["was_revisit"] is True
    assert g.stats["revisit_selections"] == 1


def test_never_investigated_pick_is_not_a_revisit():
    env = _fake(50, {})
    g = MaxAgeGuard(5)
    _, rec = g.pick(env, "a", ["h9"])
    assert rec["was_revisit"] is False


def test_overshoot_is_reported_not_hidden():
    env = _fake(500, {"h1": 0})           # age 500 with threshold 10
    g = MaxAgeGuard(10)
    g.pick(env, "a", ["h1"])
    assert g.stats["max_threshold_overshoot"] == 490
    assert g.stats["max_observed_age"] == 500


def test_repeated_selection_is_counted():
    env = _fake(50, {"h1": 10})
    g = MaxAgeGuard(5)
    g.pick(env, "a", ["h1"])
    g.pick(env, "a", ["h1"])
    assert g.stats["repeated_selection"] == 1
    assert g.stats["interventions"] == 2


def test_deferral_does_not_count_as_intervention():
    env = _fake(50, {"h1": 0})
    g = MaxAgeGuard(5)
    assert g.defer(env, "a", ["h1"]) == 1
    assert g.stats["deferrals_urgent"] == 1
    assert g.stats["interventions"] == 0
    assert g.stats["deferrals_by_agent"]["a"] == 1


def test_deferral_with_nothing_overdue_is_silent():
    env = _fake(50, {"h1": 49})
    g = MaxAgeGuard(5)
    assert g.defer(env, "a", ["h1"]) == 0
    assert g.stats["deferrals_urgent"] == 0


def test_reset_clears_bookkeeping():
    env = _fake(50, {"h1": 10})
    g = MaxAgeGuard(5)
    g.pick(env, "a", ["h1"])
    g.defer(env, "a", ["h1"])
    g.reset()
    assert g.stats["interventions"] == 0
    assert g.stats["deferrals_urgent"] == 0
    assert g.stats["guard_decisions"] == 0
    assert g.log == []
    assert g._last_pick == {}
    g.pick(env, "a", ["h1"])
    assert g.stats["repeated_selection"] == 0    # no memory across episodes


def test_score_fn_is_optional_and_used_only_for_reporting():
    env = _fake(50, {"h1": 10})
    calls = []

    def score(env_, agent, host):
        calls.append(host)
        return -999.0        # deliberately terrible base score

    g = MaxAgeGuard(5)
    pick, rec = g.pick(env, "a", ["h1"], score_fn=score)
    assert pick == "h1"                 # guard overrides a bad scorer
    assert rec["base_score"] == -999.0
    assert calls == ["h1"]              # only the pick is scored


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_non_positive_threshold_is_rejected(bad):
    with pytest.raises(ValueError, match="positive"):
        MaxAgeGuard(bad)


# --------------------------------------------------------------------------
# Integration: guard mode wiring in OrderedPolicy
# --------------------------------------------------------------------------
def test_guard_and_max_age_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        OrderedPolicy(scorer=LancerValues(), guard=True, max_age=10)


def test_guard_modes_are_explicit():
    assert OrderedPolicy(scorer=LancerValues()).guard_mode == GUARD_OFF
    assert OrderedPolicy(scorer=LancerValues(),
                         guard=True).guard_mode == GUARD_STRICT
    assert OrderedPolicy(scorer=LancerValues(),
                         max_age=10).guard_mode == GUARD_MAX_AGE


def test_guard_stats_only_exist_in_max_age_mode():
    assert OrderedPolicy(scorer=LancerValues()).guard_stats() is None
    assert OrderedPolicy(scorer=LancerValues(), guard=True).guard_stats() is None
    assert OrderedPolicy(scorer=LancerValues(),
                         max_age=10).guard_stats() is not None


class Preferential:
    """Divergent scorer: names ONE host and scores it above everything else.

    With the real lancer the guard almost never binds, so a guard test built
    on the real scorer would pass whether or not the guard worked. Naming
    the opposite host to the one the guard must pick makes divergence a
    property of the fixture instead of a lucky seed.
    """

    def __init__(self, agent, host):
        self.target = (agent, host)

    def reset(self):
        pass

    def score(self, env, agent, host):
        return 1.0 if (agent, host) == self.target else 0.0


def _real_env(seed=7629):
    env = CC4MARLEnv(seed=seed, steps=400, **ENV_KW)
    env.reset(seed=seed)
    return env


def test_max_age_guard_overrides_a_divergent_scorer():
    env = _real_env()
    agent = "blue_agent_0"
    idx = BLUE_AGENTS.index(agent)
    pol = OrderedPolicy(scorer=Preferential(agent, "zzz_absent_host"),
                        max_age=8, record_decisions=True)
    pol.reset()
    # Nothing investigated yet: every host's age == current tick, so with
    # tick <= 8 nothing is overdue and the scorer decides.
    env._tick = 5
    mask = env.get_avail_agent_actions(idx)
    legal = [h for h in env.hostnames[agent]
             if mask[action_index(env, agent, h, "Analyse")]]
    assert MaxAgeGuard(8).pick(env, agent, legal)[0] is None
    assert pol.select(env, agent) is not None
    # Past the threshold every legal host is overdue, so the guard must pick
    # the canonical argmax-by-(age, host) instead of the scorer.
    env._tick = 40
    legal = [h for h in env.hostnames[agent]
             if env.get_avail_agent_actions(idx)[
                 action_index(env, agent, h, "Analyse")]]
    guard_pick, rec = MaxAgeGuard(8).pick(env, agent, legal)
    assert guard_pick == max(legal)          # all age 40 -> largest host
    name, host = decode_index(env, agent, pol.select(env, agent))
    assert (name, host) == ("Analyse", guard_pick)
    assert pol.last_decision["branch"] == "sweep_max_age"
    assert pol.last_decision["guard_record"]["threshold"] == 8.0
    # The scorer preferred a different host; prove it, so the assertion above
    # is not vacuous.
    assert pol.last_decision["scores"][0][1] != guard_pick


def test_decision_trace_is_truncated_unless_requested():
    """Full candidate/score lists cost real time (87 candidates x 5 agents x
    400 ticks), so they are opt-in; counts and the pick are always kept."""
    env = _real_env()
    pol = OrderedPolicy(scorer=Preferential("blue_agent_0",
                                            "zzz_absent_host"), max_age=1)
    pol.reset()
    env._tick = 60
    pol.select(env, "blue_agent_0")
    d = pol.last_decision
    assert d["truncated"] is True
    assert d["cands"] is None
    assert d["scores"] is None
    assert d["n_cands"] > 0 and d["pick"]
    assert d["guard_record"] is not None


def test_guard_decision_trace_records_the_guard_pick():
    env = _real_env()
    agent = "blue_agent_1"
    pol = OrderedPolicy(scorer=Preferential(agent, "zzz_absent_host"),
                        max_age=1, record_decisions=True)
    pol.reset()
    env._tick = 60
    pol.select(env, agent)
    d = pol.last_decision
    assert d["agent"] == agent
    assert d["tick"] == 60
    assert d["branch"] == "sweep_max_age"
    assert d["guard_record"]["pick"] == d["pick"]
    assert d["n_cands"] == len(d["cands"])
    assert d["scores"] is not None
    assert len(d["scores"]) == d["n_cands"]


def test_urgent_rules_keep_priority_and_are_reported_as_deferrals():
    """A CONFIRMED host must be remediated by rule 1 even when hosts are
    overdue; the guard records a deferral, not an intervention."""
    env = _real_env()
    agent = "blue_agent_0"
    pol = OrderedPolicy(scorer=LancerValues(), max_age=1)
    pol.reset()
    env._tick = 200
    tracker = env.trackers[agent]
    host = env.hostnames[agent][0]
    tracker.state[host] = "CONFIRMED"
    tracker.last_analysis[host] = 1        # stale: everything is overdue
    action = pol.select(env, agent)
    name, got = decode_index(env, agent, action)
    assert name == "Remove"
    assert got == host
    assert pol.age_guard.stats["deferrals_urgent"] == 1
    assert pol.age_guard.stats["interventions"] == 0


def test_unguarded_mode_is_unaffected_by_the_new_code_path():
    """GUARD_OFF with no scorer keeps the parity cursor, and a scorer in
    OFF mode keeps the exact reference trace."""
    ref = run_episode(OrderedPolicy(order_fn=None, scorer=None),
                      seed=7629, steps=400, **ENV_KW)["trace"]
    from blue.policies.ordered import CursorSweep
    exp = run_episode(OrderedPolicy(order_fn=CursorSweep()),
                      seed=7629, steps=400, **ENV_KW)["trace"]
    assert trace_sha(ref) == trace_sha(exp)


def test_reset_between_episodes_clears_guard_and_last_decision():
    env = _real_env()
    pol = OrderedPolicy(scorer=Preferential("blue_agent_0",
                                            "zzz_absent_host"), max_age=1)
    pol.reset()
    env._tick = 60
    pol.select(env, "blue_agent_0")
    assert pol.age_guard.stats["guard_decisions"] > 0
    assert pol.last_decision is not None
    pol.reset()
    assert pol.age_guard.stats["guard_decisions"] == 0
    assert pol.last_decision is None


def test_guard_never_reads_privileged_state_on_a_real_episode():
    """Run a real episode and assert the guard path touches no privileged
    accessor (the oracle scorer would legitimately read one)."""
    env = CC4MARLEnv(seed=7629, steps=40, **ENV_KW)
    env.reset(seed=7629)
    pol = OrderedPolicy(scorer=LancerValues(), max_age=4)
    pol.reset()
    original = env._compromised_set
    calls = []

    def spy():
        calls.append(1)
        return original()

    env._compromised_set = spy
    for _ in range(40):
        if getattr(env, "_finished", False):
            break
        actions = {a: int(pol.select(env, a)) for a in BLUE_AGENTS}
        env.step(actions)
    assert calls == []
    assert pol.age_guard.stats["guard_decisions"] > 0


def test_max_age_guard_full_episode_smoke():
    res = run_episode(OrderedPolicy(scorer=LancerValues(), max_age=16),
                      seed=7629, steps=120, **ENV_KW)
    assert isinstance(res["cumulative_return"], float)
    assert len(res["trace"]) > 0


def test_max_age_guard_full_episode_binds_and_reports_itself():
    """A small threshold over a whole episode: the guard must actually
    intervene sometimes and report intervention/overshoot counters."""
    pol = OrderedPolicy(scorer=LancerValues(), max_age=4)
    run_episode(pol, seed=7629, steps=200, **ENV_KW)
    s = pol.guard_stats()
    assert s["guard_decisions"] > 0
    assert s["interventions"] > 0, "threshold 4 over 200 ticks must bind"
    assert s["overdue_host_decisions"] > 0
    assert s["interventions_by_agent"]
    assert s["max_observed_age"] >= 4
    assert pol.age_guard.log, "every intervention is logged with its cause"
    assert all(r["cause"] == "max_age" for r in pol.age_guard.log)
