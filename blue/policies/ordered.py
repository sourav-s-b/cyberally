"""Clean heuristic-guided defender (Phase 1 fresh build).

Same BEHAVIORAL CONTRACT as HybridBluePolicy (rules 1-3 + ordered sweep)
with none of its scar tissue: no fragile feature indices here (features
stay in blue.core), no dead model paths, no cache quirks. Differences:

- Rules, scheduler, and scorers are separate, testable units.
- The sweep runs through an explicit ``SweepScheduler`` with a
  coverage guard: in guard mode, selection is restricted to hosts
  unvisited in the current sweep (resweep when exhausted). Guard OFF
  reproduces the reference behavior bit-exactly (acceptance test);
  guard ON is the learner's training regime (verified separately).
- Scorers share the ``priority_fn(env, agent, host) -> float`` signature,
  so existing scorers plug in unchanged for cross-checks.

Reference for bit-exact verification: HybridBluePolicy with
priority_fn=None (parity) and with LancerPriority(fruitless_decay=0.5)
(lancer). Time-box: 90 min / 20 verify cycles; on failure the fallback
is wrapping the reference rule code (recorded, not silent).
"""

import blue.core.wrapper as wrapper
from blue.core.baselines import action_index

GUARD_OFF = "off"        # reference behaviour (argmax over all legal hosts)
GUARD_STRICT = "strict"  # unvisited-this-sweep restriction + resweep
GUARD_MAX_AGE = "max_age"  # oldest-overdue intervention (new, configurable)


def _legal(mask, env, agent, host, action):
    return bool(mask[action_index(env, agent, host, action)])


def argmax_pick(scored):
    """Canonical argmax with reference tie-break: max by (score, host),
    i.e. ties go to the LARGEST hostname. Every decision path (reference
    scheduler, sampler hooks, greedy eval) must use this; torch argmax
    takes the FIRST candidate and diverges on early-episode init ties."""
    return max(scored, key=lambda sh: (sh[0], sh[1]))[1]


class CursorSweep:
    """Stateless-rotation parity scorer (fresh implementation).

    Carries a per-agent cursor; each call returns the position AFTER the
    given host so the scheduler's argmax-over-unvisited reproduces
    cursor order. Simpler: score = -rotation rank. Equivalent: the
    scheduler asks for the full ordered candidate list instead.
    Implemented as an ordering provider, not a scalar scorer.
    """

    def __init__(self):
        self._cursor = {}

    def reset(self):
        self._cursor = {}

    def order(self, cands, hosts, agent):
        """Candidate hosts in sweep order starting after the cursor."""
        start = self._cursor.get(agent, 0)
        n = len(hosts)
        ranked = []
        for offset in range(n):
            host = hosts[(start + offset) % n]
            if host in cands:
                ranked.append(host)
        return ranked

    def advance(self, agent, hosts, picked):
        self._cursor[agent] = (hosts.index(picked) + 1) % len(hosts)


class LancerValues:
    """Clean-room lancer-style carried values.

    Same dynamics as the reference (init, touch-decay on completed
    actions, detect/novelty boosts, sticky suspicion with
    fruitless-decay) with explicit per-event accounting. Scalar scorer:
    ``score(env, agent, host)`` + ``order()`` helper for the scheduler.
    """

    def __init__(self, init=1.0, touch_decay=0.5, detect_boost=2.0,
                 novelty_boost=1.0, suspicious_bonus=2.0,
                 fruitless_decay=0.5):
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
        from blue.core.obs_features import host_to_vector
        subnets = env.subnets[agent]
        own = (subnets.get(host, []) if isinstance(subnets, dict)
               else subnets)
        vec = host_to_vector(env.views[agent].get(host, {}), agent, own)
        return (vec[6], vec[7], vec[9])

    def _sync(self, env, agent):
        tracker = env.trackers[agent]
        for host in env.hostnames[agent]:
            key = (agent, host)
            last_an = tracker.last_analysis.get(host)
            last_re = tracker.last_remediation.get(host)
            if last_an != self._acct_an.get(key):
                self._acct_an[key] = last_an
                self._v[key] = (self._v.get(key, self.init)
                                * self.touch_decay)
                now = tracker.state.get(host)
                if (self._state.get(key) != "CONFIRMED"
                        and now == "CONFIRMED"):
                    self._fruitless[key] = 0
                else:
                    self._fruitless[key] = self._fruitless.get(key, 0) + 1
            if last_re != self._acct_re.get(key):
                self._acct_re[key] = last_re
                self._v[key] = (self._v.get(key, self.init)
                                * self.touch_decay)
                self._fruitless[key] = 0
            state = tracker.state.get(host)
            if (state == "CONFIRMED"
                    and self._state.get(key) != "CONFIRMED"):
                self._v[key] = (self._v.get(key, self.init)
                                + self.detect_boost)
            self._state[key] = state
            sig = self._signals(env, agent, host)
            old = self._sig.get(key)
            if old is not None and any(s > o for s, o in zip(sig, old)):
                self._v[key] = (self._v.get(key, self.init)
                                + self.novelty_boost)
            self._sig[key] = sig

    def _suspicious(self, env, agent, host):
        unknown, density, n_ext = self._sig.get((agent, host), (0, 0.0, 0))
        state = env.trackers[agent].state.get(host)
        return (unknown > 0 or density > 0.9 or n_ext > 0
                or state in ("CONFIRMED", "VERIFY"))

    def score(self, env, agent, host):
        """Scalar priority (matches priority_fn signature)."""
        self._sync(env, agent)
        value = self._v.get((agent, host), self.init)
        if self._suspicious(env, agent, host):
            value += (self.suspicious_bonus
                      * (self.fruitless_decay
                         ** self._fruitless.get((agent, host), 0)))
        return value

    def __call__(self, env, agent, host):
        return self.score(env, agent, host)


class SweepScheduler:
    """Explicit sweep state with optional coverage guard.

    Guard OFF: pick = argmax score (reference behavior).
    Guard ON: pick = argmax score among hosts unvisited this sweep;
    when all zone hosts are visited, the sweep resets (resweep).
    Completed analyses are observed via tracker stamps (idempotent
    fold); urgent rules bypass the scheduler entirely (they fire first
    in the policy).
    """

    def __init__(self, guard=False):
        self.guard = guard
        self._visited = {}
        self._seen_an = {}

    def reset(self):
        self._visited = {}
        self._seen_an = {}

    def _fold(self, env, agent):
        tracker = env.trackers[agent]
        for host in env.hostnames[agent]:
            last_an = tracker.last_analysis.get(host)
            key = (agent, host)
            if last_an != self._seen_an.get(key):
                self._seen_an[key] = last_an
                if last_an is not None:
                    self._visited.setdefault(agent, set()).add(host)

    def pick(self, env, agent, cands, score_fn=None, order_fn=None,
             choose_fn=None):
        """Return the chosen host from legal sweep candidates.

        Guard filtering applies first (learner regime stays guarded);
        then choose_fn(cands, scored) if given, else order_fn/score
        argmax. choose_fn receives the GUARDED candidate list.
        """
        self._fold(env, agent)
        hosts = env.hostnames[agent]
        if self.guard:
            visited = self._visited.get(agent, set())
            fresh = [h for h in cands if h not in visited]
            if fresh:
                cands = fresh
            else:
                self._visited[agent] = set()  # resweep
        scored = ([(score_fn(env, agent, h), h) for h in cands]
                  if score_fn is not None else None)
        if choose_fn is not None:
            return choose_fn(cands, scored)
        if order_fn is not None:
            ordered = order_fn(cands, hosts, agent)
            return ordered[0] if ordered else None
        return argmax_pick(scored)


class OrderedPolicy:
    """Fresh rules 1-3 + scheduled sweep. Drop-in for HybridBluePolicy.

    ``scorer``: None (use order_fn), a scalar priority_fn, or a
    LancerValues. ``order_fn``: full-ordering provider (parity cursor).
    ``guard``: coverage-guard the sweep (learner regime; OFF for
    bit-exact reference reproduction). Kept as a boolean for historical
    reproduction: True == GUARD_STRICT.
    ``max_age``: when given (positive ticks) the sweep uses the
    :class:`MaxAgeGuard` intervention instead of the strict guard, which
    preserves scorer-driven revisits while bounding investigation age.
    Mutually exclusive with a truthy ``guard``.
    ``record_decisions``: keep the full per-decision candidate/score lists in
    ``last_decision``. Off by default: building them for every sweep decision
    (``~87`` candidates x 5 agents x 400 ticks) roughly doubled episode
    wall time, which is not something a training loop should pay for a
    diagnostic. Counts, the pick, the branch and the guard record are always
    kept.
    """

    def __init__(self, scorer=None, order_fn=None, guard=False, max_age=None,
                 guard_log_limit=200000, record_decisions=False):
        self.scorer = scorer
        self.order_fn = order_fn
        if max_age is not None and guard:
            raise ValueError("max_age and the strict coverage guard are "
                             "mutually exclusive; pick one guard mode")
        self.guard_mode = (GUARD_MAX_AGE if max_age is not None
                           else (GUARD_STRICT if guard else GUARD_OFF))
        self.scheduler = SweepScheduler(guard=(self.guard_mode == GUARD_STRICT))
        self.age_guard = (MaxAgeGuard(max_age, log_limit=guard_log_limit)
                          if self.guard_mode == GUARD_MAX_AGE else None)
        self.record_decisions = bool(record_decisions)
        self._cursor = {}
        self._tick_seen = {}
        # Phase 4 sampler hook: hook(env, agent, cands, scored) -> picked
        # host, where scored = [(score, host)] on the GUARDED candidate
        # list (coverage guard applies before the learner ever sees the
        # choice). None = argmax (reference). Rules 1-3 never consult the
        # hook: only sweep decisions are learner-visible, so PPO records
        # probabilities solely for actions it actually sampled. Parity
        # tests run hook=None (unaffected).
        self.hook = None
        # Last decision record for the current tick (guard/scorer/hook), for
        # decision-trace diagnostics. Read-only; never feeds back.
        self.last_decision = None

    def reset(self):
        self.scheduler.reset()
        if self.age_guard is not None:
            self.age_guard.reset()
        self._cursor.clear()
        self._tick_seen.clear()
        self.last_decision = None
        for obj in (self.scorer, self.order_fn):
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()

    def guard_stats(self):
        return None if self.age_guard is None else dict(self.age_guard.stats)

    def _parity_pick(self, env, agent, cands):
        hosts = env.hostnames[agent]
        start = self._cursor.get(agent, 0)
        for offset in range(len(hosts)):
            host = hosts[(start + offset) % len(hosts)]
            if host in cands:
                self._cursor[agent] = (start + offset + 1) % len(hosts)
                return host
        return None

    def _sweep_cands(self, env, agent, mask):
        return [h for h in env.hostnames[agent]
                if _legal(mask, env, agent, h, "Analyse")]

    def _record(self, env, agent, branch, cands, pick, guard_record=None,
                scores=None, hook_pick=None):
        full = self.record_decisions
        self.last_decision = {
            "tick": env._tick, "agent": agent, "branch": branch,
            "n_cands": len(cands), "pick": pick,
            "guard_record": guard_record,
            "guard_eligible": (guard_record["n_overdue"] if guard_record
                               else 0),
            "hook_pick": hook_pick,
            "truncated": not full,
            "cands": list(cands) if full else None,
            "scores": (None if scores is None or not full
                       else [(float(s), h) for s, h in scores]),
        }
        return pick

    def _finish(self, env, agent, pick, branch, cands, guard_record=None,
                scores=None, hook_pick=None):
        if pick is None:
            return 0
        self._record(env, agent, branch, cands, pick, guard_record, scores,
                     hook_pick)
        return action_index(env, agent, pick, "Analyse")

    def _defer(self, env, agent, mask):
        """An urgent rule took the decision; if the max-age guard has
        overdue hosts, record the deferral (interventions delayed by
        urgent responses) instead of pretending the guard ran."""
        if self.age_guard is None:
            return 0
        return self.age_guard.defer(env, agent, self._sweep_cands(env, agent,
                                                                 mask))

    def select(self, env, agent):
        tick = getattr(env, "_tick", None)
        if tick is not None:
            if tick < self._tick_seen.get(agent, -1):
                self.reset()
            self._tick_seen[agent] = tick
        idx = wrapper.BLUE_AGENTS.index(agent)
        mask = env.get_avail_agent_actions(idx)
        if int(mask.sum()) <= 1:
            return 0
        tracker = env.trackers[agent]
        hosts = env.hostnames[agent]

        # Rule 1+3: remediate CONFIRMED, escalate on re-detection.
        for host in hosts:
            if tracker.state.get(host) != "CONFIRMED":
                continue
            last_an = tracker.last_analysis.get(host)
            last_re = tracker.last_remediation.get(host)
            re_detected = (last_an is not None and last_re is not None
                           and last_an > last_re)
            if re_detected and _legal(mask, env, agent, host, "Restore"):
                self._defer(env, agent, mask)
                return action_index(env, agent, host, "Restore")
            if _legal(mask, env, agent, host, "Remove"):
                self._defer(env, agent, mask)
                return action_index(env, agent, host, "Remove")
            if _legal(mask, env, agent, host, "Restore"):
                self._defer(env, agent, mask)
                return action_index(env, agent, host, "Restore")

        # Rule 2: verify oldest-remediated first.
        verify = [h for h in hosts
                  if tracker.state.get(h) == "VERIFY"
                  and _legal(mask, env, agent, h, "Analyse")]
        if verify:
            host = min(verify,
                       key=lambda h: tracker.last_remediation.get(h) or 0)
            self._defer(env, agent, mask)
            return action_index(env, agent, host, "Analyse")

        # Sweep (guarded or reference).
        cands = self._sweep_cands(env, agent, mask)
        if not cands:
            return 0
        if self.order_fn is not None:
            pick = self.scheduler.pick(env, agent, cands,
                                       order_fn=self.order_fn.order)
            if isinstance(self.order_fn, CursorSweep):
                self.order_fn.advance(agent, hosts, pick)
            return self._finish(env, agent, pick, "order_fn", cands)
        if self.scorer is None:
            # No scorer at all: the parity cursor. Historical behaviour is
            # preserved exactly, including the fact that the strict guard
            # never applied to this configuration.
            if self.guard_mode == GUARD_MAX_AGE:
                pick, guard_record = self.age_guard.pick(env, agent, cands)
                if pick is not None:
                    return self._finish(env, agent, pick, "sweep_max_age",
                                        cands, guard_record=guard_record)
            return self._finish(env, agent,
                                self._parity_pick(env, agent, cands),
                                "parity_cursor", cands)
        score_fn = (self.scorer.score if hasattr(self.scorer, "score")
                    else self.scorer)
        guard_record = None
        if self.guard_mode == GUARD_MAX_AGE:
            pick, guard_record = self.age_guard.pick(
                env, agent, cands, score_fn=score_fn)
            if pick is not None:
                # Score the whole candidate list only when the trace is
                # actually being kept: the guard record already carries the
                # base score of the host it overrode.
                return self._finish(
                    env, agent, pick, "sweep_max_age", cands,
                    guard_record=guard_record,
                    scores=([(score_fn(env, agent, h), h) for h in cands]
                            if self.record_decisions else None))
        if self.hook is not None:
            scored = [(score_fn(env, agent, h), h) for h in cands]
            pick = self.scheduler.pick(
                env, agent, cands, score_fn=score_fn,
                choose_fn=lambda c, s: self.hook(env, agent, c, s))
            return self._finish(env, agent, pick, "sweep_hook", cands,
                                guard_record=guard_record, scores=scored,
                                hook_pick=pick)
        scored = [(score_fn(env, agent, h), h) for h in cands]
        pick = self.scheduler.pick(env, agent, cands, score_fn=score_fn)
        return self._finish(env, agent, pick, "sweep_scorer", cands,
                            guard_record=guard_record, scores=scored)


class MaxAgeGuard:
    """Configurable max-age intervention (Phase 5).

    Behaviour, in order:

    1. Urgent response rules (CONFIRMED remediation / VERIFY re-analysis)
       are handled by :class:`OrderedPolicy` *before* the sweep and keep
       their priority; the guard never pre-empts them. When they fire while
       hosts are overdue the guard records a **deferral**, not an
       intervention.
    2. On an investigation decision the caller passes the currently LEGAL
       hosts. The guard computes each host's age from **completed**
       investigations read out of ``BlueZoneTracker.last_analysis``, which
       is Blue-visible state (the deployed rules already consume it). It
       never reads hidden compromise state, never reads the simulator's
       true state and never reads privileged labels.
    3. If no eligible host exceeds the threshold ``A``, the guard abstains
       and normal scoring runs, including revisits of already-investigated
       hosts.
    4. If some hosts exceed ``A``, the guard selects the oldest overdue
       eligible host with the canonical tie-break (ties go to the LARGEST
       hostname, matching the reference implementation) and records the
       intervention with its cause.

    ``A`` is an **intervention threshold, not a guaranteed maximum age**.
    Busy agents (one pending action per agent, agent-wide), the 2-tick
    Analyse duration, hosts whose Analyse is not currently legal, and the
    urgent rules all cause overshoot. ``stats()`` reports the observed
    overshoot rather than pretending the bound holds.

    Age of a never-investigated host is ``env._tick`` (time since the
    episode's first post-step observation), so a host nobody has looked at
    is always the most overdue. That is deliberate: it is the failure mode
    the intervention exists to prevent.
    """

    def __init__(self, threshold, log_limit=200000):
        threshold = float(threshold)
        if not (threshold > 0):
            raise ValueError("max_age threshold must be a positive number "
                             "of ticks")
        self.threshold = threshold
        self.log_limit = int(log_limit)
        self.reset()

    def reset(self):
        self.log = []
        self.stats = {
            "guard_decisions": 0,
            "interventions": 0,
            "interventions_by_agent": {},
            "deferrals_urgent": 0,
            "deferrals_by_agent": {},
            "overdue_host_decisions": 0,
            "max_overdue_hosts": 0,
            "sum_overdue_hosts": 0,
            "max_observed_age": 0,
            "max_threshold_overshoot": 0,
            "repeated_selection": 0,
            "revisit_selections": 0,
            "sweep_selections": 0,
        }
        self._last_pick = {}
        self._investigated_at_pick = {}

    # -- ages -----------------------------------------------------------
    @staticmethod
    def age_of(env, agent, host):
        """Ticks since the last COMPLETED investigation on ``host``.

        ``env._tick`` for a host never investigated (censored upstream, but
        here the age is well defined: it keeps growing from episode start).
        """
        last = env.trackers[agent].last_analysis.get(host)
        tick = env._tick
        return tick - last if last is not None else tick

    def ages(self, env, agent, cands):
        return {h: self.age_of(env, agent, h) for h in cands}

    def overdue(self, env, agent, cands, ages=None):
        ages = self.ages(env, agent, cands) if ages is None else ages
        return {h: a for h, a in ages.items() if a > self.threshold}

    # -- the decision ---------------------------------------------------
    def pick(self, env, agent, cands, score_fn=None, ages=None):
        """Return ``(pick_or_None, record_or_None)``.

        ``None`` means "abstain, let the scorer decide".
        """
        ages = self.ages(env, agent, cands) if ages is None else ages
        over = {h: a for h, a in ages.items() if a > self.threshold}
        s = self.stats
        s["guard_decisions"] += 1
        if over:
            s["overdue_host_decisions"] += 1
            s["max_overdue_hosts"] = max(s["max_overdue_hosts"], len(over))
            s["sum_overdue_hosts"] += len(over)
            s["max_observed_age"] = max(s["max_observed_age"], max(over.values()))
            s["max_threshold_overshoot"] = max(
                s["max_threshold_overshoot"], max(over.values()) - self.threshold)
            pick = max(over.items(), key=lambda ha: (ha[1], ha[0]))[0]
            if self._last_pick.get(agent) == pick:
                s["repeated_selection"] += 1
            self._last_pick[agent] = pick
            already = env.trackers[agent].last_analysis.get(pick)
            if already is not None:
                s["revisit_selections"] += 1
            s["interventions"] += 1
            s["interventions_by_agent"][agent] = (
                s["interventions_by_agent"].get(agent, 0) + 1)
            s["sweep_selections"] += 1
            record = {
                "tick": env._tick, "agent": agent,
                "cause": "max_age", "threshold": self.threshold,
                "n_candidates": len(cands), "n_overdue": len(over),
                "pick": pick, "picked_age": over[pick],
                "max_overdue_age": max(over.values()),
                "min_overdue_age": min(over.values()),
                "oldest_candidate_age": max(ages.values()),
                "was_revisit": already is not None,
                "base_score": (float(score_fn(env, agent, pick))
                               if score_fn is not None else None),
            }
            if len(self.log) < self.log_limit:
                self.log.append(record)
            return pick, record
        return None, None

    def defer(self, env, agent, cands):
        """Record that an urgent rule took the decision while hosts were
        overdue. Returns the number of overdue hosts (0 = nothing owed)."""
        if not cands:
            return 0
        over = self.overdue(env, agent, cands)
        if over:
            self.stats["deferrals_urgent"] += 1
            self.stats["deferrals_by_agent"][agent] = (
                self.stats["deferrals_by_agent"].get(agent, 0) + 1)
        return len(over)

    def completed_investigation_ages(self, env, agent):
        """Age of every host with a completed investigation, plus the
        never-investigated count. Reported after the episode, never used
        inside the decision."""
        return completed_investigation_age_summary(env, agent)


def completed_investigation_age_summary(env, agent):
    """Module-level so ANY policy arm can report the same age profile.

    Age is ticks since the last COMPLETED investigation (Blue-visible
    ``last_analysis``). Report-only: nothing here feeds a decision.
    """
    tracker = env.trackers[agent]
    ages, never = [], 0
    for host in env.hostnames[agent]:
        last = tracker.last_analysis.get(host)
        if last is None:
            never += 1
        else:
            ages.append(env._tick - last)
    ages.sort()
    return {"n": len(ages), "n_never_investigated": never,
            "median": (ages[len(ages) // 2] if ages else None),
            "p90": (ages[int(round(0.9 * (len(ages) - 1)))] if ages
                    else None),
            "max": ages[-1] if ages else None}


class ResidualAnchor:
    """Phase 4 contract: teacher-bonused residual scorer (no learner yet).

    Wraps a frozen ``score_fn`` as ``score = base(...) + M*tanh(r(...))``
    with fixed ``M > 0``: a zero residual reproduces the base policy
    exactly under argmax, and ``M`` is the single dial bounding how far
    training can drift from the heuristic. Deployment stays masked
    argmax via the scheduler. Merged from the parallel guided.py track
    (bit-exact parity verified there before the merge; canonical track
    is this file + test_ordered_parity.py).

    The learner lives in ``blue.training.residual``; this class only owns
    the ``bonus`` (residual amplitude ``M``) contract so the scoring
    equation has exactly one definition site.
    """

    def __init__(self, base, bonus=1.0):
        self.base = base
        self.bonus = float(bonus)

    def reset(self):
        reset = getattr(self.base, "reset", None)
        if callable(reset):
            reset()

    def score(self, env, agent, host):
        import math
        base = (self.base.score(env, agent, host)
                if hasattr(self.base, "score")
                else self.base(env, agent, host))
        # The residual term is supplied by the learner via set_residual;
        # with no residual attached it is exactly zero (math.tanh(0.0)).
        return base + self.bonus * math.tanh(self._residual(env, agent, host))

    def _residual(self, env, agent, host):
        fn = getattr(self, "residual_fn", None)
        return 0.0 if fn is None else float(fn(env, agent, host))

    def set_residual_fn(self, fn):
        """Attach ``fn(env, agent, host) -> r`` (raw residual, pre-tanh)."""
        self.residual_fn = fn
        return self

    residual_fn = None

    __call__ = score
