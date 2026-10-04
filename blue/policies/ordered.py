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


def _legal(mask, env, agent, host, action):
    return bool(mask[action_index(env, agent, host, action)])


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

    def pick(self, env, agent, cands, score_fn=None, order_fn=None):
        """Return the chosen host from legal sweep candidates."""
        self._fold(env, agent)
        hosts = env.hostnames[agent]
        if self.guard:
            visited = self._visited.get(agent, set())
            fresh = [h for h in cands if h not in visited]
            if fresh:
                cands = fresh
            else:
                self._visited[agent] = set()  # resweep
        if order_fn is not None:
            ordered = order_fn(cands, hosts, agent)
            return ordered[0] if ordered else None
        scored = [(score_fn(env, agent, h), h) for h in cands]
        return max(scored, key=lambda sh: (sh[0], sh[1]))[1]


class OrderedPolicy:
    """Fresh rules 1-3 + scheduled sweep. Drop-in for HybridBluePolicy.

    ``scorer``: None (use order_fn), a scalar priority_fn, or a
    LancerValues. ``order_fn``: full-ordering provider (parity cursor).
    ``guard``: coverage-guard the sweep (learner regime; OFF for
    bit-exact reference reproduction).
    """

    def __init__(self, scorer=None, order_fn=None, guard=False):
        self.scorer = scorer
        self.order_fn = order_fn
        self.scheduler = SweepScheduler(guard=guard)
        self._cursor = {}
        self._tick_seen = {}

    def reset(self):
        self.scheduler.reset()
        self._cursor.clear()
        self._tick_seen.clear()
        for obj in (self.scorer, self.order_fn):
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()

    def _parity_pick(self, env, agent, cands):
        hosts = env.hostnames[agent]
        start = self._cursor.get(agent, 0)
        for offset in range(len(hosts)):
            host = hosts[(start + offset) % len(hosts)]
            if host in cands:
                self._cursor[agent] = (start + offset + 1) % len(hosts)
                return host
        return None

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
                return action_index(env, agent, host, "Restore")
            if _legal(mask, env, agent, host, "Remove"):
                return action_index(env, agent, host, "Remove")
            if _legal(mask, env, agent, host, "Restore"):
                return action_index(env, agent, host, "Restore")

        # Rule 2: verify oldest-remediated first.
        verify = [h for h in hosts
                  if tracker.state.get(h) == "VERIFY"
                  and _legal(mask, env, agent, h, "Analyse")]
        if verify:
            host = min(verify,
                       key=lambda h: tracker.last_remediation.get(h) or 0)
            return action_index(env, agent, host, "Analyse")

        # Sweep (guarded or reference).
        cands = [h for h in hosts if _legal(mask, env, agent, h, "Analyse")]
        if not cands:
            return 0
        if self.order_fn is not None:
            pick = self.scheduler.pick(env, agent, cands,
                                       order_fn=self.order_fn.order)
            if isinstance(self.order_fn, CursorSweep):
                self.order_fn.advance(agent, hosts, pick)
        elif self.scorer is None:
            pick = self._parity_pick(env, agent, cands)
        else:
            score_fn = (self.scorer.score if hasattr(self.scorer, "score")
                        else self.scorer)
            pick = self.scheduler.pick(env, agent, cands, score_fn=score_fn)
        if pick is None:
            return 0
        return action_index(env, agent, pick, "Analyse")
