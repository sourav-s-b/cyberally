"""Heuristic-guided Blue agent, clean reimplementation (Phase 1).

Fresh build of the rules + scorer-interface architecture (see Phase -1
decision and the lancer audit): fixed evidence-gated response rules that
a learner can never violate, with exactly one learnable decision — the
sweep ordering. Bit-exact trace compatibility with
``blue.policies.hybrid.HybridBluePolicy(priority_fn='lancer',
fruitless_decay=0.5)`` (== hybrid_lancer_v2) is the acceptance gate
(tests_blue/test_guided_parity.py); any divergence there is a bug here,
not a design difference.

Rules (audited semantics, relabeled per lancer-audit-20261004):
  1+3. CONFIRMED hosts are remediated; a post-remediation re-detection
       escalates Remove -> Restore.
  2.   VERIFY hosts are re-analysed, oldest remediation first.
  sweep. Remaining Analyse candidates are ordered by ``priority_fn``;
       None reproduces the cursor parity sweep.

Phase 4 anchor contract (specified here, built there): the learned
residual adds a teacher logit bonus M to this policy's scores and its
output is clipped, so a zero residual reproduces this policy exactly
under argmax. ``ResidualAnchor`` below records that contract; the
learner arrives in Phase 4.
"""

from blue.core.baselines import action_index
from blue.core.wrapper import BLUE_AGENTS


def _legal(mask, env, agent, host, action):
    return bool(mask[action_index(env, agent, host, action)])


class GuidedBluePolicy:
    """Evidence-gated defender with a pluggable host-priority scorer."""

    def __init__(self, priority_fn=None):
        self.priority_fn = priority_fn
        self._cursor = {}
        self._tick_seen = {}

    def reset(self):
        self._cursor.clear()
        self._tick_seen.clear()
        if hasattr(self.priority_fn, "reset"):
            self.priority_fn.reset()

    def select(self, env, agent):
        tick = getattr(env, "_tick", None)
        if tick is not None:
            if tick < self._tick_seen.get(agent, -1):
                self.reset()
            self._tick_seen[agent] = tick
        idx = BLUE_AGENTS.index(agent)
        mask = env.get_avail_agent_actions(idx)
        if int(mask.sum()) <= 1:
            return 0
        tracker = env.trackers[agent]
        hosts = env.hostnames[agent]

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

        verify = [h for h in hosts if tracker.state.get(h) == "VERIFY"
                  and _legal(mask, env, agent, h, "Analyse")]
        if verify:
            host = min(verify,
                       key=lambda h: tracker.last_remediation.get(h) or 0)
            return action_index(env, agent, host, "Analyse")

        cands = [h for h in hosts if _legal(mask, env, agent, h, "Analyse")]
        if not cands:
            return 0
        if self.priority_fn is None:
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
        return action_index(env, agent, pick, "Analyse")


class GuidedPriority:
    """Per-host scan priority with lancer touch dynamics, reimplemented.

    Carried value per (agent, host): ``init``, multiplicatively decayed
    once per completed Analyse or remediation (tracker-stamp accounting),
    boosted on CONFIRMED transitions and novel suspicious view signals,
    plus a sticky suspicion bonus decaying with consecutive fruitless
    re-analyses. Plain floats/dicts (picklable); ``reset()`` clears.
    """

    DENSITY_THRESHOLD = 0.9

    def __init__(self, init=1.0, touch_decay=0.5, detect_boost=2.0,
                 novelty_boost=1.0, suspicious_bonus=2.0,
                 fruitless_decay=0.5):
        self.init = init
        self.touch_decay = touch_decay
        self.detect_boost = detect_boost
        self.novelty_boost = novelty_boost
        self.suspicious_bonus = suspicious_bonus
        self.fruitless_decay = fruitless_decay
        self._v = {}
        self._acct_an = {}
        self._acct_re = {}
        self._state = {}
        self._sig = {}
        self._fruitless = {}

    def reset(self):
        self._v = {}
        self._acct_an = {}
        self._acct_re = {}
        self._state = {}
        self._sig = {}
        self._fruitless = {}

    def _signals(self, env, agent, host):
        from blue.policies.hybrid import host_risk_features
        vec = host_risk_features(env, agent, host)
        return (vec[6], vec[7], vec[9])

    def _sync(self, env, agent):
        tracker = env.trackers[agent]
        for host in env.hostnames[agent]:
            key = (agent, host)
            last_an = tracker.last_analysis.get(host)
            last_re = tracker.last_remediation.get(host)
            if last_an != self._acct_an.get(key):
                self._acct_an[key] = last_an
                self._v[key] = self._v.get(key, self.init) * self.touch_decay
                now = tracker.state.get(host)
                if (self._state.get(key) != "CONFIRMED"
                        and now == "CONFIRMED"):
                    self._fruitless[key] = 0
                else:
                    self._fruitless[key] = self._fruitless.get(key, 0) + 1
            if last_re != self._acct_re.get(key):
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
        return (unknown_files > 0 or density > self.DENSITY_THRESHOLD
                or n_ext > 0 or state in ("CONFIRMED", "VERIFY"))

    def __call__(self, env, agent, host):
        self._sync(env, agent)
        value = self._v.get((agent, host), self.init)
        if self._suspicious_now(env, agent, host):
            value += (self.suspicious_bonus
                      * (self.fruitless_decay
                         ** self._fruitless.get((agent, host), 0)))
        return value


class ResidualAnchor:
    """Phase 4 contract placeholder (no learner yet).

    The learned residual ``r(env, agent, host)`` combines with a frozen
    scorer ``base`` as ``score = base(...) + M * tanh(r(...))`` with
    ``M > 0`` fixed: a zero residual reproduces the base policy exactly
    under argmax, and ``M`` is the single dial bounding how far training
    can drift from the heuristic. Deployment stays masked argmax.
    """

    def __init__(self, base, bonus=1.0):
        self.base = base
        self.bonus = float(bonus)

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()

    def __call__(self, env, agent, host):
        import math
        return self.base(env, agent, host) + self.bonus * math.tanh(0.0)
