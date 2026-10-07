"""Completion-based investigation / detection metrics (Phase 2 repair).

WHY THIS MODULE EXISTS
----------------------
``blue/analysis/ordering.py`` (pre-repair) derived every coverage, age and
delay number from the *requested* action and from a ``setdefault``-once
per-host record. Three independent defects followed:

1. It recorded ``Analyse`` at the tick the action was **issued**. Analyse has
   a simulator duration of 2 ticks and the wrapper stamps completion in
   ``BlueZoneTracker.note_analyse_result`` *after* the step, so every
   recorded analysis tick was 2 ticks early.
2. ``analysed_at.setdefault((agent, host), tick)`` kept only the **first**
   analysis ever issued on a host. A host first analysed before it was
   compromised could therefore never register a later analysis, so it was
   scored "undetected" for the rest of the episode.
3. A **failed** Analyse (wrapper: ``success == "FALSE"`` ->
   ``note_failure``) does not update ``last_analysis``, but the old code
   counted the request anyway, refreshing a successful-investigation age
   that never happened.

Those three are why ``max_age`` was the constant 398 in every cell of the
2026-10-04 manifests and why ``n_undetected`` was ~80 % of
``n_compromised``. Returns are native simulator rewards and are unaffected.

Everything below is computed from **execution outcomes**, never from
requests, and every quantity is a pure function of an explicit event list
so it can be tested without a simulator.

TIME ORIGIN AND PRE/POST-STEP CONVENTION
-----------------------------------------
The canonical tick is the wrapper's post-step ``env._tick``: "tick k" is
the state *after* k completed simulator steps. The wrapper increments
``_tick`` immediately after ``env.step`` and only then consumes the
action result, so a completed Analyse of duration 2 issued during the step
that advances ``_tick`` to ``k`` is stamped ``last_analysis = k + 1``.

Consequences, all deliberate and testable:

- All recorded events live on the same post-step tick axis, so a
  compromise first observed at tick 5 and an analysis completed at tick 5
  give delay 0 (same-tick is not "after").
- An action issued at pre-step tick 3 and completing at post-step tick 5
  counts at tick 5. Issue time is recorded too, but only for
  diagnostics/guard bookkeeping, never for a completion-based metric.
- Episode end is the post-step ``_tick`` of the final step.

HOST UNIVERSE, OWNERSHIP, EXCLUSIONS
------------------------------------
- The **defendable universe** is ``env.hostnames[agent]`` for every Blue
  agent, i.e. hosts a Blue agent can legally act on.
- Privileged ``_compromised_set()`` is network-wide and also contains hosts
  no Blue agent owns (``contractor_network_*``, ``root_internet_host_0``
  at seed 7629). Those are **excluded** from every delay/coverage
  denominator and counted separately as
  ``excluded_non_defendable_compromised_hosts``. Including them inflated
  ``n_compromised`` and ``n_undetected`` in the historical manifests.
- Ownership is not assumed disjoint. ``(agent, host)`` is the identity for
  anything tracker-derived (each agent has its own tracker and its own
  view); ``host`` is the identity for network-wide aggregation, which
  de-duplicates over owners by taking the **latest** completed analysis.
  At seed 7629 the five Blue zones happen to be disjoint (87 hosts, no
  shared host), but the code does not rely on that.

COMPROMISE EPISODES (RECOVERY + REINFECTION)
-------------------------------------------
A compromise *episode* is a maximal interval during which the host appears
in the privileged compromised set. The first observed tick of each episode
is an ``onset``; the first tick of absence is its ``close``. Reinfection is
a new episode with its own onset, so a detection is attributed to exactly
one episode and event-level metrics keep the events distinct. An episode
still open at episode end has ``close = None`` and is analysed to the end.

A host compromised on the very first observed tick is flagged
``initially_compromised``; it is an ordinary episode with onset = that
tick and is counted, not dropped.

CENSORING
---------
An episode with no attributable completed analysis, or no attributable
observable detection, is **censored**: the delay is ``None`` and a reason
string is recorded. Censored episodes are never turned into a zero delay
and are never silently dropped; ``n_censored`` and the reason histogram are
always reported next to every delay summary.

NULL POLICY
-----------
Undefined metrics are ``None`` plus a ``*_reason`` string. A metric is
never coerced to 0 to make an aggregate computable.

WHAT COUNTS AS "DETECTION"
--------------------------
A detection is the Blue-**observable** event ``tracker.state[host] ==
"CONFIRMED"``. That transition happens only inside
``BlueZoneTracker.note_analyse_result`` when a completed Analyse actually
revealed malicious content (``detection_hit``), and the tracker state is
already consumed by the deployed policy's rules 1-3, so it is Blue-visible
and not privileged. It is *not* the same fact as "a post-compromise
analysis happened": the analysis can complete and find nothing. The two
are therefore reported as separate quantities
(``post_compromise_investigation_delay`` and ``detection_delay``) and the
investigation delay is never called "detection".

If only analysis completions were recorded (no tracker state), detection
would be unavailable and must be reported as ``None`` with reason
``"no_observable_detection_channel"`` rather than substituted by the
investigation proxy. ``EpisodeTimeline`` supports that: build it without
``detect`` events and ``detection_delay`` reports the reason.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

# Event kinds.
INVESTIGATION_START = "investigation_start"      # requested (diagnostics)
INVESTIGATION_COMPLETE = "investigation_complete"  # succeeded, stamped
INVESTIGATION_FAILED = "investigation_failed"      # requested, FAILED
REMEDIATION_COMPLETE = "remediation_complete"
DETECTION = "detection"                            # observable CONFIRMED
COMPROMISE_ONSET = "compromise_onset"              # privileged, eval-only
COMPROMISE_CLOSE = "compromise_close"              # privileged, eval-only

CENSORED_NEVER_INVESTIGATED = "never_investigated"
CENSORED_EPISODE_CLOSED = "episode_closed_before_event"
CENSORED_NO_DETECTION_CHANNEL = "no_observable_detection_channel"
CENSORED_NO_DETECTION = "no_detection_before_episode_end"
CENSORED_NO_INVESTIGATION = "no_investigation_before_episode_end"


class EpisodeTimeline:
    """Append-only, de-duplicated event log for one episode.

    Identity is ``(tick, kind, agent, host)``. Re-observing the same
    identity (several wrapper observations arriving in one tick, a stale
    tracker stamp read twice) is counted in ``duplicate_events`` and does
    not create a second completion or detection.
    """

    def __init__(self, universe=None, detection_channel=True):
        # agent -> tuple of hosts that agent can act on (defendable universe)
        self.universe = {a: tuple(h) for a, h in (universe or {}).items()}
        # Whether an observable-detection channel was actually recorded.
        # True means DETECTION events are trustworthy; False means the
        # producer never observed tracker state, so detection must be
        # reported unavailable rather than proxied by the investigation.
        self.detection_channel = bool(detection_channel)
        self.host_owners = defaultdict(list)
        for agent, hosts in self.universe.items():
            for host in hosts:
                self.host_owners[host].append(agent)
        self.events = []
        self._seen = set()
        self.duplicate_events = 0
        self.end_tick = None
        self.first_observed_tick = None

    # -- recording -----------------------------------------------------
    def add(self, tick, kind, host, agent=None, **payload):
        key = (int(tick), kind, agent, host)
        if key in self._seen:
            self.duplicate_events += 1
            return False
        self._seen.add(key)
        self.events.append({"tick": int(tick), "kind": kind, "agent": agent,
                            "host": host, **payload})
        return True

    def seal(self, end_tick):
        self.end_tick = int(end_tick)
        return self

    # -- queries -------------------------------------------------------
    def hosts(self):
        return sorted(self.host_owners)

    def _of_kind(self, kind):
        return [e for e in self.events if e["kind"] == kind]

    def investigations(self, agent=None, host=None):
        """Completed investigations, ascending tick."""
        out = [e for e in self._of_kind(INVESTIGATION_COMPLETE)
               if (agent is None or e["agent"] == agent)
               and (host is None or e["host"] == host)]
        return sorted(out, key=lambda e: (e["tick"], e["agent"] or "",
                                          e["host"]))

    def network_investigations(self, host):
        """Completed investigations on ``host`` across all owning agents,
        de-duplicated by tick so two agents scanning one host cannot double
        count a network-wide completion."""
        by_tick = {}
        for e in self._of_kind(INVESTIGATION_COMPLETE):
            if e["host"] == host:
                by_tick[e["tick"]] = e
        return [by_tick[t] for t in sorted(by_tick)]

    def host_latest_investigation(self, host):
        """(tick, agent) of the latest completed investigation on ``host``
        across owners, or (None, None) if never investigated."""
        best = (None, None)
        for agent in self.host_owners.get(host, ()):
            evs = self.investigations(agent=agent, host=host)
            if evs and (best[0] is None or evs[-1]["tick"] >= best[0]):
                best = (evs[-1]["tick"], agent)
        return best

    def compromise_episodes(self):
        """Maximal privileged-compromise intervals per host.

        Returns ``{host: [{"onset": t, "close": t|None,
                           "initially_compromised": bool}, ...]}``.
        """
        opens = defaultdict(dict)   # host -> {tick: kind}
        for e in self.events:
            if e["kind"] in (COMPROMISE_ONSET, COMPROMISE_CLOSE):
                opens[e["host"]][e["tick"]] = e["kind"]
        first = self.first_observed_tick
        out = {}
        for host, marks in opens.items():
            episodes, cur = [], None
            for tick in sorted(marks):
                if marks[tick] == COMPROMISE_ONSET:
                    if cur is not None:
                        raise ValueError(
                            f"{host}: onset at {tick} inside open episode "
                            f"opened at {cur['onset']}")
                    cur = {"onset": tick, "close": None,
                           "initially_compromised": first is not None
                           and tick == first}
                else:
                    if cur is None:
                        raise ValueError(
                            f"{host}: close at {tick} with no open episode")
                    cur["close"] = tick
                    episodes.append(cur)
                    cur = None
            if cur is not None:
                episodes.append(cur)
            out[host] = episodes
        return out

    def excluded_hosts(self):
        """Compromised hosts that no Blue agent can act on."""
        return sorted(h for h, eps in self.compromise_episodes().items()
                      if eps and h not in self.host_owners)


# ---------------------------------------------------------------------
# Investigation quantities
# ---------------------------------------------------------------------
def first_investigation(tl, host):
    """First **completed** analysis on ``host`` (network-wide), or None."""
    evs = tl.network_investigations(host)
    return evs[0]["tick"] if evs else None


def latest_investigation(tl, host):
    """Most recent **completed** analysis on ``host`` (network-wide)."""
    tick, _ = tl.host_latest_investigation(host)
    return tick


def investigation_age_at(tl, host, tick):
    """Age at post-step ``tick`` = tick - latest completion at or before.

    Returns ``(age, reason)``. ``reason`` is set when the host has no
    completed analysis yet: the age is censored, not 0.
    """
    best = None
    for e in tl.network_investigations(host):
        if e["tick"] <= tick and (best is None or e["tick"] > best):
            best = e["tick"]
    if best is None:
        return None, CENSORED_NEVER_INVESTIGATED
    return tick - best, None


def end_of_episode_age(tl, host, end_tick=None):
    """Episode end minus latest completed analysis on ``host``."""
    end = tl.end_tick if end_tick is None else end_tick
    last = latest_investigation(tl, host)
    if last is None:
        return None, CENSORED_NEVER_INVESTIGATED
    return end - last, None


def max_age_during_episode(tl, host):
    """Largest investigation age observed over all ticks of the episode.

    Distinct from ``end_of_episode_age``: a host analysed early and late
    can end young while being much older in between. Never-investigated
    hosts are censored here (their age grows without bound from tick 0),
    never reported as the numeric maximum.
    """
    evs = tl.network_investigations(host)
    end = tl.end_tick
    if not evs or end is None:
        return None, CENSORED_NEVER_INVESTIGATED
    best = 0
    idx = 0
    last = None
    for tick in range(0, end + 1):
        while idx < len(evs) and evs[idx]["tick"] <= tick:
            last = evs[idx]["tick"]
            idx += 1
        if last is not None:
            best = max(best, tick - last)
    return best, None


# ---------------------------------------------------------------------
# Compromise-episode quantities
# ---------------------------------------------------------------------
def _episode_window(tl, episode):
    onset = episode["onset"]
    close = episode["close"]
    end = onset if close is None else close
    if close is None:
        end = tl.end_tick if tl.end_tick is not None else onset
    return onset, end


def post_compromise_investigation_delay(tl, host, episode):
    """First completed analysis at/after onset, inside this compromise
    episode, minus onset. ``(delay, reason)``; ``(None, reason)`` when
    censored. NOT a detection measure.
    """
    onset, end = _episode_window(tl, episode)
    for e in tl.network_investigations(host):
        if onset <= e["tick"] < end:
            return e["tick"] - onset, None
    reason = (CENSORED_EPISODE_CLOSED if episode["close"] is not None
              else CENSORED_NO_INVESTIGATION)
    return None, reason


def detection_delay(tl, host, episode):
    """First **observable detection** at/after onset inside this episode,
    minus onset. ``(delay, reason)``.

    A detection is a ``DETECTION`` event (tracker CONFIRMED transition).
    With no such event recorded the honest answer is censored/unavailable,
    never the investigation proxy.
    """
    if not tl.detection_channel:
        return None, CENSORED_NO_DETECTION_CHANNEL
    onset, end = _episode_window(tl, episode)
    dets = sorted(e["tick"] for e in tl._of_kind(DETECTION)
                  if e["host"] == host)
    for tick in dets:
        if onset <= tick < end:
            return tick - onset, None
    reason = (CENSORED_EPISODE_CLOSED if episode["close"] is not None
              else CENSORED_NO_DETECTION)
    return None, reason


def episode_records(tl):
    """One record per compromise episode, in the defendable universe only."""
    excluded = set(tl.excluded_hosts())
    out = []
    for host, episodes in sorted(tl.compromise_episodes().items()):
        if host in excluded:
            continue
        for i, ep in enumerate(episodes):
            inv, inv_reason = post_compromise_investigation_delay(
                tl, host, ep)
            det, det_reason = detection_delay(tl, host, ep)
            out.append({
                "host": host, "episode": i, "onset": ep["onset"],
                "close": ep["close"],
                "initially_compromised": ep["initially_compromised"],
                "investigation_delay": inv,
                "investigation_delay_censored_reason": inv_reason,
                "detection_delay": det,
                "detection_delay_censored_reason": det_reason,
            })
    return out


def unresolved_investigations(tl):
    """Investigation requests with no completion AND no failure by episode end.

    Matched per ``(agent, host)``: a start at tick ``s`` is resolved by the
    first completion/failure of the SAME agent and host at a tick ``> s``
    (Analyse latency is positive, so a same-tick resolution cannot belong to
    this request). A global ``started - completed - failed`` subtraction is
    wrong -- completions can belong to different hosts than the starts.
    Returns ``(count, detail)``.
    """
    resolutions = {}
    for kind in (INVESTIGATION_COMPLETE, INVESTIGATION_FAILED):
        for e in tl._of_kind(kind):
            resolutions.setdefault((e["agent"], e["host"]), []).append(e["tick"])
    for pair in resolutions:
        resolutions[pair].sort()
    by_pair = {}
    for e in tl._of_kind(INVESTIGATION_START):
        by_pair.setdefault((e["agent"], e["host"]), []).append(e["tick"])
    detail = []
    for pair, starts in sorted(by_pair.items()):
        pending = list(resolutions.get(pair, []))
        for s in sorted(starts):
            take = next((i for i, t in enumerate(pending) if t > s), None)
            if take is None:
                detail.append({"agent": pair[0], "host": pair[1],
                               "start_tick": s})
            else:
                pending.pop(take)
    return len(detail), detail


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def _describe(values, reasons):
    """``values`` and ``reasons`` are parallel lists; a ``None`` value is
    censored and its reason string is counted, never replaced by 0."""
    pairs = list(zip(values, reasons))
    vals = sorted(v for v, _ in pairs if v is not None)
    return {
        "n": len(vals),
        "n_censored": sum(1 for v, _ in pairs if v is None),
        "censored_reasons": dict(Counter(
            r for v, r in pairs if v is None and r)),
        "mean": statistics.fmean(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "p90": (vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))]
                if vals else None),
        "max": vals[-1] if vals else None,
        "min": vals[0] if vals else None,
    }


def summarize(tl, end_tick=None, extra=None):
    """Full metric record for one episode.

    Every delay/age aggregate carries its own ``n`` **and** ``n_censored``
    so a reader can see how much of the population the number describes.
    """
    end = tl.end_tick if end_tick is None else end_tick
    hosts = tl.hosts()
    n_hosts = len(hosts)

    investigated = [h for h in hosts if first_investigation(tl, h) is not None]
    never = [h for h in hosts if h not in investigated]

    end_ages, end_reasons = [], []
    for h in hosts:
        age, reason = end_of_episode_age(tl, h, end)
        end_ages.append(age)
        end_reasons.append(reason)
    max_ages, max_reasons = [], []
    for h in hosts:
        age, reason = max_age_during_episode(tl, h)
        max_ages.append(age)
        max_reasons.append(reason)

    records = episode_records(tl)
    inv_desc = _describe([r["investigation_delay"] for r in records],
                         [r["investigation_delay_censored_reason"]
                          for r in records])
    det_desc = _describe([r["detection_delay"] for r in records],
                         [r["detection_delay_censored_reason"]
                          for r in records])

    started = len(tl._of_kind(INVESTIGATION_START))
    completed = len(tl._of_kind(INVESTIGATION_COMPLETE))
    failed = len(tl._of_kind(INVESTIGATION_FAILED))
    remediations = len(tl._of_kind(REMEDIATION_COMPLETE))
    detections = len(tl._of_kind(DETECTION))
    per_agent = {}
    for agent in tl.universe:
        ah = set(tl.universe[agent])
        done = {h for h in ah if tl.investigations(agent=agent, host=h)}
        per_agent[agent] = {
            "n_hosts": len(ah),
            "n_investigated": len(done),
            "coverage": len(done) / len(ah) if ah else None,
        }

    unresolved, unresolved_detail = unresolved_investigations(tl)
    out = {
        "end_tick": end,
        "n_defendable_hosts": n_hosts,
        "n_investigated_hosts": len(investigated),
        "n_never_investigated_hosts": len(never),
        "never_investigated_hosts": sorted(never),
        "coverage": len(investigated) / n_hosts if n_hosts else None,
        "coverage_per_agent": per_agent,
        "investigations_started": started,
        "investigations_completed": completed,
        "investigations_failed": failed,
        "investigations_unresolved_at_end": unresolved,
        "investigations_unresolved_detail": unresolved_detail,
        "remediations_completed": remediations,
        "detections_observed": detections,
        "duplicate_events_suppressed": tl.duplicate_events,
        "excluded_non_defendable_compromised_hosts": tl.excluded_hosts(),
        "n_compromise_episodes": len(records),
        "n_initially_compromised_episodes": sum(
            1 for r in records if r["initially_compromised"]),
        "post_compromise_investigation_delay": inv_desc,
        "detection_delay": det_desc,
        "end_of_episode_age": _describe(end_ages, end_reasons),
        "max_age_during_episode": _describe(max_ages, max_reasons),
        "episode_records": records,
    }
    if extra:
        out.update(extra)
    return out