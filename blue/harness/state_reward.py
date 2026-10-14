"""State-based training reward shaping for the harness.

Why this exists. The native Blue reward is computed entirely from *other*
agents: ``BlueRewardMachine`` scores Green ``LocalWork`` failure (LWF), Green
``AccessService`` failure (ASF) and successful Red ``Impact`` (RIA). Blue's
own actions never score. So the quantity the harness optimises is sparse,
delayed, shared across all five Blue agents, and driven by an adversary the
policy cannot observe.

The October 2026 shaping attempts scored *actions* -- +1 per Analyse that
found something, +3 per Remove that evicted an attacker. Those are dense
per-action terms, and a dense term mixed with a sparse true signal biases the
gradient toward farming the term: shaped training return climbed -435 -> -68
while native return got worse. Both runs degraded.

These terms are functions of STATE, not of action counts:

  ``detection_debt``  per tick, severity x (hosts Red has compromised that
                      Blue has not yet had CONFIRMED). Reduced only by
                      investigating the right host soon enough.
  ``confirm_debt``    per tick, severity x (hosts Blue has CONFIRMED and not
                      yet remediated).

Neither can be farmed by acting more: you cannot clear debt without changing
the state that creates it. The harness hook chooses which host to Analyse and
nothing else -- remediation and verification are owned by fixed rules -- so
these two terms are exactly the levers it controls.

Severities come from the **public** CC4 phase schedule, not from privileged
state. The only privileged read is the compromise bit, obtained through
``CC4MARLEnv._compromised_set()``, which the wrapper already documents as
reward-channel only. Nothing here may be read by an actor, a mask or a
feature builder.

Evaluation must never use this module. ``harness_eval`` scores native reward
only, and a test enforces that it does not import this file. The shaped
signal reaches training solely through ``mappo_guide.run_team_episode``'s
``reward_hook`` argument, which leaves the reported ``"return"`` native.
"""
from __future__ import annotations

import math

VERSION = "blue-harness-state-reward-v1"

TERMS = ("detection", "confirm", "both")

# The CC4 phase schedule splits the horizon three ways, remainder assigned to
# the earlier phases. Mirrors blue/harness/features.py exactly so the shaping
# severity and the actor's phase one-hot can never disagree.
def mission_phase(tick, horizon):
    if not isinstance(horizon, int) or horizon < 3:
        raise ValueError("horizon must be an int >= 3 for three phases")
    q, r = divmod(horizon, 3)
    bounds = (q + (r > 0), 2 * q + (r > 0) + (r > 1))
    return int(tick >= bounds[0]) + int(tick >= bounds[1])


def _phase_schedule():
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
    return tuple(BlueRewardMachine("Blue").get_phase_rewards(p)
                 for p in range(3))


_SCHEDULE = None


def zone_severity(host, phase):
    """Public mission severity of a host, in [0, 1].

    The largest penalty magnitude the published schedule can charge in this
    host's zone and mission phase, normalised by 10 (the steepest entry in
    the table). A host in ``operational_zone_a`` during phase 1 scores 1.0;
    a contractor-network host in phase 2 scores 0.0.
    """
    global _SCHEDULE
    if _SCHEDULE is None:
        # Blue's wrapper establishes the pinned simulator import path.
        import blue.core.wrapper  # noqa: F401
        _SCHEDULE = _phase_schedule()
    if phase not in (0, 1, 2):
        raise ValueError("phase must be 0, 1 or 2")
    schedule = _SCHEDULE[phase]
    zone = next((z for z in schedule if host.startswith(z + "_")), None)
    if zone is None:
        raise ValueError("host has no published mission zone: %r" % host)
    worst = max(-float(v) for v in schedule[zone].values())
    return min(max(worst / 10.0, 0.0), 1.0)


def _confirmed(env):
    out = set()
    for tracker in env.trackers.values():
        for host, state in tracker.state.items():
            if state in ("CONFIRMED", "VERIFY"):
                out.add(host)
    return out


def detection_debt(env, horizon):
    """Severity-weighted hosts Red has compromised and Blue has not confirmed."""
    phase = mission_phase(env._tick, horizon)
    confirmed = _confirmed(env)
    return sum(zone_severity(h, phase) for h in env._compromised_set()
               if h not in confirmed)


def confirm_debt(env, horizon):
    """Severity-weighted hosts Blue has confirmed and not yet remediated."""
    phase = mission_phase(env._tick, horizon)
    total = 0.0
    for tracker in env.trackers.values():
        for host, state in tracker.state.items():
            if state == "CONFIRMED":
                total += zone_severity(host, phase)
    return total


def make_reward_hook(term="both", weight=1.0, horizon=400, cap=10.0):
    """Training-only per-tick reward shaper.

    Returns ``hook(env=..., tick=..., native=...) -> float``. The added term
    is a *penalty* (never a bonus), bounded by ``cap``, and scaled by the
    published severity so a host in a cheap zone is not worth the same
    attention as one in the phase's critical zone.
    """
    if term not in TERMS:
        raise ValueError("unknown shaping term %r; known: %s"
                         % (term, list(TERMS)))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and non-negative")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be finite and positive")

    def hook(env, tick, native):
        debt = 0.0
        if term in ("detection", "both"):
            debt += detection_debt(env, horizon)
        if term in ("confirm", "both"):
            debt += confirm_debt(env, horizon)
        if not math.isfinite(debt):
            raise ValueError("non-finite shaped debt")
        return native - weight * min(debt, cap)

    hook.version = VERSION
    hook.term = term
    hook.weight = float(weight)
    hook.cap = float(cap)
    hook.horizon = int(horizon)
    return hook