"""Pre-decision *exposure proxies* from Blue evidence and public objectives.

These values are not incurred penalties or host-level causal effects. Native
CAGE4 reward uses hidden attack/Green state and is a shared team signal;
feeding its per-zone breakdown to a Blue actor would leak simulator truth.
The static mission cost schedule is public. Inspection and event ages come
solely from Blue's own tracking and observed telemetry.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from blue.harness.features import NAMES

VERSION = "blue-consequence-context-v1"
NAMES_EXTRA = ("public_mission_severity", "inspection_debt",
               "unreviewed_alert_debt")
PHASE_INDEX = tuple(NAMES.index(f"phase_{i}") for i in range(3))
ANALYSIS_AGE = NAMES.index("analysis_age")
ANALYSIS_AVAILABLE = NAMES.index("analysis_available")


@lru_cache(maxsize=1)
def public_reward_schedule():
    # Blue's wrapper establishes the pinned simulator import path.
    import blue.core.wrapper  # noqa: F401
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine

    machine = BlueRewardMachine("Blue")
    return tuple(machine.get_phase_rewards(i) for i in range(3))


def consequence_context(host, raw, memory, *, horizon=400, max_age=80):
    """Return three bounded Blue-computable values for one target host.

    ``raw`` is the exact pre-decision BlueFeatures row and ``memory`` is the
    eight-column EvidenceMemory row. A never-inspected host carries full
    inspection debt; a just-inspected host carries zero. Positive changes
    retain an age-weighted debt until a completed Analyse reviews them.
    """
    raw = np.asarray(raw, dtype=float)
    memory = np.asarray(memory, dtype=float)
    if raw.shape != (len(NAMES),) or memory.shape != (8,):
        raise ValueError("consequence input geometry mismatch")
    if horizon <= 0 or max_age <= 0 or not math.isfinite(max_age):
        raise ValueError("invalid consequence timescale")
    phase_bits = raw[list(PHASE_INDEX)]
    if not np.isfinite(phase_bits).all() or not np.array_equal(
            phase_bits, (phase_bits == 1).astype(float)) or phase_bits.sum() != 1:
        raise ValueError("invalid public mission phase")
    phase = int(np.argmax(phase_bits))
    schedule = public_reward_schedule()[phase]
    zone = next((z for z in schedule if host.startswith(z + "_")), None)
    if zone is None:
        raise ValueError("target host has no published mission zone")
    public_severity = min(max(-float(v) for v in schedule[zone].values()) / 10., 1.)

    available = raw[ANALYSIS_AVAILABLE]
    age = raw[ANALYSIS_AGE]
    if available not in (0., 1.) or not math.isfinite(age) or not 0 <= age <= 1:
        raise ValueError("invalid Blue inspection history")
    inspection_debt = (1. if available == 0 else
                       min(age * horizon / max_age, 1.))
    active, event_ages = memory[:4], memory[4:]
    if (not np.isfinite(memory).all() or
            not np.isin(active, (0., 1.)).all() or
            (event_ages < 0).any() or (event_ages > 1).any()):
        raise ValueError("invalid Blue event memory")
    unreviewed_alert_debt = float(np.mean(
        active * (1. + np.minimum(event_ages * horizon / max_age, 1.))))
    return (float(public_severity), float(inspection_debt),
            unreviewed_alert_debt)
