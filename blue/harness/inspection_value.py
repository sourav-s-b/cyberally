"""Opt-in investigation inputs and a bounded diagnostic ranking hook.

The compromise score answers "how likely is this host compromised now?".
For selecting an *inspection*, the same score should carry less weight
immediately after a completed Analyse and regain weight as that evidence
ages.  These two extra actor inputs make that distinction explicit.

This is a mechanism hypothesis, not a calibrated value-of-information model
or a guaranteed policy improvement.  The guard threshold supplies the only
timescale; no privileged state or post-action result enters the calculation.
"""
from __future__ import annotations

import math

import numpy as np

from blue.policies.ordered import argmax_pick

VERSION = "blue-inspection-value-v1"
FEATURE_NAMES = ("inspection_age_fraction", "freshness_adjusted_risk")


def inspection_value_rows(env, agent, hosts, risk, max_age):
    """Return age/guard-threshold and risk×age, both in [0, 1]."""
    risk = np.asarray(risk, dtype=np.float32)
    if risk.shape != (len(hosts),) or not np.isfinite(risk).all() \
            or (risk < 0).any() or (risk > 1).any():
        raise ValueError("invalid calibrated risk vector")
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("invalid guard threshold")
    tracker = env.trackers[agent]
    tick = int(env._tick)
    ages = []
    for host in hosts:
        last = tracker.last_analysis.get(host)
        if last is not None and not 0 <= last <= tick:
            raise ValueError("invalid completed analysis timestamp")
        ages.append(1. if last is None else min((tick-last)/max_age, 1.))
    ages = np.asarray(ages, dtype=np.float32)
    return np.stack((ages, risk*ages), axis=1)


class FreshnessRiskHook:
    """Fixed diagnostic heuristic; the remediation rules stay untouched."""

    def __init__(self, scorer, max_age=80., bonus=.25):
        if not math.isfinite(bonus) or bonus < 0:
            raise ValueError("invalid risk bonus")
        self.scorer, self.max_age, self.bonus = scorer, float(max_age), float(bonus)

    def reset(self):
        self.scorer.reset()

    def observe(self, env, agent):
        self.scorer.observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        risk = self.scorer.score(env, agent, cands)[:, 0]
        age_risk = inspection_value_rows(env, agent, cands, risk,
                                         self.max_age)[:, 1]
        return argmax_pick([(float(s) + self.bonus*(float(age_risk[i])-.5), h)
                            for i, (s, h) in enumerate(scored)])
