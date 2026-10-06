"""Coverage-protected policies; masks and evidence restrictions stay separate."""
from __future__ import annotations

import math

from blue.core.baselines import decode_index
from blue.core.wrapper import BLUE_AGENTS
from blue.policies.ordered import LancerValues, OrderedPolicy, argmax_pick


class HarnessPolicy(OrderedPolicy):
    """Existing remediation/verification rules, then age guard, then learner.

    The threshold is an intervention trigger, not a promised maximum age:
    busy agents and urgent work can defer investigation. Never filters by risk.
    ``last_request`` is a Blue-visible audit record, not a policy rationale.
    """

    def __init__(self, hook=None, max_age=80):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age must be finite and positive")
        super().__init__(scorer=LancerValues(fruitless_decay=0.5), max_age=max_age)
        self.hook = hook
        self.last_request = None

    def reset(self):
        super().reset()
        reset = getattr(self.hook, "reset", None)
        if callable(reset):
            reset()
        self.last_request = None

    def select(self, env, agent):
        observe = getattr(self.hook, "observe", None)
        if callable(observe):
            observe(env, agent)  # advance snapshots even while busy/guarded
        self.last_decision = None  # forced branches must not inherit old sweep records
        busy = agent in env._awaiting
        action = super().select(env, agent)
        mask = env.get_avail_agent_actions(BLUE_AGENTS.index(agent))
        if not mask[action]:
            raise RuntimeError("harness returned an invalid action")
        name, host = decode_index(env, agent, action)
        decision = self.last_decision
        branch = (decision["branch"] if decision else
                  "pending_wait" if busy else
                  "remediation" if name in ("Remove", "Restore") else
                  "verification" if name == "Analyse" else "idle")
        if decision is not None and name != "Analyse":
            raise RuntimeError("investigation hook reached remediation")
        self.last_request = {"tick": env._tick, "agent": agent,
                             "action": name, "host": host,
                             "branch": branch, "busy": busy,
                             "pending": env._awaiting.get(agent),
                             "guard": decision.get("guard_record") if decision else None}
        return action


class MLRankingHook:
    """Bounded ML-assisted heuristic baseline, distinct from RL.

    Anomaly percentile and compromise probability retain distinct semantics.
    These coefficients are frozen experiment settings, not tuned results.
    """

    def __init__(self, scorer, risk_bonus=0.25, novelty_bonus=0.0):
        self.scorer = scorer
        self.risk_bonus = float(risk_bonus)
        self.novelty_bonus = float(novelty_bonus)
        if any(not math.isfinite(v) or v < 0 for v in (self.risk_bonus, self.novelty_bonus)):
            raise ValueError("ML bonuses must be finite and nonnegative")

    def reset(self):
        self.scorer.reset()

    def observe(self, env, agent):
        self.scorer.observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        scores = self.scorer.score(env, agent, cands)
        return argmax_pick([(s + self.risk_bonus * (float(scores[i, 0]) - 0.5)
                             + self.novelty_bonus * (float(scores[i, 1]) - 0.5), h)
                            for i, (s, h) in enumerate(scored)])


class HarnessActorHook:
    def __init__(self, actor, scorer, bonus=0.25, ml_inputs="both"):
        if ml_inputs not in ("both", "risk", "zero"):
            raise ValueError("invalid ML ablation")
        if not math.isfinite(bonus) or bonus < 0:
            raise ValueError("invalid residual bonus")
        self.actor, self.scorer, self.bonus = actor, scorer, float(bonus)
        self.ml_inputs = ml_inputs

    def reset(self):
        self.scorer.reset()

    def observe(self, env, agent):
        self.scorer.observe(env, agent)

    def features(self, env, agent, cands):
        F = self.scorer.actor_rows(env, agent, cands)
        if self.ml_inputs == "zero":
            F[:, -2:] = 0.0
        if self.ml_inputs == "risk":
            F[:, -1] = 0.0
        return F

    def __call__(self, env, agent, cands, scored):
        import torch as th
        F = th.from_numpy(self.features(env, agent, cands))
        with th.no_grad():
            residual = th.tanh(self.actor(F))
        return argmax_pick([(s + self.bonus * float(residual[i]), h)
                            for i, (s, h) in enumerate(scored)])
