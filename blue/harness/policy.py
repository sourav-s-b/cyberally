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

    def __init__(self, hook=None, max_age=80, record_requests=False,
                 candidate_audit_agent=None, record_choices=False):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age must be finite and positive")
        super().__init__(scorer=LancerValues(fruitless_decay=0.5),
                         max_age=max_age, record_choices=record_choices)
        self.hook = hook
        self.last_request = None
        self.record_requests = bool(record_requests)
        self.record_all_choices = bool(record_choices)
        self.candidate_audit_agent = candidate_audit_agent
        self.last_candidate_audit = None
        from blue.harness.diagnostics import RequestDigest
        self.request_digest = RequestDigest()

    def reset(self):
        super().reset()
        reset = getattr(self.hook, "reset", None)
        if callable(reset):
            reset()
        self.last_request = None
        self.last_candidate_audit = None
        from blue.harness.diagnostics import RequestDigest
        self.request_digest = RequestDigest()

    def select(self, env, agent):
        self.last_candidate_audit = None
        self.record_decisions = agent == self.candidate_audit_agent
        self.record_choices = (self.record_all_choices
                               or agent == self.candidate_audit_agent)
        if hasattr(self.hook, "last_candidate_audit"):
            self.hook.last_candidate_audit = None
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
        if (decision is not None and decision.get("selected_action")
                and (decision["selected_action"] != name
                     or decision.get("selected_host") != host)):
            raise RuntimeError("policy decision record differs from selection")
        if (decision is not None
                and decision.get("branch") in {
                    "sweep_max_age", "sweep_scorer", "sweep_hook",
                    "order_fn", "parity_cursor"}
                and name != "Analyse"):
            raise RuntimeError("investigation hook reached remediation")
        self.last_request = {"tick": env._tick, "agent": agent,
                             "action": name, "host": host,
                             "branch": branch, "busy": busy,
                             "pending": env._awaiting.get(agent),
                             "guard": decision.get("guard_record") if decision else None,
                             "proposal": {"action": name, "host": host},
                             "selection_rule": (decision.get("selection_rule")
                                                if decision else None)}
        if (agent == self.candidate_audit_agent and decision
                and decision.get("scores")):
            scored = decision["scores"]
            lancer_pick = argmax_pick(scored)
            guard_record = decision.get("guard_record") or {}
            guard_pick = guard_record.get("pick", lancer_pick)
            hook_rows = getattr(self.hook, "last_candidate_audit", None) or {}
            actor_ranks = hook_rows.get("actor_ranks", {})
            lancer_ranks = {
                host: rank for rank, (_, host) in enumerate(
                    sorted(scored, key=lambda sh: (sh[0], sh[1]), reverse=True), 1)
            }
            actor_pick = hook_rows.get("pick")
            candidates = []
            for base_score, candidate_host in scored:
                row = {"host": candidate_host,
                       "lancer_score": float(base_score),
                       "lancer_rank": lancer_ranks[candidate_host]}
                if candidate_host in hook_rows.get("rows", {}):
                    row.update(hook_rows["rows"][candidate_host])
                    row["actor_rank"] = actor_ranks[candidate_host]
                candidates.append(row)
            self.last_candidate_audit = {
                "tick": env._tick, "agent": agent,
                "branch": decision["branch"],
                "selected_pick": host if name == "Analyse" else None,
                "actor_pick": actor_pick,
                "lancer_pick": lancer_pick,
                "guard_pick": guard_pick,
                "actor_equals_lancer": actor_pick == lancer_pick,
                "actor_equals_guard": actor_pick == guard_pick,
                "guard_intervened": bool(guard_record.get("cause") == "max_age"),
                "candidates": candidates,
            }
        if self.record_requests:
            self.request_digest.add(env._tick, agent, name, host)
        return action


class StrictHarnessPolicy(HarnessPolicy):
    """The same strict coverage guard used by the Qwen headroom control.

    Retains HarnessPolicy's observation bookkeeping and validity checks, but
    initializes OrderedPolicy with its strict guard instead of the separate
    max-age intervention used by the historical GPT-Blue experiments.
    """

    def __init__(self, hook=None, record_requests=False):
        OrderedPolicy.__init__(self, scorer=LancerValues(fruitless_decay=0.5),
                               guard=True)
        self.hook = hook
        self.last_request = None
        self.record_requests = bool(record_requests)
        self.record_all_choices = False
        self.candidate_audit_agent = None
        self.last_candidate_audit = None
        from blue.harness.diagnostics import RequestDigest
        self.request_digest = RequestDigest()


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
    def __init__(self, actor, scorer, bonus=0.25, ml_inputs="both",
                 candidate_audit_agent=None, sample=False, generator=None,
                 sample_temp=0.5):
        if ml_inputs not in ("both", "risk", "zero"):
            raise ValueError("invalid ML ablation")
        if not math.isfinite(bonus) or bonus < 0:
            raise ValueError("invalid residual bonus")
        self.actor, self.scorer, self.bonus = actor, scorer, float(bonus)
        self.ml_inputs = ml_inputs
        self.candidate_audit_agent = candidate_audit_agent
        self.last_candidate_audit = None
        self.sample = bool(sample)
        self.generator = generator
        self.sample_temp = float(sample_temp)
        if self.sample and (self.generator is None or self.sample_temp <= 0):
            raise ValueError("sampled hook requires a generator and positive temperature")

    def reset(self):
        self.scorer.reset()
        self.last_candidate_audit = None

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
        self.last_candidate_audit = None
        F = th.from_numpy(self.features(env, agent, cands))
        with th.no_grad():
            residual = th.tanh(self.actor(F))
        residuals = residual.detach().cpu().numpy().reshape(-1)
        combined = [(s + self.bonus * float(residuals[i]), h)
                    for i, (s, h) in enumerate(scored)]
        if self.sample:
            from blue.training.residual import masked_choice
            logits = th.tensor([s for s, _ in combined]) / self.sample_temp
            idx, _, _ = masked_choice(logits, [1.0] * len(cands),
                                      sample=True, seed_rng=self.generator)
            pick = cands[idx]
        else:
            pick = argmax_pick(combined)
        if agent == self.candidate_audit_agent:
            ranks = {host: rank for rank, (_, host) in enumerate(
                sorted(combined, key=lambda sh: (sh[0], sh[1]), reverse=True), 1)}
            rows = {}
            for i, (base_score, host) in enumerate(scored):
                rows[host] = {
                    "actor_residual": float(residuals[i]),
                    "actor_score": float(combined[i][0]),
                }
            self.last_candidate_audit = {"rows": rows, "actor_ranks": ranks,
                                         "pick": pick}
        return pick
