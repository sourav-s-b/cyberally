"""Bounded-deviation deployment for the anchored Lancer pilot.

The anchor pilot (`blue/results/finite_red_anchor_pilot_v1`, branch
`blue/finite-red-check`) deployed the anchored actor greedily at reference
mass 0.9 and measured zero deviation from guard-only across 72 paired
episodes (+1.58 vs Lancer is the guard-only difference, not learned gain).

This module implements the next experiment that pilot called for: measure
candidate logit margins at each greedy sweep decision, then deploy ONE
preregistered bounded-deviation rule. The rule keeps Lancer's candidate
unless the anchored logits prefer another legal host by at least `tau`
logits AND the episode-level override budget is not exhausted.

Deployment facts recorded distinctly (see AGENTS.md):
- a margin rule pick is an observed action, not evidence the host is compromised;
- native return is the only reported score;
- margin rows are Blue-visible (Lancer scores and model inputs only).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from blue.harness.policy import HarnessActorHook, HarnessPolicy
from blue.harness.scoring import MLScorer
from blue.policies.ordered import argmax_pick
from blue.training.harness_anchor import AnchoredActorHook, reference_log_probs
from blue.training.harness_rl import build_actor
from blue.training.harness_eval import episode

VERSION = "margin-bounded-deploy-v1"

_MASS_LIMITS = (0.0, 1.0)


def bounded_choice(margin, tau, budget_left, teacher_pick, alternative_pick):
    """One margin rule decision. Returns (pick, overrode, budget_left).

    A nonfinite margin is a numeric anomaly and stops the run rather than
    silently choosing the teacher."""
    if not np.isfinite(margin):
        raise ValueError("nonfinite margin recorded")
    if budget_left > 0 and margin >= 0.0 and margin >= tau:
        return alternative_pick, True, budget_left - 1
    return teacher_pick, False, budget_left


class MarginActorHook(AnchoredActorHook):
    """Anchored greedy with per-decision margin recording and a bounded
    deviation budget per episode. Overrides Lancer's candidate only when the
    anchored logits prefer the best alternative host by at least `tau`
    logits, and at most `budget` times per episode. Sampling is disabled."""

    def __init__(self, actor, scorer, mass, tau, budget,
                 ml_inputs="both", candidate_limit=None):
        if not _MASS_LIMITS[0] < mass < _MASS_LIMITS[1]:
            raise ValueError("margin deploy requires 0 < mass < 1")
        if tau is None or not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be a non-negative finite number")
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative int")
        super().__init__(actor, scorer, mass=mass, ml_inputs=ml_inputs)
        self.tau = float(tau)
        self.budget_head = int(budget)
        self.used = 0
        self.decisions = []
        self.overrode = 0

    def reset(self):
        super().reset()
        self.used = 0
        self.decisions = []

    @property
    def budget_left(self):
        return self.budget_head - self.used

    def __call__(self, env, agent, cands, scored):
        import torch as th
        F = th.from_numpy(self.features(env, agent, cands))
        prior = reference_log_probs(scored, self.mass, self.temperature)
        with th.no_grad():
            logits = policy_logits(self.actor, F, prior)
        scores = [(float(logits[i]), h) for i, h in enumerate(cands)]
        teacher = argmax_pick(scored)
        teacher_idx = next(i for i, h in enumerate(cands) if h == teacher)
        alt = [(v, h) for v, h in scores if h != teacher]
        alt_by_logit = dict((h, v) for v, h in scores)
        if alt:
            best_alt = argmax_pick(alt)
            best_alt_logit = alt_by_logit[best_alt]
        else:
            best_alt, best_alt_logit = teacher, float(scores[teacher_idx][0])
        alt_idx = next(i for i, h in enumerate(cands) if h == best_alt)
        teacher_logit = float(scores[teacher_idx][0])
        margin = best_alt_logit - teacher_logit
        # Logits contain the reference mass; the actor's own margin over the
        # teacher is mass-invariant and lets one diagnostic pass be reused
        # at any deploy mass without new simulator runs.
        residual_margin = (best_alt_logit - float(prior[alt_idx])
                           - (teacher_logit - float(prior[teacher_idx])))
        row = {"tick": int(env._tick), "agent": agent, "k": len(cands),
               "teacher": teacher, "best_alt": best_alt,
               "teacher_logit": teacher_logit,
               "best_alt_logit": best_alt_logit,
               "ref_teacher_prob": float(prior[teacher_idx].exp()),
               "ref_alt_prob": float(prior[alt_idx].exp()),
               "residual_margin": residual_margin,
               "margin": margin}
        pick, overrode, _ = bounded_choice(
            row["margin"], self.tau, self.budget_left, teacher, best_alt)
        row["pick"] = pick
        row["overrode"] = overrode
        if overrode:
            self.used += 1
            self.overrode += 1
        self.decisions.append(row)
        return pick


def policy_logits(actor, features, prior_log):
    """Same definition as the anchored arm (mass included in the logits)."""
    return prior_log + actor(features).reshape(-1)


def margin_profile(rows):
    """Distributions over recorded margins, Blue-visible only."""
    if not rows:
        raise ValueError("no margin rows recorded")
    m = np.array([r["margin"] for r in rows], dtype=float)
    for x in m:
        if not np.isfinite(x):
            raise ValueError("nonfinite margin recorded")
    return {"n": int(m.size),
            "max": float(m.max()), "p99": float(np.quantile(m, .99)),
            "p95": float(np.quantile(m, .95)), "p50": float(np.quantile(m, .5)),
            "n_positive": int((m >= 0).sum()),
            "n_gt_2": int((m > 2).sum()), "n_gt_3": int((m > 3).sum())}


class MarginCurrentHook(HarnessActorHook):
    """Margin-bounded deployment of the CURRENT residual arm.

    The deployed score is exactly the current arm's combined score
    (Lancer base + bonus * tanh(actor)); the margin rule overrides the
    Lancer candidate only when the best alternative exceeds it by at
    least `tau` score units, within an episode-level `budget`."""

    def __init__(self, actor, scorer, bonus, tau, budget,
                 ml_inputs="both", candidate_limit=None):
        if not math.isfinite(bonus) or bonus <= 0:
            raise ValueError("bonus must be a positive finite number")
        if tau is None or not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be a non-negative finite number")
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative int")
        super().__init__(actor, scorer, bonus=bonus, ml_inputs=ml_inputs)
        self.tau = float(tau)
        self.budget_head = int(budget)
        self.used = 0
        self.overrode = 0
        self.decisions = []

    def reset(self):
        super().reset()
        self.used = 0
        self.decisions = []

    @property
    def budget_left(self):
        return self.budget_head - self.used

    def __call__(self, env, agent, cands, scored):
        import torch as th
        F = th.from_numpy(self.features(env, agent, cands))
        with th.no_grad():
            residual = th.tanh(self.actor(F))
        residuals = residual.detach().cpu().numpy().reshape(-1)
        combined = [(s + self.bonus * float(residuals[i]), h)
                    for i, (s, h) in enumerate(scored)]
        teacher = argmax_pick(scored)
        teacher_combined = next(v for v, h in combined if h == teacher)
        alt = [(v, h) for v, h in combined if h != teacher]
        if alt:
            best_alt = argmax_pick(alt)
            best_alt_score = next(v for v, h in alt if h == best_alt)
        else:
            best_alt, best_alt_score = teacher, teacher_combined
        row = {"tick": int(env._tick), "agent": agent, "k": len(cands),
               "teacher": teacher, "best_alt": best_alt,
               "teacher_logit": teacher_combined,
               "best_alt_logit": best_alt_score,
               "ref_teacher_prob": None, "ref_alt_prob": None,
               "residual_margin": None,
               "margin": best_alt_score - teacher_combined}
        pick, overrode, _ = bounded_choice(
            row["margin"], self.tau, self.budget_left, teacher, best_alt)
        row["pick"] = pick
        row["overrode"] = overrode
        if overrode:
            self.used += 1
            self.overrode += 1
        self.decisions.append(row)
        return pick


def load_margin_model(model_dir, scorer_path, policy_version, mass, bonus):
    """Load one frozen residual checkpoint and return a fresh greedy hook
    factory bound to its actor; refusals are loud on any drift."""
    import torch as th
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete model: {model_dir}")
    version = manifest["config"]["policy_version"]
    if version != policy_version:
        raise ValueError(f"policy_version drift: expected {policy_version}, got {version}")
    actor = build_actor(manifest["config"].get("hidden", 64))
    actor.load_state_dict(th.load(model_dir / "actor.th",
                                  map_location="cpu", weights_only=True))
    actor.eval()
    if manifest["actor_sha256"] != actor_sha256_of(model_dir / "actor.th"):
        raise ValueError("modified actor file")
    scorer = MLScorer(scorer_path)
    return actor, scorer, manifest


def actor_sha256_of(path):
    from blue.harness.scoring import sha256
    return sha256(path)


def make_hook(policy_version, actor, scorer, mass, bonus, tau, budget):
    if policy_version == "anchored-categorical-v1":
        return MarginActorHook(actor, scorer, mass=mass, tau=tau, budget=budget)
    if policy_version == "current-residual-v1":
        return MarginCurrentHook(actor, scorer, bonus=bonus, tau=tau,
                                 budget=budget)
    raise ValueError(f"unsupported policy version for margin deploy: {policy_version}")


def _resolve_policy_version(model_dirs, policy_version=None):
    """Infer or check the frozen arm shared by every model directory."""
    versions = set()
    for model_dir in model_dirs.values():
        manifest = json.loads((Path(model_dir) / "manifest.json").read_text())
        versions.add(manifest["config"]["policy_version"])
    if len(versions) > 1:
        raise ValueError(f"mixed policy versions in one run: {sorted(versions)}")
    got = versions.pop()
    if policy_version is not None and got != policy_version:
        raise ValueError(f"policy version drift: {got} != {policy_version}")
    return got


def evaluate(root, model_dirs, seeds, steps, mass, tau, budget, scorer_path,
             out, red_agent="finite", steps_frozen=True, bonus=None,
             policy_version=None):
    """Run bounded-deploy greedy for each model on `seeds`; apply the rule
    exactly; record margins, per-tick penalties and native returns."""
    import torch as th
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    policy_version = _resolve_policy_version(model_dirs, policy_version)
    if policy_version == "anchored-categorical-v1" and mass is None:
        raise ValueError("anchored deployment requires --mass")
    if policy_version == "current-residual-v1":
        if bonus is None:
            bonus = 0.25
        if mass is not None:
            raise ValueError("current-arm deployment must not set --mass")
    report = {"version": VERSION, "policy_version": policy_version,
              "mass": mass, "bonus": bonus, "tau": tau, "budget": budget,
              "steps": steps, "seeds": seeds, "red_agent": red_agent,
              "cells": {}, "margin_profile": {}}
    for tag, model_dir in model_dirs.items():
        actor, scorer, manifest = load_margin_model(
            model_dir, scorer_path, policy_version, mass, bonus)
        if steps_frozen and manifest["config"]["steps"] != steps:
            raise ValueError("horizon must match the frozen model")
        cells = {}
        rows_all = []
        for seed in seeds:
            th.manual_seed(0)
            hook = make_hook(policy_version, actor, scorer, mass, bonus,
                             tau, budget)
            policy = HarnessPolicy(hook, manifest["config"]["max_age"])
            trace = out / f"{tag}_{seed}.jsonl"
            result = episode(policy, seed, steps, trace, red_agent=red_agent)
            rows = hook.decisions
            for r in rows:
                r["seed"] = seed
            rows_all.extend(rows)
            cells[str(seed)] = result
        report["cells"][tag] = cells
        report["margin_profile"][tag] = margin_profile(rows_all)
        (out / f"{tag}_margins.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows_all))
        overrides = sum(r["overrode"] for r in rows_all)
        report["overrides_total"] = report.get("overrides_total", 0) + overrides
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="blue/results/margin_bounded_deploy_v1/dev")
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="tag=dir pairs for frozen residual actors")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--mass", type=float, default=None)
    ap.add_argument("--bonus", type=float, default=None)
    ap.add_argument("--policy-version", default=None)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--red-agent", default="finite")
    args = ap.parse_args()
    model_dirs = {}
    for item in args.models:
        tag, _, p = item.partition("=")
        if not tag or not p:
            raise ValueError(f"models must be tag=dir, got {item}")
        model_dirs[tag] = p
    evaluate(args.out, model_dirs, args.seeds, args.steps, args.mass,
             args.tau, args.budget, args.scorer, args.out,
             red_agent=args.red_agent, bonus=args.bonus,
             policy_version=args.policy_version)


if __name__ == "__main__":
    main()
