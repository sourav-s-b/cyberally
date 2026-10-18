"""Lancer-anchored categorical sweep policy and its fresh-rollout PPO update."""
from __future__ import annotations

import copy

import numpy as np

from blue.harness.policy import HarnessActorHook
from blue.policies.ordered import argmax_pick
from blue.training import mappo_guide as mg
from blue.training.residual import masked_choice

VERSION = "anchored-categorical-v1"


def reference_log_probs(scored, mass=0.9, temperature=0.5):
    """Full-support reference with an exact Lancer argmax and legal hosts only."""
    import torch as th
    if not scored or not 0 < mass < 1 or temperature <= 0:
        raise ValueError("invalid reference distribution")
    if len(scored) == 1:
        return th.zeros(1, dtype=th.float32)
    teacher = argmax_pick(scored)
    base = th.tensor([s for s, _ in scored], dtype=th.float32)
    if not bool(th.isfinite(base).all()):
        raise ValueError("nonfinite reference scores")
    teacher_idx = next(i for i, (_, host) in enumerate(scored) if host == teacher)
    other = th.ones(len(scored), dtype=th.bool)
    other[teacher_idx] = False
    q = th.empty(len(scored), dtype=th.float32)
    q[teacher_idx] = mass
    q[other] = (1.0 - mass) * th.softmax(base[other] / temperature, dim=0)
    return th.log(q)


def policy_logits(actor, features, prior_log):
    """Same logits for sampled training, greedy deployment, and PPO."""
    return prior_log + actor(features).reshape(-1)


class AnchoredActorHook(HarnessActorHook):
    def __init__(self, actor, scorer, ml_inputs="both", mass=0.9,
                 temperature=0.5, sample=False, generator=None,
                 candidate_audit_agent=None):
        super().__init__(actor, scorer, bonus=0.0, ml_inputs=ml_inputs,
                         candidate_audit_agent=candidate_audit_agent)
        self.mass = float(mass)
        self.temperature = float(temperature)
        self.sample = bool(sample)
        self.generator = generator
        if self.sample and self.generator is None:
            raise ValueError("sampled evaluation requires an explicit generator")

    def __call__(self, env, agent, cands, scored):
        import torch as th
        self.last_candidate_audit = None
        F = th.from_numpy(self.features(env, agent, cands))
        prior = reference_log_probs(scored, self.mass, self.temperature)
        with th.no_grad():
            logits = policy_logits(self.actor, F, prior)
        if self.sample:
            idx, _, _ = masked_choice(logits, [1.0] * len(cands),
                                      sample=True, seed_rng=self.generator)
            pick = cands[idx]
        else:
            pick = argmax_pick([(float(logits[i]), host)
                                for i, host in enumerate(cands)])
        if agent == self.candidate_audit_agent:
            scores = [(float(logits[i]), host) for i, host in enumerate(cands)]
            ranks = {host: rank for rank, (_, host) in enumerate(
                sorted(scores, key=lambda sh: (sh[0], sh[1]), reverse=True), 1)}
            correction = self.actor(F).detach().reshape(-1)
            rows = {host: {"actor_residual": float(correction[i]),
                           "actor_score": float(logits[i]),
                           "reference_probability": float(prior[i].exp())}
                    for i, host in enumerate(cands)}
            self.last_candidate_audit = {"rows": rows, "actor_ranks": ranks,
                                         "pick": pick}
        return pick


class AnchoredRecorder(mg.JointRecorder):
    def __init__(self, actor, scorer, gen, ml_inputs="both", mass=0.9,
                 temperature=0.5):
        super().__init__(actor, bonus=0.0, gen=gen, temp=temperature)
        self.adapter = HarnessActorHook(actor, scorer, bonus=0.0,
                                        ml_inputs=ml_inputs)
        self.mass = float(mass)
        self.sampled_disagree = 0
        self.reference_probabilities = []
        self.base_gaps = []

    def reset(self):
        self.adapter.reset()

    def observe(self, env, agent):
        self.adapter.observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        import torch as th
        F = th.from_numpy(self.adapter.features(env, agent, cands))
        prior = reference_log_probs(scored, self.mass, self.temp)
        with th.no_grad():
            logits = policy_logits(self.actor, F, prior)
        idx, logp, ent = masked_choice(logits, [1.0] * len(cands),
                                       sample=True, seed_rng=self.gen)
        teacher = argmax_pick(scored)
        greedy = argmax_pick([(float(logits[i]), h)
                              for i, h in enumerate(cands)])
        self.total += 1
        self.agree += int(greedy == teacher)
        self.sampled_disagree += int(cands[idx] != teacher)
        self.reference_probabilities.append(float(prior[idx].exp()))
        base = dict((h, s) for s, h in scored)
        self.base_gaps.append(float(base[teacher] - base[cands[idx]]))
        row = {"feats": F.numpy(),
               "joint": mg.joint_context(env, self._policy_ref, self._jcache),
               "k": len(cands), "mask": [1.0] * len(cands),
               "base": np.array([s for s, _ in scored], dtype=np.float32),
               "prior_log": prior.numpy(),
               "old_log_probs": th.log_softmax(logits, dim=0).numpy(),
               "choice": idx, "logp": logp, "ent": ent,
               "tick": env._tick, "agent": agent, "ret": 0.0,
               "temp": self.temp}
        self.rows.append(row)
        self.pending.setdefault(agent, []).append(len(self.rows) - 1)
        self.per_agent[agent] = self.per_agent.get(agent, 0) + 1
        return cands[idx]

    def close_episode(self, rewards):
        start = self._ep_start
        super().close_episode(rewards)
        for row in self.rows[start:]:
            pad = mg.MAX_CANDS - row["k"]
            row["prior_log"] = np.pad(row["prior_log"], (0, pad),
                                       constant_values=-1e9)
            row["old_log_probs"] = np.pad(row["old_log_probs"], (0, pad),
                                           constant_values=-1e9)


def _batch_logits(actor, features, prior_log, mask):
    import torch as th
    return (prior_log + actor(features).squeeze(-1)).masked_fill(mask == 0,
                                                                  -1e9)


def ppo_update(actor, critic, opt, buf, reference_kl_coef=0.05,
               max_update_kl=0.02, epochs=4, minibatch=256,
               clip=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO with reference KL and epoch rollback on exact categorical KL."""
    import torch as th
    if not buf:
        raise ValueError("empty rollout")
    F = th.from_numpy(np.stack([b["feats"] for b in buf]).astype(np.float32))
    J = th.from_numpy(np.stack([b["joint"] for b in buf]).astype(np.float32))
    M = th.from_numpy(np.stack([b["mask"] for b in buf]).astype(np.float32))
    P = th.from_numpy(np.stack([b["prior_log"] for b in buf]).astype(np.float32))
    OL = th.from_numpy(np.stack([b["old_log_probs"] for b in buf]).astype(np.float32))
    CH = th.tensor([b["choice"] for b in buf], dtype=th.int64)
    OLD = th.tensor([b["logp"] for b in buf], dtype=th.float32)
    RET = th.tensor([b["ret"] for b in buf], dtype=th.float32)
    with th.no_grad():
        adv = RET - critic(J).squeeze(-1)
        std = adv.std() if len(adv) > 1 else th.tensor(0.0)
        adv = ((adv - adv.mean()) / std if bool(th.isfinite(std)) and std >= 1e-6
               else th.zeros_like(adv))
    totals = {k: 0.0 for k in ("pol", "vf", "ent", "kl", "reference_kl",
                               "actor_grad_norm", "critic_grad_norm",
                               "grad_clip_fraction", "clip_fraction")}
    n_batches = accepted = rejected = 0
    update_kl = 0.0
    ids = np.arange(len(buf))
    for _ in range(epochs):
        saved_actor = copy.deepcopy(actor.state_dict())
        saved_critic = copy.deepcopy(critic.state_dict())
        saved_opt = copy.deepcopy(opt.state_dict())
        epoch_totals = {k: 0.0 for k in totals}
        epoch_batches = 0
        np.random.shuffle(ids)
        for start in range(0, len(buf), minibatch):
            bi = ids[start:start + minibatch]
            logits = _batch_logits(actor, F[bi], P[bi], M[bi])
            log_all = th.log_softmax(logits, dim=1)
            probs = log_all.exp() * M[bi]
            logp = log_all.gather(1, CH[bi].unsqueeze(1)).squeeze(1)
            ratio = (logp - OLD[bi]).exp()
            pg = -th.minimum(ratio * adv[bi],
                             ratio.clamp(1.0 - clip, 1.0 + clip) * adv[bi]).mean()
            vf = th.nn.functional.mse_loss(critic(J[bi]).squeeze(-1), RET[bi])
            ent = -(probs * log_all).sum(1).mean()
            refkl = (probs * (log_all - P[bi])).sum(1).mean()
            loss = pg + vf_coef * vf - ent_coef * ent + reference_kl_coef * refkl
            opt.zero_grad()
            loss.backward()
            actor_norm = th.nn.utils.clip_grad_norm_(actor.parameters(), float("inf"))
            critic_norm = th.nn.utils.clip_grad_norm_(critic.parameters(), float("inf"))
            combined = th.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 1.0)
            opt.step()
            vals = {"pol": float(pg.detach()), "vf": float(vf.detach()),
                    "ent": float(ent.detach()), "reference_kl": float(refkl.detach()),
                    "kl": float((OLD[bi] - logp).mean().detach()),
                    "actor_grad_norm": float(actor_norm),
                    "critic_grad_norm": float(critic_norm),
                    "grad_clip_fraction": float(combined > 1.0),
                    "clip_fraction": float(((ratio - 1.0).abs() > clip).float().mean())}
            for key, value in vals.items():
                epoch_totals[key] += value
            epoch_batches += 1
        with th.no_grad():
            current = th.log_softmax(_batch_logits(actor, F, P, M), dim=1)
            update_kl = float(((OL.exp() * M) * (OL - current)).sum(1).mean())
        if not np.isfinite(update_kl) or update_kl > max_update_kl:
            actor.load_state_dict(saved_actor)
            critic.load_state_dict(saved_critic)
            opt.load_state_dict(saved_opt)
            rejected += 1
            with th.no_grad():
                current = th.log_softmax(_batch_logits(actor, F, P, M), dim=1)
                update_kl = float(((OL.exp() * M) * (OL - current)).sum(1).mean())
            break
        for key in totals:
            totals[key] += epoch_totals[key]
        n_batches += epoch_batches
        accepted += 1
    result = {key: value / max(n_batches, 1) for key, value in totals.items()}
    result.update(update_kl=update_kl, accepted_epochs=accepted,
                  rejected_epochs=rejected)
    return result
