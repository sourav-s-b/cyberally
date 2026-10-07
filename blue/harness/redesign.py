"""Opt-in investigation scheduler v2. Only Blue-visible inputs; v1 untouched.

A teacher mixture limits *sampling* departures, not defense loss. Its complete
marginal probability is used in training and stochastic execution. Greedy ranking
is a separately named diagnostic policy, never called the trained distribution.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from blue.harness.features import ACTOR_NAMES, NAMES, neural_rows
from blue.harness.policy import HarnessActorHook
from blue.policies.ordered import argmax_pick
from blue.harness.numerics import stable_pick

VERSION = 'blue-harness-redesign-v2'
EXTRA_NAMES = tuple(f'unreviewed_positive_change_{i}' for i in range(4)) + tuple(
    f'positive_change_age_{i}' for i in range(4)) + ('risk_binary_entropy',)
FEATURE_NAMES = ACTOR_NAMES + EXTRA_NAMES
CRITIC_DIM = 5 * (2 * (len(ACTOR_NAMES) - 2) + 1) + 5 + 1


class EvidenceMemory:
    """Retain positive telemetry changes until a completed analysis reviews them.

    Ages are observability bookkeeping, not probability or true compromise.
    Busy-tick observations advance memory. Exact same-tick reads are idempotent.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.events = {}
        self.seen = {}

    def rows(self, env, agent, raw):
        tick = int(env._tick)
        if tick < self.seen.get(agent, -1):
            raise ValueError('reset evidence memory at episode boundary')
        if self.seen.get(agent) != tick:
            delta_idx = [NAMES.index(n) for n in ('delta_unknown_files', 'delta_file_density',
                                                 'delta_connections', 'delta_external_connections')]
            for host, row in zip(env.hostnames[agent], raw):
                key = (agent, host)
                stamps = self.events.setdefault(key, [None] * 4)
                completed = env.trackers[agent].last_analysis.get(host)
                for i, d in enumerate(row[delta_idx]):
                    if np.isfinite(d) and d > 0:
                        stamps[i] = tick
                    if stamps[i] is not None and completed is not None and completed >= stamps[i]:
                        stamps[i] = None
            self.seen[agent] = tick
        result = []
        for host in env.hostnames[agent]:
            stamps = self.events[(agent, host)]
            result.append([float(t is not None) for t in stamps] + [
                min((tick-t)/env.episode_limit, 1.) if t is not None else 1. for t in stamps])
        return np.asarray(result, dtype=np.float32)


class SetActor(nn.Module):
    """Permutation-equivariant local candidate ranking with mean/max context."""
    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES) + 1, hidden), nn.Tanh())
        self.body = nn.Sequential(nn.Linear(3*hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, features, base, mask):
        single = features.ndim == 2
        if single:
            features, base, mask = features[None], base[None], mask[None]
        z = self.encoder(torch.cat((features, (base/4).unsqueeze(-1)), dim=-1))
        m = mask.bool().unsqueeze(-1)
        count = m.sum(1).clamp(min=1)
        mean = (z*m).sum(1)/count
        maximum = z.masked_fill(~m, -torch.inf).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        context = torch.cat((z, mean[:, None].expand_as(z), maximum[:, None].expand_as(z)), dim=-1)
        out = self.head(self.body(context)).squeeze(-1)
        return out[0] if single else out


class CentralValue(nn.Module):
    """Agent-conditioned V of all five visible zones; native-return units."""
    def __init__(self, hidden=64, value_scale=20.):
        super().__init__()
        if not np.isfinite(value_scale) or value_scale <= 0:
            raise ValueError('invalid value scale')
        self.value_scale = float(value_scale)
        self.net = nn.Sequential(nn.Linear(CRITIC_DIM, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, context):
        return self.net(context).squeeze(-1)*self.value_scale


def probabilities(actor, features, base, mask, teacher, bonus=.25, temperature=.5, explore=.2):
    """Exact marginal categorical: teacher point mass + learned softmax.

    An epsilon here controls departures on unguarded sweep decisions. It is
    not an action validity mask, a risk filter, or a performance guarantee.
    """
    if not np.isfinite([bonus, temperature, explore]).all() or bonus < 0 or temperature <= 0 or not 0 < explore <= 1:
        raise ValueError('invalid policy settings')
    single = features.ndim == 2
    if single:
        features, base, mask = features[None], base[None], mask[None]
        teacher = torch.as_tensor([teacher], dtype=torch.long)
    else:
        teacher = torch.as_tensor(teacher, dtype=torch.long, device=base.device)
    if not mask.bool().any(-1).all() or not mask.bool().gather(1, teacher[:, None]).all():
        raise ValueError('empty candidates or masked teacher')
    residual = torch.tanh(actor(features, base, mask))
    logits = ((base+bonus*residual)/temperature).masked_fill(~mask.bool(), -torch.inf)
    p = explore*torch.softmax(logits, dim=-1)
    anchor = torch.zeros_like(p).scatter(1, teacher[:, None], 1.)
    p = p + (1-explore)*anchor
    return p[0] if single else p


class RedesignHook(HarnessActorHook):
    def __init__(self, actor, scorer, bonus=.25, ml_inputs='risk', temperature=.5,
                 explore=.2, policy_seed=0, execution='sample'):
        super().__init__(actor, scorer, bonus, ml_inputs)
        if execution not in ('sample', 'greedy-diagnostic'):
            raise ValueError('unknown execution policy')
        self.temperature, self.explore = temperature, explore
        self.policy_seed, self.execution = int(policy_seed), execution
        self.memory = EvidenceMemory()
        self.reset()

    def reset(self):
        super().reset()
        if hasattr(self, 'memory'):
            self.memory.reset()
        self.generator = torch.Generator().manual_seed(self.policy_seed)

    def observe(self, env, agent):
        raw = self.scorer.observe(env, agent)
        self.memory.rows(env, agent, raw)

    def features(self, env, agent, cands):
        self.observe(env, agent)
        F = super().features(env, agent, cands)
        raw = self.scorer.features.rows(env, agent)
        memory = self.memory.rows(env, agent, raw)
        ix = [env.hostnames[agent].index(h) for h in cands]
        p = np.clip(F[:, -2], 1e-6, 1-1e-6)
        entropy = -(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2.)
        if self.ml_inputs == 'zero':
            entropy[:] = 0.
        return np.concatenate((F, memory[ix], entropy[:, None]), axis=1).astype(np.float32)

    def distribution(self, env, agent, cands, scored):
        features = torch.from_numpy(self.features(env, agent, cands))
        base = torch.tensor([s for s, _ in scored], dtype=torch.float32)
        mask = torch.ones(len(cands))
        teacher = cands.index(argmax_pick(scored))
        p = probabilities(self.actor, features, base, mask, teacher, self.bonus,
                          self.temperature, self.explore)
        return features, base, mask, teacher, p

    def __call__(self, env, agent, cands, scored):
        with torch.no_grad():
            F, base, mask, teacher, p = self.distribution(env, agent, cands, scored)
            if self.execution == 'sample':
                idx = int(torch.multinomial(p, 1, generator=self.generator))
                return cands[idx]
            residual = torch.tanh(self.actor(F, base, mask))
        return stable_pick([(s+self.bonus*float(residual[i]), h) for i,(s,h) in enumerate(scored)])


def central_context(env, scorer, agent, cache=None):
    """Never empty a busy zone, never use hidden truth, no ML in critic ablation."""
    from blue.core.wrapper import BLUE_AGENTS
    if cache is not None and env._tick in cache:
        shared = cache[env._tick]
    else:
        parts = []
        for a in BLUE_AGENTS:
            X = scorer.observe(env, a)
            F = np.concatenate((neural_rows(X, scorer.bundle['mean'], scorer.bundle['scale']),
                                np.isfinite(X).astype(np.float32)), axis=-1)
            parts.extend((F.mean(0), F.max(0), np.array([float(a in env._awaiting)])))
        shared = np.concatenate(parts).astype(np.float32)
        if cache is not None:
            cache[env._tick] = shared
    out = np.concatenate((shared, np.array([float(agent == a) for a in BLUE_AGENTS]),
                          np.array([env._tick/env.episode_limit]))).astype(np.float32)
    if out.shape != (CRITIC_DIM,) or not np.isfinite(out).all():
        raise ValueError('invalid centralized visible context')
    return out
