"""Architecture study: fixed visible features, execution and protected rules.

Memory summarizes ALL owned hosts each tick, including busy ticks. PPO replays
complete per-agent episode sequences; saved rollout hidden states are NOT used
as detached training inputs. Graph edges come exclusively from Blue views.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from blue.harness.redesign import FEATURE_NAMES

WIDTH = len(FEATURE_NAMES)


def observed_edges(env, agent, hosts):
    views = env.views[agent]
    addresses = {}
    for host in hosts:
        for interface in views.get(host, {}).get('Interface', []) or []:
            address = interface.get('ip_address')
            if address is not None:
                addresses[str(address)] = host
    ix = {host: i for i, host in enumerate(hosts)}
    adjacency = np.eye(len(hosts), dtype=np.float32)
    for host in hosts:
        for process in views.get(host, {}).get('Processes', []) or []:
            for connection in process.get('Connections', []) or []:
                other = addresses.get(str(connection.get('remote_address')))
                if other in ix:
                    adjacency[ix[host], ix[other]] = 1.
                    adjacency[ix[other], ix[host]] = 1.
    return adjacency


def pool(z, mask):
    present = mask.bool().unsqueeze(-1)
    mean = z.masked_fill(~present, 0).sum(-2) / present.sum(-2).clamp(min=1)
    maximum = z.masked_fill(~present, -torch.inf).max(-2).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return mean, maximum


class StudyActor(nn.Module):
    def __init__(self, kind='mlp', hidden=64):
        super().__init__()
        if kind not in ('mlp', 'attention', 'gru', 'graph') or hidden < 1:
            raise ValueError('invalid architecture')
        self.kind, self.hidden = kind, hidden
        self.encoder = nn.Sequential(nn.Linear(WIDTH+1, hidden), nn.Tanh())
        if kind == 'attention':
            if hidden % 4:
                raise ValueError('attention width must divide four heads')
            layer = nn.TransformerEncoderLayer(hidden, 4, 2*hidden, dropout=0.,
                                               batch_first=True, activation='gelu')
            self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        if kind == 'graph':
            self.messages = nn.ModuleList([nn.Linear(2*hidden, hidden) for _ in range(2)])
        if kind == 'gru':
            self.memory = nn.GRU(2*WIDTH, hidden, batch_first=True)
        self.body = nn.Sequential(nn.Linear((4 if kind == 'gru' else 3)*hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def memory_step(self, summary, state=None):
        if self.kind != 'gru':
            raise ValueError('no recurrent state')
        _, next_state = self.memory(summary.reshape(1, 1, -1),
                                   None if state is None else state.reshape(1, 1, -1))
        return next_state.reshape(-1)

    def history_states(self, summaries):
        return self.memory(summaries.unsqueeze(0))[0].squeeze(0)

    def forward(self, features, base, mask, adjacency=None, memory=None):
        single = features.ndim == 2
        if single:
            features, base, mask = features[None], base[None], mask[None]
            if adjacency is not None:
                adjacency = adjacency[None]
            if memory is not None:
                memory = memory[None]
        if not mask.bool().any(-1).all():
            raise ValueError('empty candidate set')
        # Mask BEFORE encoding, not only after attention/pooling.
        features = features.masked_fill(~mask.bool().unsqueeze(-1), 0.)
        base = base.masked_fill(~mask.bool(), 0.)
        z = self.encoder(torch.cat((features, (base/4).unsqueeze(-1)), -1))
        if self.kind == 'attention':
            z = self.attention(z, src_key_padding_mask=~mask.bool())
        if self.kind == 'graph':
            if adjacency is None:
                raise ValueError('graph requires declared visible edges')
            valid = mask[:, :, None]*mask[:, None, :]
            a = adjacency*valid
            a = a / a.sum(-1, keepdim=True).clamp(min=1)
            for layer in self.messages:
                z = (z + torch.tanh(layer(torch.cat((z, a@z), -1)))) / 2**.5
        mean, maximum = pool(z, mask)
        parts = [z, mean[:, None].expand_as(z), maximum[:, None].expand_as(z)]
        if self.kind == 'gru':
            if memory is None:
                raise ValueError('GRU requires replayed observation memory')
            parts.append(memory[:, None].expand_as(z))
        out = self.head(self.body(torch.cat(parts, -1))).squeeze(-1)
        return out[0] if single else out


def marginal(logits, base, mask, teacher, bonus=.25, temperature=.5, explore=.2):
    residual = torch.tanh(logits)
    scores = ((base + bonus*residual)/temperature).masked_fill(~mask.bool(), -torch.inf)
    p = explore*torch.softmax(scores, -1)
    anchor = torch.zeros_like(p)
    if p.ndim == 1:
        anchor[int(teacher)] = 1.
    else:
        anchor.scatter_(1, torch.as_tensor(teacher, device=p.device)[:, None], 1.)
    return p + (1-explore)*anchor
