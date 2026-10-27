"""State-conditioned router over set pooling and candidate attention.

The router emits one gate per decision state, then blends both branches'
candidate residual scores. A gate near zero favors SetActor; a gate near one
favors candidate self-attention. Only candidate features, heuristic scores,
and the validity mask are inputs.
"""
from __future__ import annotations

import torch
from torch import nn

from blue.harness.attention_actor import CandidateAttentionActor
from blue.harness.redesign import FEATURE_NAMES, SetActor


class SituationRouterActor(nn.Module):
    """Mix set and attention residuals using a learned state-level router."""

    def __init__(self, set_hidden: int = 64, attention_hidden: int = 42,
                 attention_heads: int = 3, router_hidden: int = 32,
                 fixed_gate: float | None = None):
        super().__init__()
        self.set_branch = SetActor(set_hidden)
        self.attention_branch = CandidateAttentionActor(
            attention_hidden, attention_heads)
        # Masked mean/max candidate context, mean/std of the heuristic score,
        # and log candidate count. The gate is shared across candidates for
        # each decision state, so it routes mechanisms by situation.
        context_dim = 2 * len(FEATURE_NAMES) + 3
        self.router = nn.Sequential(
            nn.Linear(context_dim, router_hidden), nn.GELU(),
            nn.Linear(router_hidden, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        if fixed_gate is not None and not 0.0 <= fixed_gate <= 1.0:
            raise ValueError("fixed_gate must be in [0,1]")
        self.fixed_gate = fixed_gate
        self._capture_gates = False
        self._gate_history: list[float] = []

    def begin_gate_capture(self):
        self._gate_history = []
        self._capture_gates = True

    def end_gate_capture(self):
        self._capture_gates = False
        values = list(self._gate_history)
        self._gate_history = []
        return values

    def forward(self, features: torch.Tensor, base: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        single = features.ndim == 2
        if single:
            features, base, mask = features[None], base[None], mask[None]
        if (features.ndim != 3 or
                features.shape[-1] != len(FEATURE_NAMES)):
            raise ValueError("features must have shape (batch,candidates,feature_dim)")
        if base.shape != features.shape[:2] or mask.shape != features.shape[:2]:
            raise ValueError("base and mask must match batch/candidate dimensions")
        valid = mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("each decision needs at least one valid candidate")

        set_scores = self.set_branch(features, base, mask)
        attention_scores = self.attention_branch(features, base, mask)
        safe_features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
        count = valid.sum(dim=1, keepdim=True).to(features.dtype)
        mean = safe_features.sum(dim=1) / count
        maximum = features.masked_fill(~valid.unsqueeze(-1), -torch.inf).max(1).values
        score = (base / 4.0).masked_fill(~valid, 0.0)
        score_mean = score.sum(dim=1, keepdim=True) / count
        score_var = (((score - score_mean) ** 2) * valid).sum(
            dim=1, keepdim=True) / count
        context = torch.cat((mean, maximum, score_mean,
                             score_var.clamp_min(0).sqrt(),
                             torch.log1p(count)), dim=1)
        gate = (torch.sigmoid(self.router(context)) if self.fixed_gate is None
                else torch.full_like(score_mean, self.fixed_gate))
        mixed = (1.0 - gate) * set_scores + gate * attention_scores
        mixed = mixed.masked_fill(~valid, 0.0)
        if self._capture_gates:
            self._gate_history.extend(gate.detach().cpu().flatten().tolist())
        return mixed[0] if single else mixed
