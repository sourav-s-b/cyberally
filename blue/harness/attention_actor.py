"""Small candidate-set attention actor for the architecture comparison.

This module is deliberately separate from the frozen SetActor/Qwen runner.
It has the same forward contract (features, heuristic base scores, validity
mask) and returns a per-candidate residual score. It consumes only the
Blue-visible candidate features supplied by the caller.
"""
from __future__ import annotations

import torch
from torch import nn

from blue.harness.redesign import FEATURE_NAMES


class CandidateAttentionActor(nn.Module):
    """Permutation-equivariant masked attention over legal host candidates."""

    def __init__(self, hidden: int = 64, heads: int = 4):
        super().__init__()
        if hidden < 1 or heads < 1 or hidden % heads:
            raise ValueError("hidden must be positive and divisible by heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.input = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES) + 1, hidden), nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.ff_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        # The untrained model exactly reduces to the supplied heuristic base.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor, base: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        single = features.ndim == 2
        if single:
            features, base, mask = features[None], base[None], mask[None]
        if features.ndim != 3 or features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("features must have shape (batch, candidates, feature_dim)")
        if base.shape != features.shape[:2] or mask.shape != features.shape[:2]:
            raise ValueError("base and mask must match batch/candidate dimensions")
        valid = mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("each row needs at least one valid candidate")

        token = self.input(torch.cat((features, (base / 4.0).unsqueeze(-1)),
                                     dim=-1))
        attended, _ = self.attn(token, token, token,
                                key_padding_mask=~valid,
                                need_weights=False)
        token = self.norm(token + attended)
        token = self.ff_norm(token + self.ff(token))
        residual = self.head(token).squeeze(-1)
        # Invalid candidates are neutral at this layer; the caller still owns
        # final masking when constructing the action distribution.
        residual = residual.masked_fill(~valid, 0.0)
        return residual[0] if single else residual
