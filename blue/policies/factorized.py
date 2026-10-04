"""Factorized host-then-command actor (proposal 02).

Same EPyMARL interface as RNNAgent -- ``forward(inputs, hidden)`` returns
(155 logits, next hidden) with the same GRU recurrence over the flat obs --
but the final head is factorized instead of one dense 155-way layer:

- shared per-slot encoder over the 51 host slots,
- host selector (one score per slot),
- conditional command head (Analyse/Remove/Restore per slot),
- global Sleep/Monitor head.

Combined flat logits (action order mirrors the wrapper: 0 Sleep, 1 Monitor,
then 2+3*slot+cmd):
  flat[0] = sleep, flat[1] = monitor,
  flat[2+3*s+c] = host_score[s] + cmd_score[s, c]

Because the output is still plain 155-way logits, masking, Boltzmann
sampling, PPO log-probs, and agent.th save/load all work unchanged; no
vendor edit is needed (register at runtime, see blue.training.mappo).

Slot width derives from input_shape so root-session ablations keep working:
slot_feats = (input_shape - n_agents) // 51. Requires args.n_agents
(BasicMAC's config namespace always carries it).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SLOTS = 51
N_CMD = 3  # Analyse, Remove, Restore
SLOT_EMB = 32


class FactorizedRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        # Optional return-to-go conditioning (proposal 14, RvS): trailing
        # rtg_dim dims are consumed by fc1/GRU only, never by the slot/id
        # split. Default 0 reproduces the exact original layout.
        self.rtg_dim = int(getattr(args, "rtg_dim", 0) or 0)
        slot_width = input_shape - self.n_agents - self.rtg_dim
        assert slot_width % N_SLOTS == 0, (
            f"input {input_shape} with {self.n_agents} id dims and "
            f"{self.rtg_dim} RTG dims is not {N_SLOTS} slots wide")
        self.slot_feats = slot_width // N_SLOTS
        hidden = args.hidden_dim
        ctx = hidden + self.n_agents

        self.fc1 = nn.Linear(input_shape, hidden)
        if args.use_rnn:
            self.rnn = nn.GRUCell(hidden, hidden)
        else:
            self.rnn = nn.Linear(hidden, hidden)
        self.slot_enc = nn.Linear(self.slot_feats, SLOT_EMB)
        self.host_head = nn.Linear(SLOT_EMB + ctx, 1)
        self.cmd_head = nn.Linear(SLOT_EMB + ctx, N_CMD)
        self.global_head = nn.Linear(ctx, 2)
        # Optional cross-slot self-attention (proposal 03, supervised use):
        # lets slots compare staleness/suspicion pairwise instead of routing
        # through the dense bottleneck. 0 = off (behavior identical).
        # Attention runs on slot embeddings only (ctx is slot-constant).
        self.attn_layers = int(getattr(args, "attn_layers", 0) or 0)
        if self.attn_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=SLOT_EMB, nhead=4, dim_feedforward=128,
                batch_first=True)
            self.slot_attn = nn.TransformerEncoder(
                layer, num_layers=self.attn_layers)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def decompose(self, inputs, hidden_state):
        """(flat_logits, next_hidden, host, cmd, glob) for aux losses."""
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        b = inputs.shape[0]
        flat_slots = N_SLOTS * self.slot_feats
        slots = inputs[:, :flat_slots].reshape(b, N_SLOTS, self.slot_feats)
        # Id block sits between slots and the optional trailing RTG dims;
        # heads see exactly the n_agents id dims as before.
        ids = inputs[:, flat_slots:flat_slots + self.n_agents]
        emb = F.relu(self.slot_enc(slots))
        ctx = torch.cat([h.unsqueeze(1).expand(-1, N_SLOTS, -1),
                         ids.unsqueeze(1).expand(-1, N_SLOTS, -1)], dim=-1)
        if self.attn_layers > 0:
            emb = self.slot_attn(emb)
        he = torch.cat([emb, ctx], dim=-1)
        host = self.host_head(he).squeeze(-1)          # (b, 51)
        cmd = self.cmd_head(he)                        # (b, 51, 3)
        glob = self.global_head(torch.cat([h, ids], dim=-1))  # (b, 2)
        flat = torch.empty(b, 2 + N_SLOTS * N_CMD, device=inputs.device,
                           dtype=inputs.dtype)
        flat[:, :2] = glob
        flat[:, 2:] = (host.unsqueeze(-1) + cmd).reshape(b, -1)
        return flat, h, host, cmd, glob

    def forward(self, inputs, hidden_state):
        flat, h, _, _, _ = self.decompose(inputs, hidden_state)
        return flat, h
