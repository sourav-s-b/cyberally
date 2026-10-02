# Proposal 03: Entity-attention encoder (1–2 layers)

- Status: proposed (design only, no code)
- Date: 2026-10-02

## Description

Shared per-slot encoder over the 51 x 17 observation (+ agent-id context),
1–2 multi-head self-attention layers across slots, per-slot scorer, masked
softmax over slots. Critic reuses the encoder + mean pool. Feeds proposal
02's factorized heads (or the hybrid scorer in 01). Padding slots masked
throughout, REFIL-style.

## Motivation / evidence

- Set Transformer for MARL (NeurIPS 2022), Theorem 1: pairwise self-attention
  is exponentially hard for DeepSets/MLP-bottleneck forms — the formal
  version of our diagnosis. PAC-Bayes bounds independent of slot count N.
- REFIL (ICML 2021): entity-wise feedforward + `MHA(A,X,M)` with
  observability mask M gives variable-count handling and decentralized
  execution; PyMARL-based code ports toward EPyMARL. Honest caveat from
  their ablations: the gain came from the auxiliary subgroup loss, not
  attention alone — attention-only transfer is the smaller-gain path, and
  there is no PPO equivalent of that aux loss, so we transfer architecture
  only.
- Cost is trivial: 51^2 = 2601 attention weights per head.

## Experiment

Not yet run. Planned with 02: attention-only encoder, no aux loss; compare
vs MLP encoder under identical factorized heads, >= 8 seeds.

## Pros

- Gives the policy pairwise slot comparison (staleness/suspicion *relative*
  to other slots) instead of forcing it through a dense bottleneck.
- Mask pattern handles padding cleanly and generalizes to variable host
  counts for free.
- Small, composable with 01 and 02.

## Cons

- Pure permutation invariance discards slot identity (subnet/zone/host-index
  matter here) — requires typed slot embeddings, or it re-creates a
  symmetric bottleneck of a different kind.
- Attention is less sample-efficient under sparse native reward; extra params
  can hurt before they help. 1–2 layers max, with an MLP-encoder control.
- No on-policy PPO evidence at our scale; theory is offline/pessimistic.

## Verdict

BUILD WITH 02 as the encoder candidate, against an MLP-encoder control.
Do not build standalone — its value is only measurable through a head that
uses cross-slot comparison.
