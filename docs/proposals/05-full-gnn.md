# Proposal 05: Full GNN policy (cybermonic-style)

- Status: parked (valid, not now)
- Date: 2026-10-02
- Source: King, Bowman & Huang, arXiv:2509.16151; code
  `github.com/cybermonic/cage-4-submission` (GPL-2.0, retain attribution).

## Description

Per-agent temporal attributed graph (Hosts/Routers/Ports/Files/Internet),
2-layer GCN + self-attention global vector g, actor emitting per-node /
per-edge / per-global action matrices (V_a x 3, E_a x 2, 1 x 1), critic on
g, independent PPO per agent. ~50k episodes to converge.

## Motivation / evidence

Highest-performing non-heuristic CC4 entry, 5th overall (-193.68; paper
reports -193 ± 84). The per-node factorized action head is exactly the
family proposal 02 adopts without the graph encoder. Transductive (flatten)
variants collapse on node reorder (-1950); inductive variants survive
(<2% change) — relational bias is real for topology shift.

## Experiment

Not run by us. Published numbers only (see comparability rules: 100 x 500
episodes, official build).

## Pros

- Best published learned policy on CC4; open code to crib the head design.
- Genuine topology generalization (variable sizes, reorder-robust) — the
  only family with that property demonstrated.

## Cons

- Still ~80 points behind heuristics on base (-193 vs -113), and the worst
  stealthy-red collapse in the field (-996): the graph does not buy
  robustness to observation degradation.
- ~50k-episode convergence; ~10x training slowdown vs MLP (reported for the
  MPNN sibling); heavy obs-graph bookkeeping in the wrapper.
- Topology-variance benefit we do not need: our obs is fixed 51-pad with
  masks, and our eval is fixed-topology until the generalization suite (03rd
  priority) exists.

## Verdict

PARKED. Revisit if/when variable-topology or degraded-observation scenarios
become the eval target and proposals 01–03 stall there. If revisited, port
the factorized head first (02), encoder second.
