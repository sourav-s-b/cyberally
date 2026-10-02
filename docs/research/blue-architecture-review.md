# Blue architecture review: what to build after the flat MLP

Date: 2026-10-02. Branch: `blue/mappo-training`.
Companion: `docs/research/blue-sparse-reward-literature.md` (why fine-tuning
fails); this doc answers *what architecture replaces the flat head*.

## 1. The decision in one paragraph

The flat 155-way MLP/GRU head cannot express cross-slot argmax ("which of 51
slots is stalest / most suspicious") except by routing a global comparison
through a shared dense bottleneck — BC stalls at ~12% non-sleep accuracy and
MAPPO fine-tuning never beats the teacher. The official CAGE-4 analysis (Kiely
et al., AAAI 2025) shows all three winning teams used sweep-and-remediate
heuristics, and the best MARL entry (a GNN with factorized per-node action
heads) still lost by ~80 points. So the evidence says: **keep the heuristic
skeleton as a hard floor, learn only the ordering (which host next), and fix
the actor shape (host-then-command factorization, optionally with a small
entity-attention encoder) so the learned part can express what the heuristic
does.** Full GNNs, learned hierarchy masters, curiosity, and shaped training
rewards are all rejected or deferred with reasons below.

## 2. Ground truth: official CAGE-4 results (primary source)

Kiely et al., "Exploring the Efficacy of MARL for Autonomous Cyber Defence:
A CAGE Challenge 4 Perspective", AAAI 2025, 39(28):28907–28913.
Eval: 100 episodes x 500 steps, FiniteStateRed. Extracted from the paper PDF.

| Team | Type | CC4 base | Const-size | +Phishing | StealthyRed | AggressiveRed |
|---|---|---|---|---|---|---|
| UC | heuristic | **-113 ± 35** | -101 ± 36 | -534 ± 347 | **-153 ± 112** | -508 ± 277 |
| lancer | heuristic | -118 ± 40 | **-71 ± 23** | -232 ± 79 | -871 ± 627 | -801 ± 236 |
| punch | heuristic | -142 ± 44 | -94 ± 20 | **-205 ± 78** | -158 ± 46 | -837 ± 203 |
| cybermonic | MARL (GNN) | -193 ± 84 | -176 ± 51 | -350 ± 182 | -996 ± 482 | -654 ± 244 |

Findings that constrain our design (all quotes/paraphrases verified vs PDF):

- **Heuristics swept the podium.** "Across all experiments ... the
  heuristic-based agents outperformed all MARL-based submissions."
- **Team UC (winner):** rule-based; event filtering; revised obs vector with
  one-hot malicious process/connection/file flags persisting until
  Restore/Remove; reactive Restore on connection-flagged hosts,
  Restore/Remove on process/file flags; proactive Allow/BlockTrafficZone per
  comms policy; else Analyse/Monitor/DeployDecoy (random host); Analyse
  targets the longest-since-analysed/restored host not subject to Remove.
  Only team that encoded per-episode present users/servers in obs, eliminating
  invalid actions from random network sizes.
- **Team lancer (2nd, no ML):** decoy on every host first; round-robin
  Analyse; Restore on `escalate.sh`, Remove on `cmd.sh`; per-host float
  priority (init configurable, decayed after Analyse/Restore/Remove, boosted
  by Monitor observations); always serves the highest-priority host. This is
  the closest published analog of our `HybridBluePolicy(priority_fn)`.
- **Team punch (3rd, "much better than our RL agents"):** identical
  analyse-then-restore round-robin per subnet; extra wrapper reads raw CybORG
  obs and sets a host flag when any file has density > 0.9; state machine on
  step count + state. Concrete feature-engineering precedent: a file-density
  signal belongs in our belief features.
- **Team cybermonic (4th, best non-heuristic):** temporal attributed graph
  (Hosts/Routers/Ports/Files/Internet), per-agent partial graph, actions as
  graph edits, GNN + global vector g, actor outputs node/edge action
  probabilities, critic on g, independent PPO per agent.
- **Invalid actions decided rankings.** Random users/servers per episode made
  every team except UC act on nonexistent hosts; fixing size to max reshuffled
  the board (lancer -71 best). "All submitted MARL solutions failed to learn
  how to determine the network configuration at the beginning of each
  episode." Our padded-51 + validity-mask design already handles this; keep it.
- **Generalization variants break everyone** (parameters never seen by teams):
  phishing rate 0.01->0.75, stealthy red (FP 0.01->0.75, discovery detection
  collapsed), aggressive red (all red durations -> 1 step). No agent, heuristic
  or MARL, holds up. These variants are our future held-out suite — the real
  differentiator, since base-scenario scores are saturating.

Calibration to our numbers: our round-robin is -93.5 ± 27.8 over 400-step
episodes (-0.234/step) vs UC -113/500 (-0.226/step). Same band, with the
caveat that simulator version, steps, and seeds differ — do not claim parity,
only that we are in the winner's band on the base scenario and must look to
generalization + learned ordering for further gains.

## 3. Candidate families (all primary-sourced)

### 3a. Factorized host-then-command actor — SHORTLIST #1 (build second)

- **SR-DRL on CAGE-2** (Nyberg & Johnson, arXiv:2407.05775, implements
  Janisch et al. 2009.12462): joint policy factored as host selector
  `pi1(host|H) = softmax(phi1(h))` over N nodes, then command head
  `pi2(cmd|h_i)`; critic on pooled g (train-only). MPNN (max-agg + residual,
  3-4 layers best), SB3 PPO, 500k steps, trained vs Meander only. Zero-shot
  over 7 topology variants (10-16 hosts): L-3 -113 ± 63 vs per-variant MLP
  ensemble -78 ± 42 vs untrained MLP -2104 (cannot even run on new sizes).
  Claim is zero-shot function, not beating a specialist. 10x slower training
  than MLP; no red-shift coverage.
- **GACD** (Ramamurthy & Dhir, arXiv:2506.22706): same host->action family
  with two PPO nets + Graphormer/VGAE + fused-Gromov-Wasserstein OT map.
  "On par" with non-OT GNNs; wins only on reorder-robustness. Heavy machinery
  for a fixed-topology problem — reject the OT half, keep the factorization.
- Mapping to us: replace one 155-way softmax with a 51-way host selector +
  small conditional command head (+ global Sleep/Monitor gate). Preserves
  per-slot validity masking (mask inside pi1). BC-distillable from
  round-robin traces, then short MAPPO. Effort M. Directly tests the
  cross-slot-argmax hypothesis with learning.

### 3b. Entity attention encoder (REFIL-lite / Set-Transformer-lite) — SHORTLIST #2 (build with #1)

- **REFIL** (Iqbal et al., ICML 2021): entity-wise feedforward + multi-head
  attention `MHA(A,X,M)` gated by observability mask M; padding/nonexistent
  entities are masked columns, so one weight set handles variable counts.
  The reported gain came from the auxiliary imagined-subgroup loss, *not*
  attention alone; attention-only transfer is the smaller-gain path. Code is
  PyMARL-based (EPyMARL's predecessor) — portable. Effort M (S for
  attention-only, no aux loss: there is no PPO equivalent of the QMIX aux
  loss; defer it).
- **Set Transformer for MARL** (NeurIPS 2022): Theorem 1 — pairwise
  self-attention is exponentially hard for DeepSets/MLP-bottleneck
  (`rho(sum phi)`) forms. This is the formal version of our diagnosis and
  the theory cover for a shared per-slot encoder + 1-2 attention layers +
  masked softmax. 51^2 = 2601 attention weights: cheap. Caveat: pure
  permutation invariance discards slot identity (subnet/zone matters) — add
  typed slot embeddings. Attention is less sample-efficient under sparse
  reward; keep it to 1-2 layers.
- Mapping to us: 51 slots = entities; reshape 872 -> [51,17] + id context;
  `eFF -> MHA -> eFF`, per-slot scorer, masked softmax; critic = same encoder
  + mean pool. Pairs naturally with #1 (attention encoder feeds factorized
  heads). Effort S/M.

### 3c. TERLA semantic action collapse — DEFER (adopt only 2 pieces now)

- **TERLA** (Dudman & Bull, CAMLIS 2025, PMLR 299:87-109): 2 HGT layers + sum
  pool -> 70-dim latent from a Mission/Subnet/Host graph with 2 IDS bits per
  host; policy picks among **5 semantic actions** resolved to hosts by IDS-rank
  targeting (first match; drops Monitor and Block/Allow). Action waiting
  (withhold obs while an action is pending, keep streaming rewards) fixes the
  silent-Sleep MDP corruption. Trains on shaped per-agent segment-health
  deltas (OT x2), evaluates on native shared reward. RLlib PPO, ~1M steps.
  Results (their build; absolutes NOT comparable to ours): sleep -6650 <
  shaping-only -6300 < random -5150 < vanilla PPO -2825 ~= separate TERLA
  -2773 < single shared TERLA -2048; TERLA acts 6-7% of steps vs PPO ~33%.
- Adopt now: **action waiting** (audit ours — we already fail explicitly on
  overdue actions, keep that behavior) and **IDS-ranked targeting as an
  idea**. Defer the 5-action collapse (drops Monitor/firewall control; crude
  first-match targeting caps the ceiling) and the shaped training reward
  (uses privileged red-session counts — defensible only as training-only
  auxiliary with native-reward selection, needs an explicit decision, never
  silent adoption). Effort M for the encoder port to EPyMARL.

### 3d. Full GNN policy (cybermonic / TERLA encoder) — PARK

- Best non-heuristic, yet -80 behind heuristics on base and collapses worst
  on stealthy red (-996). ~50k-episode convergence, ~10x training slowdown,
  topology-variance benefit we do not need (fixed 51-pad). Revisit only if
  generalization scenarios become the target and #1/#2 stall there.

### 3e. Hierarchical RL master — REJECT (keep our rules = the expert master)

- **Mindrake CAGE-1 winners** (-30.07, HPPO + curiosity): 2 per-red PPO
  sub-agents + per-step PPO selector on one raw step; ICM curiosity on
  sub-agents ("nearly double" on b-line discovery). Pipeline: tune -> train
  sub -> freeze -> train controller.
- **Mindrake CAGE-2** (3rd): per-episode controller; RL -57.05 vs heuristic
  first-4-steps classifier -57.29 vs bandit -57.46 — learned gating is noise
  when the choice is fingerprintable.
- **CardiffUni CAGE-2 winners** (-54.57): per-red HPPO + greedy decoys, where
  brute-forced greedy decoys did more work than the RL.
- **H-MARL on CC4** (Singh et al., AAMAS'25; arXiv:2410.17351; code
  adityavs14/Hierarchical-MARL): Investigate/Recover/ControlTraffic
  sub-policies + master. Expert rule master (IOCs -> Recover) -129.53 beats
  flat IPPO -181.62; learned Meta master ~= -181.62 but trains 3-5x faster;
  never co-train both levels (Collective failed).
- Verdict: per-red experts are N/A (single FiniteStateRed); learned gating
  adds ~0 where rules suffice; our fixed CONFIRMED/VERIFY rules *are* the
  expert master and the learned priority scorer *is* the learned low level.
  Only cheap extension queued: a bandit/heuristic sweep-vs-remediate mode
  gate (S effort) if the hybrid plateaus.

### 3f. Rejected outright

- **Curiosity (ICM/RND):** solves discovery sparsity we do not have
  (round-robin sweep covers discovery; -93.5 proves it).
- **Factored-additive Q** (NeurIPS 2022 offline RL): assumes simultaneously
  executed sub-actions; our 1-of-155 exclusive choice makes `Q = q_slot +
  q_cmd` misspecified exactly where it matters. Usable only as critic
  variance-reduction lens.
- **CyberDreamcatcher GAT+REINFORCE** (arXiv:2501.14700, WITHDRAWN):
  weaker than an MLP baseline on its own task, custom fork, REINFORCE-only.

## 4. Decision matrix

| # | Candidate | Evidence | Effort | Risk | Verdict |
|---|---|---|---|---|---|
| 1 | Hybrid: fixed rules + learned per-host priority (Lancer++/H-MARL-Expert-shaped) | 3/3 CAGE-4 winners; H-MARL Expert +52 over flat IPPO; lancer priority ~= our `priority_fn` slot | S | Low; scorer can underperform round-robin (measurable in isolation) | **BUILD FIRST** |
| 2 | Host-then-command factorized actor in EPyMARL MAPPO | SR-DRL/GACD; tests argmax hypothesis with learning; BC-distillable | M |pad/mask plumbing; PPO joint-logprob bookkeeping | **BUILD SECOND** |
| 3 | 1-2 layer entity-attention encoder feeding #2 | Set-Transformer Thm 1 (formal diagnosis); REFIL mask pattern; cheap (51^2) | S/M | sample efficiency; keep slot identity via typed embeddings | **WITH #2** |
| 4 | TERLA-style 5-action semantic collapse | -2048 vs -2825 relative; 6-7% action rate | M | drops Monitor/firewall; crude targeting; shaped-reward honesty flag | DEFER; take action-waiting audit only |
| 5 | Full GNN policy | best non-heuristic, -80 behind; 50k eps; 10x slow | L | worst stealthy-red collapse (-996) | PARK |
| 6 | Learned hierarchy master / bandit gate | Mindrake null result; H-MARL Meta ~= flat | S (bandit) / M (PPO) | non-stationarity if co-trained | REJECT PPO master; queue bandit gate |
| 7 | Curiosity / factored-additive-Q / withdrawn-GAT | — | — | — | REJECT with reasons above |

## 5. Recommended sequence

1. **Harden the hybrid (now, S).** Implement lancer-style per-host priority
   (decay on touch, boost on Monitor hits) + punch-style file-density>0.9
   host flag + UC-style persistent malicious-event flags in Blue-visible
   features. Prove `HybridBluePolicy(None)` == round-robin parity, then train
   a tiny risk scorer on Blue-visible 17-feat vectors with privileged labels
   as *training targets only*. >=8-seed native eval. Target: beat -93.5 and
   approach lancer's constant-size -71 band on equal footing.
2. **Factorized actor (next, M).** Host selector + conditional command head
   (+ global gate), optional 1-layer entity attention encoder; BC-distill
   from round-robin traces, short MAPPO, compare vs hybrid on >=8 seeds.
3. **Generalization suite (held-out).** Phishing/stealthy/aggressive-style red
   variants as eval-only scenarios — where every published agent fails; this
   is the actual leaderboard differentiator, not base-scenario points.
4. **Non-goals:** full GNN, learned PPO master, curiosity, shaped training
   rewards (native-only unless explicitly approved), any privileged input to
   actors/masks.

## 6. Score-comparability caveats (read before citing any number)

- Our episodes are 400 steps; official CAGE-4 is 500. Compare per-step rates
  (-0.234 vs -0.226/step), never raw totals, and never claim parity.
- TERLA (-2773/-2048) and H-MARL (-129.53/-181.62) absolutes come from
  different builds/configs; only their *relative deltas* transfer.
- All 3-seed comparisons remain inside the ±313 noise band (see
  blue-sparse-reward-literature §noise floor); >=8 seeds for every claim.

## Sources

- Kiely et al., AAAI 2025 (35158), paper PDF via
  `https://ojs.aaai.org/index.php/AAAI/article/download/35158/37313`
  (team sections + Table 2 + generalization experiments; extracted locally).
- King, Bowman & Huang, arXiv:2509.16151 (cybermonic GNN; CC4 §5.3; code
  `https://github.com/cybermonic/cage-4-submission`, GPL-2.0).
- Dudman & Bull, PMLR 299:87-109 (TERLA;
  `https://proceedings.mlr.press/v299/dudman25a.html`,
  `https://arxiv.org/html/2511.09114v2`).
- Nyberg & Johnson, arXiv:2407.05775 (SR-DRL/MPNN on CAGE-2).
- Ramamurthy & Dhir, arXiv:2506.22706 (GACD).
- CyberDreamcatcher, arXiv:2501.14700 (WITHDRAWN v4 — findings discounted).
- Zhang et al., NeurIPS 2022 (Set Transformer relational reasoning).
- Iqbal et al., ICML 2021, PMLR v139 (REFIL; code
  `https://github.com/shariqiqbal2810/REFIL`, PyMARL-based).
- Tang et al., NeurIPS 2022 (factored action spaces, offline RL).
- Mindrake: `alan-turing-institute/cage-challenge-1-public`,
  `cage-challenge-2-public`; CAGE-1/2 result tables;
  CardiffUni HPPO + greedy decoys (CAGE-2 winners).
- Singh et al., AAMAS 2025 / arXiv:2410.17351 (H-MARL on CC4; code
  `https://github.com/adityavs14/Hierarchical-MARL`).
