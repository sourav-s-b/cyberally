# Blue extraction-lottery literature survey (2026-10-02, second sweep)

Purpose: the first survey (`blue-rl-recovery-literature.md`, 10 tracks,
~260 papers) attacked "how to beat the teacher." This one attacks what
actually happened: the extraction lottery, the advantage spike/collapse
duality, the transfer gap, and the missing validation discipline. Seven
of eight tracks returned, ~200 entries.

Method: 8 researcher subagents, deep web search + page fetch,
instructed to DROP unverifiable citations rather than invent them.

VERIFICATION CAVEAT (same as sweep 1): entries passed a
search-existence check, not a read-the-paper check. Fetch the DOI/arXiv
ID before citing, sourcing, or implementing. Duplicates with sweep 1 are
noted in §6.

## Our failure modes (labels used throughout)

- F1 Extraction lottery: identical data/budget, train seeds 0/1/2 give
  regression -82.0 / -105.6 / -108.1; Bellman error spreads 4-5x
  (1.6 / 8.4 / 6.4); single-seed conclusions keep reversing.
- F2 Advantage duality: unstabilized Q/V spikes (adv_max 176, std 2.9,
  1.5% clipped) → fixation tails (-826, -319); stabilized Q/V
  (LayerNorm + 1/100 scale) collapses advantages to ~0 (std 0.01) →
  extraction degenerates to BC (which also tails: -543, -230).
  Static `--awr-alpha 0.5` blend is a hack, not a solution.
- F3 Transfer gap: IQL +11.5 paired on regression → -10.9 on held-out
  (uniform mild degradation, no catastrophe); wins land on hard seeds
  (7805 -34, 7806 -139), losses on easy ones. RvS/attention same shape.
- F4 No validation discipline: BC, RvS, IQL extraction all train fixed
  30 epochs on all data; in-sample match % is the only metric and does
  not predict return.
- F5 Coverage/decisiveness tradeoff: pure AWR narrows sweeps (66 vs 86
  hosts); uniform cloning preserves coverage but can't exceed.
- F6 Suspected recurrent drift: feedforward Q/V degrade mildly, GRU
  policy tails catastrophically — hidden-state OOD hypothesized,
  untested.

## Highest-leverage bets (synthesis)

1. Install validation splits + early stopping FIRST (F4, attacks all).
   Mandlekar (best val loss 50-100% worse than best rollout),
   Kumar Q-peak stopping (no env needed), Codevilla (match% ~0.39
   correlation — our metric is proven unreliable), Bergmeir/Roberts
   (trajectory-level blocked splits; random splits leak). Cheapest,
   de-lotteries everything downstream. Experiment V1.
2. ESS monitor + adaptive temperature (F2). Martino ESS
   ((∑w)²/∑w²) in one line; MPO dual-solved per-batch temperature
   replaces static alpha; CRR binary/exponential filter caps spikes;
   RLPD ESS-triggered rescaling escapes uniformity. Experiment V2.
3. FQE-V(s0) + BVFT selection (F1, held-out hygiene). Paine (rank 256
   policies rollout-free), Zhang-Jiang BVFT (hyperparameter-free,
   discrete-action ideal), Tang-Wiens two-stage (WIS prune → FQE
   rerank). Ends test-env selection and held-out contamination.
   Experiment V3.
4. Snapshot/SWA ensemble (F1). Huang snapshots / Izmailov SWA: bank
   diverse checkpoints from ONE run, weight-average the -82/-161
   swings away; distill to one policy for deployment (Rusu/Hinton).
   Experiment V4.
5. Q-ensemble pessimism (F1/F2). EDAC (min-over-N + diversity),
   MSG (independent targets — our single Q is structurally optimistic),
   REDQ/DroQ subset-min. Experiment V5.
6. Recurrence ablations (F6). Ruiz-Gu state-passing/fitted-noise init
   (unexplored-states hypothesis = our 400-step drift), Kapturowski
   burn-in vs zero-start, Pleines seed-scaling + hidden-refresh,
   Miceli-Barone RND contraction of drifted states. Feedforward-policy
   ablation decides GRU-vs-MLP. Experiment V6.
7. Support-constraint extraction (F5). SPOT (support, not density —
   zero-advantage sweeps stay legal), TD3+BC additive-BC knob
   (replaces alpha-hack tunably), OPAL (latent sweep primitives lock
   support). Experiment V7.
8. Transfer-side fixes (F3). Mediratta (diversity > size — KEEP weak
   tails, now with citation), QDT (CQL-relabeled returns + DT keeps
   stitching and transfer), Weltevrede symmetric consistency.
   Experiment V8.

## Problem → track matrix

| Problem | Primary tracks | Supporting |
|---|---|---|
| F1 lottery | 7 ensembles, 1 OPE/selection | 8 validation |
| F2 duality | 3 AWR mechanics | 7 ensembles, 5 coverage |
| F3 transfer | 4 OOD transfer | 6 recurrence, 5 coverage |
| F4 validation | 8 validation splits, 1 OPE | — |
| F5 coverage | 6 coverage cloning | 3 AWR, 4 transfer |
| F6 drift | 5 recurrence | 8 validation (hidden leakage) |

## Track catalogs

### 1. OPE and offline checkpoint selection (25)

1. Eligibility Traces for Off-Policy Policy Evaluation — Precup, Sutton,
   Singh, 2000, ICML. Relevance: trajectory/per-decision IS baselines
   that explode under long-horizon sparse logs.
2. Doubly Robust Off-policy Value Evaluation — Jiang, Li, 2016, ICML.
   Relevance: model value + IS correction for unbiased low-variance
   checkpoint scoring.
3. Data-Efficient Off-Policy Policy Evaluation — Thomas, Brunskill, 2016,
   ICML. Relevance: WDR+MAGIC blend minimizing MSE for noisy-checkpoint
   selection.
4. High-Confidence Off-Policy Evaluation — Thomas, Theocharous,
   Ghavamzadeh, 2015, AAAI. Relevance: lower confidence bounds rejecting
   lucky/unsafe checkpoints.
5. Breaking the Curse of Horizon — Liu, Li, Tang, Zhou, 2018, NeurIPS.
   Relevance: stationary density ratios replace exploding trajectory
   weights for 400-step episodes.
6. More Robust Doubly Robust OPE — Farajtabar, Chow, Ghavamzadeh, 2018,
   ICML. Relevance: variance-minimizing DR model for seed-robust
   selection.
7. Batch Policy Learning under Constraints — Le, Voloshin, Yue, 2019,
   ICML. Relevance: introduces FQE for rollout-free checkpoint scoring.
8. DualDICE — Nachum, Chow, Dai, Li, 2019, NeurIPS. Relevance:
   behavior-agnostic weights when the logging policy is an unknown
   mixture (ours: 3 teachers).
9. Off-Policy Evaluation via Off-Policy Classification — Irpan et al.,
   2019, NeurIPS. Relevance: sparse-reward ranking without IS or model;
   robust early-stopping signal (SoftOPC).
10. Towards Optimal OPE with Marginalized IS — Xie, Ma, Wang, 2019,
    NeurIPS. Relevance: polynomial-in-horizon MIS theory for 155-action
    long episodes.
11. GenDICE — Zhang, Dai, Li, Schuurmans, 2020, ICLR. Relevance:
    behavior-agnostic stationary-ratio OPE.
12. Minimax Weight and Q-Function Learning for OPE — Uehara, Huang,
    Jiang, 2020, ICML. Relevance: unified MWL/MQL+DR theory for
    estimator choice.
13. OPE via the Regularized Lagrangian (BestDICE) — Yang et al., 2020,
    NeurIPS. Relevance: stabilized DICE optimization for reliable
    ranking.
14. Understanding the Curse of Horizon in OPE — Liu, Bacon, Brunskill,
    2020, ICML. Relevance: when PDIS/MIS fail to beat IS; caution for
    sparse long-horizon selection.
15. Model Selection in RL — Farahmand, Szepesvari, 2011. Relevance:
    naive validation Bellman error misleads; complexity-penalized
    correction.
16. Hyperparameter Selection for Offline RL — Paine et al., 2020, NeurIPS
    Offline RL Workshop. Relevance: FQE V(s0) ranks 256 policies
    without env — direct template.
17. D4RL — Fu et al., 2020. Relevance: fixed-log benchmark exposing the
    seed/hyperparameter lottery.
18. A Deep RL Approach to MIS with Successor Representation — Fujimoto,
    Meger, Precup, 2021, ICML. Relevance: scales MIS ratios to
    high-dim obs.
19. Benchmarks for Deep OPE — Fu et al., 2021, ICLR. Relevance: FQE best
    but brittle; rank-correlation + regret@k protocol for our study.
20. Empirical Study of OPE for RL — Voloshin, Le, Jiang, Yue, 2021,
    NeurIPS. Relevance: 33-method stress test; no universal winner —
    ensemble validators.
21. Towards Hyperparameter-free Policy Selection (BVFT) — Zhang, Jiang,
    2021, NeurIPS. Relevance: projected-Bellman tournament, no FQE
    hyperparameters, discrete-action ideal.
22. Model Selection for Offline RL in Healthcare — Tang, Wiens, 2021,
    MLHC. Relevance: WIS prune + FQE rerank two-stage balances
    cost/accuracy.
23. COMBO — Yu et al., 2021, NeurIPS. Relevance: conservative
    model-rollout pessimistic checkpoint values.
24. Supervised Off-Policy Ranking — Jin et al., 2022, ICML. Relevance:
    ranking (not values) robust to 4-5x Bellman scale shifts.
25. Model Selection for OPE (LSTD-Tournament) — Liu et al., 2025,
    NeurIPS. Relevance: statistically stronger BVFT successor for
    tuning OPE itself.

### 2. AWR mechanics and weight degeneracy (30)

1. Advantage-Weighted Regression — Peng, Kumar, Zhang, Levine, 2019.
   Relevance: canonical AWR with beta + clipping for our spike regime.
2. AWAC — Nair et al., 2020. Relevance: TD-advantage AWR without
   behavior modelling.
3. Critic Regularized Regression — Wang et al., 2020, NeurIPS.
   Relevance: binary/exponential filtering as stable raw-exp
   alternative.
4. IQL — Kostrikov, Nair, Levine, 2021/2022. Relevance: our exact
   extraction stack with beta-controlled AWR.
5. MARWIL — Wang et al., 2018, NeurIPS. Relevance: monotonic exp
   weighting, beta interpolates BC and improvement.
6. Reward-Weighted Regression — Peters, Schaal, 2007, ICML. Relevance:
   seminal EM ancestor of AWR degeneracy.
7. Fitted Q-iteration by AWR — Neumann, Peters, 2008. Relevance: first
   soft-greedy advantage-weighted FQI.
8. Relative Entropy Policy Search — Peters et al., 2010. Relevance:
   KL-bounded E-step yielding exp-advantage policy.
9. MPO — Abdolmaleki et al., 2018, ICLR. Relevance: dual-temperature
   E-step + KL M-step fixing manual beta — replaces alpha-hack.
10. V-MPO — Song et al., 2019/2020. Relevance: top-k advantages + dual
    temperature for variance-robust weighting.
11. TRPO — Schulman et al., 2015. Relevance: monotonic KL trust-region
    foundation for conservative extraction.
12. PPO — Schulman et al., 2017. Relevance: clipped surrogate + advantage
    normalization practice.
13. What Matters in On-Policy RL — Andrychowicz et al., 2020. Relevance:
    per-minibatch advantage/value normalization ablations.
14. Implementation Matters — Engstrom et al., 2020, ICLR. Relevance:
    code-level normalization/clipping drive gains.
15. CQL — Kumar et al., 2020, NeurIPS. Relevance: conservative
    lower-bound Q preventing overoptimistic spikes.
16. BEAR — Kumar et al., 2019, NeurIPS. Relevance: support-constrained
    backups diagnosing fixation collapse.
17. BRAC — Wu et al., 2019. Relevance: KL/MMD behavior-regularization
    framework, fixed/adaptive strengths.
18. TD3+BC — Fujimoto, Gu, 2021, NeurIPS. Relevance: Q-normalized BC
    regularization replacing brittle exp weights.
19. ReBRAC — Tarasov et al., 2023, NeurIPS. Relevance: LayerNorm + depth
    interaction — directly relevant to our LayerNorm-scale finding.
20. RLPD — Ball et al., 2023, ICML. Relevance: LayerNorm critic bounds
    extrapolation; ESS-triggered rescaling escapes uniformity.
21. Layer Normalization — Ba et al., 2016. Relevance: mechanism behind
    our std 2.9→0.01 compression.
22. PopArt (values across magnitudes) — van Hasselt et al., 2016,
    NeurIPS. Relevance: adaptive rescaling alternative to static 1/100.
23. Multi-task RL with PopArt — Hessel et al., 2019. Relevance:
    scale-invariant updates across reward magnitudes.
24. Extreme Q-Learning — Garg et al., 2023, ICLR. Relevance: Gumbel
    analysis of beta sensitivity in AWR extraction.
25. Double Gumbel Q-Learning — Hui et al., 2023, NeurIPS. Relevance:
    heteroscedastic noise model for advantage miscalibration.
26. In-Sample Learning via Implicit Value Regularization — Xu et al.,
    2023, ICLR. Relevance: unifies IQL/XQL/AWR as implicit behavior
    regularizers.
27. In-Sample Softmax — Xiao et al., 2023. Relevance:
    temperature-controlled in-sample softmax without collapse.
28. IDQL — Hansen-Estruch et al., 2023. Relevance: unimodal AWR cannot
    fit multimodal actor; ESS diagnosis + resampling.
29. AlignIQL — He et al., 2024. Relevance: formal condition for AWR
    weight matching Q-implied policy.
30. Effective Sample Size for IS — Martino et al., 2017. Relevance: ESS
    diagnostics ((∑w)²/∑w², perplexity) for weight degeneracy —
    one-line monitor.

### 3. Offline transfer and OOD generalization (33)

1. CQL — Kumar et al., 2020. (Pessimism baseline; transfer cost.)
2. IQL — Kostrikov et al., 2022. (In-sample values, seed overfit.)
3. BRAC — Wu et al., 2019. (Distribution vs support constraints.)
4. BEAR — Kumar et al., 2019. (Support constraint allows beyond-support
   generalization.)
5. BCQ — Fujimoto et al., 2019, ICML. (Extrapolation error as transfer
   failure cause.)
6. MOPO — Yu et al., 2020, NeurIPS. (Uncertainty-penalized model
   rollouts for OOD states.)
7. MOReL — Kidambi et al., 2020, NeurIPS. (Pessimistic MDP
   known/unknown partition.)
8. COMBO — Yu et al., 2021. (Model rollouts + CQL penalty.)
9. Is Pessimism Provably Efficient — Jin et al., 2021, ICML.
   (Minimax-optimal under single-policy coverage.)
10. REM — Agarwal et al., 2020, ICML. (Ensembles generalize where
    constraints hurt.)
11. Offline RL Tutorial — Levine et al., 2020. (Shift/train-test
    taxonomy.)
12. RL Unplugged — Gulcehre et al., 2020, NeurIPS. (Offline selection
    brittleness benchmarks.)
13. Generalization Gap in Offline RL — Mediratta et al., 2024, ICLR.
    (BC beats CQL/BCQ/DT on held-out levels; DIVERSITY BEATS SIZE —
    keep weak tails.)
14. Structure > Pessimism Amount — Weltevrede et al., 2026.
    (Symmetric consistency fixes seed-biased pessimism.)
15. Latent Distribution Representation — Wang et al., 2025, AAAI.
    (Adversarial min-max invariant representations across splits.)
16. Data Diversity, Posterior Sampling — Nguyen-Tang, Arora, 2023,
    NeurIPS. (Diversity subsumes concentrability for extrapolation.)
17. Realizability + Single-Policy Concentrability — Zhan et al., 2022,
    COLT. (Regularized primal-dual under minimal coverage.)
18. State Aggregation and Trajectory Data — Jia et al., 2024, COLT.
    (Function-class aggregation governs error, not raw coverage.)
19. Quantifying Generalization (CoinRun) — Cobbe et al., 2019, ICML.
    (Train-test seed protocol; regularization closes gap.)
20. Procgen — Cobbe et al., 2020, ICML. (Cross-seed robustness; scale
    aids generalization.)
21. Decision Transformer — Chen et al., 2021, NeurIPS. (Return-cloning
    often transfers better than DP methods.)
22. When Does RCSL Work — Brandfonbrener et al., 2022, NeurIPS.
    (Needs near-deterministic dynamics + return coverage; no stitch.)
23. SPOT — Wu et al., 2022, NeurIPS. (Density support constraint
    balancing optimality vs OOD.)
24. PRDC — Ran et al., 2023, ICML. (Nearest-neighbor dataset constraint
    permits unseen pairs, improving transfer.)
25. MACAW — Mitchell et al., 2021, ICML. (Offline-to-unseen-task
    transfer.)
26. SMAC — Pong et al., 2022, ICML. (Context-shift diagnosis
    offline→online.)
27. CORRO — Yuan et al., 2022, ICML. (Contrastive behavior-robust task
    representations.)
28. BOSA — Liu et al., 2024, AAAI. (Dual action+dynamics support for
    cross-domain logs.)
29. QDT — Yamagata et al., 2023, ICML. (CQL-relabeled returns + DT:
    stitching WITH transfer.)
30. Cal-QL — Nakamoto et al., 2023, NeurIPS. (Miscalibrated pessimism
    unlearns; calibration preserves gains.)
31. Distribution Shift/Generalization/OOD Survey — Samani et al., 2026.
    (Value-penalty vs policy-constraint taxonomy.)
32. SRDP — Ada et al., 2024. (State-reconstruction auxiliary for OOD
    extrapolation.)
33. Dynamic Uncertainty Estimation — Wang et al., 2025, AAAI.
    (Adaptive pessimism avoids uniform degradation on easy seeds —
    our exact held-out shape.)

### 4. Recurrent policies and hidden-state drift (30)

1. R2D2 Recurrent Replay — Kapturowski et al., 2019, ICLR. (Stored-state
   staleness; burn-in protocol — test burn-in vs zero-start.)
2. DRQN — Hausknecht, Stone, 2015. (LSTM vs frame-stack under
   observability shift.)
3. Recurrent PPO Generalization — Pleines et al., 2022. (Seed-scaling
   flips memorization→generalization; hidden-refresh cookbook.)
4. PPO — Schulman et al., 2017. (Multi-epoch staleness breaks RNN trust
   region.)
5. A3C — Mnih et al., 2016, ICML. (A3C-LSTM truncated-rollout template
   for 400-step GRU.)
6. IMPALA — Espeholt et al., 2018, ICML. (Policy lag + LSTM shipping
   initial states.)
7. GTrXL — Parisotto et al., 2020, ICML. (Gated Transformer-XL beats
   LSTM on memory, matches on reactive — recurrence-ablation test.)
8. MERLIN — Wayne et al., 2018. (Memory alone fails without predictive
   shaping.)
9. Recurrent Model-Free RL Strong Baseline — Ni et al., 2022, ICML.
   (Tuning unlocks recurrent generalization on 18/21 POMDPs.)
10. Decision Transformer — Chen et al., 2021. (Finite-window
    alternative: no hidden carryover.)
11. HELM — Paischer et al., 2022, ICML. (Frozen Transformer memory
    beats LSTM on memory tasks.)
12. VariBAD — Zintgraf et al., 2020, ICLR. (Test whether GRU encodes
    seed identity vs belief.)
13. RL² — Duan et al., 2016. (RNN fast adaptation risks training-MDP
    overfit.)
14. Epistemic POMDPs — Ghosh et al., 2021, NeurIPS. (Limited seeds turn
    MDP into POMDP; memory + ensembles approximate Bayes-optimal.)
15. CoinRun — Cobbe et al., 2019. (Train-test seed gaps; regularization
    fixes.)
16. Procgen — Cobbe et al., 2020. (Fixation on training seeds; scale
    cures.)
17. Assessing Generalization in Deep RL — Packer et al., 2018.
    (Memory can HURT OOD — vanilla beats adaptive variants.)
18. Overfitting in Deep RL — Zhang et al., 2018. (Robust memorization
    despite stochasticity.)
19. EPOpt — Rajeswaran et al., 2017, ICLR. (CVaR-over-ensemble
    alternative to memory for seed robustness.)
20. What Matters for On-Policy AC — Andrychowicz et al., 2021.
    (Implementation details incl. hidden refresh dominate.)
21. RNN Regularization — Zaremba et al., 2014. (Dropout non-recurrent
    connections only.)
22. Variational RNN Dropout — Gal, Ghahramani, 2016, NeurIPS. (Same
    mask per timestep; testable on 872-dim GRU.)
23. State-Regularized RNNs — Wang, Niepert, 2019, ICML. (Finite
    centroids stop unbounded drift.)
24. Distributionally Robust Recurrent Decoders — Miceli-Barone et al.,
    2021. (RND OOD detector contracts drifted GRU states.)
25. How Do Sequence Models Generalize — Bau, Andreas, 2021.
    (History dropout favors local cues; input noise favors global.)
26. Length Generalization in Recurrent Models — Ruiz, Gu, 2025.
    (Unexplored-states hypothesis = our 400-step drift;
    state-passing/fitted-noise init.)
27. Memory Traces — Eberhard et al., 2025, ICML. (EMA traces replace
    GRU; scale better than windows.)
28. Flow-based Recurrent Belief — Chen et al., 2022, ICML. (Test GRU
    belief collapse OOD with flows.)
29. DQN frames — Mnih et al., 2015, Nature. (Finite-window baseline the
    GRU must beat to justify memory.)
30. Atari DRL — Mnih et al., 2013. (Fixed-history mild-degradation
    pattern our feedforward Q/V replicates.)

### 5. Coverage-preserving cloning (30)

1. Behavior Transformers — Shafiullah et al., 2022, NeurIPS. (K-mode
   mixture fixes multi-modal sweep collapse.)
2. Diffusion Policy — Chi et al., 2023, RSS. (Full support preserved,
   expressive.)
3. Implicit BC — Florence et al., 2022, CoRL. (Energy-based; no
   averaging away rare branches.)
4. OPAL — Ajay et al., 2021, ICLR. (VAE sweep primitives delineate
   support; improve selection only.)
5. AWR — Peng et al., 2019. (Canonical source of our collapsing
   weighting.)
6. AWAC — Nair et al., 2021, ICLR. (Implicit-constraint AWR for
   offline→online.)
7. BCQ — Fujimoto et al., 2019. (Batch support restriction vs OOD
   collapse.)
8. TD3+BC — Fujimoto, Gu, 2021, NeurIPS. (Additive BC term = tunable
   coverage knob replacing alpha-hack.)
9. CQL — Kumar et al., 2020. (Decisiveness without deleting support.)
10. IQL — Kostrikov et al., 2022. (Stitching without OOD queries.)
11. SPOT — Wu et al., 2022, NeurIPS. (SUPPORT-not-density constraint
    keeps zero-advantage sweeps legal.)
12. Closed-Form Policy Improvement — Li et al., 2023. (Mixture behavior
    avoids mode-dropping.)
13. SMODICE — Ma et al., 2022, ICML. (State-occupancy matching offline;
    coverage not action likelihood.)
14. Skew-Fit — Pong et al., 2020, ICML. (Max-entropy goal distribution
    for uniform visitation.)
15. EDL — Campos et al., 2020, ICML. (Separate explore/discover/learn;
    MI skills collapse coverage otherwise.)
16. State Entropy Regularization — Ashlag et al., 2025. (K-NN
    state-entropy protects alternate sweep paths.)
17. InfoGAIL — Li et al., 2017, NeurIPS. (Diverse-mode cloning via
    latent MI.)
18. Out-of-Dynamics IL — Qiu et al., 2023, CoRL. (Cluster modes before
    transfer so minorities survive.)
19. QD-IL — Yu et al., 2024. (Archive of diverse high-quality policies
    from limited demos.)
20. Wasserstein QDIL — Yu et al., 2025, AAMAS. (Fixes adversarial QDIL
    instability + overfitting.)
21. G-QDIL — Wan et al., 2025. (Generative diverse repertoire beyond
    single expert.)
22. Counterfactual BC — Sagheb, Losey, 2025. (Recover intended coverage
    from noisy sweeps via counterfactual expansion.)
23. MACAW — Mitchell et al., 2021. (AWR invertibility failure →
    support-aware weighting.)
24. SIL — Oh et al., 2018, ICML. (Clone only outperformance; keep
    zero-advantage maintenance.)
25. Coverage Paths with Deep RL — Jonnarth et al., 2023. (Multi-scale
    sweep-coverage learning.)
26. Sim-to-Real Coverage Planning — Jonnarth et al., 2024. (IL+RL
    coverage transfer baseline.)
27. Voronoi Multi-Robot Exploration — Hu et al., 2020. (Partitioned
    sweep preserving coverage.)
28. SCORE — Yu et al., 2026. (Support-constrained improvement without
    real-world experience.)
29. EBGAN-MDN — Li et al., 2025. (Benchmarks mode coverage vs collapse
    in BC.)
30. Coverage Path Planning Survey — Galceran, Carreras, 2013.
    (Taxonomy our policy must preserve: boustrophedon and co.)

### 6. Ensembles against the lottery (25)

1. REM — Agarwal et al., 2020, ICML. (Random Q-mixtures; ensembles
   generalize where constraints hurt.)
2. Maxmin Q-Learning — Lan et al., 2020, ICLR. (N tunes over/under
   estimation; shrinks Bellman lottery.)
3. EDAC — An et al., 2021, NeurIPS. (Min-over-N + gradient diversity;
   10x fewer nets for stable pessimism.)
4. SAC-N — An et al., 2021. (Raising N with min-clipping collapses OOD
   overestimation + seed spread.)
5. Deep Ensembles — Lakshminarayanan et al., 2017, NeurIPS.
   (Seed-diverse ensembles + OOD uncertainty for selection.)
6. Snapshot Ensembles — Huang et al., 2017, ICLR. (M checkpoints from
   ONE run — our single-seed-budget answer.)
7. REDQ — Chen et al., 2021, ICLR. (Subset-min stabilizes high-update
   Q-learning.)
8. SUNRISE — Lee et al., 2021, ICML. (Variance-weighted Bellman loss
   tames single-seed instability.)
9. Averaged-DQN — Anschel et al., 2017. (Averaging past Q-nets cuts
   target variance.)
10. SWA — Izmailov et al., 2018. (Weight-averaged flatter minima,
    seed-insensitive.)
11. FGE — Garipov et al., 2018, NeurIPS. (Late-run diverse models for
    cheap.)
12. Bootstrapped DQN — Osband et al., 2016, NeurIPS. (K-head voting,
    cheap posterior.)
13. Policy Distillation — Rusu et al., 2016, ICLR. (N teachers → one
    deployable policy.)
14. Distillation (Hinton) — Hinton et al., 2015. (Temperature-softened
    ensemble → student.)
15. MSG — Ghasemipour et al., 2022, NeurIPS. (Independent targets
    required; shared-target optimism — our single Q indicted.)
16. DroQ — Hiraoka et al., 2022, ICLR. (Dropout+LayerNorm small
    ensemble; seed-stable on budget — pairs with our LayerNorm.)
17. TQC — Kuznetsov et al., 2020, ICML. (Quantile-drop bias control +
    distributional ensembling.)
18. RL That Matters — Henderson et al., 2018, AAAI. (5-seed splits
    reverse conclusions — our reporting mandate.)
19. Statistical Precipice — Agarwal et al., 2021, NeurIPS. (IQM +
    bootstrap CIs stop single-seed reversals.)
20. BatchEnsemble — Wen et al., 2020, ICLR. (Rank-one per-member
    near-free ensembles.)
21. Double Q-Learning — van Hasselt, 2010. (Cross-evaluated argmax vs
    max-operator bias.)
22. UWAC — Wu et al., 2021, ICML. (Dropout-variance inverse-weights
    OOD backups across seeds.)
23. Offline Hyperparameter Selection — Paine et al., 2020. (FQE V(s0),
    not training Q, for selection.)
24. CQL — Kumar et al., 2020. (Lower-bound baseline ensembles must
    beat without behavior collapse.)
25. D4RL — Fu et al., 2020. (Seed-spread shrinkage benchmark suite.)

### 7. Validation splits and early stopping (26)

1. Offline Hyperparameter Selection — Paine et al., 2020. (FQE V(s0)
   ranking without env.)
2. Healthcare Model Selection — Tang, Wiens, 2021. (WIS prune + FQE
   rerank two-stage.)
3. Deep OPE Benchmarks — Fu et al., 2021, ICLR. (Rank correlation +
   Regret@k; no method dominates.)
4. COBS OPE Study — Voloshin et al., 2021. (33 estimators; match
   estimator to domain.)
5. Robot Manipulation from Offline Demos — Mandlekar et al., 2021,
   CoRL. (Val loss 50-100% worse than best rollout; CHECKPOINT
   SELECTION dominates — install proxy-rollout selection.)
6. BVFT — Xie, Jiang, 2021, ICML. (Hyperparameter-free tournament
   selection from holdout.)
7. Time-Series CV — Bergmeir, Benitez, 2012. (Blocked CV over random
   splits for dependent transitions.)
8. Spatial/Temporal CV Strategies — Roberts et al., 2017. (Block by
   trajectory/time/group; random CV selects non-causal predictors.)
9. Causal Confusion in IL — de Haan et al., 2019, NeurIPS. (Lower val
   loss, far worse return — canonical match-up/return-down.)
10. Offline Model-Free Workflow — Kumar et al., 2021, CoRL. (Q-peak/TD
    early stopping without env.)
11. DAgger Reduction — Ross et al., 2011. (Why iid action-match
    validation fails under induced shift.)
12. Fundamental Limits of IL — Rajaraman et al., 2020, NeurIPS.
    (H-squared compounding bound; match% can't guarantee return.)
13. Offline Driving Evaluation — Codevilla et al., 2018, ECCV. (MSE
    correlation 0.39 with success — our metric indicted numerically.)
14. Doubly Robust OPE — Jiang, Li, 2016. (Unbiased low-variance
    validator for trajectory-split selection.)
15. RvS Essentials — Emmons et al., 2022, ICLR. (Val error unreliable
    for RvS; tune capacity explicitly.)
16. IQL — Kostrikov et al., 2022. (Validate temperature and steps
    separately from values.)
17. Early Stopping with CV — Prechelt, 1998. (Strips-of-rising-error
    rules beat fixed epochs; restore best weights.)
18. Model Selection in RL — Farahmand, Szepesvari, 2011.
    (Complexity-penalized Bellman selection with oracle inequality.)
19. OPC/SoftOPC — Irpan et al., 2019. (Success-classification rank
    without models or IS.)
20. Pessimistic Model Selection — Yang et al., 2021. (LCB selection
    recovers true best among 70 models; point estimates overfit.)
21. Horizon in IL — Foster et al., 2024, NeurIPS. (Trajectory log-loss
    tracks return; per-step accuracy decouples.)
22. Offline RL Without OPE — Brandfonbrener et al., 2021, NeurIPS.
    (One-step improvement more robust validator than iterative OPE.)
23. Forecasting Performance Estimation — Cerqueira et al., 2020.
    (Blocked CV if stationary; repeated holdout if not.)
24. Validity of CV for Autoregressive Prediction — Bergmeir et al.,
    2018. (K-fold valid only with uncorrelated residuals; else leaks.)
25. Bias in CV Model Selection — Varma, Simon, 2006. (Nested splits:
    inner tunes epochs, outer estimates — reuse biases.)
26. Copycat Agents in BC — Wen et al., 2020, NeurIPS. (History BC
    cheats via previous-action correlation; RESET hidden state per
    split — directly our GRU leakage hazard.)

## Ordered next-experiment program

- V1 Validation splits + early stopping (F4; cheapest, unlocks all):
  trajectory-blocked 80/20 split of log episodes, checkpoint every 2
  epochs, select on validation CE + perturbed-trajectory metric; nested
  split for reporting. Applies to BC/RvS/IQL-extraction uniformly.
- V2 ESS-gated adaptive extraction (F2): online ESS monitor, MPO-style
  per-batch temperature or CRR filter when ESS collapses, RLPD-style
  rescaling when uniform. Kills static alpha.
- V3 Rollout-free selection (F1/hygiene): FQE-V(s0) rerank of all saved
  ckpts (a05/s0-s2/stab*), BVFT tournament as hyperparameter-free
  cross-check. Ends test-env selection.
- V4 Snapshot/SWA (F1): cyclic-LR snapshots from one run,
  weight-average, distill to one factorized policy; measure seed-spread
  shrinkage on regression.
- V5 Ensemble Q (F1/F2): EDAC-style min-over-N + DroQ-small on the
  24-ep log; MSG-independent-targets check.
- V6 Recurrence ablation (F6): feedforward-policy extraction vs GRU
  (same advantages); burn-in vs zero-start; hidden-reset and
  state-passing-init tests; RND OOD-state probe.
- V7 Support constraint (F5): SPOT-style support penalty and TD3+BC
  additive knob as alpha-hack replacements; OPAL primitives as
  larger build.
- V8 Transfer log rule (F3): diversity-over-size logging (keep weak
  tails per Mediratta), QDT relabel-then-distill, symmetric
  consistency auxiliary.

## Gap

Track 8 (BC fragility/seed-variance case studies) returned empty;
re-run post-compact. Partially covered by tracks 7 (validation) and 6
(ensembles). Unique papers ≈ 190 after cross-track dedup.
