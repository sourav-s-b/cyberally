# Blue RL recovery literature survey (2026-10-02)

Purpose: second, larger literature sweep after the program exhausted its
first ladder (hybrid ordering closed, MAPPO fine-tunes diverge, attention
distill at teacher parity). Ten parallel research tracks, 25–36 papers
each, ~300 total, every entry mapped to a measured Blue weakness below.

Method: 10 researcher subagents, each instructed to use deep web search +
page fetch, 2020–2026 priority with seminal older work, and to DROP any
citation they could not verify rather than invent it. Track catalogs in
§5 are reproduced as returned (provenance preserved per track).

VERIFICATION CAVEAT: entries passed a search-existence check, not a
read-the-paper check. Years/venues may be off by one in places; fetch the
DOI/arXiv ID before citing, sourcing, or implementing. Duplicates across
tracks are noted, not deduplicated, in §6.

## Our weaknesses (labels used throughout)

- W1 RL fine-tuning diverges: flat and factorized MAPPO collapse from good
  inits (-205 → -1709); KL-to-teacher only slows it; critic 3.8 → 61.8.
- W2 Sparse/noisy reward: ~41% of steps carry reward; ±313 noise band;
  3-seed deltas prove nothing; long Analyse→Remove→Restore chains.
- W3 Coverage collapse: learned policies fixate (27 vs 67 hosts); learned
  risk scoring abandoned coverage for confidence.
- W4 Demo ceiling: BC 21.8% match, attention clone ties but cannot exceed
  the teacher; no stitching of suboptimal fragments.
- W5 Seed/red brittleness: lancer_v1 -63.2 → -110.1 ± 130.4 held-out with a
  -414 episode; finite red -139 vs discovery -59.
- W6 Architecture gap: flat MLPs fail on 51×17 slots; attention works for
  cloning but has no RL training recipe here yet.
- W7 Belief gap: memoryless policies; Red invisible until detected (true 34
  sessions vs Blue-visible 0); hand-built suspicion works, learned fails.

## Highest-leverage bets (synthesis across all tracks)

Ordered by expected value per implementation cost for THIS program. Each
names the weakness it attacks and the concrete next experiment.

1. IQL / XQL offline stitching on heuristic logs (W4, W1). In-sample
   expectile values + advantage-weighted extraction: match-then-EXCEED
   from fixed logs with no OOD queries. Our 8101+ demo logs already exist.
   Experiment: IQL-discrete on lancer_v2 traces; gate = beat teacher on
   held-out. (Track C #15/#16; also Oryx #17 track E for offline MARL.)
2. AWAC fine-tuning (W1). Off-policy critic + implicit advantage-weighted
   actor constraint; the literature's direct answer to "SAC-from-BC dips."
   Replaces our KL-anchored PPO with an update that cannot drift far from
   demo support. Experiment: AWAC from attention-distill ckpt with
   stop-rules. (Track D #14.)
3. JSRL guide curriculum (W1, W2). Frozen heuristic owns early-episode
   steps with a shrinking horizon; RL only decides late episodes. Sidesteps
   early-collapse entirely. Experiment: lancer_v2 guide + learned explorer.
   (Track D #21.)
4. PEX policy expansion (W1). Keep BC/attention policy frozen, grow a
   separate online head with value-based selection — divergence becomes
   architecturally impossible. Experiment: PEX head on attention ckpt.
   (Track D #22.)
5. HAPPO/HATRPO sequential updates (W1). Advantage decomposition with
   per-agent sequential trust regions replaces the simultaneous MAPPO
   updates behind our divergence. Experiment: HAA2C/HAPPO on factorized
   actor. (Track E #13/#14.)
6. History/dec critic choice (W1). Lyu et al. prove state-only central
   critics bias gradients under partial observability — our 1190-dim critic
   blowup, diagnosed. Experiment: history-state critic + decentralized
   critic ablation before any new algorithm. (Track E #22/#23/#21.)
7. PLR⊥ / ACCEL red curricula (W5). Minimax-regret replay curricula with
   Nash guarantees; ACCEL mutates reds at the capability frontier. Directly
   targets the -59/-139 red gap and -414 tail episodes. Needs Environment
   hooks (extends coordination/blue-red-variants.md). (Track H #5/#6.)
8. PSRO league (W5). Blue/red population co-evolution with exploitability
   metric instead of single-red overfitting; AlphaStar-league practice.
   (Track H #15/#16, cyber line Hammar T-FP track A #13/#14.)
9. E3B / NovelD episodic exploration (W2, W3). Elliptical episodic bonus
   works when states never repeat (our noisy obs); boundary-difference
   bonus fixes single-corridor fixation. Cheapest coverage fix.
   (Track B #15/#14.)
10. Dynamic potential-based shaping (W2). The ONLY provably
    optimality-preserving densification (Ng; Devlin–Kudenko time-indexed).
    Pairs with Bates 2025 "sparse healthy bonus beats dense scaffolds."
    (Track B #21/#22, A #24/#25.)
11. Kickstarting with decaying schedule (W1). Upgrades our fixed-KL attempt:
    decaying teacher cross-entropy + PBT weight schedule. (Track D #1.)
12. Shielding + options (W3, W2). Temporal-logic shield (Alshiekh; Carr
    POMDP extension) keeps coverage sane by construction; options/HAC
    formalize Analyse→Remove→Restore chains and densify via hindsight.
    Matches our hybrid philosophy exactly. (Track I #15/#1/#7.)
13. Entity-RL actor + GTrXL stabilization (W6). Thompson entity-Transformer
    generalizes 10→40 nodes; SR-DRL is the host-then-command GNN+RL
    blueprint; GTrXL gating is the recipe to move our slot-attention from
    cloning to RL. (Track F #28/#16/#14, A #28.)
14. Learned belief: DVRL / VariBAD (W7). Particle-filter belief replaces
    brittle hand suspicion; VariBAD formalizes probe-vs-remediate under
    unknown Red. (Track G #12/#16.)
15. Eval rigor now (program hygiene). Adopt rliable IQM + performance
    profiles + stratified bootstrap (Agarwal) alongside our mean±std; power
    analysis (Colas) to size seed counts. Cheap, immediate. (Track J #1/#3.)

## Weakness → track matrix

| Weakness | Primary tracks | Supporting |
|---|---|---|
| W1 diverge | D fine-tune, E MARL | B shaping, G memory |
| W2 sparse reward | B exploration | J security rewards, I hierarchy |
| W3 coverage | I hierarchy/shield, B exploration | F architectures |
| W4 demo ceiling | C offline/IL | D fine-tune |
| W5 brittleness | H robustness/UED | A cyber co-training, J eval |
| W6 architecture | F architectures | E MAT/sequence MARL |
| W7 belief | G world models/memory | A cyber games |

## Track catalogs

### A. Autonomous cyber-defence RL/MARL (35)

1. CybORG: A Gym for the Development of Autonomous Cyber Agents — Standen,
   Lucas, Bowman, Richer, Kim, Marriott, 2021, IJCAI ACD Workshop.
   Relevance: base simulator for CC2–CC4 Blue work with partial
   observability and sparse penalties.
2. CybORG: An Autonomous Cyber Operations Research Gym — Baillie, Standen,
   Schwartz, Docking, Bowman, Kim, 2020, arXiv. Relevance: original CybORG
   design motivating wrapper, masks, fixed action ordering.
3. Autonomous Network Defence Using Reinforcement Learning — Foley, Hicks,
   Highnam, Mavroudis, 2022, ACM AsiaCCS. Relevance: winning CAGE1
   hierarchical PPO that first beat flat PPO. Mechanism: per-attacker
   sub-policies + controller predicting Red type.
4. CAGE Challenge 2 Winning Entry: PPO + Greedy Decoys — CardiffUni team,
   2022, TTCP CAGE Challenge 2. Relevance: learned policy + greedy decoy
   heuristic beats pure RL.
5. Mindrake HIPPO Float Observations / Bandit Controller — Foley, Hicks et
   al. (Team Mindrake), 2022, CAGE Challenge 2. Relevance: 3rd-place CAGE2;
   compact float obs + bandit controller stabilize PPO.
6. On Autonomous Agents in a Cyber Defence Environment — Kiely, Bowman,
   Standen, Moir, 2023, Adaptive Cyber Defence Workshop. Relevance:
   official CAGE2 analysis ranking hierarchical DRL above single-agent,
   ensembles, non-DRL.
7. Beyond CAGE: Investigating Generalization of Learned Autonomous Network
   Defense Policies — Wolk et al., 2022, NeurIPS RL for Real Life Workshop.
   Relevance: 2nd-place CAGE2 ensemble tested on unseen networks/attackers.
   Mechanism: weighted majority vote over diverse PPO + transfer training.
8. Network Environment Design for Autonomous Cyberdefense (FARLAND) —
   Molina-Markham et al., 2021, arXiv. Relevance: curricula of harder
   networks + RL-targeted Red adversaries via generative programs.
9. Autonomous Penetration Testing using RL (NASim) — Schwartz, Kurniawati,
   2019, arXiv. Relevance: lightweight attack-graph sim ancestor of
   generalization experiments.
10. NASimEmu — Janisch, Pevny, Lisy, 2023, SECAI@ESORICS. Relevance:
    sim→emulation transfer + novel topologies with invariant architectures.
11. Finding Effective Security Strategies Through RL and Self-Play — Hammar,
    Stadler, 2020, CNSM. Relevance: first self-play cyber response;
    co-trained defenders beat static-attacker policies.
12. Learning Intrusion Prevention Policies through Optimal Stopping —
    Hammar, Stadler, 2021, CNSM. Relevance: noisy alerts → threshold
    stop/continue rules under partial observability.
13. Learning Security Strategies through Game Play and Optimal Stopping —
    Hammar, Stadler, 2022, ICML ML4Cyber Workshop. Relevance: dynamic Red
    via fictitious self-play; threshold T-FP Nash equilibria.
14. Learning Near-Optimal Intrusion Responses Against Dynamic Attackers —
    Hammar, Stadler, 2024, IEEE TNSM. Relevance: journal version proving
    threshold best-responses, beating SOTA self-play baselines.
15. Optimal Security Response to Network Intrusions in IT Systems — Hammar,
    2024, KTH PhD thesis. Relevance: unifies stopping/tolerance/APT games
    with sim-to-emulation validation.
16. CSLE: A Reinforcement Learning Platform for Autonomous Security
    Management — Hammar, 2026, MLSys. Relevance: full platform, 50+
    scenarios, 34 RL algorithms, twin refinement.
17. Incorporating Deception into CyberBattleSim for Autonomous Defense —
    Walter et al., 2021, arXiv. Relevance: decoy count/placement stalls RL
    attackers; Blue decoy design data.
18. CyberBattleSim — Microsoft Defender Research Team, 2021. Relevance:
    minimal lateral-movement gym for fast RL iteration.
19. A Multiagent CyberBattleSim for RL Cyber Operation Agents — Kunz et
    al., 2023, CSCI/arXiv. Relevance: co-trained Blue beats isolated Blue.
20. Autonomous Network Defence Using RL Within CyberBattleSim — Lubbock et
    al., 2022, AFCEA. Relevance: ATT&CK-grounded MARL Blue, centralized vs
    decentralized communication ablations.
21. Deep Reinforcement Learning for Cyber Security — Nguyen, Reddi, 2019,
    IEEE TNNLS/arXiv. Relevance: seminal survey; reward sparsity +
    adversarial robustness framing.
22. Adversarial RL under Partial Observability in Autonomous Computer
    Network Defence — Han et al., 2019, arXiv. Relevance: Blue under hidden
    Red + adversarial observation manipulation; belief-state modeling.
23. Adversarial RL in a Cyber Security Simulation — Elderman et al., 2017,
    ICAART. Relevance: early Red–Blue coevolution; defenders must train vs
    learning attackers.
24. Reward Shaping for Happier Autonomous Cyber Security Agents — Bates,
    Mavroudis, Hicks, 2023, ACM AISec. Relevance: CAGE2 penalty magnitude,
    positive densification, curiosity for sparse signals.
25. Less Is More? Rewards in RL for Cyber Defence — Bates, Hicks, Mavroudis,
    2025, arXiv. Relevance: sparse healthy-network bonuses beat dense
    scaffolds on stability and effectiveness.
26. CybORG++ — Emerson, Bates, Hicks, Mavroudis, 2024, arXiv. Relevance:
    fixed CAGE2 bugs + 1000x MiniCAGE for rapid reward/mask ablations.
27. Developing Optimal Causal Cyber-Defence Agents via Cyber Security
    Simulation (Yawning Titan) — Andrew et al., 2022, ICML ML4Cyber
    Workshop. Relevance: lightweight graph gym for fast topology/reward
    iteration.
28. Entity-based RL for Autonomous Cyber Defence — Thompson, Caron, Hicks,
    Mavroudis, 2024, arXiv. Relevance: Transformer over host entities
    generalizes 10–40 nodes where MLPs collapse. Mechanism: Entity-Gym +
    RogueNet attention under random topologies.
29. Automated Cyber Defence: A Review — Vyas et al., 2023, arXiv. Relevance:
    40-paper review separating gym building from agent learning.
30. Deep RL for Autonomous Cyber Defence: A Survey — Palmer et al., 2023,
    arXiv. Relevance: compares CybORG, FARLAND, CyberBattleSim, NASim,
    Yawning Titan, PrimAITE on fidelity.
31. Multi-Objective RL for Automated Resilient Cyber Defence — Adams et al.,
    2024, arXiv. Relevance: splits noisy reward into defence vs
    green-availability objectives (MOPPO, Pareto-conditioned nets).
32. RL for Autonomous Resilient Cyber Defence — Miles et al., 2024,
    Frazer-Nash/Dstl white paper. Relevance: MARL role specialization
    beating experts; topology generalization to unseen nets.
33. TTCP CAGE Challenge 3 — TTCP CAGE Working Group, 2022, GitHub.
    Relevance: decentralized 18-drone MARL precedent for 5-agent
    coordination.
34. TTCP CAGE Challenge 4 — TTCP CAGE Working Group, 2024, GitHub.
    Relevance: canonical 5-Blue enterprise MARL task (zones, decoys,
    messaging).
35. PrimAITE — Dstl/QinetiQ ARCD team, 2023, GitHub. Relevance:
    configurable packet-level alternative for validating CC4 policies beyond
    CybORG abstraction.

### B. Sparse reward and exploration (26)

1. Exploration by Random Network Distillation — Burda et al., 2019, ICLR.
   Relevance: dense novelty bonus for 400-step sparse sweeps.
2. Curiosity-driven Exploration by Self-supervised Prediction — Pathak et
   al., 2017, ICML. Relevance: drives Analyse/Detect when extrinsic reward
   absent. Mechanism: forward-model error in inverse-dynamics features.
3. Large-Scale Study of Curiosity-Driven Learning — Burda et al., 2019,
   ICLR. Relevance: warns noisy CAGE alerts cause noisy-TV trap.
4. Unifying Count-Based Exploration and Intrinsic Motivation — Bellemare et
   al., 2016, NeurIPS. Relevance: pseudocount bonus for rarely-revisited
   partially-observed states.
5. Count-Based Exploration with Neural Density Models — Ostrovski et al.,
   2017, ICML. Relevance: pseudocounts + Monte-Carlo propagation for
   sparse long-horizon credit.
6. #Exploration — Tang et al., 2017, NeurIPS. Relevance: cheap SimHash
   counts scale to 155-dim joint actions as baseline.
7. VIME — Houthooft et al., 2016, NeurIPS. Relevance: information-gain
   bonus when extrinsic reward sparse/noisy.
8. First Return, Then Explore — Ecoffet et al., 2021, Nature. Relevance:
   fixes detachment/derailment over 400-step chains. Mechanism: archive
   cells, return-then-explore.
9. DTSIL — Guo et al., 2020, NeurIPS. Relevance: return-to-diverse-past-
   trajectories without simulator reset for systematic sweeping.
10. Never Give Up — Badia et al., 2020, ICLR. Relevance: lifelong+episodic
    novelty that never vanishes suits 59%-zero-reward episodes.
11. Agent57 — Badia et al., 2020, ICML. Relevance: adaptive explore/exploit
    family + long horizons for divergent fine-tunes.
12. RIDE — Raileanu, Rocktäschel, 2020, ICLR. Relevance: rewards
    state-changing actions under procedural variation, not obs noise.
13. BeBold — Zhang et al., 2020, arXiv. Relevance: uniform frontier bonus
    prevents single-subnet fixation.
14. NovelD — Zhang et al., 2021, NeurIPS. Relevance: breadth-first boundary
    exploration for multi-host coverage.
15. E3B — Henaff et al., 2022, NeurIPS. Relevance: episodic elliptical
    bonus when every observation is unique/noisy.
16. Self-Supervised Exploration via Disagreement — Pathak et al., 2019,
    ICML. Relevance: ensemble variance survives stochastic dynamics where
    prediction-error curiosity sticks.
17. BYOL-Explore — Guo et al., 2022, NeurIPS. Relevance: latent multi-step
    prediction curiosity for partially-observable histories.
18. RE3 — Seo et al., 2021, ICML. Relevance: stable cheap coverage bonus
    robust to ±313 noise.
19. DIAYN — Eysenbach et al., 2019, ICLR. Relevance: unsupervised diverse
    skills pretrain sweep primitives before sparse reward.
20. UPSIDE — Kamienny et al., 2022, ICLR. Relevance: directed skills +
    diffusion model Analyse→Restore chains as composable tree.
21. Policy Invariance Under Reward Transformations — Ng et al., 1999, ICML.
    Relevance: theory for shaping without changing the optimal defender.
22. Dynamic Potential-Based Reward Shaping — Devlin, Kudenko, 2012, AAMAS.
    Relevance: time-varying potentials preserving policy/Nash guarantees.
23. Hindsight Experience Replay — Andrychowicz et al., 2017, NeurIPS.
    Relevance: relabel failed episodes as successes for alternate host
    goals.
24. Hindsight Credit Assignment — Harutyunyan et al., 2019, NeurIPS.
    Relevance: credit to early Analyse via hindsight likelihood of late
    Restore payoff.
25. COMA — Foerster et al., 2018, AAAI. Relevance: difference baseline
    isolates each Blue agent's contribution. (Also track E.)
26. Scheduled Intrinsic Drive — Zhang et al., 2019, arXiv. Relevance:
    scheduler separates exploration/exploitation phases to avoid divergence.

### C. Offline RL and imitation (30)

1. Efficient Reductions for Imitation Learning — Ross, Bagnell, 2010,
   AISTATS. Relevance: BC O(T²) compounding vs interactive O(T) fix for
   sweep drift.
2. DAgger (no-regret reduction) — Ross, Gordon, Bagnell, 2011, AISTATS.
   Relevance: core fix for 22% BC covariate shift via teacher queries.
3. Feedback in Imitation Learning: Three Regimes of Covariate Shift —
   Spencer et al., 2021, arXiv. Relevance: when flat BC suffices vs fails
   under partial observability.
4. Causal Confusion in Imitation Learning — de Haan et al., 2019, NeurIPS.
   Relevance: why more history hurts cloning under shift.
5. HG-DAgger — Kelly et al., 2018/2019. Relevance: gate teacher queries on
   doubt/unsafe states; cheap sweep fixes.
6. Query-Efficient IL (SafeDAgger) — Zhang, Cho, 2016/2017. Relevance:
   cuts labeling cost via safety classifier gating.
7. DART — Laskey et al., 2017, CoRL. Relevance: noise-injected expert logs
   teach recovery without online expert.
8. GAIL — Ho, Ermon, 2016, NeurIPS. Relevance: occupancy matching escapes
   the 22% per-step ceiling.
9. AIRL — Fu et al., 2017/2018. Relevance: transferable reward recovery
   from heuristic logs for re-optimization.
10. MaxEnt IRL — Ziebart et al., 2008, AAAI. Relevance: reward recovery
    from noisy suboptimal demos.
11. Algorithms for IRL — Ng, Russell, 2000, ICML. Relevance: foundational
    reward-identification framing for exceeding the teacher.
12. BCQ — Fujimoto et al., 2019, ICML. Relevance: fixed-log control staying
    near demonstrated sweeps.
13. BEAR — Kumar et al., 2019, NeurIPS. Relevance: support-set matching for
    sparse noisy cyber rewards.
14. CQL — Kumar et al., 2020, NeurIPS. Relevance: exceed a heuristic
    mixture from fixed logs via conservative lower bound.
15. IQL — Kostrikov et al., 2021/2022. Relevance: stitch sweep fragments
    into better-than-teacher policy without OOD queries; discrete-action
    AWR extraction.
16. XQL — Garg et al., 2023, ICLR. Relevance: IQL successor, stronger
    stitching on sparse long-horizon logs via Gumbel regression.
17. CRR — Wang et al., 2020, NeurIPS. Relevance: filtered BC cloning only
    high-advantage actions.
18. AWAC (offline→online) — Nair et al., 2020. Relevance: bridges heuristic
    pretrain to online fine-tune without collapse. (Also track D.)
19. TD3+BC — Fujimoto, Gu, 2021, NeurIPS. Relevance: minimalist BC+Q
    baseline to test before complex offline RL.
20. D4RL — Fu et al., 2020/2021. Relevance: narrow/mixture/replay log
    evaluation protocol for our heuristic dataset.
21. Decision Transformer — Chen et al., 2021, NeurIPS. Relevance: clones
    periodic sweep policies via return-prompted history modeling.
22. Trajectory Transformer — Janner et al., 2021, NeurIPS. Relevance:
    long-horizon prediction + reward-guided beam-search planning.
23. Planning with Diffusion — Janner et al., 2022, ICML. Relevance:
    stitches suboptimal trajectories for sparse delayed reward.
24. Decision Diffuser — Ajay et al., 2022/2023. Relevance: return-
    conditioned diffusion to exceed teacher without DP.
25. Diffusion Policy — Chi et al., 2023, RSS. Relevance: multimodal
    sweep-branching where MLP BC averages modes.
26. RvS — Emmons et al., 2021/2022. Relevance: tuned return/goal MLP can
    match DT/IQL for parity — cheapest next rung.
27. When Does Return-Conditioned SL Work? — Brandfonbrener et al., 2022.
    Relevance: warns return-BC cannot stitch; needs near-optimal coverage.
28. IQ-Learn — Garg et al., 2021, NeurIPS. Relevance: non-adversarial
    reward+policy from few demos; cyber reward recovery.
29. Gato — Reed et al., 2022, TMLR. Relevance: template for multi-scenario
    tokenized defender policy.
30. Cal-QL — Nakamoto et al., 2023. Relevance: fixes CQL-to-online
    degradation (our flat-BC MAPPO collapse pattern) via calibration.

### D. RL fine-tuning of BC policies (36)

1. Kickstarting — Schmitt et al., 2018. Relevance: decaying distillation
   weight + PBT schedule; direct upgrade of our fixed-KL attempt.
2. Hierarchical Kickstarting — Matthews et al., 2022, CoLLA. Relevance:
   multiple teachers with per-state weighting (5 Blue agents).
3. Residual RL for Robot Control — Johannink et al., 2019, ICRA. Relevance:
   freeze base controller, RL-train additive residual only.
4. Residual Policy Learning — Silver et al., 2018. Relevance: RL residual
   on frozen nondifferentiable controller.
5. ResFiT — Ankile et al., 2025. Relevance: why direct RL fine-tuning of
   BC policies destabilizes; residual fix.
6. TRPO — Schulman et al., 2015. Relevance: monotonic trust region our
   MAPPO clipping only approximates.
7. PPO — Schulman et al., 2017. Relevance: our diverging baseline;
   adaptive-KL vs clip variants matter.
8. Implementation Matters in Deep Policy Gradients — Engstrom et al.,
   2020, ICLR. Relevance: code-level tricks explain PPO gaps; audit ours.
9. What Matters in On-Policy RL? — Andrychowicz et al., 2021, ICLR.
   Relevance: 250k-agent study ranking which PPO choices stop collapse.
10. PPG — Cobbe et al., 2021, ICML. Relevance: separate policy/value phases
    vs value-loss interference under noisy rewards.
11. MPO — Abdolmaleki et al., 2018, ICLR. Relevance: KL-constrained EM
    alternative to clipping for anchored fine-tuning.
12. V-MPO — Song et al., 2019. Relevance: on-policy MPO scaling to
    discrete multi-task control.
13. AWR — Peng et al., 2019. Relevance: supervised-style weighted
    regression that cannot drift far from BC data.
14. AWAC — Nair et al., 2020. Relevance: off-policy critic + implicit
    advantage-weighted constraint; closest fix to BC-init dip.
15. CRR — Wang et al., 2020. Relevance: value-filtered BC. (Also track C.)
16. BRAC — Wu et al., 2019. Relevance: KL/MMD-to-behavior framework
    generalizing our teacher-KL anchor.
17. BEAR — Kumar et al., 2019. Relevance: OOD bootstrapping-error
    diagnosis + support fix. (Also track C.)
18. TD3+BC — Fujimoto, Gu, 2021. Relevance: simplest Q-anchoring recipe.
    (Also track C.)
19. SAC — Haarnoja et al., 2018. Relevance: stable off-policy
    entropy-regularized alternative to PPO for sparse rewards.
20. SAC + autotune — Haarnoja et al., 2019. Relevance: temperature tuning
    stabilizing fine-tuning across seeds.
21. JSRL — Uchendu et al., 2023, ICML. Relevance: frozen heuristic guide
    curriculum with shrinking horizon; avoids init collapse.
22. PEX — Zhang et al., 2023, ICLR. Relevance: frozen BC + expandable
    online head; divergence architecturally impossible.
23. Progressive Nets — Rusu et al., 2016. Relevance: freeze teacher column,
    train new column with lateral adapters.
24. Primacy Bias — Nikishin et al., 2022, ICML. Relevance: early-experience
    overfitting; periodic last-layer resets as cheap test.
25. Dormant Neurons — Sokar et al., 2023, ICML. Relevance: diagnostic for
    our collapse; ReDo recycling.
26. Implicit Under-Parameterization — Kumar et al., 2021, ICLR. Relevance:
    rank collapse under bootstrapping in sparse tasks.
27. Capacity Loss — Lyle et al., 2022, ICLR. Relevance: sparse targets
    destroy value-fitting ability; InFeR anchor.
28. Plasticity in NNs — Lyle et al., 2023, ICML. Relevance: sharpness/norm/
    rank diagnostics for collapse.
29. Loss of Plasticity (continual) — Abbas et al., 2023, CoLLA. Relevance:
    mirrors long fine-tuning decay; CReLU mitigation.
30. Loss of Plasticity (Nature) — Dohare et al., 2024. Relevance:
    catastrophic plasticity loss incl. PPO; continual-backprop fixes.
31. InstructGPT — Ouyang et al., 2022. Relevance: canonical SFT→PPO with
    per-token KL anchor + pretraining mix.
32. Learning to Summarize from Human Feedback — Stiennon et al., 2020.
    Relevance: KL-anchored PPO of supervised policy beats pure BC.
33. Fine-Tuning LMs from Human Preferences — Ziegler et al., 2019.
    Relevance: original KL-to-prior loop vs reward overoptimization.
34. Secrets of RLHF I: PPO — Zheng et al., 2023. Relevance: PPO-max recipe
    (token-KL, normalization, critic pretraining).
35. DAPG — Rajeswaran et al., 2018, RSS. Relevance: decaying demo-
    augmented NPG from good demonstrators without divergence.
36. Parrot — Singh et al., 2020/2021. Relevance: flow-based behavioral
    prior structuring exploration around demo support.

### E. MARL credit assignment and CTDE (27)

1. COMA — Foerster et al., 2018, AAAI. Relevance: counterfactual baseline
   for shared sparse team reward. (Also track B.)
2. VDN — Sunehag et al., 2018, AAMAS. Relevance: first joint→per-agent
   decomposition; lazy-agent failures.
3. QMIX — Rashid et al., 2018, ICML. Relevance: CTDE baseline with
   monotonic mixer for decentralized Blue.
4. MADDPG — Lowe et al., 2017, NeurIPS. Relevance: original centralized
   critic; variance growth with agents (our 1190-dim blowup precursor).
5. QTRAN — Son et al., 2019, ICML. Relevance: drops QMIX monotonicity when
   best action depends on teammates.
6. MAVEN — Mahajan et al., 2019, NeurIPS. Relevance: committed joint
   exploration for multi-step chains via shared latent variable.
7. Weighted QMIX — Rashid et al., 2020, NeurIPS. Relevance: keeps
   decentralization, recovers optimal joints under nonmonotonic payoffs.
8. QPLEX — Wang et al., 2021, ICLR. Relevance: full IGM expressiveness for
   155-dim joints; duplex dueling + attention weights.
9. DOP — Wang et al., 2021, ICLR. Relevance: off-policy decomposed critic,
   per-agent tree-backup gradients O(n|A|).
10. LICA — Zhou et al., 2020, NeurIPS. Relevance: implicit credit via
    hypernetwork critic; no explicit per-host shaping.
11. RODE — Wang et al., 2021, ICLR. Relevance: clusters Analyse/Remove/
    Restore effects into roles, shrinking 100+ actions per agent.
12. Surprising Effectiveness of PPO in Cooperative MARL — Yu et al., 2022,
    NeurIPS. Relevance: why our MAPPO diverges — value norm, state inputs,
    clipping, death-masking are load-bearing.
13. HAPPO/HATRPO — Kuba et al., 2021. Relevance: sequential per-agent
    trust-region updates; monotonic; replaces simultaneous MAPPO.
14. HARL — Zhong et al., 2024, JMLR. Relevance: heterogeneous-agent mirror
    learning; HAA2C/HADDPG/HATD3 for unequal Blue zones.
15. MAT — Wen et al., 2022, NeurIPS. Relevance: joint policy as
    autoregressive sequence; linear in agents; monotonic guarantee.
16. Sable — Mahjoub et al., 2025, ICML. Relevance: retention-based sequence
    MARL to 1000+ agents with full-episode memory.
17. Oryx — Formanek et al., 2025. Relevance: offline autoregressive
    coordination from static logs; sequential IQ-learning.
18. SHAQ — Wang et al., 2022, NeurIPS. Relevance: Shapley Q-values; dummy
    detection for idle sweepers.
19. Shapley Counterfactual Credits — Li et al., 2021, KDD. Relevance:
    coalition credit for Analyse-enabling-Remove across hosts.
20. FACMAC — Peng et al., 2021, NeurIPS. Relevance: factored critic +
    joint-action gradient; avoids monolithic critic variance.
21. Centralized vs Decentralized Critics — Lyu et al., 2021, AAMAS.
    Relevance: central critic raises actor variance (multi-action,
    multi-observation); directly diagnoses ours.
22. State-Based Critics — Lyu et al., 2022, AAAI. Relevance: state-only
    critics bias gradients under partial observability — our divergence
    mechanism.
23. On Centralized Critics — Lyu et al., 2023, JAIR. Relevance: prescribes
    history-state critics; critic-choice tradeoffs for cyber POMDPs.
24. Actor-Attention-Critic — Iqbal, Sha, 2019, ICML. Relevance: attention
    critic scales to many obs by attending to relevant teammates.
25. Joint-Constraint MATRPO — Shek et al., 2025. Relevance: KKT KL
    allocation for heterogeneous agents (unequal Blue improvement).
26. Counterfactual Shapley Credit — Li et al., 2026. Relevance: skill-vs-
    luck disentanglement for sparse delayed noisy rewards (phi-PPO).
27. UneVEn — Gupta et al., 2021, ICML. Relevance: successor-feature GPI vs
    relative overgeneralization locking in suboptimal coverage.

### F. Architectures: attention, entities, GNNs (25)

1. Pointer Networks — Vinyals et al., 2015, NeurIPS. Relevance: host-
   selection head over 51 variable slots vs fixed 155-softmax.
2. HyperNetworks — Ha et al., 2016, ICLR. Relevance: generate per-host/
   command heads from shared context.
3. Deep Sets — Zaheer et al., 2017, NeurIPS. Relevance: permutation-
   invariant 51-slot treatment; fixes MLP order-sensitivity.
4. GCN — Kipf, Welling, 2017, ICLR. Relevance: backbone of KEEP GNN
   defender; host embeddings over topology.
5. Neural Combinatorial Optimization with RL — Bello et al., 2017, ICLR.
   Relevance: pointer selection trained with REINFORCE + critic; cloning
   →RL bridge.
6. GAT — Velickovic et al., 2018, ICLR. Relevance: weight compromised
   neighbors over healthy ones on cyber graphs.
7. Graph Networks (relational bias) — Battaglia et al., 2018. Relevance:
   entity/relation bias over flattened 872-vector.
8. Relational Deep RL — Zambaldi et al., 2018/2019. Relevance: entity
   self-attention trained MODEL-FREE with RL; sample efficiency + transfer.
9. Action Branching — Tavakoli et al., 2018, AAAI. Relevance: shared trunk
   + host/command branches; linear scaling alternative to 155-head.
10. QMIX (factored value) — Rashid et al., 2018. (Cross-listed track E.)
11. Transformer-XL — Dai et al., 2019, ACL. Relevance: recurrent memory
    backbone for long attacker-dwell horizons (see GTrXL).
12. Set Transformer — Lee et al., 2019, ICML. Relevance: DeepSets +
    cross-slot attention; matches our 1-layer parity result.
13. AlphaStar — Vinyals et al., 2019, Nature. Relevance: entity Transformer
    + LSTM + autoregressive pointer heads under RL at 10^26 actions.
14. GTrXL — Parisotto et al., 2020, ICML. Relevance: GRU-gated residuals
    make Transformer policies RL-trainable; required to move our attention
    actor from distill to RL.
15. Slot Attention — Locatello et al., 2020, NeurIPS. Relevance:
    competitive binding of 51 slots to isolate compromised hosts.
16. SR-DRL — Janisch, Pevny, Lisy, 2020. Relevance: EXACT blueprint — GNN
    + host-then-command autoregressive factorization WITH RL and zero-shot
    size generalization.
17. Perceiver — Jaegle et al., 2021, ICML. Relevance: distill 51×17 +
    topology into fixed latent bottleneck before reasoning.
18. UPDeT — Hu et al., 2021, ICLR. Relevance: decouples observation-
    entities from action-groups; one policy, varying host counts, transfer.
19. Decision Transformer — Chen et al., 2021. (Cross-listed track C.)
20. MAPPO tuning baseline — Yu et al., 2022. (Cross-listed track E.)
21. MAT — Wen et al., 2022. (Cross-listed track E.)
22. Attention for Routing (Kool et al.) — 2019, ICLR. Relevance: Transformer
    encoder + pointer decoder with REINFORCE rollout baseline for selection.
23. Mamba — Gu, Dao, 2023. Relevance: linear-time selective SSM
    alternative to slot-Transformer for 400-step memory.
24. Generalizable Graph-based RL (KEEP) — King, Bowman, Huang, 2025,
    arXiv. Relevance: CAGE-4 5th-place GNN-PPO; node/edge/global action
    matrices match host×command structure.
25. CC4 MARL efficacy — Kiely et al., 2025, AAAI. Relevance: defines CC4
    MARL env, baselines, why flat MLPs fail variable topology.

### G. World models, memory, belief (33)

1. DreamerV3 — Hafner et al., 2025, Nature. Relevance: sparse-reward
   400-step latent imagination for remediation chains.
2. DreamerV2 — Hafner et al., 2021, ICLR. Relevance: discrete latent
   attacker-state hypotheses suit discrete isolate/restore actions.
3. DreamerV1 — Hafner et al., 2020, ICLR. Relevance: long-horizon credit
   over weak early alerts via analytic value gradients.
4. PlaNet — Hafner et al., 2019, ICML. Relevance: fast latent planning
   from noisy obs without full state.
5. World Models — Ha, Schmidhuber, 2018, NeurIPS. Relevance: compress
   history into memory remembering weak signals when view shows 0.
6. R2D2 (recurrent replay) — Kapturowski et al., 2019, ICLR. Relevance:
   recipe for LSTMs over 400-step episodes: burn-in, 80-step sequences.
7. DRQN — Hausknecht, Stone, 2015. Relevance: recurrent baseline vs
   frame-stacking for flickering Red presence.
8. GTrXL — Parisotto et al., 2020. (Cross-listed track F.)
9. Transformer-XL — Dai et al., 2019. (Cross-listed track F.)
10. HELM — Paischer et al., 2022, ICML. Relevance: frozen pretrained
    memory as history compressor when learned suspicion fails.
11. Decision Transformer — Chen et al., 2021. (Cross-listed track C.)
12. DVRL — Igl et al., 2018, ICML. Relevance: particle-filter SMC belief
    encoder jointly with policy; explicit belief over hidden compromise.
13. MERLIN — Wayne et al., 2018. Relevance: external memory + variational
    predictor; prediction shapes representations when reward cannot.
14. RL² — Duan et al., 2016. Relevance: meta-learn per-host suspicion
    update adapting within-episode to unseen Reds.
15. MAML — Finn et al., 2017, ICML. Relevance: few-gradient adaptation to
    novel attacker without retraining.
16. VariBAD — Zintgraf et al., 2020, ICLR. Relevance: Bayes-optimal
    probing-vs-remediation under unknown Red via latent task posterior.
17. PEARL — Rakelly et al., 2019, ICML. Relevance: off-policy belief over
    attacker type; posterior sampling for extended hunts.
18. Predictive State Representations — Littman et al., 2001, NeurIPS.
    Relevance: track compromise as predictions of future alerts/spread,
    not unobservable state.
19. POMDP planning (Kaelbling et al.) — 1998. Relevance: belief-MDP +
    finite-memory controller framing for Blue vs invisible Red.
20. Bayesian RL survey — Ghavamzadeh et al., 2015. Relevance:
    suspicion-as-posterior toolkit; BAMDPs, Thompson sampling.
21. Optimal Learning (BAMDP thesis) — Duff, 2002. Relevance: probing/
    remediation as planning over hyperstates.
22. BA-POMDPs — Ross et al., 2007, NeurIPS. Relevance: jointly learn
    detection model + plan sensing vs cleaning.
23. POMCP — Silver, Veness, 2010, NeurIPS. Relevance: online remediation
    planning with the CAGE simulator as black-box model; no explicit
    compromise probabilities.
24. SARSOP — Kurniawati et al., 2008, RSS. Relevance: offline solver for
    large security POMDPs over likely compromise beliefs.
25. MuZero — Schrittwieser et al., 2020, Nature. Relevance:
    value-equivalent spread model for multi-step containment + MCTS.
26. PPO (recurrent baseline) — Schulman et al., 2017. (Cross-listed D.)
27. Optimal Stopping (intrusion prevention) — Hammar, Stadler, 2021, CNSM.
    Relevance: belief-threshold stopping for monitor→block escalation.
    (Also track A.)
28. Multi-stop extension — Hammar, Stadler, 2022. Relevance: remediation
    chains as sequential stops under attacker uncertainty.
29. E-GraphSAGE — Lo et al., 2022, NOMS. Relevance: lateral-spread
    prediction over topology from flow edges.
30. Anomal-E — Caville et al., 2022, KBS. Relevance: unsupervised
    self-supervised weak-signal detector for zero-day Red.
31. NID data-driven survey — Chou, Jiang, 2021, ACM CSUR. Relevance: warns
    sandbox datasets mislead; sim-vs-real evaluation guidance.
32. CC4 hierarchical IPPO — Kiely et al., 2025, AAAI. (Cross-listed A #25
    context; master/sub-policies for investigate/recover/block.)
33. CybORG gym — Standen et al., 2021. (Cross-listed A.)

### H. Robustness, UED, opponent curricula (32)

1. POET — Wang et al., 2019, GECCO. Relevance: co-evolve attacker variants
   instead of fixed discovery/finite reds.
2. Enhanced POET — Wang et al., 2020, ICML. Relevance: sustain open-ended
   curriculum past local optima of single-red training.
3. UED / PAIRED — Dennis et al., 2020, NeurIPS. Relevance: core regret
   curriculum formalism for unseen attackers.
4. PLR — Jiang et al., 2021, ICML. Relevance: replay high-TD-error
   seeds/variants; cheap seed-brittleness fix.
5. DCD / PLR⊥ — Jiang et al., 2021, NeurIPS. Relevance: minimax-regret
   guarantee; train-only-on-curated rule vs held-out collapse.
6. ACCEL — Parker-Holder et al., 2022, ICML. Relevance: evolve reds at
   capability frontier; closes -59/-139 red gap.
7. SAMPLR — Jiang et al., 2022, NeurIPS. Relevance: grounded fictitious
   transitions preserve Bayes-optimality; fixes curriculum covariate
   shift behind ±313 noise.
8. ADR (OpenAI Rubik's) — Akkaya et al., 2019. Relevance: auto-expand
   red-parameter ranges on competence; model for our suite.
9. Domain Randomization — Tobin et al., 2017, IROS. Relevance: treat
   held-out seeds/reds as training variation.
10. Active DR — Mehta et al., 2020, CoRL. Relevance: sample informative
    variations, not uniform seeds yielding -414 episodes.
11. RARL — Pinto et al., 2017, ICML. Relevance: minimax Blue-vs-
    disturbance co-training; minimal adversarial red loop.
12. Robust RL (actor-disturber) — Morimoto, Doya, 2005. Relevance:
    foundational minimax policy learning.
13. EPOpt — Rajeswaran et al., 2017, ICLR. Relevance: CVaR over ensemble;
    survive tail -414 episodes, not just the mean.
14. Double Oracle — McMahan et al., 2003, ICML. Relevance: iterative
    Blue/red population expansion to equilibrium; antidote to single-red
    overfit.
15. PSRO — Lanctot et al., 2017, NeurIPS. Relevance: double oracle for deep
    RL Blue vs red populations + exploitability metric.
16. Pipeline PSRO — McAleer et al., 2020, NeurIPS. Relevance:
    parallelized co-training; makes many-red + ≥8-seed eval affordable.
17. XDO — McAleer et al., 2021, NeurIPS. Relevance: efficient equilibrium
    for sequential cyber defence.
18. RMDO — Tang et al., 2023, ICML. Relevance: polynomial-sample oracle for
    faster equilibrium with theory.
19. α-Rank — Omidshafiei et al., 2019, Sci Rep. Relevance: rank Blue/red
    populations under intransitivity where means mislead.
20. Emergent Tool Use (hide-and-seek) — Baker et al., 2019/2020. Relevance:
    self-play alone generates novel attacker tricks; no hand-designed reds.
21. AlphaStar league — Vinyals et al., 2019, Nature. Relevance: main/
    exploiter league prevents collapse to unseen counters; Blue-league
    blueprint.
22. MAML — Finn et al., 2017. (Cross-listed track G.)
23. Procgen — Cobbe et al., 2020, ICML. Relevance: train/test seed splits
    and scaling laws; justifies ≥8-seed reporting.
24. CoinRun — Cobbe et al., 2019, ICML. Relevance: fixed-seed overfit demo;
    exactement our lancer_v1 regression-vs-held-out pattern.
25. mixreg — Wang et al., 2020, NeurIPS. Relevance: smooth policy across
    seeds; reduce ±130 variance.
26. Network Randomization — Lee et al., 2020, ICLR. Relevance: invariant
    features robust to unseen obs perturbations.
27. UCB-DrAC — Raileanu et al., 2021, NeurIPS. Relevance: auto-select
    augmentation per task; decoupled policy/value regularization.
28. DAAC/IDAAC — Raileanu, Fergus, 2021, ICML. Relevance: separate
    policy/value nets fix shared-encoder overfit inflating regression
    scores — check OUR encoder sharing.
29. IRM — Arjovsky et al., 2019. Relevance: causal features stable across
    reds, not spurious seed correlations.
30. IPO — Sonar et al., 2021, L4DC. Relevance: RL instantiation of
    invariance for cross-domain Blue transfer.
31. Test-Time Training — Sun et al., 2020, ICML. Relevance: online
    adaptation to novel red at deployment without labels.
32. Deep RL That Matters — Henderson et al., 2018, AAAI. Relevance:
    multi-seed mean±std + significance mandate. (Also track J.)

### I. Hierarchy, safety, programmatic policies (30)

1. Options framework — Sutton, Precup, Singh, 1999. Relevance: formalizes
   Analyse→Remove→Restore as options; fixes flat coverage collapse.
2. Option-Critic — Bacon et al., 2017, AAAI. Relevance: learns intra-option
   policies + terminations end-to-end, no hand subgoals.
3. Deliberation cost — Harb et al., 2018, AAAI. Relevance: stops options
   degenerating to single-step actions that kill sweeps.
4. DAC — Zhang, Whiteson, 2019, NeurIPS. Relevance: trains master+options
   with standard PPO/SAC for 155-dim discrete actions.
5. FeUdal Nets — Vezhnevets et al., 2017, ICML. Relevance: slow scheduler
   + workers; learned ordering over fixed sweeps.
6. HIRO — Nachum et al., 2018, NeurIPS. Relevance: off-policy two-level
   goal-conditioned learning for sparse multi-step remediation.
7. HAC — Levy et al., 2019, ICLR. Relevance: parallel 3-level hierarchy +
   hindsight densifies sparse rewards over long chains.
8. DIAYN — Eysenbach et al., 2019. (Cross-listed track B.)
9. OPAL — Ajay et al., 2021, ICLR. Relevance: offline primitives from Blue
   logs; shorten horizon, constrain shift.
10. SPiRL — Pertsch et al., 2020, CoRL. Relevance: skill prior guides
    exploration toward promising hosts.
11. LOVE — Jiang et al., 2022, NeurIPS. Relevance: compression finds true
    Analyse→Remove boundaries.
12. CMDPs — Altman, 1999. Relevance: formalism for safety/coverage
    constraints alongside reward.
13. CPO — Achiam et al., 2017, ICML. Relevance: trust-region safe updates
    keep learned ordering inside guardrails during training.
14. Safe RL survey — Garcia, Fernandez, 2015, JMLR. Relevance: maps our
    masks/shields to safety taxonomy.
15. Shielding — Alshiekh et al., 2018, AAAI. Relevance: symbolic shield
    overrides unsafe actions, minimal interference, preserves convergence.
16. Shielding under PO — Carr et al., 2023, AAAI. Relevance: shields for
    POMDP sparse-reward settings; bootstrapping then removable.
17. Invalid-action masking — Huang, Ontañón, 2020. Relevance: theory
    legitimizing our validity masks as correct gradients for 155 actions.
18. PIRL — Verma et al., 2018, ICML. Relevance: verifiable programmatic
    Blue policies from neural oracle.
19. VIPER — Bastani et al., 2018, NeurIPS. Relevance: distill DNN scheduler
    to decision trees preserving coverage, verifiable.
20. LEAPS — Trivedi et al., 2021, NeurIPS. Relevance: latent program space
    for sweep/remediation routines from reward only.
21. HPRL — Liu et al., 2023, ICML. Relevance: compose short programs into
    longer OOD remediation chains.
22. HIPO — Lin et al., 2024, NeurIPS. Relevance: programs-as-options for
    long repetitive sweeps; CEM + diversity + compatibility retrieval.
23. DreamCoder — Ellis et al., 2021, PLDI. Relevance: grow reusable routine
    library jointly with neural search for hybrid rules.
24. Behavior Trees — Colledanchise, Ögren, 2018. Relevance: modular sweep
    structure with learned parameter nodes.
25. Hierarchical Deep MARL — Tang et al., 2018. Relevance: low-level skills
    + high-level coordination for 5-agent sparse-delayed teams.
26. OPRE — Vezhnevets et al., 2020, ICML. Relevance: separate attacker-
    latent estimation from best-response execution.
27. ROMA — Wang et al., 2020, ICML. Relevance: emergent specialized roles
    (subnet defenders) with shared learning.
28. Adaptive MoE — Jacobs et al., 1991. Relevance: gating+experts template
    for routing obs to fixed Blue skills.
29. Sparse MoE — Shazeer et al., 2017, ICLR. Relevance: sparse router over
    fixed heuristic sub-policies without proportional compute.
30. HRL survey — Pateria et al., 2021, ACM CSUR. Relevance: taxonomy
    linking feudal/options/subgoal/MARL choices for our hybrid.

### J. Evaluation rigor + security signal design (32)

1. Statistical Precipice (rliable) — Agarwal et al., 2021, NeurIPS.
   Relevance: IQM, optimality gap, performance profiles, stratified
   bootstrap — adopt alongside mean±std.
2. Deep RL That Matters — Henderson et al., 2018. (Cross-listed H.)
3. How Many Random Seeds? — Colas et al., 2018. Relevance: power analysis;
   why 3 seeds prove nothing; size N from effect/variance.
4. Hitchhiker's Guide to Statistical Comparisons — Colas et al., 2019.
   Relevance: Welch t-test choice for non-normal CC4 returns.
5. Overfitting in Deep RL — Zhang et al., 2018. Relevance: train/test
   splits mandatory; training reward can memorize.
6. Quantifying Generalization (CoinRun) — Cobbe et al., 2019.
   (Cross-listed H.)
7. Procgen — Cobbe et al., 2020. (Cross-listed H.)
8. Procgen Benchmark — Mohanty et al., 2021, PMLR. Relevance: manifest
   template — fixed timesteps, hidden test levels, trace hashes.
9. Performance Profiles — Dolan, Moré, 2002. Relevance: empirical CDFs
   instead of rank flips from single means.
10. Reproducibility of Benchmarked Tasks — Islam et al., 2017. Relevance:
    pinned commits + artifact hashes (our manifests).
11. Illuminating Generalization via PLG — Justesen et al., 2018. Relevance:
    train across level distributions, not one topology/seed.
12. PBRS — Ng et al., 1999. (Cross-listed track B.)
13. CybORG gym — Standen et al., 2021. (Cross-listed track A.)
14. CC4 efficacy — Kiely et al., 2025, AAAI. (Cross-listed track A.)
15. CC4 AI Magazine — Kiely et al., 2025. Relevance: extended reference for
    per-step rates, cross-paper comparison.
16. CyberBattleSim — Microsoft, 2021. (Cross-listed track A.)
17. Deception in CyberBattleSim — Andrew et al., 2021. (Cross-listed A.)
18. Adaptive Honeypot Engagement (SMDP) — Huang, Zhu, 2019. Relevance:
    intel-vs-risk tradeoff for CC4 Decoy use.
19. Optimal Strategy Selection for Cyber Deception via Deep RL — Zhang et
    al., 2022. Relevance: honeypot deployment vs attack graphs with
    deployment-cost penalty.
20. Honeypot Allocation over Attack Graphs — Anwar et al., 2020, ICNC.
    Relevance: game-theoretic decoy placement under budget.
21. Concealing Honeypot Functionality — Dowling et al., 2019, ECML PKDD
    Workshop. Relevance: RL-tuned honeypot replies that look real.
22. Adaptive Honeypot Deployment — Wang et al., 2025, IEEE ICCBDAI.
    Relevance: deception-success reward minus waste; dense honest shaping
    template.
23. Optimal Stopping (intrusion prevention) — Hammar, Stadler, 2021.
    (Cross-listed track A/G.)
24. Multi-stop extension — Hammar, Stadler, 2022. (Cross-listed G.)
25. Security Strategies via Self-Play — Hammar, Stadler, 2020.
    (Cross-listed A.)
26. FARLAND — Molina-Markham et al., 2021. (Cross-listed A.)
27. Network Defense is Not a Game — Winder, 2021. Relevance: evaluate over
    drifting distributions, not one fixed game.
28. Partitioned DQN responders — Cardellini et al., 2022. Relevance:
    per-subnet DQN scaling like multi-Blue without sharing privilege.
29. Cyber Alert Allocation Markov Games — Shah et al., 2019, GameSec.
    Relevance: alert-to-analyst allocation under overload; triage model.
30. Two Can Play That Game — Shah et al., 2020, ACM TIST. Relevance:
    red-team the triage, retrain via double-oracle.
31. Meta-AAD — Zha et al., 2020, ICDM. Relevance: anomaly outputs + analyst
    feedback as RL state; no true labels as inputs (matches our privilege
    rule).
32. RAPID — Liu et al., 2022, ACSAC. Relevance: context/history alert
    embeddings; RL ranks multi-step chains — denser honest Blue signal.

## Suggested program (ordered, cheapest first)

- P0 Adopt rliable-style reporting (J1–J4) on existing manifests. No new
  runs.
- P1 IQL-discrete on lancer_v2 logs (C15) + RvS return-MLP baseline (C26);
  gate on held-out vs teacher. Uses existing 8101+ demos.
- P2 AWAC from attention ckpt (D14) with stop-rules; PEX variant (D22) if
  it dips.
- P3 JSRL with lancer_v2 guide (D21); kickstarting schedule (D1) as the
  cheaper cousin.
- P4 Critic audit: history-state + decentralized critics (E21–E23), then
  HAPPO (E13) on the factorized actor.
- P5 Exploration: E3B/NovelD bonus (B14/B15) + dynamic PBRS densification
  (B22) + Bates sparse-healthy bonus (A25) — one combined shaping track
  with optimality proofs attached.
- P6 Red curriculum: PLR⊥ replay over seed/variant pool first (H4/H5, no
  sim changes), ACCEL/PSRO after Environment hooks land (H6/H15).
- P7 Actor: entity-Transformer (A28) → SR-DRL autoregressive RL (F16) →
  GTrXL stabilization (F14) to carry the attention win into RL.
- P8 Belief: DVRL particle belief (G12) or VariBAD context (G16) as the
  learned replacement for hand suspicion; POMCP (G23) as a planning
  baseline using the sim as model.
- P9 Programmatic: shield (I15/I16) around whatever wins; LEAPS/HIPO
  (I20/I22) only if interpretability becomes the binding constraint.

## Duplicates across tracks (intentional)

COMA (B25/E1), DT (C21/F19/G11), GTrXL (F14/G8), Transformer-XL (F11/G9),
MAML (G15/H22), AWAC (C18/D14), CRR (C17/D15), BEAR (C13/D17), TD3+BC
(C19/D18), PPO (D7/G26), QMIX (E3/F10), MAT (E15/F21), MAPPO-tuning (E12/
F20), Hammar stopping line (A12–A14/G27–G28/J23–J24), Henderson (H32/J2),
CybORG (A1–A2/G33/J13), CC4-Kiely (A34–A35/F25/G32/J14–J15), CyberBattleSim
(A18–A19/J16–J17), FARLAND (A8/J26), DIAYN (B19/I8), Cobbe CoinRun/Procgen
(H23–H24/J6–J7), Ng PBRS (B21/J12). Kept in each track for provenance;
count unique papers ≈ 260.
