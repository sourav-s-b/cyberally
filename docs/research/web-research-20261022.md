# Web research pass — field evidence + budget feasibility (2026-10-22)

Owner: Blue. Branch: `blue/web-research-20261022` (worktree `worktrees/research`).

Method: 4-phase plan approved by the user (both goals equally, ~1 week effort budget,
broad ~20-entry citation verification). Stop rule applied throughout: a paper that could
not be located/verified in ~15 minutes is marked unverified rather than cited. No
experiments were launched and no seed block was consumed; seed-audit of 8433–8499 is
required only if a pre-registered experiment from Section E is started.

Note on archive figures: score numbers quoted from the AAAI-25 review below were
verified against the extracted paper text during the session
(`/tmp/opencode/kiely-cc4-aaai25.txt`, session-local); re-check against the published
PDF before they appear in any external deck.

## A. Field evidence — what CAGE4 participants actually did

Primary source: Kiely et al., *Cyber Security as a Multi-Agent Game: A Review of CAGE
Challenge 4*, AAAI-25 workshop (DOI 10.1609/aaai.v39i28.35158). Verified findings:

- Table 2 (CC4, 100 episodes x 500 steps, generalization variant): best heuristic
  scores are far ahead of the best MARL agent on every environment variant; the best
  MARL entry (Cybermonic, GNN-based) converged only after ~10^4-scale episodes per
  configuration and still clearly trailed the heuristic leaders.
- 15 participants / 65 submissions; **no RL submission beat the top scripted
  heuristics**; official failure causes were (1) invalid actions on non-existent hosts
  and (2) inability to infer network configuration.
- Team behaviours that worked: UC — hostile-observation rewrite ("malicious flags")
  plus proactive traffic control phased with the mission; Lancer — decoy covering
  every host with priority raised on Monitor observations; Punch — a wrapper exposing
  raw file/density statistics (threshold ~0.9).

Secondary sources (verified):

- Singh et al., "Hierarchical MARL..." arXiv 2410.17351, published as RLJ vol. 6
  (2025) pp. 790–810, RLC 2025 Paper77: hierarchical PPO with sub-policies plus
  observation enhancements; reports strong returns but **contains no heuristic
  baseline comparison**, so it is not counter-evidence to the AAAI-25 result.
- TERLA (Dudman & Bull, "Towards a Generalisable Cyber Defence Agent for Real-World
  Computer Networks", arXiv 2511.09114, CAMLIS 2025, PMLR 299:87–109) attacks the
  topology-transfer axis; SR-DRL (arXiv 2407.05775, IEEE CSR 2024) attacks the
  structure axis; posterior-conditioned MARL (Kazeminajafabadi et al., AIJ 360:104606,
  2026) attacks the nonstationary-attack axis on CybORG/CAGE4.

**Absolute finding: we found no published RL or imitation-learning agent that beats
the strong scripted heuristics in the CAGE4 setting.** Local corroboration: our own
IQL off-policy study on Blue-branch logs scored −96.0 vs the round-robin heuristic
−85.1 on held-out seeds, and the finished margin-bounded override arm lost its
pre-registered dev gate (see `docs/research/margin-bounded-deploy-20261022.md`,
branch `blue/margin-bounded-override`).

Constraint interaction: our local wrapper currently has no Deploy/Decoy/Traffic-Control
actions, so the winning participants' main levers cannot be expressed today. Any RL
attempt in this wrapper inherits a handicap that the literature explicitly names as a
top failure cause (invalid actions / missing configuration cues).

## B. Budget feasibility — does imitation→RL ever beat imitation at small compute?

Question posed: is there verified evidence that offline-RL/fine-tuning on top of
imitation beats imitation alone under a <1-shard / one-night-local budget?

Verified evidence, converging:

- RL's Razor (arXiv 2509.04259): on-policy filtering of old data, not KL or clip
  tuning, is what separates RL fine-tuning from SFT-style regression.
- Retaining by Doing (arXiv 2510.18874, ICML 2026): RL forgets less than SFT because
  of mode-seeking on-policy data; approximately-on-policy data suffices.
- Catastrophic forgetting in fine-tuning (arXiv 2402.02868, ICML 2024 spotlight):
  poor fine-tuning transfer is mostly forgetting of pre-trained behaviour on
  subspaces not visited early; retention tricks mitigate it.
- Three Regimes of Offline-to-Online RL (arXiv 2510.01460): in the "Superior" regime
  (small imitation-log gap) pi0-style anchored fine-tuning yields small effects by
  construction — this matches our own null anchor pilots.
- Naive Q-function pretraining gives little or no benefit vs random init for online
  fine-tuning (arXiv 2607.27203, Dong/Polonsky/Sadigh/Finn 2026) — critic-pretrain
  paths are off the table.
- When offline RL beats BC (Kumar et al., arXiv 2204.05618): conditions are sparse
  rewards, noisy/suboptimal data, long horizons. Our measured case inside the
  simulator: only 41% of reward entries are nonzero, the first nonzero arrives around
  step ~135, and ~69% of steps are forced-Sleep — i.e. exactly the sparse/noisy
  profile. So the conditions side is met.
- Scale reality check: Cybermonic's MARL convergence at ~10^4-scale episodes per
  configuration says a MAPPO-class re-run at our budget is underpowered; this is a
  datum, not a guarantee.

**Verdict: the sparse-reward/noisy-data conditions favour RL-over-BC in principle, but
the only budget-feasible designs are (a) offline RL that reuses already-collected logs
(AWAC/ReBRAC family) and (b) shrinking-horizon warm-start from a scripted guide
(JSRL/kickstarting). A fresh MAPPO-class regime-conditioned run is off-table at this
budget.** This preserves the AGENTS.md constraint that fresh rollouts, not replayed
transitions, feed an on-policy update — offline methods sidestep that rule rather than
violating it.

## C. Verified citation ledger (verification corrections included)

Verdict key: [OK] = cited ID/venue correct; [FIX-ID] = wrong identifier or venue,
correct one given; [DROP] = claim not supported, do not use.

| # | Claim/paper | Verdict | Correct citation / note |
|---|---|---|---|
| 1 | AWAC "Efficient RL with Previously Learned Policies" | [OK] | arXiv 2006.09359 |
| 2 | PEX (pessimistic exploration fine-tune) | [FIX-ID] | arXiv **2302.00935**, Zhang/Xu/Yu, ICLR 2023; offline head is AWR-like, not BC; "no divergence" is our gloss |
| 3 | Kickstarting + HKS | [OK] | Kickstarting arXiv 1803.03835 (decay sched. already in paper); Hierarchical Kickstarting arXiv 2207.11584 |
| 4 | MPO | [OK] | arXiv 1806.06920; "dual temperature/bisection" is paraphrase, not paper wording |
| 5 | Snapshot Ensembles / SWA | [OK] | 1704.00109 (ICLR 2017) / 1803.05407 (UAI 2018); SWA is weight average, not snapshot ensemble |
| 6 | Blocked CV (Bergmeir/Roberts) | [FIX-ID] | Bergmeir, Hyndman & Koo, CSDA 120:70–83 (2018); blocked-CV advocacy itself = Bergmeir & Benítez, Inf. Sci. 191 (2012); Roberts et al., Ecography 40(8) 2017 (not PNAS) |
| 7 | SPOT | [FIX-ID] | **2202.06239**, Wu et al. "Supported Policy Optimization" (2022); VAE support estimator, not an IQL variant. 2407.10840 is wrong |
| 8 | TD3+BC | [OK] | 2106.06860, Fujimoto & Gu, NeurIPS 2021 |
| 9 | EDAC / MSG | [FIX-ID x2] | EDAC **2110.01548** (NeurIPS 2021; 2110.03375 = GPL, different paper); MSG **2205.13703** (NeurIPS 2022, Ghasemipour et al.) |
| 10 | DVRL / VariBAD | [FIX-ID / OK] | DVRL = **1806.02426**, Igl et al. 2018 (not Zintgraf 2019); VariBAD = 1910.08348, ICLR 2020 [OK] |
| 11 | JSRL | [FIX-ID] | **2204.02372**, Uchendu et al., ICML 2023 (2206.07101 is astrophysics); guide policy need not be demos-trained; shrinking-horizon curriculum confirmed |
| 12 | RvS / Brandfonbrener | [FIX-ID x2] | RvS = **2112.10751**, Emmons et al. (ICLR 2022); Brandfonbrener "return-conditioned supervised learning does not stitch" = **2206.01079**, NeurIPS 2022 |
| 13 | PBRS theory pair | [OK] | Ng/Harada/Russell ICML 1999; Devlin & Kudenko, "Dynamic Potential-Based Reward Shaping", AAMAS 2012 pp. 433–440 (do not conflate with their 2011 AAMAS multi-agent theory paper) |
| 14 | HAC | [FIX-venue] | arXiv 1712.00948 correct, venue **ICLR 2019** (not ICML 2018) |
| 15 | CQL | [OK] | arXiv 2006.04779, NeurIPS 2020 |
| 16 | Forgetting in fine-tuning | [OK] | 2402.02868, Wołczyk et al., ICML 2024 spotlight |
| 17 | Retaining by Doing | [OK] | 2510.18874, ICML 2026 |
| 18 | "Do you really need to pretrain Q-functions" | [FIX-ID] | arXiv **2607.27203** (2026), Dong/Polonsky/Sadigh/Finn; author-string "Chase Kiefer" does not exist |
| 19 | Bates reward-shaping paper | [FIX-TITLE] | "Less is more? Rewards in RL for Cyber Defence", **2503.03245** (4 pp.); sibling: 2310.13565 (ACM AISec 2023); long version 2602.04809 |
| 20 | Martino ESS / RLPD | [OK / FIX-ID] | Martino ESS = 1602.03572 (Signal Processing 131, 2017) [OK]; RLPD = **2302.02948**, Ball et al., ICML 2023 (2305.09830 = unrelated); do NOT attribute "ESS-triggered critic rescaling" to RLPD — not in the paper |
| 21 | Lyu centralized-critic | [FIX-yr] | "Contrasting Centralized and Decentralized Critics" = **2102.04402**, AAMAS **2021**; the partial-observability bias/variance analysis for state-based critics is in the JAIR 2023 extension, arXiv 2408.14597; sibling AAAI-22 paper is 2201.01221 |
| 22 | Posterior-conditioned MARL CAGE4 | [OK, naming] | Kazeminajafabadi et al., AIJ 360:104606 (2026), DOI 10.1016/j.artint.2026.104606, open access |
| 23 | Singh hierarchical MARL venue | [OK] | RLJ vol. 6 (2025) 790–810, RLC 2025 Paper77, arXiv 2410.17351 |
| 24 | TERLA | [FIX-TITLE] | "Towards a Generalisable Cyber Defence Agent for Real-World Computer Networks", arXiv 2511.09114, CAMLIS 2025, PMLR 299:87–109; "TERLA" is the method name, not the title |
| 25 | SR-DRL | [OK] | 2407.05775, IEEE CSR 2024; "SR-DRL" is the method name in the paper, not the title |
| 26 | Codevilla offline-eval driving | [FIX-ID] | "On Offline Evaluation of Vision-based Driving Models", **1809.04843**, ECCV 2018; the "~0.39" figure is not stated as such — cite as "weak/unreliable correlation" per abstract |
| 27 | robomimic/L3 checkpoint selection | [FIX-ID] | "What Matters in Learning from Offline Human Demonstrations", **2108.03298**, CoRL 2021; valid-loss/last checkpoint selected 50–100% worse than oracle |
| 28 | RL's Razor | [OK] | arXiv 2509.04259 |
| 29 | Implicit BC seed variance | [OK] | 2109.00137, CoRL 2021: explicit BC unstable across seeds, implicit BC stable |
| 30 | ReBRAC / CORL / Wang | [OK] | 2305.09836 (10-unseen-seed protocol); 2210.07105 (CORL benchmark code); 2306.00972 (Wang, ICML 2023) |
| 31 | SMODICE | [FIX-ID] | **2202.02433**, Ma et al., ICML 2022: offline IL via state-occupancy matching, no expert actions needed; f-divergence Fenchel-dual matching (no "Gaussian"); 2210.10775 is an unrelated segmentation paper — never cite it as SMODICE |
| 32 | Q-Transformer | [FIX-ID, DROP] | **2309.10150**, Chebotar et al., CoRL 2023 (2302.01542 is unrelated); the "horizon-dependent regret bounds" attribute is NOT in the paper — drop it. If a horizon-bound claim is needed, verify "Q-learning Decision Transformer" 2306.11372 separately first |
| 33 | RL-vs-BC conditions | [FIX-AUTHOR] | Conditions for offline RL > BC = **Kumar et al. 2204.05618** (ICLR 2022), not Brandfonbrener; Brandfonbrener's no-stitching theorem = 2206.01079 (NeurIPS 2022) |
| 34 | Rigaki "stochastic parrots" | [FIX-TITLE] | "Out of the Cage: How Stochastic Parrots Win in Cyber Security Environments", **2308.12086**, ICAART 2024 (GPT-4 red agent matches/beats DQN with zero training; environment is NetSecGame, not CAGE). The title "Model Exfiltration and Trust Issues with LLM-assisted Cyber Agent" could not be verified anywhere — do not use |
| 35 | Yan mentoring-LLM CAGE | [OK] | "Depending on yourself when you should: Mentoring LLM with RL agents...", arXiv 2403.17674 (2024, no confirmed venue): SecurityBot, LLM mentored by RL experts, CAGE Challenge 1 |

## D. Novelty scan (negative results are findings)

1. **LLM as RED agent inside CAGE2/3/4 — not found** in any publication. All LLM
   agents actually run inside CAGE environments are blue/defender side (e.g.
   2505.04843; LLM+RL training acceleration 2509.05311). LLM red agents exist only in
   NetSecGame (2308.12086; Hackphyr 2409.11276; 2603.10041) and ad-hoc pentest
   frameworks. Medium-high confidence; absence can never be proven.
2. **Belief/hypothesis-testing-gated remediate-or-not decisions — not found.** Closest
   works learn remediation mappings under uncertainty (2302.01595; AISA 2507.07416;
   PenHeal). The articulation "a successful Remove does not prove the host is clean —
   record action completion, observed verification, and privileged evaluation as
   distinct facts" appears in our own docs and is not in any found paper.
3. **Attacker-regime-conditioned imitation in cyber defense — not found.** The
   supporting theory (SMODICE occupancy matching; Kumar conditions;
   Brandfonbrener no-stitching) is robotics/D4RL-based. Frame novelty as
   application/validation in cyberspace, not as inventing the mechanism.
4. LLM defenders remain weak (Castro et al. 2505.04843: GPT-4o-mini blue −2547 vs
   all-RL −493; 104x slower) — supports keeping LLM-only blue agents out of our path.

## E. One-pagers (budget-feasible, each pre-registrable; run in this order)

**OP1 — Invalid-action / host-coverage contract suite (no training, do first).**
Matches the AAAI-25 official failure cause. Cold-run the wrapper contract: action
validity masks, host-preservation, pending-action attribution, Remove-then-verify
evidence separation, episode-boundary reproducibility. Accept if there is no invalid
action on any seed set; fix visible defects before any training run.

**OP2 — Scripted-guide warm start (JSRL 2204.02372 / Kickstarting 1803.03835).**
Use the existing heuristic (guard/Lancer-style) as a frozen guide policy with a
shrinking guide-step curriculum. Addresses our 41%-nonzero/first-nonzero-at-~135
credit-assignment problem by giving early dense guidance. Budget fits: freeze guide,
PPO fine-tune, eval vs guard. Pre-register tau/K; require seed-audit of 8433–8499.

**OP3 — Offline RL reusing already-collected logs (AWAC 2006.09359, ReBRAC 2305.09836
protocol).** Target every measured failure mode at once; zero new rollouts needed at
run time and no AGENTS.md conflict with on-policy retraining (we're not reverting
MAPPO). Justify with the Implemenation-BC seed-variance result (2109.00137).
Pre-register: clean held-out set, comparison vs guard + BC; ReBRAC's 10-unseen-seed
protocol for reporting.

**OP4 — Regime-conditioned BC baseline (SMODICE 2202.02433 spirit).** Condition the
imitation policy on red-regime features; SMODICE shows state occupancy matching is
enough for the imitation part. Run two-level: (i) train per-regime heads, (ii)
re-dial SW variants; if the device variance already fixes the Oracle/CC gap, this is
the mapping instead. Evaluate against the no-baseline shift pairs (Bronkhorst
2206.01079 theorem for why failing here is expected).

**OP5 — Seed-variance protocol (mandatory for every future Blue experiment).**
Based on 2109.00137, 2305.09836, and 2210.07105 (CORL real-code): no single-seed
claims; report mean +/- std over >=10 unseen seeds; if only 3 seeds are affordable,
state it as a bounded-CI estimate, not as a headline. This is process, not code.

**OP6 — Critic design under partial observability (design note, no run).** Lyu et al.
2408.14597: state-based centralized critics can inject bias/variance under
partial observability in Dec-POMDP settings like CAGE4. Required to read before any
MAPPO retrain. Deliverable: one-page design note in docs/ with which critic we
would/wouldn't use, per our wrapper.

**OP7 — Belief-gated remediation (novelty candidate, research note first).**
Our measured drivers: (i) B4 says the mechanism seems novel, and (ii) our own docs
articulate the Remove != clean-host danger. Feasibility: implement the "belief-
gated France" mechanism via tutor/belief-defined gates: record Remove-success-only
vs Verify-success-only vs both; emit "action-observed/evaluated" triples in the
update machinery only; evaluate the latch left margin vs the margin-bounded
containment machinery if/when adopted to training (branch
`blue/margin-bounded-override`).

**OP8 — LLM-Red agent inside CAGE4 (long-shot novelty play, park unless the user
unblocks).** B3: not found in literature; the exact gap our LLM-Red branch targets.
Not budget-feasible this week; needs a Cybermonic-scale rerun per pair, and gate
success on the pre-registered LLM-Red progression condition we already set.

## F. Risks, non-actions, and branch plan

- No fresh MAPPO retraining (AGENTS.md), no new reward shaping during this
  verification pass, and no seed-block consumption until a pre-reg + seed-audit of
  8433–8499 go first.
- Do-not-cite list after verification: 2210.10775 (not SMODICE), 2302.01542 (not
  Q-Transformer), 2407.10840 (not SPOT), 2110.03375 (not EDAC = GPL), 2206.07101 (not
  JSRL), 2305.09830 (not RLPD), 2302.00935 must be looked up as PEX, plus
  "Model Exfiltration..." title, and every "~0.39 correlation" phrasing as a point
  estimate.
- Deliverable branch: this file + the blue-session status update live on
  `blue/web-research-20261022`. Integration: PR this to main after review, then
  teammates rebase per docs/team-workflow.md.
