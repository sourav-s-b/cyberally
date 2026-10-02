# Proposal 01: Hybrid rules + learned scan priority

- Status: tested — all ordering variants exhausted. Lancer_v2 parity
  stands; risk hybrids fail the held-out gate. Program: attention
  distill (02 Exp3) or generalization suite (15).
- Draft: `cage-challenge-4/blue_hybrid.py` (`HybridBluePolicy`)
- Date: 2026-10-01 / reviewed 2026-10-02

## Description

Split the policy in two. Fixed, unviolable rules handle what rules do best:
CONFIRMED -> Remove (escalate to Restore on re-detection), VERIFY ->
re-Analyse, oldest remediation first. A learned per-host priority scorer
(`priority_fn(env, agent, host) -> float`) decides only the scan order —
which host to Analyse next. `priority_fn=None` degrades to host order, so the
learnable part is measurable in isolation against round-robin (-93.5 ± 27.8).

## Motivation / evidence

- CAGE-4 winner family, all three (Kiely et al., AAAI 2025): UC's
  flag-driven remediate + Analyse-longest-stale; lancer's round-robin +
  per-host float priority (decay on touch, boost on Monitor) — the closest
  published analog of `priority_fn`; punch's analyse-restore round-robin,
  "much better than our RL agents".
- H-MARL on CC4 (Singh et al., AAMAS'25): expert-rule master (IOCs ->
  Recover) -129.53 beats flat IPPO -181.62 (+52). Rules-first wins measured.
- Our `MASK_RULES`/`BlueZoneTracker` already encode CONFIRMED/VERIFY states;
  the hybrid only adds the ordering head. Privileged labels allowed as
  *training targets* for the risk scorer, never as inputs (contracts).

## Experiment

PARITY PROVEN 2026-10-02 (this change: `priority_fn=None` now runs a cursor
round-robin identical to `RoundRobinBaseline`; the first-draft `cands[0]`
would have stuck on one host, so it was fixed before measuring).

- Regression: `cage-challenge-4/tests_blue/test_hybrid_parity.py` — 2 seeds,
  asserts equal native returns AND identical per-tick traces. Passes (69 s).
- Full manifest, 8 seeds x 400 steps, native reward (scratch
  `/tmp/opencode/parity_8seed.py`, results/ is gitignored so numbers live
  here):

| seed | round_robin | hybrid_none | traces identical |
|---|---|---|---|
| 7629 | -85 | -85 | yes (1995/1995) |
| 7630 | -105 | -105 | yes |
| 7640 | -123 | -123 | yes |
| 7701 | -85 | -85 | yes |
| 7702 | -135 | -135 | yes |
| 7703 | -64 | -64 | yes |
| 7704 | -53 | -53 | yes |
| 7705 | -98 | -98 | yes |

Mean -93.5 both. Sim suite after change: 82 passed, 5 skipped (was 81/5).

Pool rerun 2026-10-02 (`blue_eval_parallel.py --policies round_robin
hybrid_none`, 6 workers, fork, 66 s for 16 cells): identical returns and
trace hashes; manifest `docs/proposals/manifests/parity-pool-20261002.json`;
`blue_compare.py --baseline round_robin` reports mean +0.0, trace MATCH.

## Experiment 2: lancer scorer v1 (2026-10-02, TESTED, beats teacher)

- Implementation: `LancerPriority` in `blue_hybrid.py` (per-host float,
  init 1.0, x0.5 decay per completed Analyse/Remove/Restore counted once,
  +2.0 on CONFIRMED transition, +1.0 on novel view signals, +2.0 sticky
  suspicion bonus while unknown-files/density>0.9/ext-conns/CONFIRMED-VERIFY
  persist). String spec `"lancer"` so registry entries stay picklable;
  `HybridBluePolicy.reset` propagates. Also fixed `host_risk_features`
  (passed 5 args to the 3-arg `host_to_vector`; was dead code, now used).
  Registry: `hybrid_lancer`. Unit tests
  `tests_blue/test_hybrid_priority.py` (6 tests: resolution, reset,
  once-only decay, CONFIRMED boost, pickling, legal actions).
- Bug found by the pool: `env.subnets[agent]` is a host->subnets dict, not
  a list; fixed to per-host entries (matches wrapper obs construction).
- Pool run `lancer-v1-20261002` (8 seeds x 400, native; manifest committed):

| policy | n | mean | std | per-seed returns |
|---|---|---|---|---|
| hybrid_lancer | 8 | -63.2 | 26.0 | -50 -41 -63 -48 -75 -44 -64 -121 |
| round_robin | 8 | -93.5 | 27.8 | -85 -105 -123 -85 -135 -64 -53 -98 |

Paired diff mean +30.2 (+35 +64 +60 +37 +60 +20 -11 -23), 6/8 seeds
better. Paired t ~= 2.6, p ~= 0.04 UNCORRECTED, first variant tried:
encouraging, not a victory claim — needs held-out-seed replication.
Per-step rate -0.158 vs winner-band -0.226 (usual build caveats).
- Mechanism (trace mix, seeds 7704/7705/7630): lancer wins by remediating
  LESS (fewer Restore disruptions; cf. TERLA's 6-7% action rate) while
  keeping coverage. Failure mode on 7704/7705: sticky suspicion bonus
  re-analyses already-known hosts (starves coverage) and re-detection
  chains cause Restore storms (7705: 56 Restores vs 38, final 26 vs 19
  compromised). v2 direction: decay the suspicion bonus with fruitless
  re-analyses (tie to `empty_strikes`), or suppress it for recently
  analysed hosts.

## Experiment 3: v2 + held-out replication (2026-10-02, TESTED, humbling)

- v2: suspicion bonus x `fruitless_decay**n_fruitless` (`fruitless_decay`
  0.5; v1 frozen as `fruitless_decay=1.0`). Fruitless = completed Analyse
  with no new CONFIRMED transition; remediation resets. Registry
  `hybrid_lancer_v2` via new `priority_kwargs` plumbing. First version of
  the logic (reset on any CONFIRMED state) was caught by the new unit test
  and fixed: re-analysing an already-known host must count fruitless.
- Regression, same 8 seeds (`lancer-v2-20261002`):

| policy | mean | std | per-seed |
|---|---|---|---|
| hybrid_lancer (v1) | -63.2 | 26.0 | -50 -41 -63 -48 -75 -44 -64 -121 |
| hybrid_lancer_v2 | -75.2 | 33.6 | -52 -54 -63 -54 -90 -57 -81 -151 |
| round_robin | -93.5 | 27.8 | -85 -105 -123 -85 -135 -64 -53 -98 |

v2 still beats the teacher (+18.2) but is uniformly worse than v1 —
the decay hurts most on the failure seeds (7705: -151 vs -121).
- Held-out, 8 FRESH seeds 7801-7808 chosen before running
  (`lancer-heldout-20261002`):

| policy | mean | std | per-seed |
|---|---|---|---|
| hybrid_lancer (v1) | -110.1 | 130.4 | -138 -414 -42 -134 -32 -35 -41 -45 |
| hybrid_lancer_v2 | -91.2 | 48.5 | -108 -94 -59 -167 -47 -155 -51 -49 |
| round_robin | -85.1 | 46.1 | -88 -90 -52 -144 -51 -163 -53 -40 |

Paired diffs vs teacher: v1 -25.0, v2 -6.1, both inside noise — NO
improvement claim stands.
- Interpretation: v1's +30.2 was selection bias on the regression seeds.
  v1 is seed-fragile (held-out std 130 vs 26): sticky suspicion locks onto
  the right hosts early on some seeds (7806: -35 vs -163) and catastrophically
  neglects coverage on others (7802: -414). v2's decay regularizes toward
  teacher behavior (std 48.5 ~= teacher 46.1) — kills catastrophes AND big
  wins, net teacher parity. The teacher itself is stable across seed sets
  (-93.5 -> -85.1); the learners are not.
- Lesson recorded: fixed constants cannot win the ordering game across
  seeds — they trade catastrophe for mediocrity. Next is the adaptive risk
  scorer (learned per-host priority from Blue-visible features), which can
  condition on *why* a host looks suspicious instead of applying one bonus
  schedule everywhere.

## Experiment 4: snapshot risk scorer (2026-10-02, TESTED, kills the path)

- Dataset `blue_risk_data.py`: per-tick per-host 17-feat rows (10 base +
  ages/belief) + privileged compromised-now labels, round-robin episodes on
  fresh seeds 7901-7908 (244,986 rows, 27.8% pos). v2 adds 4 view-delta
  features (lancer's Monitor-novelty in feature form) -> 21 feats.
- Trainer `blue_train_risk.py`: numpy-only L2 logreg, class-balanced,
  deterministic; no sklearn in either venv (uv install timed out), which
  turned out fine. Two bugs caught by tests before trusting numbers: a
  misranked `auc()` (reported 0.52; true 0.67-0.68) and a NameError.
  Manifests `risk-train-v1/v2/v2now-20261002.json`.
- Gate results: AUC 0.67 undetected / 0.68 now (bar was >= 0.65) — real but
  modest signal. Deltas added nothing (0.670 vs 0.668).
- Scorer `RiskPriority` (batched per agent/tick, pure proba, 8 unit tests):
  pool `risk-v1-20261002` / `risk-heldout-20261002`:

| policy | regr mean | held-out mean |
|---|---|---|
| hybrid_risk | -196.6 ± 150.8 | -190.8 ± 153.1 |
| round_robin | -93.5 ± 27.8 | -85.1 ± 46.1 |

Worse on 15/16 seeds. Mechanism confirmed on seed 7630: same analysis
count (803 vs 793) but only 27 distinct hosts vs 67 — pure proba has no
touch-decay/recency dynamics, so it fixates and starves coverage. AUC
0.67 does not convert to episode return without ordering dynamics.
- Verdict: snapshot risk scoring CLOSED (would need risk x recency
  dynamics — noted, not built). The program's best remains lancer_v2 at
  teacher parity; ordering edge must come from architecture (proposal 02),
  not snapshot classification.

## Experiment 5: risk x recency hybrids (2026-10-02, TESTED, gate failed)

- Implementation: `RiskRecencyPriority` (subclasses lancer, embeds
  `RiskPriority` as proba engine): (a) decay = proba x 0.5^fruitless;
  (b) bonus = full lancer + 2.0 x proba; (c) gate = lancer-ordered among
  proba >= 0.3, abstains to lancer values if none pass. Registry
  `hybrid_rx_decay/bonus/gate`. Fixed a real init-order trap found by
  tests (overridden `reset()` running mid-construction — inlined field
  init in both scorer `__init__`s). 4 unit tests.
- Pool, 5 policies x 8 seeds x 2 sets (80 cells), manifests
  `riskx-20261002.json` / `riskx-heldout-20261002.json`:

| policy | regr | held-out |
|---|---|---|
| round_robin | -93.5 ± 27.8 | -85.1 ± 46.1 |
| hybrid_lancer_v2 | -75.2 ± 33.6 | -91.2 ± 48.5 |
| hybrid_rx_decay | -203.4 ± 221.8 | -104.5 ± 45.8 |
| hybrid_rx_bonus | -186.2 ± 176.5 | -251.1 ± 214.1 |
| hybrid_rx_gate | -95.5 ± 23.8 | -164.9 ± 123.5 |

- Gate (beat lancer_v2 on held-out): FAILED by all three. Decay is the
  safest learned variant (uniformly ~-20, never catastrophic) but adds
  noise, not signal. Bonus inherits risk fixations (-715, -367). Gate
  does not abstain safely on hard seeds (-398, -318 at tau 0.3).
- Lesson: AUC 0.67 snapshot signal is too weak to survive the
  coverage/return trade at decision time, with or without dynamics.
  Only the attention distill — which learns the ordering function
  itself, supervised, with cross-slot comparison — reached parity.
  Proposal 01 closed: no further ordering-variant spend.

## Pros

- Smallest change that can beat the teacher: keeps everything the teacher
  does right, learns only the ordering it does arbitrarily.
- Learning is safely bounded: the scorer can never trigger a wrong
  remediation (rules gate all Remove/Restore).
- Each piece is ablatable (rules alone, +priority, +flags, +scorer).
- Directly validated by 3/3 competition winners + H-MARL Expert.

## Cons

- Ceiling: if optimal play needs strategy the rules forbid (e.g. proactive
  firewall play per UC, decoy play per lancer), the hybrid cannot reach it
  without rule changes — rules become the new bottleneck.
- The risk scorer trains on sparse, delayed labels; small-model discipline
  needed to avoid re-learning round-robin with extra steps.
- Draft is tracked (`blue_hybrid.py`) with parity, priority, and risk
  tests (`test_hybrid_parity.py`, `test_hybrid_priority.py`,
  `test_risk_priority.py`).

## Verdict

Done (was: BUILD FIRST). Parity proven, lancer v1/v2 and snapshot-risk
tested: best fixed result is lancer_v2 at teacher parity; adaptive snapshot
scoring killed by coverage collapse. No further constant-tuning or snapshot
scoring — ordering edge must come from architecture (proposal 02).
