# Proposal 01: Hybrid rules + learned scan priority

- Status: drafted — parity proven; lancer v1 TESTED at -63.2 ± 26.0
  (+30.2 vs teacher, 6/8 seeds; needs held-out replication)
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

Remaining, not yet run: (1) lancer-style priority + punch-style
file-density>0.9 flag + UC-style persistent malicious-event flags, same
seeds; (2) tiny risk scorer on Blue-visible 17-feat vectors. Target: beat
-93.5, approach lancer's constant-size -71 band on equal footing.

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
- Draft is uncommitted and has no tests yet.

## Verdict

BUILD FIRST. Concrete next task: parity proof, then priority + flags,
>= 8-seed native evals throughout.
