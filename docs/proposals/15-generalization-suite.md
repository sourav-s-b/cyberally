# Proposal 15: Generalization suite (red/behavior variants)

- Status: tested (phase 1: red variety; rate/duration variants pending
  Environment — see `docs/coordination/blue-red-variants.md`)
- Date: 2026-10-02

## Description

Base-scenario points are saturated (teacher ≈ published winner band) and
every published agent collapses on behavior variants. This suite makes
generalization the scored dimension: fixed protocol, fresh seeds
8001-8008 (never used for selection), 400 steps, native reward, plus
diagnostics (detection latency, remediation precision, Restore rate —
to be added to `blue_compare.py` output next).

## Motivation / evidence

Kiely et al., AAAI 2025, Table 2: IncreasedPhishing / StealthyRed /
AggressiveRed reorder or break all top agents (e.g. lancer -118 -> -871
stealthy; cybermonic -193 -> -996). Our program has never evaluated off
the base red. Phase 1 uses only in-tree reds (no sim changes).

## Experiment 1: red variety (2026-10-02, TESTED)

- Wrapper `red_agent=` flag (`discovery` default = all prior results;
  `finite` / `verbose` / `random` / `sleep`), recorded in `get_env_info`
  and manifests. Pool `--red-agent`. Tests
  `tests_blue/test_red_variants.py` (3 tests).
- 5 reds x (sleep, round_robin, hybrid_lancer_v2) x 8 seeds, manifests
  `red-<red>-20261002.json`:

| red | sleep | round_robin | hybrid_lancer_v2 |
|---|---|---|---|
| discovery | -2265 ± 733 | -58.8 ± 14.3 | -67.6 ± 22.2 |
| finite | -4208 ± 545 | -139.1 ± 49.0 | -124.0 ± 19.5 |
| verbose | -4208 ± 545 | -139.1 ± 49.0 | -124.0 ± 19.5 |
| random | -261 ± 362 | -131.8 ± 148.8 | -118.9 ± 135.7 |
| sleep | 0 | 0 | 0 |

- Findings: FiniteStateRed is much harsher than Discovery (-139 vs -59);
  verbose ≡ finite bit-for-bit (it only adds terminal printing — expected,
  and a nice determinism cross-check); random is mild with many zero
  episodes; sleep-red gives exactly 0 (sanity: no adversary, no penalties;
  also validates the flag actually swaps the adversary). lancer_v2 ≈
  teacher on every red (±15, inside noise) — no ordering edge off-base
  either; the hybrid's regression win does not transfer across reds.
- Verdict: suite stands as the standing generalization benchmark. Next:
  green-rate variants (pending Environment plumb-through), duration
  overrides (pending), then re-run all policies.

## Pros

- First off-base measurement in the program; already differentiates reds
  (sleep 0 vs finite -4208 floor).
- Mostly Blue-side; only rates/durations need Environment.

## Cons

- Red-swap ≠ published variants (phishing/stealth/aggressive are
  rate/duration changes, not agent swaps) — this is coverage, not the
  paper's suite yet.
- Fresh-seed set 8001-8008 now consumed as suite seeds (by design).

## Verdict

Keep as standing suite. Phase 1 complete; phases for rates/durations are
blocked on the coordination proposal, not on Blue code.
