# Plan: close the metric-evidence gap, then write the final report

Branch `blue/metric-repair-maxage`. Answers the one open item in the GPT
review of 2026-10-05: the delay/max-age numbers were cited without showing
the calculation or proving the old `max_age` bug is gone.

## Already verified (read-only check, 2026-10-05)

- Production path is onset-aware **by construction**:
  `recorder.post_step` (recorder.py:66-110) emits `COMPROMISE_ONSET` /
  `COMPROMISE_CLOSE` from the privileged set, `DETECTION` on the first
  observable `CONFIRMED`, and `INVESTIGATION_COMPLETE` only when the
  tracker stamp actually changes; `metrics.detection_delay`
  (metrics.py:339-357) returns the first detection at/after onset inside
  the episode, else censors with a reason.
- The old defect is fenced: `ordering.py:118-124` withholds
  `coverage / max_age / det_delay_median / n_undetected` by default;
  `--emit-invalid-metrics` only reproduces them for audit.
- On real data the old bug is gone: 137 distinct
  `max_age_during_episode.max` values across the 256 matrix cells, range
  90-397; the constant 398 appears nowhere.

## Step 1 - make the calculation visible (no new claims)

1. Worked example in the metric docs: one real episode record with onset
   tick, analysis tick, detection tick, computed delay, censoring reason —
   recomputable by hand.
2. Add `test_max_age_varies_on_real_sim` to
   `blue/tests_blue/test_recorder_live.py`: two real-sim episodes give
   different `max_age_during_episode.max`, and no value equals the
   episode-end tick (the old bug's signature).
3. Run metric + recorder-live suites in both venvs; report counts.

## Step 2 - final report (`docs/status/blue-results-report.md`)

For a reviewer who has not seen the logs:

1. What was built - MAPPO-guide (shared residual actor over local host
   rows, central-V critic over joint Blue-visible context, guard off,
   explicit-generator sampling, resume, reloadable checkpoints), heuristic
   backbone Lancer + margin-0.5 fallback shield.
2. Every experiment with numbers - 8-arm guard matrix; residual PPO pilots
   1-2; Phase A/B/C attribution; margin sweep; 32-seed heuristic screen;
   detection-headroom mining; response-rule ablations; agent-4 suspicion
   skill (8-seed +23.75 -> 32-seed +0.44, CI crossing zero); mode-vs-regime
   matrix. Each row: tried / result / verdict.
3. What RL did and did not do - implemented and verified, never beat the
   heuristic; two named causes with evidence (native reward sparse - only
   41% of steps nonzero; no allocation trade-off - coverage 1.000 under
   every policy despite 53% forced lockout).
4. Metric semantics - onset-aware definitions, four censoring rules, proof
   the constant-398 bug is gone.
5. Limitations, narrowly stated - ablations show removing Restore /
   verification is harmful; that does NOT prove retained rules are optimal.
   Nothing shows every Blue decision was tested.
6. Untested / next - reward shaping exists but unused; capacity-constrained
   allocation is the only untried lever, with the explicit caveat that the
   constraint is synthetic (Blue-imposed), so any result is a coordination
   claim, not a threat-model claim.
7. Reproduction - exact commands, seed lists, commit hashes.

## Step 3 - decision point

After the report: (a) capacity + shaping experiment (~4h, synthetic
caveat), or (b) stop and hand over.

## Discipline

Docs + one test only; no training; no change to any production metric;
`git diff --check` before each commit; dev seeds only.