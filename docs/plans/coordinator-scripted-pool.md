# Detailed plan: MARL coordinator over scripted scenario pool

## 0. Goal and non-goals

**Goal:** give RL one honest, narrow chance to earn a gain: coordinate zone
agents across scripted attack regimes, with Lancer as fallback. Stop rules
at every phase; a documented limit is a complete result.

**Non-goals (explicitly out):** LLM attacker work (Red role); mid-episode
Red switching (needs Environment-owned simulator hook — recorded as
deferred); general residual-RL or investigation-order retraining (frozen
after paired -81.4 / -399.6); touching `cage-challenge-4/`; training on
seeds 7809–8200.

## 1. Standing facts this plan relies on

- Lancer is the best deployable ordering (32-seed screen,
  `blue/analysis/heuristic_screen.py --red-agent {discovery,finite}`);
  per-seed wins elsewhere = selection noise (expected max ≈ +47 ≈
  observed +46 over 128 draws).
- All deployable arms miss ~35% of detections identically; urgent rules are
  load-bearing (`no_restore_escalation` → mean ≈ -3570;
  `no_verify` → paired -24.4 on finite dev seeds).
- `mappo_guide2` frozen as failed; shield-0.5 ties Lancer on both regimes
  (paired +0.00, `deviation.py --hook shield`).
- Red is fixed at construction (`blue/core/wrapper.py:131,190`) — regimes
  vary per episode only.
- Remove/Restore legal only on CONFIRMED/VERIFY
  (`blue/core/masking.py:20-26`) — preemptive-Remove needs a Blue-owned
  mask change.
- Onset-aware metrics + 23 tests green
  (`blue/tests_blue/test_metric_semantics.py`); timeline in
  `docs/status/blue-timeline.md`.

## 2. Phase A — close the measurement record (no runs)

1. Re-run `test_metric_semantics.py` (sim venv) to confirm 23 green after
   all later edits.
2. Commit a one-paragraph measurement-status note in
   `docs/status/blue-session.md` mapping the four onset cases to test
   names + the timeline pointer.
3. Commit. Budget: minutes.

## 3. Phase B — mine the Lancer weakness (no training)

1. From `blue/results/matrix-a/cells.jsonl` episode records +
   `deviation_finite.json` traces, tabulate repeated missed-compromise
   clusters by zone/host-pattern/timing using onset-aware delay
   definitions (`blue/analysis/metrics.py:325-357`).
2. Required output: ONE specified weakness (location, onset-to-miss
   pattern, seeds it repeats on, which existing action could address it)
   **or** a written stop verdict.
3. Commit note either way. Budget: minutes (data on disk).

## 4. Phase C — versioned scripted pool (no training)

1. Write `docs/proposals/manifests/scenario-pool-v2.json` extending v1:
   regimes `{discovery, finite}` (+ `random`/`sleep` degenerate controls
   if useful), per-episode block schedules (mixed pools, alternating block
   orders — the switching proxy), seed lists, wrapper `foundation-v3`,
   flag versions, trace pointers.
2. Variants must target the Phase-B weakness only — no padding.
3. Commit. Budget: minutes.

## 5. Phase D — fixed emergency skill + headroom gate (no training)

1. Implement preemptive-Remove-on-suspicion as an `OrderedPolicy` flag
   (same pattern as `no_restore_escalation`/`no_verify` in
   `ordered.py:343-369`): defaults preserve behavior; parity test in
   `test_response_variants.py` style (flags-off == Lancer exactly,
   2 seeds).
2. Fixed router: Lancer-response vs emergency skill on unambiguous
   triggers first.
3. Eval via `heuristic_screen.py --red-agent {discovery,finite} --seeds
   <8 dev>` + skill arm, paired.
4. **Gate:** skill wins some regime by a clear margin AND a Blue-visible
   signal (compromise rate, delay trend, lockout rate) distinguishes the
   regimes. Fail → stop, report, no training. Budget: ~15 min on 4 shards.

## 6. Phase E — MAPPO coordinator (only if Phase D gates)

1. Reuse `mappo_guide.py`: `JointRecorder` (line 135),
   `ppo_central_update` (line 283), `run_team_episode` (line 216),
   `ShieldHook` (line 340); zone agents choose Lancer-response vs
   emergency skill with local obs (+ claimed-host message); central critic
   on joint Blue-visible state; Lancer fallback armed; explicit-generator
   sampling + `--temp-end` anneal (`mappo_guide.py:582-606`); dirty-tree
   manifest.
2. Train on mixed-regime dev-seed episodes only:
   `--iters 10 --eps-per-iter 6`, `blue/results/mappo_coord1/`, detached
   with live per-iter forced-split + entropy logging. Budget ~20 min.
3. New tests: skill-choice validity (choice indexes legal skill set),
   fallback parity (fallback-only == Lancer), save/reload + resume
   (existing patterns in `test_mappo_guide.py`).

## 7. Phase F — adaptation eval (smoke labels until frozen block)

1. Arms: Lancer, Lancer+shield-0.5, coordinator, strongest fixed skill —
   via `deviation.py --hook {greedy,shield} --shield-margin M` and
   `mappo_guide.py --mode eval` (`deviation.py:178-188`).
2. Splits: static-regime blocks (forgetting check) +
   alternating-regime blocks (switching check), paired held-out dev seeds;
   per-regime tables (return, compromise duration, corrected delays,
   intervention rate).
3. Gate (`docs/proposals/manager-gate-20261005.md`): paired mean ≥ +5,
   95% CI excludes zero → replication; lower bound > +5 after replication
   → "at least five better". Miss → keep fallback, publish limits.

## 8. Seed, eval, and commit discipline

- Dev seeds only (7629–7729) until Phase F frozen design; 7809–8200
  untouched (minus recorded consumed sub-ranges 7801–7808, 8001–8008,
  8201–8216).
- Small-n results labeled smoke/point-estimates; no advance/regress calls
  outside §4.2 rules.
- Every phase ends committed or explicitly deferred; `git diff --check`
  before each commit; results stay gitignored, manifests committed.

## 9. Risks

- Phase B finds no repeating weakness → plan ends at documentation
  (already the modal outcome twice).
- Preemptive-Remove FP cost exceeds containment gain → skill fails gate.
- Finite-regime variance (paired sd ~50–116) swallows +5 effects at
  8 seeds → wider seed spreads needed before any claim.
- Claimed-host messaging may prove unnecessary (zones already disjoint
  per seed) → drop it rather than gold-plate.

## 10. Done criteria

- Coordinator with measured +5 (gated) → proceed to frozen eval on
  granted block.
- Anything else → Lancer + shield stands as the defended result with
  documented limits.
