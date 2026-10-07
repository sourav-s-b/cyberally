# Plan: improving the risk-as-training-signal lead

Status: plan only, 2026-10-05. Nothing below has been run. Branch
`blue/metric-repair-maxage` pushed and clean.

## What we have

Residual PPO actor (Lancer-initialised, 15 inputs = 14 Blue-visible host
rows + learned P(compromised)), guard off, central critic, explicit
generator sampling. Eval: **+4.50** vs Lancer on 8 dev seeds, CI
[-22.9, +31.9], worst seed -44. Pre-registered gate (+5, CI excluding
zero) NOT cleared. Status: promising training-method result, not a win.

## Verified premise (measured 2026-10-05, privileged diagnostic only)

Precision of the hosts the actor *chose* — the actual PPO population:

| seed | picks with proba>=0.5 | picks with proba<0.5 |
|---|---:|---:|
| 7629 | 61% compromised | 13% |
| 7630 | 36% | 7% |
| 7640 | 45% | 18% |
| 7701 | 31% | 15% |

Filtering concentrates signal ~3-4x, but the kept set is still majority
noise on 3 of 4 seeds. This bounds how much the filter can possibly buy
and explains +4.5 rather than more.

## Two honest caveats to carry into any write-up

1. **The filter biases the gradient.** It is not standard PPO over the
   full experience; it is a selected subset. Do not describe it as
   ordinary PPO. A selected-experience update is defensible only if
   described accurately.
2. **Thresholds were chosen after inspecting a collapse seed.** Treat the
   current +4.50 as exploratory, not dev-confirmed.

## Ordered steps (cheapest / most informative first)

### Step 1 — train longer (~25 min)
Same recipe, `--iters 10 --eps-per-iter 4`, filter 0.5. Only 4 iterations
have ever been run, the return curve was still improving, and updates are
now ~10x larger (KL 0.015-0.020 vs 0.001-0.003). Fairest single test of
"undertrained vs wrong".
Decision: if mean >= +5 and CI excludes zero -> Step 4. Else keep as the
new reference point.

### Step 2 — threshold from the measured precision (~15 min)
Train-time filter 0.65 instead of 0.5. The calibration table shows
0.6-0.8 is ~52% precise vs 0.4-0.6 at ~30%, so a higher cut should keep
cleaner examples. Note: this filter applies to TRAINING only; execution is
unaffected, so unlike the eval gate it cannot disable the policy.
Decision rule: pick the better of 0.5 / 0.65 on dev; report both.

### Step 3 — remove the bias instead of filtering (~40 min)
Two standard, unbiased replacements, one at a time so attribution stays
clean:
- **3a. Denser reward.** Training already supports shaping
  (`SHAPING_DEFAULTS` clear +3 / confirm +1 / vandalism -3, tested,
  never used). Training on shaped reward with the FULL buffer is standard
  PPO and directly attacks the measured cause (native reward nonzero on
  only 41% of steps). Evaluation stays native, as always.
- **3b. Risk in the critic, not the filter.** Add P(compromised) summaries
  to the central critic input. Variance reduction without selection bias —
  the correct fix for high advantage variance (paired SD 72-116), and it
  matches the MAPPO literature's finding that critic input quality is
  load-bearing.

### Step 4 — fresh-seed confirmation (only if dev clears +5)
8 seeds that have never been used, NOT the reserved block. If confirmed,
then and only then spend 7809-8200 on the frozen evaluation.

## What NOT to do
- Do not re-tune the shield margin (bounded at Lancer already).
- Do not revisit ordering / regime-routing (both measured, no headroom).
- Do not spend the reserved seed block before Step 4 passes.
- Do not report +4.50 as an RL gain in the deck.

## Cost
Step 1 ~25 min, Step 2 ~15 min, Step 3a ~40 min, Step 3b ~15 min (reuse
checkpoint-free training), Step 4 ~10 min. Full sequence ~1h45m.
Each step ends committed, with the dev gate stated in advance.