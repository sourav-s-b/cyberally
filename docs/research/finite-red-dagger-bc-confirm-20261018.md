# Fresh finite-Red confirmation: BC and DAgger

**Date:** 2026-10-18
**Branch/source:** `blue/finite-red-check`, frozen source commit `2f44db117d98791103fd99329005f7d56f84ffe9`
**Purpose:** Evaluate the already-trained BC and DAgger checkpoints on a new shared finite-Red block. No model was retrained or selected on these episodes.
**Result status:** Complete; all 448 planned episode rows and frozen artifacts validated.

## Protocol

The five BC and five DAgger checkpoints from `finite_red_dagger_bc_pilot_20261018` were held fixed. Each was evaluated on the same 32 fresh episode seeds, `8253–8284`, with native return, 400 ticks and the age-80 guard. Lancer, the age-80 rule, Sleep and masked-random controls used the same seeds. The scorer, frozen source files, checkpoint hashes, manifests, evaluation seed lists and confirmation source hash passed the readiness check.

The primary uncertainty unit is the five existing model-training seeds. For each model, the 32 paired episode differences were averaged, then a two-sided 95% t interval was calculated across those five means. This checks whether the earlier result repeats across a fresh episode block for these trained models. It does not measure variation from retraining on new data.

## Results

| Policy | Mean native return | p10 | Minimum | Lancer agreement | Coverage |
|---|---:|---:|---:|---:|---:|
| BC | -149.75 | -211 | -450 | 0.392 | 1.000 |
| DAgger | -148.43 | -201 | -288 | 0.395 | 1.000 |
| Lancer | -180.00 | -250.4 | -492 | — | 1.000 |
| Age-80 guard | -169.25 | -253.8 | -497 | — | 1.000 |
| Sleep | -4123.56 | -5509.5 | -6802 | — | 0.000 |
| Masked random | -780.56 | -1157.9 | -1815 | — | 1.000 |

| Paired comparison, policy minus baseline | Mean difference | 95% interval across five model means |
|---|---:|---:|
| BC − Lancer | +30.25 | [+24.65, +35.85] |
| BC − age-80 guard | +19.50 | [+13.90, +25.10] |
| DAgger − Lancer | +31.57 | [+24.94, +38.20] |
| DAgger − age-80 guard | +20.82 | [+14.19, +27.45] |
| DAgger − BC | +1.32 | [−4.13, +6.76] |

## Interpretation and next step

Both trained investigation actors repeat their positive paired difference against Lancer and the age-80 guard on this fresh episode block. This is stronger evidence for these fixed checkpoints than the original single-block pilot. The DAgger-versus-BC interval still overlaps zero, so DAgger has not shown a return advantage over simpler BC. Mean agreement is also nearly the same. BC remains the simpler default investigator; keep DAgger as a comparable candidate where its slightly better pooled p10/min merits separate risk analysis.

This does not establish generalization to other Red strategies or new model-training data. The five training replicates all use the same frozen training allocation, and the interval is conditional on this one fresh evaluation block. Preserve the result as a confirmation of fixed models, not as a broad claim that learned Blue beats Lancer generally.

For the prepared follow-ups, do not spend a run on OEC unless reduced expert-label use is a stated objective: the return comparison gives no reason to expect it to beat BC. Posterior conditioning remains contingent on new evidence that Blue-visible histories can distinguish useful Red regimes. The next useful work is the hierarchy C0 trace/data audit to identify one concrete, observable recovery, verification or coverage failure. If no such failure and adequate training data exist, stop at the audit.

## Reproduction

- Full paired episode rows and controls: `blue/results/finite_red_dagger_bc_confirm_20261018/evaluation/`.
- Frozen identity and exact seed list: `blue/results/finite_red_dagger_bc_confirm_20261018/frozen_plan.json`.
- Aggregated results: `blue/results/finite_red_dagger_bc_confirm_20261018/analysis.json`.
- Integrity/progress check: `python scripts/blue_followup_readiness.py --scorer /home/sourav/projects/cyberally/blue/results/gpt_harness/scorer.pkl`.
