# First Review deck: cross-branch accuracy audit (8 October 2026)

Edited artifact: root `First_Review.pptx` (14 slides, SHA-256
`ae28d85f78b0dd34eddd83972f0523ee5b0d3058f21633954bc147e7288b01ec`).
It is intentionally Git-ignored. The untouched, ignored backup is
`runs/first_review/First_Review.original.pptx` (SHA-256
`e5370eb5ceff840daedb65bf5578b091f3daa438a2de41577ab8416ce07af2dd`).

## Sources inspected

| Source | Exact revision or artifact | Relevance |
|---|---|---|
| `gpt-blue` | worktree HEAD `bda8da97b434397295ec6cf58fb04b970f69aabd`; main32 experiment source `6942d584f15231a7f19add0a510a04f7ff3ad229` | Blue harness, trained-arm protocol, control policies |
| Main32 Blue report | uncommitted worktree `docs/status/blue-gpt-main32-results.md`, SHA-256 `c11050b091fcdb25d8e6a01833e0a5ca0ca8ecbe80bac93f44de368fde1dfff1` | Five-seed/32-iteration results and uncertainty |
| Main32 aggregate | ignored `blue/results/gpt_harness/main32_status/aggregate.json`, SHA-256 `4c3a2e212791ff2733afc605f3278f949979062d851ec8553ce3fb4dc04e8557` | Full evaluation cells and frozen cohort |
| Blue detector audit | uncommitted `docs/status/blue-gpt-pilot-audit.md`, SHA-256 `943f52261e50eeb6495e4d4403d765427e65ca9367417e9152b8051c9084d4b9` | Eight-episode HGB/logistic comparison, runtime and caveats |
| Blue legacy | current `blue/metric-repair-maxage` HEAD `243a6fd`; `opencode/blue-diagnosis` `b9ad250` | Historical failures and direct inspection of investigation-only hook |
| Red | local `origin/red/blue-lancer-comparison` `7634cb3053c3aae61aaf0850ecb9aa75916d9b00`, `red/evidence/lancer-development-20261007.md` and adjacent JSON | Independently documented *different* three-arm LLM/CAGE-Red comparison |
| Integrated status | local `origin/main` `01140fe517c056444e4e95430b034b35bd8aadf5` | Integration status, not proof of a completed retraining loop |

`git ls-remote origin` failed because of local SSH/proxy permissions. Remote
freshness could not be checked, so `origin/*` above means the locally cached
tracking refs. The current `gpt-blue` report and aggregate are local artifacts,
not committed evidence; their hashes are recorded to make this review precise.

## Slide-by-slide findings

| Slide(s) | Review and disposition |
|---|---|
| 1-2 | Kept original title, date and problem framing. The date is the planned first-review date, not an experiment timestamp. |
| 3-4 | Objectives explicitly distinguish simulator evidence from future container, SIEM, response-latency and retraining goals. No sub-second containment claim is made. |
| 5 | Replaced stale "MAPPO pilot started" with completed short/32-iteration comparisons and their unresolved benefit. Kept zeroth-review-minutes caveat. |
| 6-8 | Updated Blue architecture and inserted an investigation pipeline. The `HarnessPolicy` delegates remediation/verification to existing rules before the learned sweep; the hook is restricted to `Analyse` investigation target choice. Simulator validity masks, Blue evidence restrictions and privileged evaluation labels are separate. A2C exists as an implemented alternative in `blue/harness/learning.py`; the main32 result uses MAPPO/PPO. |
| 9 | Corrected Blue comparative runtime to Python 3.11.16 / NumPy 1.26.4 / PyTorch 2.2.0+cpu. Red local Ollama remains a different runtime. EPyMARL is a historical implementation source, not the current main32 harness. |
| 10 | Preserved all four teammate-reported Red model cells exactly. The cited `BLUE_RED_TEST_RESULTS.md` and `runs/lancer-small-models-20261007/RESULTS.html` are absent from local branches. User says the tests were personally observed on another laptop; slide and notes mark them preliminary/source-pending. The independently documented Red comparison at `7634cb3` is a different protocol and is not substituted into this table. Removed the obsolete two-seed Blue-vs-8B table. |
| 11 | Added the controlled Blue result. Means: Lancer -79.91, guard -78.22, zero -74.59, risk -81.21, both -74.71, Sleep -2267.28, random -245.00. Primary predeclared both-vs-guard +3.50625, paired training-seed t95 [-3.14891, 10.16141]; both-vs-Lancer +5.19375 [-1.46141, 11.84891]. Intervals cross zero and the requirement to beat both controls was not met. Five independent training seeds and 32 *reused* shared development episodes are not 160 independent training runs. Final 7809-8200 seeds were untouched. |
| 11 | The detector result is HGB macro PR-AUC 0.72794 versus logistic 0.63107 over eight reused development episodes, one fitted model. Better prediction does not establish a better defense; the risk-only RL arm performed worse than the zero-input arm in main32 and its uncertainty interval includes zero. Novelty's standalone performance is weak, and it is not described as a compromise probability. |
| 12-13 | Removed the arbitrary 80% completion figure. Work still pending: failure-mechanism trace, Red-suite integration, fresh-rollout retraining, held-out test, SIEM/container deployment and response-latency measurement. Dates remain TBD. |
| 14 | Kept the original reference format; corrected Blue runtime references and added experiment-source details to speaker notes. |

## Claim limits for the presentation

- The deck reports development evidence, not final held-out or production
  performance. The four-model Red table remains provisional until its original
  source outputs are available and matched to the cells.
- More training alone has not been shown to solve Blue failure modes. Main32
  does not prove a learned policy beats the guard or Lancer. Detector ranking
  and defensive return are separate outcomes.
- Cross-branch runs use different attack suites, seed sets, and sometimes
  runtimes. Their absolute scores should not be pooled or compared as one
  benchmark.

Validation: the edited PPTX opened and rendered in PowerPoint, contains 14
slides and 14 speaker-note pages, and passes ZIP integrity. All 14 slides
were exported to ignored PNG previews; the added and heavily edited slides
were visually inspected for clipping and layout. The preserved four-model
Red numeric cells were compared against the original deck. Only the root
PPTX and its Windows Zone.Identifier are ignored by the `.gitignore` change.

## Handoff and review checklist

- Working branch: `blue/metric-repair-maxage` at `243a6fd`; this task changes
  only `.gitignore`, this audit and the ignored local presentation. No other
  branch/worktree was edited and no merge or push was performed.
- Checks run: `git diff --check`, `git check-ignore -v`, PowerPoint open/render,
  OOXML ZIP integrity, slide/note counts, stale-text scan, and cell-for-cell
  original/edited Red-table comparison. No training or simulator test was
  needed for a presentation edit.
- Review before distributing: obtain the teammate's four-model source
  outputs, verify all per-run cells and local timing, and replace the
  source-pending note only if those checks pass. Confirm the committee date
  and the zeroth-review minutes when available.
- Dependent branches: none need a code change. The results cite `gpt-blue`
  and Red commits by exact revision; their data and protocol must stay
  distinct if later integrated. A PR should include this audit and the
  `.gitignore` change, while the deck remains a local/shareable artifact.
