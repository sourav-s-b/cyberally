# Running the long training unattended on Kaggle

Why: the simulator stepping is CPU-bound and single-threaded (~20-25 s per
400-step episode), so the useful resource is wall-clock time on a machine
that stays on without you.

## What runs where

- **Code**: public GitHub repo, branch `blue/metric-repair-maxage`, pinned to
  the commit in `run.py` (`COMMIT`). Single source of truth; re-push the
  kernel to pick up new commits.
- **Artifacts**: attached Kaggle dataset `souravsreekumar02/blue-cyborg-code`
  holds the two files that are deliberately NOT in git (`scorer_mlp.npz`,
  `risk_model_v2.pkl` - trained model initialisations).
- **Runner**: `blue.training.remote_train` - checkpoints every iteration,
  append-only `metrics.jsonl`, `--resume` after a hard kill, plateau stop,
  and a final FULL-seed FULL-horizon eval that alone decides the gate (a
  cheap mid-run subset can never declare a pass).
  An actor snapshot is kept at every eval (`snapshots/iter_NNNN/`) and the
  final report scores BOTH the best subset checkpoint and the last actor, so
  a lucky stopping point cannot hide an unlucky one.

## Commands

```bash
# one-time: upload the two artifact files as a dataset
kaggle datasets create -p <dir-with-blue-results> --dir-mode zip

# push a new kernel version (run.sh is Python: Kaggle script kernels
# execute the code_file as Python, not bash)
kaggle kernels push -p ops/kaggle

# watch
kaggle kernels status souravsreekumar02/blue-long-train
```

## Gotchas hit while setting this up (2026-10-06)

1. `id` in kernel-metadata.json must be `owner/slug`, not a bare slug, or the
   push fails with a misleading `Invalid slug`.
2. `kernel_type: script` runs the code_file **as Python**; a bash script dies
   with `SyntaxError: invalid syntax`.
3. A script kernel uploads **only** the code_file. `/kaggle/src` contained
   nothing else, so an `import blue` failed. Code must arrive via git clone
   or an attached dataset.
4. `numpy<2` is required (gym 0.26.2); Kaggle's image ships numpy 2 and many
   preinstalled packages that will complain in pip output. Those warnings are
   expected and harmless for this job.
5. `--min-gain` has to be larger than the eval noise. The 4-seed subset
   swung by ~40 between evals, so the original `--min-gain 1.0` counted
   noise as progress and the plateau stop never fired.
6. Logs are only downloadable through the CLI after a run finishes
   (`kaggle kernels logs` returns nothing while it is running). The web UI
   shows the live log. The runner therefore prints per-episode lines plus a
   heartbeat (`--log-every`), because ~25 s per episode and ~12 min between
   evals look exactly like a hang otherwise.
