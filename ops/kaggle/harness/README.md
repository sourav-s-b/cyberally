This runner is separate from `blue-long-train`. The pilot uses training RNG
seeds 0/1 and zero/risk/both ML controls, eight iterations of four fresh
400-step episodes per model. No reward shaping or probability filtering.

Attach a private frozen dataset containing scorer.pkl, golden.npz and
pilot.json. Code is fetched from GitHub at the exact source_commit in pilot.json. Verify locally before submission. Resume at completed PPO iteration
boundaries: never replay interrupted on-policy buffers. To recover across Kaggle
sessions, copy prior output run directories into /kaggle/working/pilot first.
The script does not automatically retrieve prior kernel output.

All six final models need common development evaluation against Lancer and the
coverage-only policy. These are technical pilot results, not a main experiment.
The main experiment remains conditional on pilot validation. Protected final
seeds 7809–8200 are excluded. Artifact and runtime compatibility must pass first.

Audit update: source and virtualenv now live under `/tmp`, so only pilot artifacts
are exported from `/kaggle/working`. The completed pilot accidentally exported
the full runtime, forcing over 100 pages of output listing before its report.
This change applies only to future submissions; the completed run is preserved.

`requirements-legacy-candidate.txt` records an isolated Python 3.11 compatibility
candidate, not a replacement for the frozen pilot pins. NumPy 1.26.4 cannot
deserialize the pilot detector; rebuild it from unchanged training data and
validate golden predictions before using that candidate remotely. The generic
Gym startup warning appears even under NumPy 1.26.4 and alone proves no failure.

For a later main experiment, plan five independent shards, each retaining all
three ML arms for ONE training RNG seed (32 iterations, four episodes each).
Each shard needs its own frozen input/configuration and resumable output;
aggregate all five matched contrasts only after every shard finishes. Preserve
both Lancer and guard controls and native return. Source/model/runtime/evaluation
suite and success criteria must be frozen before submission. No main-experiment
configuration is launch-ready and no larger kernel has been submitted.

The future runner's `runtime_profile=legacy-candidate` selects Python 3.11 and
`requirements-legacy-lock.txt`, the complete installed compatibility-test stack.
The shorter candidate requirements file documents intentional top-level pins.
`diagnostics=true` enables requested-action hashes and fixed-state input checks.
Neither option retroactively changes the completed pilot.

Main32 preparation is frozen in `main32-plan.json`; actual uploaded `pilot.json`
pins the published source commit. Five kernels share one private dataset; each
bootstrap sets GPT_BLUE_SHARD_SEED to one authorized cohort seed. Shared master
configuration is hashed into each shard. The evaluator reports its actual n=1
locally; aggregate all FIVE final shard outputs with harness_experiment for n=5
training-seed intervals. Primary BOTH must clear both Lancer and guard; all other
comparisons descriptive. Include Sleep/random controls and preserve all tails.
