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
