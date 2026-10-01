# RED-01 scripted baseline

The adapter selects the native `DiscoveryFSRed` class with a strict versioned
configuration. It preserves the scenario's shared RNG, six Red agent names,
subnet ownership, native Green policy and full topology. Blue sleeps in this
standalone smoke factory. No simulator or Blue files are changed.

`configs/discovery-fs-red-v1.json` is a replayable policy configuration, not a
stored action sequence. Unknown versions, fields and policies are rejected.
The existing `EnterpriseScenarioGenerator(red_agent_class=...)` constructor is
the injection hook. Environment owns the future shared scenario factory; the
Blue training wrapper has not been connected to this adapter yet.

From this worktree's root in PowerShell, using an activated simulator environment:

```powershell
$env:PYTHONPATH = "$PWD/cage-challenge-4"
python -m red.smoke --output runs/red-smoke --seeds 7629 7630 --steps 40
python -m pytest -q red/tests cage-challenge-4/tests_blue
```

The existing local interpreter is
`C:/Projects/cyberally/.venv-train/Scripts/python.exe`; it can replace `python`
in these commands. This is a tested local Python 3.12 runtime, not an accepted
replacement for Environment's pending runtime specification.

Each invocation needs a new output directory. The manifest records policy/config
versions, seed list, horizon, source commit and hashes, runtime versions, native
Blue team return counted once, joint tick count and invalid executed Red actions.
`red-actions.jsonl` is a separate Red diagnostic channel with simulator-tick units.
It records controller executed actions, including Sleep during pending actions;
it does not infer requested actions, action success, pending targets or clearance.
Do not pass these Red diagnostics to Blue actors or anomaly detectors.

Tests compare the adapter with the unmodified native baseline on development
seeds 7629/7630, including all Blue-visible observations, Red executed actions and
native rewards. Fresh-instance and repeated-reset runs must match. Those smoke
seeds are not training or held-out evaluation suites. There are no measured
compromise/detection/service metrics or attacker-strength claims in this task.

Next: review the proposed injection/manifest seam with Environment, Blue and
Evaluation, then connect the reviewed factory to training/evaluation consumers.
Variants and strategy-pool sampling come after native baseline integration.
See `docs/coordination/red-strategy-factory.md` and `docs/status/red.md`.
