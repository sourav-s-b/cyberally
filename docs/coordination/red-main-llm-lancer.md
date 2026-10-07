# Proposal: default LLM Red entry point and Lancer comparison

- Status: local draft, not an accepted shared API or merged change.
- Producer: Red, branch `red/blue-lancer-comparison`.
- Consumers: Blue training/evaluation, Environment factories, Evaluation metrics.
- Exact Blue dependency: `3096f068f43aa461ac3fe5e659c2e24e6aa86188`
  (`blue/metric-repair-maxage`, fetched from GitHub).
- Prior Red prototype dependency: `6d160cb74f01364228cfeda56eb0fbb78b8f3be4`
  plus local source hashes in each report; uncommitted prototype copied explicitly.

## Proposed changes and defaults

Add `red_agent_class=None` to Blue's `CC4MARLEnv` constructor. With None,
existing named attacker behavior stays unchanged. With a class, the caller
supplies `red_agent` as a provenance label and the native scenario constructs
fresh Red instances using its existing explicit constructor inspection.
No agent IDs, topology, observations, masks, reward or episode semantics change.

Add `CC4LLMEnv` as the main Red-enabled Blue environment. Default Red uses a
bounded local LLM strategy planner and persistent observation-only tactical
executor. No DiscoveryFSRed fallback. Config/version defaults and run commands
are documented in `red/README.md`. The global request budget does not reset
with episodes; per-agent planning quotas do. Failed inference retains a valid
prior plan. Pending native actions produce Sleep without inference.

`red.main` defaults to new Red vs local `hybrid_lancer_v2`; `--compare` is the
only normal entry point that adds DiscoveryFSRed and tactical-only controls.

## Diagnostics, tests and rollout

Draft report `red-planner-development-v1` records source/model/prompt hashes,
config, actual development seeds, native return once, execution validity,
wall time and evaluation-only active compromised/privileged host counts/time.
Label definitions are local diagnostic proposals, not accepted shared metrics.
Actor input is Red-local knowledge and enabled parameters only. No privileged
session labels enter prompts or Blue observations. Reports distinguish model
inference from mocks and from the fixed spread executor ablation.

Tests: bad plan/hidden focus/timeout/budget execution, pending action and reset,
lost-access memory, custom factory injection, Lancer integration and paired
output/episode checks. Preserve existing Blue foundation/Lancer/Red-variant tests.

Rollout: review producer and additive Blue injection seam, then explicitly
select CC4LLMEnv in Blue runner/training configs, then evaluate fresh Blue
rollouts before any retraining. Preserve upstream classes and benchmark access;
removing an attacker from active configuration does not delete framework code.
No remote deployment, long training, PR or teammate message is implied.
