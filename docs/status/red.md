# Red handoff

## LLM main attacker and Blue Lancer integration, 2026-10-07

- Working branch: `red/blue-lancer-comparison`, isolated worktree
  `C:/Projects/cyberally/runs/red-blue-test`. Exact Blue dependency fetched from
  GitHub: `3096f068f43aa461ac3fe5e659c2e24e6aa86188`
  (`blue/metric-repair-maxage`). Root dirty Blue checkout preserved.
- Prior Red source: `red/llm-adapter` based on
  `6d160cb74f01364228cfeda56eb0fbb78b8f3be4`, with local prototype files copied
  explicitly. User authorized publication of this tested integration on the
  task branch on 2026-10-07; no merge into main is authorized or performed.
- Delivered: observation-only LLM strategy planning, persistent native host
  knowledge, validated tactical action execution, observed-decoy exclusion,
  per-agent/episode planning quotas, pending handling and full memory reset.
  Failed planning retains the prior plan and tactical execution; no
  DiscoveryFSRed fallback. Upstream generic knowledge transitions are reused
  under existing attribution; simulator sources remain unchanged.
- Main entry: `python -m red.main` defaults to LLM Red vs `hybrid_lancer_v2`;
  `--compare` explicitly adds DiscoveryFSRed and the fixed-spread tactical
  ablation. Lancer is the local no-decoy variant, not published-Lancer parity.
  `red.blue_env.CC4LLMEnv` is the tested default-LLM Blue environment entry;
  existing training registries/configs still need explicit selection. No
  retraining run is claimed.
- Additive Blue diff: optional `red_agent_class` injection in the wrapper.
  Existing named-agent defaults/observations/actions/rewards remain unchanged.
  Draft proposal `docs/coordination/red-main-llm-lancer.md`; affected-role
  review required before merging, not for authorized local tests.
- Validation: 41 Red tests passed in 21.02 s after the decoy fix. Earlier
  selected regression command: 73 passed in 63.34 s (37 Red plus 36 Blue
  foundation/Red-variant/Lancer tests). Three new Blue-integration tests also
  passed independently in 5.27 s and are included in the later 41 Red total.
  Python compilation and `git diff --check` passed.
- Longer live testing exposed upstream PIDSelective selector's incompatible
  priority signature. First live trace stopped at tick 118 and is incomplete:
  `runs/lancer-live-v1-20261007/`. Red-local replacement uses discovered session
  ports and Red-observed decoy ports; no privileged host-truth policy input.
  A regression test covers priority compatibility and decoy exclusion.
- Full paired live rerun completed: `runs/lancer-live-v2-20261007/`.
  Native horizon 400; existing development seeds 7629/7630/7631; unchanged
  full topology/native Green; fixed local Lancer v2 Blue. Two plans per agent,
  20 actionable turns between plans, 36 global request attempts, local
  `llama3.1:8b`. Reserved/final seed blocks are untouched.
- All nine episodes completed 399 native ticks, zero invalid Red actions.
  Native Blue returns by seeds 7629/7630/7631: LLM -676/-645/-719;
  DiscoveryFSRed -52/-54/-59; fixed-spread tactical-only -134/-116/-90.
  Means: -680 / -55 / -113.33 respectively; lower favors Red. New Red caused
  more mission damage on all three development seeds against this fixed Blue.
  This does not prove universal superiority or an optimal attacker.
- Tradeoffs: mean privileged host-time 3480 LLM vs 4823.67 scripted and
  4008.33 tactical; mean compromised host-time 11006 / 11734 / 11088.33.
  Existing Red retained more privileged host-time. Mean episode wall time
  66.23 s LLM / 19.70 s scripted / 29.75 s tactical; measurements include
  inference and background-load variation, not a controlled latency benchmark.
- Live inference: 35 requests, 32 accepted plans, three invalid-focus plans
  rejected while execution continued; no timeout/budget Sleep failures.
  Model selected spread throughout, so goal-switching effectiveness is not
  demonstrated. Soft focus differs from the fixed-spread/no-focus tactical
  control. Planning latency median 3.563 s, maximum 14.203 s; 46,901 input and
  595 output tokens. Both source hashes and trace hash verified after completion.
  `red/evidence/lancer-development-20261007.json` is the small report manifest;
  `red/evidence/lancer-development-20261007.md` gives the human-readable result.
- PR checklist: source hashes/model digest/config/seeds saved in report;
  logs/tools/weights ignored; native licensing retained; no shared schema
  declared accepted; branch push authorized by the user. No PR, teammate
  message or review request sent.
- Next: review integration seam, then select CC4LLMEnv in the intended Blue runner
  before any fresh-rollout retraining. Long training and full incident metrics
  have not been run. No universal strongest-attacker claim is made.

Inherited historical status below is superseded by this local handoff.

> Shared blockers recorded by Blue on 2026-09-30; owner to confirm or correct.
> See [`../current-state.md`](../current-state.md).
>
> - **`main` is unblocked.** BLUE-01 wrapper fixes are merged; `main` and
>   `blue/foundation` are both at `f8a9deb`. Branch from `main` normally.
> - No pinned runtime exists (Python 3.10 agreed but not installed; current venv
>   is 3.12 with no torch). Draft the adapter and fixtures meanwhile.
> - Measured baseline for calibrating any strategy: at seed 7629 under
>   masked-random Blue, true-state Red reached 13 user-level and 21 root-level
>   sessions by step 100, and 17/27 by step 200. A `strategy_id` entry must be a
>   replayable policy/config with seed, version and preconditions — a stored
>   `action_sequence` is a diagnostic trace and may be invalid once defenses or
>   topology change.

- Owner: teammate to assign
- First branch: `red/strategy-adapter`
- Status: not started
- Completed: built-in scripted Red is available in CybORG; no team adapter yet.
- Validation: no new Red experiments run.
- Blockers: ENV-01 runtime and agreed scenario injection seam.
- Dependency commits/PRs: none yet.
- Contract changes: propose strategy factory/config and replayable pool entries.
- Next: RED-01 scripted adapter/config; then pinned multi-attacker variants.
- Artifacts: none; no LLM integration or external attack execution configured.
