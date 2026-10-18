# Hierarchical specialist C0 audit: finite-Red development traces

**Date:** 2026-10-18
**Branch:** `blue/finite-red-check`
**Purpose:** Decide whether existing traces justify selecting and training a second Blue specialist.

## Decision

**Do not select or train a second specialist from these traces.** They show a useful verification diagnostic: after a verification `Analyse` completes, the Blue belief is `CONFIRMED` in about 23–25% of recorded verifications, `VERIFY` in about 38%, and `UNKNOWN` in about 37–38%. This is a repeated unresolved *observation*, not evidence that the verification rule is wrong. The traces do not record alternative eligible actions, missed opportunities, or a valid clean-host outcome. A successful `Remove` is not proof that the host is clean.

Coverage is not an evident second-specialist target in this block: every model investigated every host in all 24 episodes. The configured age-80 mechanism did not guarantee a strict maximum age (observed per-model maxima were 109–135), but no return or harm attribution in these traces shows that this overshoot is a policy failure. The existing rule remains the appropriate control pending a better-targeted test.

## Evidence and scope

The audit read only Blue-visible actor traces and non-privileged reports from:

`blue/results/finite_red_candidate_audit/{s0,s1,s2,s3,s4,s5}/`

Each of six fixed models has 24 finite-Red episodes on the same seeds, 7665–7688 (144 model-episode trajectories total). These seeds were already used for candidate-choice diagnostics and intervention selection. They are development evidence only: the repeated model runs are not independent episode blocks and must not be used to validate a new specialist or tune a held-out policy. The separately named `.privileged.jsonl` files were not read for this audit.

The frozen reports identify source commit `2f44db117d98791103fd99329005f7d56f84ffe9`, native evaluation return, 400 ticks, and the shared age-80 rule. Each model report covers 24/24 episodes and 1,791 host-episode observations; none has a never-investigated host.

## Decision-branch evidence

Counts below are per episode, averaged within each model; ranges span the six models. The same finite-Red seed block is reused across models, so these counts describe coverage of the traces, not independent statistical evidence.

| Branch | Recorded actions / opportunities | Blue-visible cue and outcome | Missing evidence for a specialist decision |
|---|---:|---|---|
| Investigation hook | 503–574 / episode | Target feature vectors, branch and chosen `Analyse` are recorded. Ranked candidate details are recorded only for the configured audit agent and hook decisions. | No error label or counterfactual return for each choice; no full candidate set for other agents/branches. |
| Age/coverage sweep | 185–207 `Analyse` / episode | Feature vector and chosen host are recorded; every host is eventually investigated in these episodes. Max observed host age is 109–135. | No eligible alternatives, intervention opportunity record, or causal attribution showing that overshoot reduced return. |
| Recovery | 65–77 decisions / episode: 26–29 `Remove`, 39–48 `Restore` | Feature vector, selected host/action, pending action and completion are present. Completed actions report simulator result `TRUE` and move Blue belief to `VERIFY`; `Remove` takes 3 ticks and `Restore` 5. Across a model's 24 episodes, 8–14 recovery requests remain unresolved at the horizon. | No eligible alternatives or missed-action labels; simulator success does not establish host cleanliness; no action-level counterfactual or reliable recovery target. |
| Verification | 103–125 `Analyse` / episode | Feature vector and selected host are recorded. Completed analyses report result `TRUE`, take 2 ticks, and leave belief `CONFIRMED` in 23–25%, `VERIFY` in about 38%, and `UNKNOWN` in 37–38%. Across a model's 24 episodes, 2–9 verification requests remain pending at the horizon. | No eligible alternatives, criterion for when another analysis is preferable, or label showing that unresolved belief caused a defense failure. |
| Pending wait | 1,067–1,083 `Sleep` / episode | Pending host/action and completion are recorded. | These are harness-enforced waits, not specialist choices. |

The branch labels, chosen action, host, timing, feature vector, pending state, and subsequent completion are present in the JSONL. The artifact does **not** persist action masks/candidate alternatives for recovery, verification, or age-sweep decisions; nor does it log a manager proposal separately from the executed action. Completion can be associated with a request by agent/host/action and time, but that is an observed transition, not evidence that another choice would have been better.

## Follow-up gate

Keep the current rule and manager fixed. Before choosing a second specialist, a new *development-only* collector should record, for each proposed decision:

1. the full Blue-visible observation and feature version;
2. the eligible action set and why each action is eligible;
3. the manager's proposal, the executed action, any override, and the override reason;
4. request and completion ticks, duration, simulator result, next Blue-visible observation, reward, and terminal/truncation status;
5. a separate privileged evaluation label only when needed to audit outcomes, never as policy input or a training target unless that target is explicitly part of a separately reviewed experiment.

First freeze the development seed block and a testable observable failure criterion. If the collector finds no repeated failure with an actionable choice and trustworthy target, stop C0 without training. If it does, select exactly one specialist and test it alone with the manager and other rules frozen before testing composition. Do not use seeds 8253–8284 or 8401–8432 for this diagnosis or selection.

No simulator run, training, held-out evaluation, policy change, or contract change was made for this audit. Validation was a read-only pass over the six frozen reports and 144 actor traces; the privileged companion traces were excluded.
