# Seed ledger update: LLM Red evaluation blocks

Date: 2026-10-07. Author: Blue role. Branch `opencode/blue-diagnosis`.

Recorded here because `docs/status/blue.md` and `docs/status/blue-session.md`
already disagree about the ledger, and `blue/training/seed_audit.py` is now
the authoritative check. Run it before claiming any block is free:

```bash
.venv-train/bin/python -m blue.training.seed_audit --check LO HI
.venv-train/bin/python -m blue.training.seed_audit --free-blocks 10
```

## Newly consumed

| block | consumed by | note |
|---|---|---|
| 8221-8252 | risk-input ablation, 6 runs (3 training seeds x 2 arms) | previously verified free, now spent |
| 7629-7631 | LLM Red baseline reproduction + first RL evaluation | already spent before this; Red used them too |
| 8221-8228 | GPT `gpt-blue` harness pilot evaluation | overlaps the row above |

The audit now reports 8221-8252 as contaminated, sourced to
`blue/training/ab_freeze_config.py` — which is correct: the frozen config
declared that block and the ablation actually ran on it.

## Newly in use

| block | purpose | status |
|---|---|---|
| 7641-7664 | LLM Red evaluation of Lancer vs guarded actor, 24 seeds | in progress 2026-10-07 |

Verified free before use (`seed_audit --check 7641 7664`). Sits between the
7629-7729 development block and nothing reserved.

## Still untouched

- **7809-8200 final evaluation block.** `llmred_eval.py` raises on any seed
  in this range, so the block is protected in code, not by convention.
- 7801-7808 (spent for decisions)
- 8301-8400 (finite Red)
- 8501+ (reserved)

## Free blocks currently available

From `seed_audit --free-blocks 10`, restricted to blocks that do **not**
overlap any declared-reserved zone:

| block | size |
|---|---:|
| 7600-7628 | 29 |
| 7641-7700 | 60 (minus 7641-7664 now in use) |
| 7730-7800 | 71 |

`8253-9250` and everything above is numerically free but overlaps the
`8301-8400` finite-Red block and the `8501+` reservation. **Treat it as
unavailable until the ledger conflict is resolved in writing.** The standing
dispute — `blue.md` says `8201+` reserved, `blue-session.md` claims
`8200-8420` free, and 8201-8216 are in fact consumed — is still unresolved
and is not something Blue should settle unilaterally.

## Unresolved ledger conflict (unchanged, still open)

1. `blue.md`: "8201+ reserved"
2. `blue-session.md`: "8200-8420 free"
3. Observed: 8201-8216 consumed by earlier BC/IQL manifests

`seed_audit.py` reports observed consumption and applies declared reservations
separately, so the conflict is visible rather than silently resolved. It needs
a decision from Blue, Evaluation and Environment together.