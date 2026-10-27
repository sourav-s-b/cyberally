# Blue architecture long run

This is a development study of whether the earlier weak results came from the
small training budget or from the policy design. It keeps the Blue harness,
MAPPO-style shared policy and centralized critic, finite Red, risk scorer,
coverage guard, reward, and PPO settings fixed while varying the actor.

The nine arms are MLP widths 64/128/256/512, attention widths 128/256, GRU-128,
graph-128, and a graph-128 self-only adjacency control. The two larger models
are new candidates; the rest repeat previously screened designs with a stronger
budget. Five training RNG seeds × three architecture groups yield fifteen Kaggle
shards. Each arm runs 32 iterations × 4 fresh on-policy episodes per iteration:
45 models and 5,760 training episodes total. Training scenarios draw from the
71-seed development block 7730–7800; each arm receives the same episode-seed
schedule within a training RNG shard. The training RNG seeds are independent
replications, not the episode rows.

Including evaluation and repeated controls, this schedules 11,520 full episodes
overall (about 92 CPU-hours at the measured 28.8 seconds per episode; wall time
depends on shard concurrency).

Evaluation uses shared development seeds 8221–8252, which have appeared in
prior development work. Each checkpoint is tested both with deterministic
greedy ranking and sampled execution; Lancer, coverage guard, Sleep, masked
random, and a common untrained harness are recorded as controls. Evaluation
means are computed within each model and episode seed, then paired across the
five training RNG shards. This does not create a held-out or final performance
claim. Seeds 7809–8200 remain protected.

The frozen scorer, golden predictions, runtime lock, complete seed lists,
configuration, and source commit are verified by preflight before training.
The runner saves progress and checkpoints under `/kaggle/working`; do not place
these artifacts in Git. Each training RNG/architecture group is a separate
Kaggle CPU kernel so a failed group can be rerun independently. Each group has
three actors and about 768 full episodes. The earlier architecture screen
averaged 28.8 seconds per rollout and ranged from 16.5 to 44.1 seconds, giving
about 6.1–9.4 hours per job at those observed rates, plus setup. Kaggle documents
a 12-hour CPU session limit ([notebook documentation](https://www.kaggle.com/docs/notebooks)).

The models within each shard execute serially. All arm results and tails must be retained; no best-checkpoint
selection is permitted.

## Local preparation

From a clean checkout of the committed task branch:

```bash
python3 ops/kaggle/architecture/prepare.py \
  --artifact-dir /path/to/frozen/kaggle_main_artifacts \
  --upload-root /tmp/blue-architecture-longrun
```

Upload the private dataset directory first, then push each generated kernel
directory in `kernels/`. After all fifteen outputs are downloaded, aggregate them:

```bash
python -m blue.training.architecture_aggregate \
  --shards /path/to/downloaded-outputs/seed-* \
  --out /path/to/architecture-report.json
```

Check `study.json`, each `plan.json`, the complete per-model result cells,
checkpoint hashes, and the aggregate report before interpreting any ranking.
