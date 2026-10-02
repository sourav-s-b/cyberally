Critic-only warmup phase for the vendored PPO learner.

Blue role, 2026-10-01. `third_party/` is gitignored, so this diff is recorded
here instead of in the vendor tree. The rebuild steps in
docs/status/blue-session.md ("Fresh-machine (WSL) rebuild") must apply it,
otherwise `--warmup-steps` silently does nothing (the driver still accepts
the flag, the learner just ignores the freeze).

Why it exists: the literature review in
docs/research/blue-sparse-reward-literature.md identified value divergence at
the onset of on-policy fine-tuning as the mechanism behind the BC degradation
(WSRL, ICLR 2025, arXiv:2412.07762). The published mitigation is to fit the
critic to on-policy data from the FROZEN policy before any actor update.

Apply against EPyMARL cbc38c09588064eab978501d0f12c2cf58fa7fc2:

    git apply environment/patches/ppo-warmup.patch

or apply the three hunks by hand. They are deliberately additive:

1. `__init__`: three attributes + a `_warmup_active()` helper, all read via
   `getattr` so upstream configs without these keys keep working.
2. `train()`: wrap the agent optimiser step in `if not self._warmup_active(t_env)`.
   The critic, target updates and logging are untouched, so the critic still
   trains during warmup.
3. `grad_norm` is initialised to a zero tensor because the clip call that
   assigned it is now inside the conditional.

Measured effect: none. The 4000-step warmup scored -455 vs -393 without it,
with behavioural KL(pi_bc || pi_ft) ~0.001 nats. Default is OFF
(`warmup_steps=0`); the flags exist so the ablation is reproducible, not
because warmup is recommended.
