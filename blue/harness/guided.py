"""Guide-policy roll-in hook for online RL curricula."""
from __future__ import annotations

from blue.policies.ordered import argmax_pick


class GuidePrefixHook:
    """Run a frozen ordered teacher until a simulator-tick handoff.

    The wrapped learner is called only after ``prefix_ticks``. Consequently,
    only learner-controlled choices can enter its policy-gradient buffer.
    Rewards are not changed: the episode recorder still receives every native
    simulator reward and computes returns from each learned decision onward.
    """

    def __init__(self, learner, prefix_ticks: int):
        if isinstance(prefix_ticks, bool) or int(prefix_ticks) != prefix_ticks or prefix_ticks < 0:
            raise ValueError("guide prefix must be a nonnegative integer")
        self.learner = learner
        self.prefix_ticks = int(prefix_ticks)
        self.reset_counts()

    def reset_counts(self):
        self.guide_decisions = 0
        self.learner_decisions = 0

    def reset(self):
        reset = getattr(self.learner, "reset", None)
        if callable(reset):
            reset()
        self.reset_counts()

    def observe(self, env, agent):
        observe = getattr(self.learner, "observe", None)
        if callable(observe):
            observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        if int(env._tick) < self.prefix_ticks:
            self.guide_decisions += 1
            return argmax_pick(scored)
        self.learner_decisions += 1
        return self.learner(env, agent, cands, scored)
