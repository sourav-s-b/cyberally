"""Fixed skill for the Phase-B weakness: agent-4 suspicion priority.

Mining showed blue_agent_4's zones (admin/office/public_access) miss ~58%
of detections vs ~28% elsewhere, invariant across regimes/guards/scorers.
This hook keeps Lancer behavior everywhere except agent_4's sweep, which
serves never-analysed/stalest hosts first (skipping hosts picked in the
agent's last COOLDOWN sweeps). Deterministic, sim-venv safe, no learning.
"""

COOLDOWN = 3
TARGET_AGENT = "blue_agent_4"


class Agent4SuspicionHook:
    """Sweep hook: Lancer argmax except agent_4 suspicion order."""

    def __init__(self, cooldown=COOLDOWN):
        self.cooldown = int(cooldown)
        self._recent = {}

    def reset(self):
        self._recent.clear()

    def __call__(self, env, agent, cands, scored):
        from blue.policies.ordered import argmax_pick
        if agent != TARGET_AGENT:
            return argmax_pick([(s, h) for s, h in scored])
        tracker = env.trackers[agent]
        recent = self._recent.setdefault(agent, [])
        fresh = [h for h in cands if h not in recent]
        pool = fresh or list(cands)
        pool.sort(key=lambda h: (
            tracker.last_analysis.get(h) is not None,
            tracker.last_analysis.get(h) or 0))
        pick = pool[0]
        recent.append(pick)
        del recent[:-self.cooldown]
        return pick
