"""Is one episode reproducible at all?

The 4-iteration pilot recipe reproduced iterations 1-2 exactly and then
diverged at iteration 3. Seeding cannot explain a mid-run divergence, so
this isolates the layer: replay the SAME seed several times with everything
fixed and compare returns. Also replays across two fresh processes, because
in-process caching (dicts keyed by object id, cached scenario graphs) would
show up as a difference between the first and later calls.

Dev seeds only; the reserved block 7809-8200 is never used.
"""
import json
import sys

REPO = "/home/sourav/Projects/cyberally"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from blue.training import mappo_guide as mg  # noqa: E402

SEEDS = [7706, 7707]
STEPS = 400


def one_pass():
    from blue.policies.ordered import LancerValues, OrderedPolicy
    out = {}
    for s in SEEDS:
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        r = mg.run_team_episode(pol, s, STEPS, **mg.ENV_KW)
        out[str(s)] = r["return"]
    return out


if __name__ == "__main__":
    passes = json.loads(sys.argv[1]) if len(sys.argv) > 1 else 3
    results = [one_pass() for _ in range(passes)]
    print(json.dumps(results))