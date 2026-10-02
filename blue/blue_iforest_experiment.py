"""IsolationForest experiment: clean-reset training set -> score Analyse-revealed hosts.

Methodology notes (all verified against live env):
- SimulationController.step() CLEARS per-agent observation sets each step, so
  there is no server-side accumulation; deltas must be merged client-side.
- Even Blue's visibility ceiling (get_agent_state) shows NO red sessions on
  truly-compromised hosts. Monitor deltas are empty ({success, action}).
- Detection data appears ONLY via Analyse(hostname): 2-step action
  (IN_PROGRESS then TRUE + host detail incl. Files/Processes/Connections).
- Labels come from TRUE state (env.state.hosts[].sessions), used for
  evaluation ONLY, never fed to the model.
"""

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # sibling blue modules
sys.path.insert(0, os.path.join(os.path.dirname(_HERE),
                                "cage-challenge-4"))  # pristine simulator

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, DiscoveryFSRed
from CybORG.Simulator.Actions.AbstractActions import Analyse

sys.path.insert(0, ".")
from blue_obs_features import obs_to_matrix, host_to_vector, host_ground_truth

OWN = "blue_agent_0"


def make_cyborg(steps=300):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=DiscoveryFSRed,
        steps=steps,
    )
    return CybORG(scenario_generator=sg, seed=7629)


def true_label(env, hostname):
    for agent, sids in env.state.hosts[hostname].sessions.items():
        if "red" in agent and sids:
            return 1
    return 0


# ---- 1. clean set: resets across 25 seeds ----
clean_vecs = []
per_seed_n = []
for seed in range(25):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=DiscoveryFSRed,
        steps=300,
    )
    c = CybORG(scenario_generator=sg, seed=seed)
    res = c.reset(agent=OWN)
    _, mat = obs_to_matrix(res.observation, OWN)
    per_seed_n.append(len(mat))
    clean_vecs.extend(mat)
X_clean = np.array(clean_vecs)
uniq = np.unique(X_clean, axis=0)
print(f"clean: {len(X_clean)} vectors from 25 seeds, hosts/seed={per_seed_n[:5]}...")
print(f"clean: {len(uniq)} UNIQUE vectors (variance check)")
print(f"clean: per-feature std = {X_clean.std(axis=0).round(3)}")

# ---- 2. fit on clean only, default contamination ----
clf = IsolationForest(random_state=0).fit(X_clean)

# ---- 3. compromised: Analyse-reveal RZA hosts at step 100 and 200 ----
cyborg = make_cyborg()
env = cyborg.environment_controller
cyborg.reset(agent=OWN)
targets = [f"restricted_zone_a_subnet_user_host_{i}" for i in range(10)]
targets.append("restricted_zone_a_subnet_server_host_0")

results = {}
for snap, n_steps in (("step100", 100), ("step200", 100)):
    for _ in range(n_steps):
        env.step()
    vecs, labels = [], []
    for h in targets:
        a = Analyse(session=0, agent=OWN, hostname=h)
        cyborg.step(agent=OWN, action=a)  # IN_PROGRESS
        r = cyborg.step(agent=OWN, action=a)  # resolves
        revealed = dict(r.observation).get(h, {})
        # merge revealed detail onto baseline shape host_to_vector expects
        host_view = {"Sessions": [{"agent": OWN, "username": "ubuntu",
                                   "Type": "UNKNOWN"}]}
        host_view.update(revealed)
        vecs.append(host_to_vector(host_view, OWN))
        labels.append(true_label(env, h))
    X = np.array(vecs)
    scores = clf.decision_function(X)  # lower = more anomalous
    auc = roc_auc_score(labels, -scores)
    results[snap] = (np.array(labels), scores)
    print(f"{snap}: n={len(labels)} compromised={sum(labels)} "
          f"AUC={auc:.3f} mean_score_clean-side={scores[np.array(labels)==0].mean():.3f} "
          f"mean_score_comp={scores[np.array(labels)==1].mean():.3f}")

# score distributions for gut check
for snap, (labels, scores) in results.items():
    print(f"{snap} scores | clean: {np.round(sorted(scores[labels==0]),3)}")
    print(f"{snap} scores | comp : {np.round(sorted(scores[labels==1]),3)}")
