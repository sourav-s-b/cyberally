"""IsolationForest v2: train on Analyse-revealed CLEAN views, score held-out seeds.

Guardrails (per review):
- Train/test seed split decided UPFRONT; headline AUC uses test seeds only.
- Clean views pulled at multiple timesteps per episode (10/40/70) so green
  legitimate activity (SSH sessions, benign processes) shows up in training.
- No hardcoded subnets: own subnets carried from each episode's reset
  snapshot into host_to_vector via extract_subnets.
- Analyse/Monitor are sensing-only (cf. Monitor==Sleep evidence), so the
  probing itself doesn't perturb Red. Labels from TRUE state, eval only.

Expected runtime: ~12 episodes x ~140 joint steps, a few minutes.
"""

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # sibling blue modules
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "cage-challenge-4"))  # pristine simulator

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, DiscoveryFSRed
from CybORG.Simulator.Actions.AbstractActions import Analyse

from blue_obs_features import (host_to_vector, extract_subnets,
                               FEATURE_NAMES, VECTOR_LEN)

OWN = "blue_agent_0"
TARGETS = ([f"restricted_zone_a_subnet_user_host_{i}" for i in range(10)]
           + ["restricted_zone_a_subnet_server_host_0"])
TRAIN_SEEDS = list(range(8))          # 0..7
TEST_SEEDS = [100, 101, 102, 103]
SNAP_STEPS = [10, 40, 70]             # multi-timestep snapshots per episode


def new_env(seed):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=DiscoveryFSRed,
        steps=400,
    )
    return CybORG(scenario_generator=sg, seed=seed)


def true_label(env, hostname):
    for agent, sids in env.state.hosts[hostname].sessions.items():
        if "red" in agent and sids:
            return 1
    return 0


def analyse_view(cyborg, env, hostname):
    """Analyse a host; return (revealed_dict, true_label_at_resolve)."""
    a = Analyse(session=0, agent=OWN, hostname=hostname)
    cyborg.step(agent=OWN, action=a)   # IN_PROGRESS -- wait, don't re-issue
    r = cyborg.step(agent=OWN, action=a)  # resolves
    return dict(r.observation).get(hostname, {}), true_label(env, hostname)


def collect(seeds):
    """Returns (clean_vecs, comp_vecs, comp_labels_note). Clean = true-state
    clean at resolve time; compromised collected alongside for scoring."""
    clean, comp = [], []
    for seed in seeds:
        cyborg = new_env(seed)
        env = cyborg.environment_controller
        base = cyborg.reset(agent=OWN).observation
        targets = [h for h in TARGETS if h in base]
        subnets = {h: extract_subnets(base[h]) for h in targets}
        last = 0
        for snap in SNAP_STEPS:
            for _ in range(snap - last):
                env.step()
            last = snap
            for h in targets:
                revealed, label = analyse_view(cyborg, env, h)
                # world advances 2 joint steps per Analyse; account for it
                last += 2
                view = {"Sessions": [{"agent": OWN, "username": "ubuntu",
                                      "Type": "UNKNOWN"}]}
                view.update(revealed)
                v = host_to_vector(view, OWN, subnets.get(h))
                assert len(v) == VECTOR_LEN
                (clean if label == 0 else comp).append(v)
    return np.array(clean), np.array(comp)


Xtr_clean, Xtr_comp = collect(TRAIN_SEEDS)
print(f"train: clean={len(Xtr_clean)} comp={len(Xtr_comp)} "
      f"unique_clean={len(np.unique(Xtr_clean, axis=0))}")
print(f"train clean per-feature std: {Xtr_clean.std(axis=0).round(3)}")

clf = IsolationForest(random_state=0).fit(Xtr_clean)

Xte_clean, Xte_comp = collect(TEST_SEEDS)
print(f"test:  clean={len(Xte_clean)} comp={len(Xte_comp)}")

Xte = np.vstack([Xte_clean, Xte_comp])
yte = np.array([0] * len(Xte_clean) + [1] * len(Xte_comp))
ste = clf.decision_function(Xte)  # lower = more anomalous
print(f"TEST AUC = {roc_auc_score(yte, -ste):.3f} "
      f"(n_clean={len(Xte_clean)}, n_comp={len(Xte_comp)})")
print(f"mean score clean={ste[yte==0].mean():.3f} comp={ste[yte==1].mean():.3f}")
print("features:", FEATURE_NAMES)
print("sample clean vec:", Xte_clean[0].tolist())
print("sample comp  vec:", Xte_comp[0].tolist())
