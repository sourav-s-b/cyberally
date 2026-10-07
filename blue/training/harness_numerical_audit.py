"""Bounded, read-only checkpoint replay with explicit near-tie intervention.

This creates a different diagnostic policy when tolerance > 0. It never
rewrites the frozen model or substitutes its results into the old evaluation.
"""
import argparse
import json
from pathlib import Path

import torch

from blue.harness.numerics import stable_pick
from blue.harness.policy import HarnessActorHook, HarnessPolicy
from blue.harness.scoring import sha256
from blue.training import mappo_guide as mg
from blue.training.harness_rl import check_seeds, load_hook


class AuditHook(HarnessActorHook):
    def __init__(self, hook, tolerance):
        super().__init__(hook.actor, hook.scorer, hook.bonus, hook.ml_inputs)
        self.tolerance = tolerance
        self.rows = []
        self.per_agent = {}

    def mark_episode(self):
        self.rows = []
        self.per_agent = {}

    def close_episode(self, rewards):
        pass

    def __call__(self, env, agent, cands, scored):
        with torch.no_grad():
            residual = torch.tanh(self.actor(torch.from_numpy(self.features(env, agent, cands))))
        values = [(s+self.bonus*float(residual[i]), h) for i, (s, h) in enumerate(scored)]
        pick = stable_pick(values, self.tolerance)
        top = sorted(values, reverse=True)
        self.rows.append({'tick':env._tick, 'agent':agent, 'host':pick,
                          'margin':top[0][0]-top[1][0] if len(top)>1 else None,
                          'near_ties':sum(top[0][0]-s <= self.tolerance for s,h in values)})
        self.per_agent[agent] = self.per_agent.get(agent, 0)+1
        return pick


def replay(model_dir, scorer, seed, threads, tolerance, out):
    check_seeds([seed])
    torch.set_num_threads(threads)
    out = Path(out)
    if out.exists():
        raise FileExistsError('preserve existing diagnostic')
    actor_path = Path(model_dir)/'actor.th'
    digest = sha256(actor_path)
    original, manifest = load_hook(model_dir, scorer)
    hook = AuditHook(original, tolerance)
    policy = HarnessPolicy(hook, max_age=manifest['config']['max_age'], record_requests=True)
    cell = mg.run_team_episode(policy, seed, manifest['config']['steps'], hook=hook,
                               recorder=hook, **mg.ENV_KW)
    if sha256(actor_path) != digest:
        raise RuntimeError('frozen checkpoint changed')
    result = {'model_sha256':digest, 'scorer_sha256':hook.scorer.sha256,
              'threads':threads, 'tolerance':tolerance, 'cell':cell,
              'requests':policy.request_digest.result(), 'decisions':hook.rows,
              'interpretation':'local reused diagnostic; tolerance changes policy; no remote reproduction or improvement claim'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='decisions'}), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--scorer',default='blue/results/gpt_harness/legacy_candidate/scorer.pkl')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--tolerance',type=float,default=0.)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    replay(a.model_dir,a.scorer,a.seed,a.threads,a.tolerance,a.out)
