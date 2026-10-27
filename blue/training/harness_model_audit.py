"""Inspect frozen trained checkpoints on reused diagnostic episodes, without updates."""
import argparse
import json
from pathlib import Path

from blue.harness.diagnostics import input_sensitivity
from blue.harness.policy import HarnessPolicy
from blue.training import mappo_guide as mg
from blue.training.harness_rl import check_seeds, load_hook


class AuditHook:
    def __init__(self, hook):
        self.hook = hook
        self.rows = []

    def reset(self):
        self.hook.reset()
        self.rows = []

    def observe(self, env, agent):
        self.hook.observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        import torch as th
        pick = self.hook(env, agent, cands, scored)
        F = th.from_numpy(self.hook.features(env, agent, cands))
        base = th.tensor([s for s, _ in scored], dtype=th.float32)
        sensitivity = input_sensitivity(self.hook.actor, F, base, self.hook.bonus)
        self.rows.append({'tick': env._tick, 'agent': agent, 'picked': pick,
                          **sensitivity})
        return pick


def audit_model(model_dir, scorer, seeds, steps, out):
    check_seeds(seeds)
    result = {'claim': 'Fixed checkpoint input interventions on reused diagnostic episodes; no reward advantage estimate.',
              'model_dir': str(model_dir), 'episodes': {}}
    for seed in seeds:
        hook, manifest = load_hook(model_dir, scorer)
        audit = AuditHook(hook)
        policy = HarnessPolicy(audit, max_age=manifest['config']['max_age'], record_requests=True)
        episode = mg.run_team_episode(policy, seed, steps, hook=audit, **mg.ENV_KW)
        episode.update(policy.request_digest.result())
        episode['sweep_decisions'] = len(audit.rows)
        episode['ml_top_index_changes'] = sum(r['ml_top_index_changed'] for r in audit.rows)
        episode['novelty_top_index_changes'] = sum(r['novelty_top_index_changed'] for r in audit.rows)
        episode['ml_logit_delta_max'] = max((r['ml_logit_delta_max'] for r in audit.rows), default=0.)
        episode['novelty_logit_delta_max'] = max((r['novelty_logit_delta_max'] for r in audit.rows), default=0.)
        episode['decisions'] = audit.rows
        result['episodes'][str(seed)] = episode
        result['actor_sha256'] = manifest['actor_sha256']
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + '\n')
        print(f"seed={seed} return={episode['return']} sweeps={len(audit.rows)} "
              f"ML_index_changes={episode['ml_top_index_changes']} "
              f"novelty_index_changes={episode['novelty_top_index_changes']}", flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--scorer', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[8226, 8228])
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    audit_model(a.model_dir, a.scorer, a.seeds, a.steps, a.out)
