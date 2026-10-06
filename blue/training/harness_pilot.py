"""Frozen pilot evaluation: common episodes, all training seeds, no selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from blue.harness.policy import HarnessPolicy
from blue.policies.ordered import OrderedPolicy, LancerValues
from blue.training import mappo_guide as mg
from blue.training.harness_rl import load_hook, check_seeds


def evaluate(config, output, scorer):
    cfg = json.loads(Path(config).read_text())
    seeds = cfg['development_eval_seeds']
    check_seeds(seeds)
    if set(seeds) & set(cfg['train_episode_seeds']):
        raise ValueError('train/eval overlap')
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    cache_path = output / 'baselines.json'
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache['configuration_sha256'] != key:
            raise ValueError('baseline configuration drift')
    else:
        cache = {'configuration_sha256': key, 'cells': {}}
    for name in ('lancer', 'guard'):
        cells = cache['cells'].setdefault(name, {})
        for seed in seeds:
            if str(seed) not in cells:
                policy = (OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5)) if name == 'lancer'
                          else HarnessPolicy(max_age=cfg['max_age']))
                cells[str(seed)] = mg.run_team_episode(policy, seed, cfg['steps'], **mg.ENV_KW)
                cache_path.write_text(json.dumps(cache, indent=2))
                print(f'eval {name} seed={seed} return={cells[str(seed)]["return"]}', flush=True)
    rows = []
    for train_seed in cfg['training_rng_seeds']:
        for arm in cfg['arms']:
            tag = f'{arm}_s{train_seed}'
            path = output / tag / 'evaluation.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result['configuration_sha256'] != key:
                    raise ValueError('evaluation configuration drift')
            else:
                result = {'configuration_sha256': key, 'arm': arm, 'train_seed': train_seed, 'cells': {}}
            for seed in seeds:
                if str(seed) not in result['cells']:
                    hook, _ = load_hook(output / tag, scorer)
                    policy = HarnessPolicy(hook, max_age=cfg['max_age'])
                    result['cells'][str(seed)] = mg.run_team_episode(policy, seed, cfg['steps'], hook=hook, **mg.ENV_KW)
                    path.write_text(json.dumps(result, indent=2))
                    print(f'eval {tag} seed={seed} return={result["cells"][str(seed)]["return"]}', flush=True)
            for baseline in ('lancer', 'guard'):
                result['mean_minus_' + baseline] = float(np.mean([
                    result['cells'][str(s)]['return'] - cache['cells'][baseline][str(s)]['return'] for s in seeds]))
            rows.append(result)
            path.write_text(json.dumps(result, indent=2))
    pairs = []
    for seed in cfg['training_rng_seeds']:
        means = {r['arm']: r['mean_minus_lancer'] for r in rows if r['train_seed'] == seed}
        pairs.append({'train_seed': seed, 'risk_minus_zero': means['risk']-means['zero'],
                      'both_minus_risk': means['both']-means['risk']})
    report = {'configuration_sha256': key, 'runs': rows, 'matched_pairs': pairs,
              'interpretation': 'Technical pilot only; independent unit is training seed (n=2). Shared evaluation episodes are not independent model replicas. No success or significance claim.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--scorer', required=True)
    a = ap.parse_args()
    evaluate(a.config, a.output, a.scorer)
