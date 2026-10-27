"""Frozen cohort/shard validation and training-seed-level final analysis."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate(config):
    if config['arms'] != ['zero', 'risk', 'both']:
        raise ValueError('unexpected arms')
    groups = [config[k] for k in ('training_rng_seeds', 'train_episode_seeds', 'development_eval_seeds')]
    if any(not g or len(g) != len(set(g)) for g in groups):
        raise ValueError('empty/duplicate seed lists')
    if any(7809 <= s <= 8200 for g in groups for s in g):
        raise ValueError('protected final seeds')
    if set(groups[1]) & set(groups[2]):
        raise ValueError('train/eval overlap')
    if config.get('env_kw') is not None:
        from blue.training.mappo_guide import ENV_KW
        if json.dumps(config['env_kw'], sort_keys=True) != json.dumps(ENV_KW, sort_keys=True):
            raise ValueError('environment configuration drift')


def shard_config(config, seed):
    validate(config)
    if seed not in config['training_rng_seeds']:
        raise ValueError('unknown training shard')
    return {**config, 'training_rng_seeds': [seed],
            'cohort_training_rng_seeds': config['training_rng_seeds'],
            'cohort_sha256': config_hash(config)}


def mean_interval(values):
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('need complete finite training replicas')
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {'n_training_seeds': len(values), 'per_training_seed': values.tolist(),
            'mean': mean, 'ci95': [mean-radius, mean+radius], 'df': len(values)-1}


def aggregate(config, folders):
    validate(config)
    expected = config['training_rng_seeds']; eval_seeds = list(map(str, config['development_eval_seeds']))
    rows, baselines, seen = {}, None, set()
    for folder in map(Path, folders):
        local = json.loads((folder/'pilot.json').read_text())
        seed = local['training_rng_seeds'][0]
        if seed in seen or local != shard_config(config, seed):
            raise ValueError('duplicate shard or cohort/configuration drift')
        seen.add(seed)
        key = config_hash(local)
        report = json.loads((folder/'report.json').read_text())
        cache = json.loads((folder/'baselines.json').read_text())
        if report['configuration_sha256'] != key or cache['configuration_sha256'] != key:
            raise ValueError('report/cache configuration drift')
        current = {b: {s: cache['cells'][b][s]['return'] for s in eval_seeds}
                   for b in config.get('baselines', ['lancer', 'guard'])}
        if any(set(cache['cells'][b]) != set(eval_seeds) for b in current):
            raise ValueError('unmatched baseline episodes')
        if baselines is not None and current != baselines:
            raise ValueError('baseline mismatch between shards')
        baselines = current
        if len(report['runs']) != 3:
            raise ValueError('incomplete model cohort')
        for row in report['runs']:
            arm = row['arm']; tag = (seed, arm)
            if arm not in config['arms'] or row['train_seed'] != seed or tag in rows or set(row['cells']) != set(eval_seeds):
                raise ValueError('unmatched/duplicate model or episode cells')
            manifest = json.loads((folder/f'{arm}_s{seed}'/'manifest.json').read_text())
            if manifest['status'] != 'complete' or len(manifest['history']) != config['iters'] or manifest['scorer_sha256'] != config['input_hashes']['scorer.pkl'] or manifest['source_commit'] != config['source_commit']:
                raise ValueError('incomplete/foreign checkpoint')
            actor_path = folder/f'{arm}_s{seed}'/'actor.th'
            if hashlib.sha256(actor_path.read_bytes()).hexdigest() != manifest['actor_sha256']:
                raise ValueError('checkpoint weight hash mismatch')
            if config.get('runtime_versions') and manifest['runtime'] != config['runtime_versions']:
                raise ValueError('checkpoint runtime drift')
            for key in ('iters','eps_per_iter','steps','hidden','lr','bonus','temp','max_age'):
                if manifest['config'][key] != config[key]:
                    raise ValueError('checkpoint training configuration drift')
            if manifest['config']['seed'] != seed or manifest['config']['ml_inputs'] != arm or manifest['config']['train_seeds'] != config['train_episode_seeds']:
                raise ValueError('checkpoint seed/arm drift')
            rows[tag] = {s: row['cells'][s]['return'] for s in eval_seeds}
    if seen != set(expected):
        raise ValueError('missing training shards; no complete-cohort claim')
    comparisons = {}
    for arm in config['arms']:
        comparisons[arm] = {b: mean_interval([np.mean([rows[(seed, arm)][s]-baselines[b][s]
                                                     for s in eval_seeds]) for seed in expected])
                            for b in ('lancer', 'guard')}
    ml = {f'{a}_minus_{b}': mean_interval([np.mean([rows[(seed,a)][s]-rows[(seed,b)][s]
                                                  for s in eval_seeds]) for seed in expected])
          for a,b in (('risk','zero'), ('both','risk'))}
    primary = comparisons['both']
    return {'cohort_sha256': config_hash(config), 'comparisons': comparisons,
            'ml_comparisons': ml, 'all_episode_returns': {f'{s}_{a}': v for (s,a),v in rows.items()},
            'baselines': baselines,
            'primary_both_beats_both_controls': all(primary[b]['ci95'][0] > 0 for b in ('lancer','guard')),
            'interpretation': 'Replication unit is training RNG seed. Intervals conditional on this reused development evaluation suite. Primary arm both, both controls required; other arm/ML intervals descriptive. Every model and tail loss retained.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--folders', nargs='+', required=True);p.add_argument('--out', required=True)
    a = p.parse_args(); result=aggregate(json.loads(Path(a.config).read_text()), a.folders)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
