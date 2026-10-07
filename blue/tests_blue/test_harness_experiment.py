import json
import hashlib

import pytest

from blue.training.harness_experiment import aggregate, config_hash, mean_interval, shard_config, validate


def config():
    return {'arms':['zero','risk','both'], 'training_rng_seeds':list(range(5)),
            'train_episode_seeds':[7706], 'development_eval_seeds':[8221,8222],
            'iters':32, 'eps_per_iter':4, 'steps':400, 'hidden':64, 'lr':.0003,
            'bonus':.25, 'temp':.5, 'max_age':80, 'source_commit':'frozen',
            'input_hashes':{'scorer.pkl':'model'}}


def test_shard_partition_and_protected_suite():
    c=config()
    for seed in range(5):
        s=shard_config(c,seed)
        assert s['training_rng_seeds']==[seed]
        assert s['cohort_sha256']==config_hash(c)
    with pytest.raises(ValueError):shard_config(c,5)
    c['development_eval_seeds']=[7900]
    with pytest.raises(ValueError,match='protected'):validate(c)


def test_complete_cohort_uses_five_training_units_and_rejects_missing_or_drift(tmp_path):
    c=config(); folders=[]
    for seed in range(5):
        p=tmp_path/str(seed);p.mkdir();folders.append(p)
        local=shard_config(c,seed);key=config_hash(local)
        (p/'pilot.json').write_text(json.dumps(local))
        cells={str(s):{'return':-10.} for s in c['development_eval_seeds']}
        (p/'baselines.json').write_text(json.dumps({'configuration_sha256':key,'cells':{'lancer':cells,'guard':cells}}))
        runs=[]
        for arm in c['arms']:
            d=p/f'{arm}_s{seed}';d.mkdir()
            (d/'actor.th').write_bytes(b'frozen-weights')
            mc={k:c[k] for k in ('iters','eps_per_iter','steps','hidden','lr','bonus','temp','max_age')}
            mc.update(seed=seed,ml_inputs=arm,train_seeds=c['train_episode_seeds'])
            (d/'manifest.json').write_text(json.dumps({'status':'complete','history':[{}]*32,'scorer_sha256':'model','source_commit':'frozen','config':mc,'actor_sha256':hashlib.sha256(b'frozen-weights').hexdigest()}))
            runs.append({'arm':arm,'train_seed':seed,'cells':{s:{'return':v['return']+seed+1} for s,v in cells.items()}})
        (p/'report.json').write_text(json.dumps({'configuration_sha256':key,'runs':runs}))
    r=aggregate(c,folders)
    assert r['comparisons']['both']['guard']['n_training_seeds']==5
    assert r['comparisons']['both']['guard']['per_training_seed']==[1,2,3,4,5]
    assert len(r['all_episode_returns'])==15
    with pytest.raises(ValueError,match='missing'):aggregate(c,folders[:-1])
    p=folders[-1]/'baselines.json'; cache=json.loads(p.read_text());cache['cells']['guard']['8221']['return']=-100
    p.write_text(json.dumps(cache))
    with pytest.raises(ValueError,match='baseline mismatch'):aggregate(c,folders)
