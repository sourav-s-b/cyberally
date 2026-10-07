import json
import pytest
from blue.training.harness_replay_compare import compare


def fixture(tmp_path):
    cfg={'source_commit':'frozen','input_hashes':{'risk_s1_actor.th':'model','scorer.pkl':'scorer'},'jobs':[]}
    local=tmp_path/'local';remote=tmp_path/'remote';local.mkdir();remote.mkdir()
    for threads in (1,2):
        for name,tolerance in [('original',0.),('stable',1e-6)]:
            tag=f'{name}_8236_t{threads}'
            cfg['jobs'].append({'tag':tag,'model':'risk_s1','seed':8236,'threads':threads,'tolerance':tolerance})
            row={'threads':threads,'tolerance':tolerance,'model_sha256':'model','scorer_sha256':'scorer',
                 'cell':{'seed':8236,'ticks':399,'return':-152.},
                 'requests':{'requested_actions_count':1995,'requested_actions_sha256':'a'*64}}
            for root in (local,remote):(root/(tag+'.json')).write_text(json.dumps(row))
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg));(remote/'replay.json').write_text(json.dumps(cfg))
    return path,local,remote


def test_replay_comparison_requires_exact_requests_even_when_returns_match(tmp_path):
    config,local,remote=fixture(tmp_path)
    assert compare(config,local,remote)['cross_machine_stable_requests_match']
    path=remote/'stable_8236_t2.json';row=json.loads(path.read_text())
    row['requests']['requested_actions_sha256']='b'*64;path.write_text(json.dumps(row))
    result=compare(config,local,remote)
    assert result['cells']['stable_8236_t2']['same_return']
    assert not result['cross_machine_stable_requests_match']
    assert not result['stable_thread_checks']['8236']['remote_thread_invariant']


def test_replay_comparison_rejects_artifact_substitution(tmp_path):
    config,local,remote=fixture(tmp_path)
    path=remote/'original_8236_t1.json';row=json.loads(path.read_text())
    row['model_sha256']='different-model';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='contract drift'):compare(config,local,remote)
