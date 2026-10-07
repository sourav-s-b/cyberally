"""Validate the frozen local/remote replay cohort, preserving all failures."""
import argparse
import json
import math
from pathlib import Path


def compare(config, local, remote):
    cfg=json.loads(Path(config).read_text())
    local,remote=Path(local),Path(remote)
    snapshot=json.loads((remote/'replay.json').read_text())
    if snapshot!=cfg:
        raise ValueError('remote replay configuration drift')
    cells={}
    for job in cfg['jobs']:
        tag=job['tag']
        pair={}
        for site,root in [('local',local),('remote',remote)]:
            d=json.loads((root/(tag+'.json')).read_text())
            if (d['threads']!=job['threads'] or d['tolerance']!=job['tolerance']
                or d['model_sha256']!=cfg['input_hashes'][job['model']+'_actor.th']
                or d['scorer_sha256']!=cfg['input_hashes']['scorer.pkl']
                or d['cell']['seed']!=job['seed'] or d['cell']['ticks']!=399
                or not math.isfinite(d['cell']['return'])
                or d['requests']['requested_actions_count']!=1995):
                raise ValueError('replay contract drift: '+site+'/'+tag)
            pair[site]={'return':d['cell']['return'], 'requests_sha256':d['requests']['requested_actions_sha256']}
        pair['same_return']=pair['local']['return']==pair['remote']['return']
        pair['same_requests']=pair['local']['requests_sha256']==pair['remote']['requests_sha256']
        cells[tag]=pair
    stable={}
    for seed in sorted({j['seed'] for j in cfg['jobs']}):
        stable[str(seed)]={}
        for site in ('local','remote'):
            a,b=[cells[f'stable_{seed}_t{t}'][site] for t in (1,2)]
            stable[str(seed)][site+'_thread_invariant']=a==b
    return {'source_commit':cfg['source_commit'],'cells':cells,
            'stable_thread_checks':stable,
            'cross_machine_stable_requests_match':all(v['same_requests'] and v['same_return'] for k,v in cells.items() if k.startswith('stable_')),
            'interpretation':'two reused diagnostic episodes; ranking tolerance is a changed policy, not a defensive improvement or universal numerical guarantee; v2 still requires its own trained-policy preflight'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--remote',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=compare(a.config,a.local,a.remote)
    Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
