"""Full-episode replay of main32 losses; labels remain in separate trace files.

Same-seed worlds cease to be matched after policy divergence. A trace locates
associations and the first differing request, not a causal action advantage.
"""
import argparse
import json
from pathlib import Path

from blue.harness.policy import HarnessPolicy
from blue.training.harness_rl import load_hook,check_seeds
from blue.training.harness_eval import episode


def analyse(paths):
    logs={k:[json.loads(l) for l in Path(p).read_text().splitlines()] for k,p in paths.items()}
    result={'claim':'diagnostic associations only; same-seed worlds may diverge after requests differ',
            'request_divergences':{},'policies':{}}
    reference=logs['guard']
    for name,rows in logs.items():
        actions={};completions={};positive={};forced={}
        for r in rows:
            for a,rec in r['agents'].items():
                key=(rec['action'],rec['host']);actions[key]=actions.get(key,0)+1
                request=rec.get('request') or {};branch=request.get('branch','unknown');forced[branch]=forced.get(branch,0)+1
                c=rec['completion']
                if c:
                    key=c['action'];completions[key]=completions.get(key,0)+1
                    if c['belief_after']=='CONFIRMED':positive.setdefault(c['host'],r['tick'])
        truth=[json.loads(l) for l in Path(paths[name]).with_suffix('.privileged.jsonl').read_text().splitlines()]
        penalties={};first={}
        for r in truth:
            for e in r['penalty_events']:
                h=e.get('host','action_cost');penalties[h]=penalties.get(h,0)+e['penalty'];first.setdefault(h,r['tick'])
        result['policies'][name]={'return':sum(r['reward'] for r in rows),'completed_actions':completions,
            'confirmed_first_at':positive,'branches':forced,'penalty_hosts':sorted(
                [{'host':h,'total_penalty':v,'first_penalty_tick':first[h]} for h,v in penalties.items()],key=lambda r:r['total_penalty'])[:5]}
        if name!='guard':
            for g,r in zip(reference,rows):
                changed=[a for a in g['agents'] if (g['agents'][a]['action'],g['agents'][a]['host']) != (r['agents'][a]['action'],r['agents'][a]['host'])]
                if changed:
                    result['request_divergences'][name]={'tick_after_step':r['tick'],
                        'changed_agents':[{ 'agent':a,'guard':g['agents'][a],'actor':r['agents'][a]} for a in changed]};break
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-seed',type=int,required=True)
    p.add_argument('--episode-seed',type=int,required=True);p.add_argument('--out',required=True)
    p.add_argument('--scorer',default='blue/results/gpt_harness/legacy_candidate/scorer.pkl')
    a=p.parse_args();check_seeds([a.episode_seed]);out=Path(a.out)
    if out.exists():raise FileExistsError('preserve existing traces')
    out.mkdir(parents=True);paths={};report={'cells':{},'training_seed':a.training_seed,'episode_seed':a.episode_seed}
    for arm in ('guard','zero','risk'):
        if arm=='guard':policy=HarnessPolicy(max_age=80)
        else:
            hook,m=load_hook(f'blue/results/gpt_harness/main32_status/s{a.training_seed}/pilot/{arm}_s{a.training_seed}',a.scorer)
            policy=HarnessPolicy(hook,m['config']['max_age'])
        paths[arm]=out/f'{arm}.jsonl'
        report['cells'][arm]=episode(policy,a.episode_seed,400,paths[arm])
        print(arm,report['cells'][arm]['return'],flush=True)
    frozen=json.loads(Path(f'blue/results/gpt_harness/main32_status/s{a.training_seed}/pilot/report.json').read_text())
    expected={r['arm']:r['cells'][str(a.episode_seed)]['return'] for r in frozen['runs']}
    report['reproduction']={arm:{'expected_remote_return':expected[arm],
        'local_return':report['cells'][arm]['return'],
        'matches':expected[arm]==report['cells'][arm]['return']} for arm in ('zero','risk')}
    report['remote_failure_reproduced']=all(r['matches'] for r in report['reproduction'].values())
    report['analysis']=analyse(paths)
    if not report['remote_failure_reproduced']:
        report['analysis']['claim']='REPRODUCTION FAILED: local traces cannot explain the remote failure; descriptive local associations only'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
