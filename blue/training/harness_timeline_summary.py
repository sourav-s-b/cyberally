"""Summarize diagnostic timelines, never turn hidden truth into policy inputs."""
from collections import defaultdict
import json
from pathlib import Path

ROOT=Path('blue/results/gpt_harness/v2_sensitivity/decision_timeline')
FOCUS=['admin_network_subnet_router','office_network_subnet_user_host_1']

def read(path):return [json.loads(l) for l in path.read_text().splitlines()]
def first_difference(left,right,key):
    for l,r in zip(left,right):
        if l[key]!=r[key]:return {'request_tick':l['request_tick'],'original':l[key],'changed':r[key]}
    return None

def run():
    blue={arm:read(ROOT/(arm+'.blue.jsonl')) for arm in ['original','single_request','untrained']}
    truth={arm:read(ROOT/(arm+'.privileged.jsonl')) for arm in blue}
    report={'scope':'fixed episode 8233 and policy RNG; diagnostic event accounting','arms':{}}
    for arm,rows in blue.items():
        penalties=defaultdict(float);events=defaultdict(int);completions=[];requests=[]
        for b,t in zip(rows,truth[arm]):
            for e in t['penalty_events']:
                k=e.get('host','action_cost')+'|'+e['event'];penalties[k]+=e['penalty'];events[k]+=1
            for agent,r in b['agents'].items():
                if r['host'] in FOCUS and r['action']!='Sleep':requests.append({'tick':b['request_tick'],'agent':agent,'action':r['action'],'host':r['host'],'branch':r['request']['branch']})
                c=r['completion']
                if c and c['host'] in FOCUS:
                    v=c['blue_after'];completions.append({'tick':b['after_tick'],'agent':agent,'host':c['host'],'action':c['action'],'result':v['last_result'],'belief':v['belief'],'files':v['view'].get('Files')})
        states={h:{'first_compromised_tick':next((r['request_tick'] for r in truth[arm] if h in r['compromised_before']),None),
                   'first_confirmed_tick':next((r['after_tick'] for r in rows if r['focus_after'][h]['belief']=='CONFIRMED'),None)} for h in FOCUS}
        report['arms'][arm]={'return':sum(r['reward'] for r in rows),'penalty_by_host_event':dict(penalties),'event_counts':dict(events),
            'focus_ground_truth_and_beliefs':states,'focus_requests':requests,'focus_completions':completions}
    o=blue['original'];c=blue['single_request'];ot=truth['original'];ct=truth['single_request']
    report['first_reward_difference']=first_difference(o,c,'reward')
    report['first_compromise_set_difference']=first_difference(ot,ct,'compromised_after')
    report['first_red_green_action_difference']=first_difference(ot,ct,'red_green_actions')
    report['first_penalty_event_difference']=first_difference(ot,ct,'penalty_events')
    report['tick48_blue_evidence']=o[48]['focus_before']
    report['request_differences_40_65']=[]
    for l,r in zip(o,c):
        tick=l['request_tick']
        if 40<=tick<=65:
            diffs=[{'agent':a,'original':{k:l['agents'][a][k] for k in ['action','host']},'changed':{k:r['agents'][a][k] for k in ['action','host']}} for a in l['agents'] if any(l['agents'][a][k]!=r['agents'][a][k] for k in ['action','host'])]
            if diffs:report['request_differences_40_65'].append({'request_tick':tick,'agents':diffs})
    (ROOT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['arms','tick48_blue_evidence','request_differences_40_65']},indent=2))
    for a,r in report['arms'].items():print(a,r['return'],r['penalty_by_host_event'],r['focus_ground_truth_and_beliefs'])

if __name__=='__main__':run()
