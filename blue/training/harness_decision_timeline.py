"""Full frozen replay timeline: Blue evidence separate from diagnostic truth."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from blue.core.wrapper import BLUE_AGENTS,CC4MARLEnv
from blue.core.baselines import decode_index
from blue.training import harness_redesign as hr
from blue.training import mappo_guide as mg
from blue.training.harness_eval import penalty_events
from blue.training.risk_data import true_compromised
from blue.training.harness_update_audit import ROOT,SCORER
from blue.training.harness_single_decision_check import SingleDecisionHook
from blue.harness.scoring import sha256

FOCUS=['admin_network_subnet_router','office_network_subnet_user_host_1']
OUT=ROOT/'decision_timeline'

def visible(env,agent,host):
    tr=env.trackers[agent]
    return {'belief':tr.state[host],'last_analysis':tr.last_analysis[host],
        'last_remediation':tr.last_remediation[host],'last_result':tr.last_result[host],
        'empty_strikes':tr.empty_strikes[host],
        'observed_at':copy.deepcopy(env.observed_at[agent][host]),
        'view':copy.deepcopy(env.views[agent][host])}

def diagnostic_actions(env):
    controller=env.env;rows=[]
    for agent,actions in controller.action.items():
        if 'red' not in agent and 'green' not in agent:continue
        if not actions:continue
        action=actions[0];host=getattr(action,'hostname',None);ip=getattr(action,'ip_address',None)
        if host is None and ip is not None:host=controller.state.ip_addresses.get(ip)
        observations=controller.observation[agent].observations
        obs=observations[0].data if observations else {}
        rows.append({'agent':agent,'action':type(action).__name__,'host':host,
            'ip':str(ip) if ip is not None else None,'success':str(obs.get('success'))})
    return rows

def run():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    frozen=json.loads((ROOT/'cells.json').read_text());report={'scope':'one reused episode; no held-out claim','cells':{}}
    for arm in ['original','single_request','untrained']:
        blue_path=OUT/(arm+'.blue.jsonl');truth_path=OUT/(arm+'.privileged.jsonl')
        if blue_path.exists():raise FileExistsError('preserve previous traces')
        model=ROOT/'more_exploration_s1';before=sha256(model/'actor.th')
        policy,manifest=hr.load(model,SCORER,8233*997+1)
        if arm=='single_request':policy.hook=SingleDecisionHook(policy.hook)
        if arm=='untrained':
            with torch.no_grad():
                for p in policy.hook.actor.parameters():p.zero_()
        env=CC4MARLEnv(seed=8233,steps=400,**mg.ENV_KW);env.reset(seed=8233);policy.reset()
        rewards=[]
        with blue_path.open('w') as blue,truth_path.open('w') as truth:
            for _ in range(400):
                tick=env._tick;pre_pending=dict(env._awaiting);truth_before=sorted(true_compromised(env))
                records={};actions={}
                focus_before={h:visible(env,'blue_agent_4',h) for h in FOCUS}
                for agent in BLUE_AGENTS:
                    action=int(policy.select(env,agent));actions[agent]=action
                    name,host=decode_index(env,agent,action)
                    records[agent]={'action':name,'host':host,'request':copy.deepcopy(policy.last_request),
                        'blue_before':visible(env,agent,host) if host else None}
                _,reward,term,trunc,_=env.step(actions);reward=float(reward[0]);rewards.append(reward)
                events=penalty_events(env)
                if not np.isclose(sum(e['penalty'] for e in events),reward):raise AssertionError('reward accounting mismatch')
                for agent,record in records.items():
                    issued=(record['host'],record['action']) if record['host'] and record['action'] in ('Analyse','Remove','Restore') else None
                    active=pre_pending.get(agent) or issued
                    record['pending_after']=env._awaiting.get(agent);record['completion']=None
                    if active and agent not in env._awaiting:
                        host,name=active
                        record['completion']={'host':host,'action':name,'blue_after':visible(env,agent,host)}
                blue.write(json.dumps({'request_tick':tick,'after_tick':env._tick,'agents':records,
                    'focus_before':focus_before,'focus_after':{h:visible(env,'blue_agent_4',h) for h in FOCUS},
                    'reward':reward},default=str,allow_nan=False)+'\n')
                truth.write(json.dumps({'request_tick':tick,'after_tick':env._tick,
                    'compromised_before':truth_before,'compromised_after':sorted(true_compromised(env)),
                    'penalty_events':events,'red_green_actions':diagnostic_actions(env)},default=str)+'\n')
                if term or trunc:break
        digest=policy.request_digest.result();total=sum(rewards)
        if arm=='single_request':expected=json.loads((ROOT/'update_audit/single_decision_check.json').read_text());expected_return=expected['intervention_return'];expected_digest=expected['requests']
        else:
            expected=frozen['more_exploration_s1_8233_1' if arm=='original' else 'control_e0.5_8233_1'];expected_return=expected['return'];expected_digest=expected['requests']
        if total!=expected_return or digest!=expected_digest:raise AssertionError('timeline replay drift')
        if sha256(model/'actor.th')!=before:raise AssertionError('checkpoint changed')
        report['cells'][arm]={'return':total,'requests':digest,'ticks':len(rewards),'blue_trace':str(blue_path),'truth_trace':str(truth_path),'actor_sha256':before}
        (OUT/'progress.json').write_text(json.dumps(report,indent=2)+'\n');env.close()
        print(arm,total,flush=True)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':run()
