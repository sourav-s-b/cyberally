"""Read-only simulator RNG fingerprints; partial diagnostic, never scored."""
import hashlib
import json
import torch
from blue.core.wrapper import BLUE_AGENTS,CC4MARLEnv
from blue.core.baselines import decode_index
from blue.training import harness_redesign as hr
from blue.training import mappo_guide as mg
from blue.training.harness_update_audit import ROOT,SCORER
from blue.training.harness_single_decision_check import SingleDecisionHook

def fingerprint(rng):
    return hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True,default=str).encode()).hexdigest()

if __name__=='__main__':
    torch.set_num_threads(1);out=ROOT/'decision_timeline';report={'scope':'80-tick diagnostic prefix; no return or performance estimate','arms':{}}
    for arm in ['original','single_request']:
        expected=[json.loads(l) for l in (out/(arm+'.blue.jsonl')).read_text().splitlines()]
        policy,m=hr.load(ROOT/'more_exploration_s1',SCORER,8233*997+1)
        if arm=='single_request':policy.hook=SingleDecisionHook(policy.hook)
        env=CC4MARLEnv(seed=8233,steps=400,**mg.ENV_KW);env.reset(seed=8233);policy.reset()
        controller=env.env
        identity={'state_shares_controller_rng':controller.state.np_random is controller.np_random,
            'npc_agents_share_controller_rng':{a:i.agent.np_random is controller.np_random for a,i in controller.agent_interfaces.items() if a.startswith(('red','green'))}}
        rows=[]
        for tick in range(80):
            before=fingerprint(controller.np_random);acts={}
            for a in BLUE_AGENTS:
                acts[a]=int(policy.select(env,a));name,host=decode_index(env,a,acts[a])
                if (name,host)!=(expected[tick]['agents'][a]['action'],expected[tick]['agents'][a]['host']):raise AssertionError('request prefix drift')
            selection=fingerprint(controller.np_random)
            env.step(acts)
            rows.append({'request_tick':tick,'before_selection':before,'after_selection':selection,'after_step':fingerprint(controller.np_random)})
        report['arms'][arm]={'identity':identity,'rows':rows};env.close()
        print('prefix complete',arm,flush=True)
    x=report['arms']['original']['rows'];y=report['arms']['single_request']['rows']
    report['first_rng_state_divergence_after_step']=next((a['request_tick'] for a,b in zip(x,y) if a['after_step']!=b['after_step']),None)
    report['policy_selection_consumes_simulator_rng']=any(r['before_selection']!=r['after_selection'] for a in report['arms'].values() for r in a['rows'])
    (out/'rng_prefix.json').write_text(json.dumps(report,indent=2)+'\n')
    print('first RNG divergence after request tick',report['first_rng_state_divergence_after_step'],flush=True)
