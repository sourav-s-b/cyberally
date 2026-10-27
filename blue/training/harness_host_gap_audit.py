"""Blue-visible timeline for one suspected detection gap; truth is separate."""
import copy,json
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
from blue.harness.scoring import sha256

HOST='operational_zone_a_subnet_user_host_0';AGENT='blue_agent_1'
OUT=ROOT/'host_gap_audit';MODEL=ROOT/'more_exploration_s1';POLICY_SEED=8233*997+1

def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
 bluepath=OUT/'blue_visible.jsonl';truthpath=OUT/'privileged_diagnostic.jsonl'
 if bluepath.exists() or truthpath.exists():raise FileExistsError('preserve existing host-gap trace')
 frozen=json.loads((ROOT/'cells.json').read_text())['more_exploration_s1_8233_1']
 before=sha256(MODEL/'actor.th');policy,m=hr.load(MODEL,SCORER,POLICY_SEED)
 env=CC4MARLEnv(seed=8233,steps=400,**mg.ENV_KW);env.reset(seed=8233);policy.reset()
 observed_events=[];native=0.;ticks=0
 with bluepath.open('w') as blog,truthpath.open('w') as tlog:
  for _ in range(400):
   tick=env._tick;pending=dict(env._awaiting);tr=env.trackers[AGENT]
   pre={'view':copy.deepcopy(env.views[AGENT][HOST]),'observed_at':copy.deepcopy(env.observed_at[AGENT][HOST]),
        'belief':tr.state[HOST],'last_analysis':tr.last_analysis[HOST],'last_remediation':tr.last_remediation[HOST],
        'last_result':tr.last_result[HOST],'empty_strikes':tr.empty_strikes[HOST],
        'busy':AGENT in pending,'pending':pending.get(AGENT)}
   true_before=HOST in true_compromised(env);requests={}
   actions={}
   for a in BLUE_AGENTS:
    action=int(policy.select(env,a));actions[a]=action;name,host=decode_index(env,a,action)
    requests[a]={'action':name,'host':host,'request':copy.deepcopy(policy.last_request)}
   _,reward,term,trunc,_=env.step(actions);reward=float(reward[0]);native+=reward;ticks+=1
   events=penalty_events(env)
   if not np.isclose(sum(e['penalty'] for e in events),reward):raise AssertionError('native penalty accounting mismatch')
   completed=pending.get(AGENT)
   completion=None
   if completed and AGENT not in env._awaiting:
    h,name=completed;tracker=env.trackers[AGENT]
    completion={'host':h,'action':name,'result':tracker.last_result[h],
       'belief':tracker.state[h],'last_analysis':tracker.last_analysis[h],
       'last_remediation':tracker.last_remediation[h],
       'target_view':copy.deepcopy(env.views[AGENT][h]),
       'target_observed_at':copy.deepcopy(env.observed_at[AGENT][h])}
   row={'tick':tick,'after_tick':env._tick,'host_before':pre,
        'blue_requests':requests,'host_completion':completion,
        'host_after':{'view':copy.deepcopy(env.views[AGENT][HOST]),
          'observed_at':copy.deepcopy(env.observed_at[AGENT][HOST]),
          'belief':env.trackers[AGENT].state[HOST],
          'last_analysis':env.trackers[AGENT].last_analysis[HOST],
          'last_remediation':env.trackers[AGENT].last_remediation[HOST],
          'last_result':env.trackers[AGENT].last_result[HOST],
          'busy':AGENT in env._awaiting,'pending':env._awaiting.get(AGENT)},
        'native_reward':reward,'host_penalties':[e for e in events if e.get('host')==HOST]}
   blog.write(json.dumps(row,default=str,allow_nan=False)+'\n')
   tlog.write(json.dumps({'tick':tick,'host_red_session_before':true_before,
       'host_red_session_after':HOST in true_compromised(env),
       'all_red_sessions_after':sorted(true_compromised(env)),
       'penalty_events':events},default=str)+'\n')
   if term or trunc:break
 digest=policy.request_digest.result()
 if native!=frozen['return'] or digest!=frozen['requests']:raise AssertionError('frozen replay mismatch')
 if sha256(MODEL/'actor.th')!=before:raise AssertionError('checkpoint changed')
 report={'scope':'reused episode and policy RNG; no held-out claim','episode':8233,
   'training_run':'more_exploration_s1','policy_seed':POLICY_SEED,'host':HOST,'blue_agent':AGENT,
   'native_return':native,'ticks':ticks,'requests':digest,'actor_sha256':before,
   'blue_trace':str(bluepath),'privileged_trace':str(truthpath),
   'claim':'diagnostic truth is excluded from policy and Blue visible timeline'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');env.close();print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
