"""One tick-38 high-risk-host investigation check; no training/score claim."""
import copy,json
from pathlib import Path
import torch
from blue.core.wrapper import BLUE_AGENTS,CC4MARLEnv
from blue.core.baselines import decode_index
from blue.training import harness_redesign as hr
from blue.training import mappo_guide as mg
from blue.training.risk_data import true_compromised
from blue.training.harness_update_audit import ROOT,SCORER
from blue.harness.scoring import sha256
HOST='operational_zone_a_subnet_user_host_0';AGENT='blue_agent_1'

SEED=8233;POLICY_SEED=SEED*997+1;MODEL=ROOT/'more_exploration_s1';N=42
class OneChoice:
 def __init__(self,hook):self.hook=hook;self.events=[]
 def reset(self):self.hook.reset();self.events=[]
 def observe(self,*a):self.hook.observe(*a)
 def __call__(self,env,agent,cands,scored):
  if env._tick==38 and agent==AGENT and HOST in cands:
   teacher=max(scored,key=lambda x:(x[0],x[1]))[1]
   risk=float(self.hook.scorer.score(env,agent,[HOST])[0,0])
   self.events.append({'tick':env._tick,'agent':agent,'original_teacher':teacher,'replacement':HOST,'risk':risk})
   return HOST
  return self.hook(env,agent,cands,scored)
if __name__=='__main__':
 torch.set_num_threads(1);out=ROOT/'host_gap_audit'/'early_detection_check.json'
 if out.exists():raise FileExistsError(out)
 before=sha256(MODEL/'actor.th');policy,m=hr.load(MODEL,SCORER,POLICY_SEED);hook=OneChoice(policy.hook);policy.hook=hook
 env=CC4MARLEnv(seed=SEED,steps=400,**mg.ENV_KW);env.reset(seed=SEED);policy.reset();rows=[];total=0.
 for _ in range(N):
  tick=env._tick;pre=dict(env._awaiting);truth_before=HOST in true_compromised(env);acts={};req={}
  for a in BLUE_AGENTS:
   ix=int(policy.select(env,a));acts[a]=ix;name,h=decode_index(env,a,ix);req[a]={'action':name,'host':h}
  _,rew,term,trunc,_=env.step(acts);total+=float(rew[0]);completion=None
  if AGENT in pre and AGENT not in env._awaiting:
   h,name=pre[AGENT];tr=env.trackers[AGENT]
   completion={'host':h,'action':name,'result':tr.last_result[h],'belief':tr.state[h],
    'last_analysis':tr.last_analysis[h],'files':copy.deepcopy(env.views[AGENT][h].get('Files')),
    'processes':copy.deepcopy(env.views[AGENT][h].get('Processes'))}
  rows.append({'tick':tick,'requests':req,'completion':completion,
    'focus_visible':{'belief':env.trackers[AGENT].state[HOST],
       'last_analysis':env.trackers[AGENT].last_analysis[HOST],
       'files':copy.deepcopy(env.views[AGENT][HOST].get('Files'))},
    'red_session_before':truth_before,'red_session_after':HOST in true_compromised(env),
    'reward':float(rew[0])})
  if term or trunc:break
 expected=[json.loads(l) for l in (ROOT/'decision_timeline/original.blue.jsonl').read_text().splitlines()[:N]]
 for i,(x,y) in enumerate(zip(rows,expected)):
  for a,v in y['agents'].items():
   got=x['requests'][a];want=(v['action'],v['host'])
   if i>38 or (i==38 and a==AGENT):continue
   assert (got['action'],got['host'])==want,f'prefix diverged before intervention tick {i}, {a}'
 if len(hook.events)!=1 or sha256(MODEL/'actor.th')!=before:raise AssertionError('intervention/checkpoint validation failed')
 report={'scope':'42-tick mechanism check on reused episode; no return claim',
  'actor_sha256':before,'episode':SEED,'policy_rng':POLICY_SEED,'intervention':hook.events,
  'rows':rows,'prefix_matches_original_before_single_request':True,'partial_reward_not_performance_metric':total}
 out.write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps({'intervention':hook.events,'completion':[r for r in rows if r['completion'] and r['completion']['host']==HOST], 'truth':[{'tick':r['tick'],'before':r['red_session_before'],'after':r['red_session_after']} for r in rows if r['tick'] in (37,38,39,40,41)],'partial_reward_not_metric':total},indent=2,default=str));env.close()
