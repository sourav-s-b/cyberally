"""Read-only short-prefix audit of risk and actor ranking at tick 38."""
import json
from pathlib import Path
import torch
from blue.core.wrapper import BLUE_AGENTS,CC4MARLEnv
from blue.core.baselines import decode_index
from blue.training import harness_redesign as hr
from blue.training import mappo_guide as mg
from blue.training.harness_update_audit import ROOT,SCORER
from blue.training.harness_single_decision_check import SingleDecisionHook
from blue.harness.scoring import sha256

MODEL=ROOT/'more_exploration_s1';SEED=8233;POLICY_SEED=SEED*997+1;HOST='operational_zone_a_subnet_user_host_0';AGENT='blue_agent_1'
class Capture:
 def __init__(self,hook):self.hook=hook;self.rows=[]
 def reset(self):self.hook.reset();self.rows=[]
 def observe(self,*args):self.hook.observe(*args)
 def __call__(self,env,agent,cands,scored):
  if env._tick in (37,38) and agent==AGENT:
   with torch.no_grad():F,B,M,T,P=self.hook.distribution(env,agent,cands,scored)
   names=self.hook.actor
   from blue.harness.redesign import FEATURE_NAMES
   i=FEATURE_NAMES.index('compromise_risk'); n=FEATURE_NAMES.index('risk_binary_entropy')
   self.rows.append({'tick':env._tick,'host_snapshot':{
      'belief':env.trackers[agent].state[HOST],
      'analysis_age':env._tick-(env.trackers[agent].last_analysis[HOST] or 0),
      'files':env.views[agent][HOST].get('Files'),
      'processes':env.views[agent][HOST].get('Processes'),
      'observed_at':env.observed_at[agent][HOST]},
      'host_in_sweep_candidates':HOST in cands,'teacher':cands[T],
      'risk_model':self.hook.scorer.score(env,agent,[HOST])[0].tolist(),
      'candidates':[{'host':h,'base':float(B[j]),'actor_residual':float(self.hook.bonus*torch.tanh(self.hook.actor(F,B,M))[j]),
        'risk':float(F[j,i]),'risk_entropy':float(F[j,n]),'probability':float(P[j])} for j,h in enumerate(cands)]})
  return self.hook(env,agent,cands,scored)
if __name__=='__main__':
 torch.set_num_threads(1);out=ROOT/'host_gap_audit'/'candidate_scores.json'
 if out.exists():raise FileExistsError(out)
 before=sha256(MODEL/'actor.th');policy,m=hr.load(MODEL,SCORER,POLICY_SEED);cap=Capture(policy.hook);policy.hook=cap
 env=CC4MARLEnv(seed=SEED,steps=400,**mg.ENV_KW);env.reset(seed=SEED);policy.reset();requests=[]
 for tick in range(40):
  acts={}
  for a in BLUE_AGENTS:
   x=int(policy.select(env,a));acts[a]=x;name,host=decode_index(env,a,x);requests.append({'tick':tick,'agent':a,'action':name,'host':host})
  env.step(acts)
 expected=[json.loads(l) for l in (ROOT/'decision_timeline/original.blue.jsonl').read_text().splitlines()[:40]]
 for i,row in enumerate(expected):
  for a,v in row['agents'].items():
   q=next(x for x in requests if x['tick']==i and x['agent']==a)
   assert (q['action'],q['host'])==(v['action'],v['host']),'prefix drift'
 if sha256(MODEL/'actor.th')!=before:raise AssertionError('checkpoint changed')
 report={'scope':'40-tick diagnostic prefix; same model/environment; no return estimate','actor_sha256':before,'observations':cap.rows}
 out.write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps(report,indent=2,default=str),flush=True);env.close()
