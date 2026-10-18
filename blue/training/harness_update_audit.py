"""Read-only trained/untrained distribution audit on matched saved episodes.

Shadow distributions share the trained trajectory's state. They are not
counterfactual returns or action values; no optimization or model write occurs.
"""
import json
from pathlib import Path
import numpy as np
import torch
from blue.training import harness_redesign as hr
from blue.harness.redesign import SetActor, FEATURE_NAMES, probabilities
from blue.harness.scoring import sha256

ROOT=Path('blue/results/gpt_harness/v2_sensitivity')
OUT=ROOT/'update_audit'
SCORER='blue/results/gpt_harness/legacy_candidate/scorer.pkl'

class AuditHook:
    def __init__(self, hook):self.hook=hook;self.records=[]
    def reset(self):self.hook.reset();self.records=[]
    def observe(self,env,agent):self.hook.observe(env,agent)
    def __call__(self,env,agent,cands,scored):
        h=self.hook
        with torch.no_grad():
            F,B,M,T,P=h.distribution(env,agent,cands,scored)
            raw=h.actor(F,B,M);res=h.bonus*torch.tanh(raw)
            zero=h.explore*torch.softmax(B/h.temperature,dim=-1)
            zero[T]+=1-h.explore
            clone=torch.Generator();clone.set_state(h.generator.get_state())
            shadow=int(torch.multinomial(zero,1,generator=clone))
            chosen=int(torch.multinomial(P,1,generator=h.generator))
            ablated=F.clone()
            for name in ('compromise_risk','risk_binary_entropy'):
                ablated[:,FEATURE_NAMES.index(name)]=0
            without_ml=probabilities(h.actor,ablated,B,M,T,h.bonus,h.temperature,h.explore)
        self.records.append({'tick':env._tick,'agent':agent,'chosen':cands[chosen],
            'shadow_zero_residual_pick':cands[shadow],'teacher':cands[T],
            'total_variation':float(.5*(P-zero).abs().sum()),
            'max_probability_shift':float((P-zero).abs().max()),
            'residual_range':float(res.max()-res.min()),
            'raw_range':float(raw.max()-raw.min()),
            'ml_zeroing_total_variation':float(.5*(P-without_ml).abs().sum()),
            'candidates':[{'host':host,'base':float(B[i]),'residual':float(res[i]),
               'trained_probability':float(P[i]),'zero_residual_probability':float(zero[i]),
               'blue_risk':float(F[i,FEATURE_NAMES.index('compromise_risk')])}
               for i,host in enumerate(cands)]})
        return cands[chosen]

def run():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    report={'scope':'read-only diagnostic; same-state shadows are not counterfactual outcomes',
            'models':{},'episodes':{}}
    for folder in sorted(ROOT.glob('*_s[01]')):
        manifest=json.loads((folder/'manifest.json').read_text());c=manifest['config']
        torch.manual_seed(c['seed']);initial=SetActor(c['hidden'])
        weights=torch.load(folder/'actor.th',weights_only=True,map_location='cpu')
        report['models'][folder.name]={'actor_sha256':sha256(folder/'actor.th'),
            'parameter_delta_l2':{name:float((weights[name]-p).norm()) for name,p in initial.state_dict().items()},
            'updates':[{k:r[k] for k in ('iteration','kl_exact','actor_gradient_norm_mean','explained_variance','target_std','policy_steps')} for r in manifest['history']]}
    frozen=json.loads((ROOT/'cells.json').read_text())
    for tag,seed,replica in [('reference_s0',8233,0),('more_exploration_s1',8233,1)]:
        key=f'{tag}_{seed}_{replica}';policy,manifest=hr.load(ROOT/tag,SCORER,seed*997+replica)
        before=sha256(ROOT/tag/'actor.th');audit=AuditHook(policy.hook);policy.hook=audit
        result=hr.evaluation_episode(policy,seed,400)
        digest=policy.request_digest.result()
        if result['return']!=frozen[key]['return'] or digest!=frozen[key]['requests']:
            raise AssertionError('instrumented trajectory does not reproduce frozen cell')
        if sha256(ROOT/tag/'actor.th')!=before:raise AssertionError('checkpoint changed')
        records=audit.records
        summary={'return':result['return'],'requests':digest,'learned_choices':len(records),
            'shadow_sample_changes':sum(r['chosen']!=r['shadow_zero_residual_pick'] for r in records)}
        for name in ('total_variation','max_probability_shift','residual_range','raw_range','ml_zeroing_total_variation'):
            v=[r[name] for r in records];summary[name]={'mean':float(np.mean(v)),'max':max(v),'p95':float(np.quantile(v,.95))}
        summary['first_shadow_changes']=[{k:v for k,v in r.items() if k!='candidates'} for r in records if r['chosen']!=r['shadow_zero_residual_pick']][:10]
        (OUT/(key+'.json')).write_text(json.dumps({'summary':summary,'decisions':records},indent=2)+'\n')
        report['episodes'][key]=summary
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'episode':key,**{k:v for k,v in summary.items() if k!='first_shadow_changes'}}),flush=True)

if __name__=='__main__':run()
