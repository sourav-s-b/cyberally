"""One frozen diagnostic intervention, no training or performance claim.

Replace the earliest same-state shadow mismatch in the selected failed replay.
Consume the trained sampler's draw normally and change only that host request;
subsequent policy/environment divergence is part of this mechanism check.
"""
import json
from pathlib import Path
import torch
from blue.training import harness_redesign as hr
from blue.training.harness_update_audit import ROOT,OUT,SCORER,AuditHook
from blue.harness.scoring import sha256

class SingleDecisionHook(AuditHook):
    def __init__(self,hook):super().__init__(hook);self.interventions=[]
    def reset(self):super().reset();self.interventions=[]
    def __call__(self,env,agent,cands,scored):
        chosen=super().__call__(env,agent,cands,scored)
        if env._tick==48 and agent=='blue_agent_4':
            record=self.records[-1]
            replacement=record['shadow_zero_residual_pick']
            self.interventions.append({'tick':env._tick,'agent':agent,'original':chosen,'replacement':replacement})
            return replacement
        return chosen

if __name__=='__main__':
    torch.set_num_threads(1)
    model=ROOT/'more_exploration_s1';before=sha256(model/'actor.th')
    policy,manifest=hr.load(model,SCORER,8233*997+1)
    audit=SingleDecisionHook(policy.hook);policy.hook=audit
    result=hr.evaluation_episode(policy,8233,400)
    if len(audit.interventions)!=1 or audit.interventions[0]['original']==audit.interventions[0]['replacement']:
        raise AssertionError('intended single change did not occur')
    if sha256(model/'actor.th')!=before:raise AssertionError('checkpoint changed')
    frozen=json.loads((ROOT/'cells.json').read_text())
    report={'claim':'one reused episode and policy RNG; mechanism check only',
        'intervention_rule':'earliest saved same-state shadow mismatch, fixed before replay',
        'actor_sha256':before,'episode':8233,'policy_seed':8233*997+1,
        'original_return':frozen['more_exploration_s1_8233_1']['return'],
        'untrained_return':frozen['control_e0.5_8233_1']['return'],
        'intervention_return':result['return'],'interventions':audit.interventions,
        'requests':policy.request_digest.result()}
    (OUT/'single_decision_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
