"""Read-only design audit of complete main32 artifacts and one frozen rollout."""
import argparse
import json
from pathlib import Path

import numpy as np


def cohort(root):
    root=Path(root);result=json.loads((root/'aggregate.json').read_text())
    audit={'claim':'read-only design diagnostics, not causal attribution',
           'training':[],'request_identity':{},'input_sensitivity':{},'tails':{}}
    for seed in range(5):
        histories=[]
        for arm in ('zero','risk','both'):
            folder=root/f's{seed}'/'pilot'/f'{arm}_s{seed}'
            m=json.loads((folder/'manifest.json').read_text());h=m['history'];histories.append(h)
            audit['source_commit']=m['source_commit']
            last=h[-1]
            audit['training'].append({'training_seed':seed,'arm':arm,
                **{k:last[k] for k in ('greedy_agreement','sampled_override_rate','vf','pol','ent')},
                'mean_guard_interventions_last_update':float(np.mean([g['interventions'] for g in last['guard']]))})
            d=json.loads((folder/'diagnostics_032.json').read_text());v=[r for e in d for r in e['sweep_sensitivity']]
            audit['input_sensitivity'][f'{seed}_{arm}']={'n_decisions':len(v),
                'ml_top_index_changes':sum(r['ml_top_index_changed'] for r in v),
                'max_ml_logit_difference':max(r['ml_logit_delta_max'] for r in v)}
        same=[len({tuple((e['seed'],e['requested_actions_sha256']) for e in h[i]['episode_request_digests']) for h in histories})==1 for i in range(32)]
        audit['request_identity'][str(seed)]={'initial_identical_updates':next((i for i,v in enumerate(same) if not v),32),'identical_updates_total':sum(same)}
    for arm in ('zero','risk','both'):
        losses=sorted([(r[e]-result['baselines']['guard'][e],tag,e,r[e],result['baselines']['guard'][e])
            for tag,r in result['all_episode_returns'].items() if tag.endswith('_'+arm) for e in r])
        audit['tails'][arm]=losses[:5]
    return audit


def shadow_gradients(model_dir,scorer_path,seed,steps):
    """One fresh rollout, backward only; no optimizer step or checkpoint write."""
    import torch as th
    from blue.training.harness_rl import load_hook,HarnessRecorder,check_seeds
    from blue.harness.policy import HarnessPolicy
    from blue.harness.scoring import sha256
    from blue.training import mappo_guide as mg
    check_seeds([seed]);d=Path(model_dir);hook,m=load_hook(d,scorer_path);c=m['config']
    if steps!=c['steps']:raise ValueError('frozen horizon required')
    before=sha256(d/'actor.th');actor=hook.actor
    critic=mg.build_central_critic(hidden=c['hidden']);critic.load_state_dict(th.load(d/'critic.th',map_location='cpu',weights_only=True))
    rec=HarnessRecorder(actor,hook.scorer,c['bonus'],th.Generator().manual_seed(123),c['temp'],c['ml_inputs'])
    policy=HarnessPolicy(rec,c['max_age'])
    result=mg.run_team_episode(policy,seed,steps,hook=rec,recorder=rec,**mg.ENV_KW)
    buf=rec.rows
    F=th.tensor(np.stack([r['feats'] for r in buf]),dtype=th.float32)
    J=th.tensor(np.stack([r['joint'] for r in buf]),dtype=th.float32)
    B=th.tensor(np.stack([r['base'] for r in buf]),dtype=th.float32)
    M=th.tensor(np.stack([r['mask'] for r in buf]),dtype=th.float32)
    choices=th.tensor([r['choice'] for r in buf]);old=th.tensor([r['logp'] for r in buf]);targets=th.tensor([r['ret'] for r in buf])
    with th.no_grad():adv=targets-critic(J).squeeze(-1);adv=(adv-adv.mean())/adv.std().clamp(min=1e-6)
    logits=((B+c['bonus']*th.tanh(actor(F)))/c['temp'])*M+(1-M)*-1e9
    logall=th.log_softmax(logits,dim=-1);p=logall.exp();chosen=logall.gather(1,choices[:,None]).squeeze(1)
    ratio=(chosen-old).exp();pol=-th.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
    vf=th.nn.functional.mse_loss(critic(J).squeeze(-1),targets);entropy=-(p*logall).sum(-1).mean()
    (pol+.5*vf-.01*entropy).backward()
    def norm(net):return float(th.sqrt(sum((p.grad**2).sum() for p in net.parameters() if p.grad is not None)))
    an,cn=norm(actor),norm(critic);joint=float(np.hypot(an,cn));factor=min(1.,1./(joint+1e-6))
    if before!=sha256(d/'actor.th'):raise AssertionError('diagnostic modified frozen checkpoint')
    return {'model_dir':str(d),'actor_sha256':before,'seed':seed,'rollout':result,
        'sampling_seed':123,'n_rows':len(buf),'actor_gradient_norm':an,'critic_gradient_norm':cn,
        'joint_clip_multiplier':factor,'actor_only_clip_multiplier':min(1.,1./(an+1e-6)),
        'value_loss':float(vf.detach()),'scope':'one full-batch shadow backward, not historical minibatch norms; no optimization'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='blue/results/gpt_harness/main32_status');p.add_argument('--out',required=True)
    p.add_argument('--model-dir');p.add_argument('--scorer',default='blue/results/gpt_harness/legacy_candidate/scorer.pkl');p.add_argument('--seed',type=int,default=7706);p.add_argument('--steps',type=int,default=400)
    a=p.parse_args();r=cohort(a.root)
    if a.model_dir:r['shadow_gradients']=shadow_gradients(a.model_dir,a.scorer,a.seed,a.steps)
    Path(a.out).write_text(json.dumps(r,indent=2)+'\n')
    if a.model_dir:print(json.dumps(r['shadow_gradients'],indent=2))
