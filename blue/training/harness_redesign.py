"""Auditable v2 restricted MAPPO/A2C experiment. No long run by default.

Same protected remediation, native reward, input contract and exploration for
both learners. Finite-task decision credit includes intervening forced actions.
Old checkpoints are deliberately incompatible; no silent weight migration.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from blue.harness.redesign import VERSION, FEATURE_NAMES, CentralValue, RedesignHook, SetActor, central_context
from blue.harness.learning import decision_credit, update
from blue.harness.policy import HarnessPolicy
from blue.harness.scoring import MLScorer, sha256
from blue.training import mappo_guide as mg
from blue.training.harness_rl import check_seeds, source_identity


class Recorder(RedesignHook):
    def __init__(self, actor, critic, scorer, args, iteration):
        super().__init__(actor, scorer, args.bonus, args.ml_inputs, args.temp,
                         args.explore, args.seed*100003+iteration+1, 'sample')
        self.critic, self.gamma, self.gae_lambda = critic, args.gamma, args.gae_lambda
        self.rows=[];self.total=0;self.agree=0;self.per_agent={}
        self.sampled_disagree=0;self._ep_start=0;self.cache={}

    def mark_episode(self):
        self._ep_start=len(self.rows);self.cache={}

    def __call__(self,env,agent,cands,scored):
        with torch.no_grad():
            F,base,mask,teacher,p=self.distribution(env,agent,cands,scored)
            idx=int(torch.multinomial(p,1,generator=self.generator))
            joint=central_context(env,self.scorer,agent,self.cache)
            value=float(self.critic(torch.from_numpy(joint)))
        self.total+=1;self.agree+=int(idx==teacher);self.sampled_disagree+=int(idx!=teacher)
        self.per_agent[agent]=self.per_agent.get(agent,0)+1
        self.rows.append({'feats':F.numpy(),'base':base.numpy(),'mask':mask.numpy(),
            'old_probs':p.numpy(),'joint':joint,'choice':idx,'teacher':teacher,
            'tick':env._tick,'agent':agent,'value':value})
        return cands[idx]

    def close_episode(self,rewards):
        episode=self.rows[self._ep_start:]
        for agent in {r['agent'] for r in episode}:
            rows=[r for r in episode if r['agent']==agent]
            adv,targets,durations=decision_credit([r['tick'] for r in rows],rewards,
                [r['value'] for r in rows],self.gamma,self.gae_lambda)
            for row,a,t,d in zip(rows,adv,targets,durations):
                row.update(advantage=float(a),target=float(t),duration=int(d))


def runtime():
    return {'python':platform.python_version(),'torch_num_threads':torch.get_num_threads(),**{p:version(p) for p in
        ('numpy','scipy','torch','scikit-learn','gym','gymnasium')}}


def provenance():
    files=['blue/harness/redesign.py','blue/harness/learning.py','blue/harness/calibration.py','blue/harness/numerics.py',
        'blue/harness/features.py','blue/harness/scoring.py','blue/harness/policy.py',
        'blue/training/harness_redesign.py','blue/training/mappo_guide.py',
        'blue/core/wrapper.py','blue/policies/ordered.py']
    return {p:sha256(p) for p in files}


def train(args):
    torch.set_num_threads(args.threads)
    check_seeds(args.train_seeds)
    if len(set(args.train_seeds))!=len(args.train_seeds) or min(args.iters,args.eps_per_iter,args.steps)<1:
        raise ValueError('invalid budget/seeds')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    scorer=MLScorer(args.scorer); actor=SetActor(args.hidden);critic=CentralValue(args.hidden,args.value_scale)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=args.lr)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=args.critic_lr)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if k not in ('out','resume','stop_after','mode','eval_seeds','model_dir','policy_replicas')}
    manifest={'version':VERSION,'features':list(FEATURE_NAMES),'config':config,
        'source_commit':source_identity(),'source_files':provenance(),'runtime':runtime(),
        'env_kw':mg.ENV_KW,'scorer_sha256':scorer.sha256,'status':'started',
        'semantics':'restricted CTDE investigation scheduling; native finite-task reward; same teacher-mixture sampling in training and primary evaluation; no V-based gate'}
    hist=[]
    if (out/'manifest.json').exists():
        if not args.resume:raise FileExistsError('new run directory required')
        old=json.loads((out/'manifest.json').read_text())
        for k in ('version','features','config','source_commit','source_files','runtime','env_kw','scorer_sha256'):
            if json.dumps(old[k],sort_keys=True)!=json.dumps(manifest[k],sort_keys=True):raise ValueError('resume drift: '+k)
        ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        actor.load_state_dict(ck['actor']);critic.load_state_dict(ck['critic'])
        actor_opt.load_state_dict(ck['actor_optimizer']);critic_opt.load_state_dict(ck['critic_optimizer'])
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        hist=ck['history']
    elif args.resume:raise FileNotFoundError('no resumable run')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for iteration in range(len(hist),args.iters):
        rec=Recorder(actor,critic,scorer,args,iteration)
        policy=HarnessPolicy(rec,max_age=args.max_age,record_requests=True)
        episodes=[]
        for e in range(args.eps_per_iter):
            seed=args.train_seeds[(iteration*args.eps_per_iter+e)%len(args.train_seeds)]
            # Explicit independent episode sampling seeds survive same-seed resets.
            rec.policy_seed=args.seed*100003+iteration*1009+e+1
            result=mg.run_team_episode(policy,seed,args.steps,hook=rec,recorder=rec,**mg.ENV_KW)
            episodes.append({**result,'requests':policy.request_digest.result(),'guard':policy.guard_stats()})
        stats=update(actor,critic,actor_opt,critic_opt,rec.rows,args.algorithm,args.bonus,
            args.temp,args.explore,args.epochs,args.minibatch,args.clip,args.entropy_coef,
            args.target_kl,args.max_grad_norm,args.critic_epochs)
        hist.append({'iteration':iteration+1,'episodes':episodes,'sampled_override_rate':rec.sampled_disagree/max(rec.total,1),**stats})
        torch.save({'actor':actor.state_dict(),'critic':critic.state_dict(),
            'actor_optimizer':actor_opt.state_dict(),'critic_optimizer':critic_opt.state_dict(),
            'history':hist,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
            'python_rng':random.getstate()},out/'checkpoint.tmp')
        os.replace(out/'checkpoint.tmp',out/'checkpoint.pt')
        torch.save(actor.state_dict(),out/'actor.th');torch.save(critic.state_dict(),out/'critic.th')
        manifest.update(history=hist,status='complete' if len(hist)==args.iters else 'paused',actor_sha256=sha256(out/'actor.th'))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({'iteration':iteration+1,'algorithm':args.algorithm,
            'return_mean':float(np.mean([r['return'] for r in episodes])),
            'override_rate':hist[-1]['sampled_override_rate'],**stats}),flush=True)
        if args.stop_after and len(hist)>=args.stop_after:break


def load(model_dir,scorer_path,policy_seed,execution='sample'):
    d=Path(model_dir);m=json.loads((d/'manifest.json').read_text());c=m['config']
    scorer=MLScorer(scorer_path)
    if (m['version']!=VERSION or m['features']!=list(FEATURE_NAMES) or m['status']!='complete'
        or sha256(d/'actor.th')!=m['actor_sha256'] or scorer.sha256!=m['scorer_sha256']):
        raise ValueError('checkpoint/artifact contract mismatch')
    if runtime()!=m['runtime']:raise ValueError('evaluation runtime drift')
    if provenance()!=m['source_files']:raise ValueError('evaluation source drift')
    actor=SetActor(c['hidden']);actor.load_state_dict(torch.load(d/'actor.th',map_location='cpu',weights_only=True));actor.eval()
    hook=RedesignHook(actor,scorer,c['bonus'],c['ml_inputs'],c['temp'],c['explore'],policy_seed,execution)
    return HarnessPolicy(hook,c['max_age'],record_requests=True),m


def evaluate(args):
    torch.set_num_threads(args.threads)
    check_seeds(args.eval_seeds)
    if args.policy_replicas<1:raise ValueError('empty policy replica list')
    from blue.policies.ordered import LancerValues,OrderedPolicy
    d=Path(args.out);d.mkdir(parents=True,exist_ok=True)
    if (d/'report.json').exists():raise FileExistsError('new eval directory required')
    results={'claim':'reused development mechanism/smoke only, no superiority claim',
        'sampling_unit':'policy replicas nested within evaluation episode and training seed; never independent training runs',
        'cells':{},'baselines':{},'source_commit':source_identity()}
    for seed in args.eval_seeds:
        policy,m=load(args.model_dir,args.scorer,seed*997)
        c=m['config']
        if seed in c['train_seeds'] or args.steps!=c['steps']:raise ValueError('train/eval overlap or horizon drift')
        results['model_manifest']=m
        controls={'lancer':OrderedPolicy(scorer=LancerValues(fruitless_decay=.5)),
                  'guard':HarnessPolicy(max_age=c['max_age'])}
        from blue.core.baselines import SleepBaseline,MaskedRandomBaseline
        controls.update(sleep=SleepBaseline(),random=MaskedRandomBaseline(seed=seed))
        for name,control in controls.items():
            results['baselines'].setdefault(name,{})[str(seed)]=mg.run_team_episode(control,seed,args.steps,**mg.ENV_KW)
        mixture=[]
        for replica in range(args.policy_replicas):
            policy,m=load(args.model_dir,args.scorer,seed*997+replica)
            # Exactly zero residual: same sampling and feature path, no learned correction.
            with torch.no_grad():
                for parameter in policy.hook.actor.parameters():parameter.zero_()
            mixture.append(mg.run_team_episode(policy,seed,args.steps,hook=policy.hook,**mg.ENV_KW))
        results['baselines'].setdefault('untrained-mixture',{})[str(seed)]=mixture
        for execution in ('sample','greedy-diagnostic'):
            cells=[]
            for replica in range(args.policy_replicas if execution=='sample' else 1):
                policy,m=load(args.model_dir,args.scorer,seed*997+replica,execution)
                cell=mg.run_team_episode(policy,seed,args.steps,hook=policy.hook,**mg.ENV_KW)
                cells.append({**cell,'policy_seed':seed*997+replica,'requests':policy.request_digest.result(),'guard':policy.guard_stats()})
            results['cells'].setdefault(execution,{})[str(seed)]=cells
        (d/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k not in ('model_manifest','cells')},indent=2))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('train','eval'))
    p.add_argument('--algorithm',choices=('mappo','a2c'),default='mappo')
    p.add_argument('--scorer',default='blue/results/gpt_harness/legacy_candidate/scorer.pkl')
    p.add_argument('--out',required=True);p.add_argument('--model-dir')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--train-seeds',type=int,nargs='+',default=[7706,7707,7708,7709])
    p.add_argument('--eval-seeds',type=int,nargs='+',default=[8241,8229])
    for name,default in [('iters',2),('eps-per-iter',2),('steps',400),('hidden',64),('epochs',4),('critic-epochs',4),('minibatch',256),('policy-replicas',2),('stop-after',0),('threads',1)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('lr',3e-4),('critic-lr',3e-4),('bonus',.25),('temp',.5),('max-age',80.),('gamma',.99),('gae-lambda',1.),('explore',.2),('clip',.2),('entropy-coef',0.),('target-kl',.02),('max-grad-norm',1.),('value-scale',20.)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--ml-inputs',choices=('zero','risk','both'),default='risk')
    p.add_argument('--resume',action='store_true')
    return p


if __name__=='__main__':
    a=parser().parse_args()
    train(a) if a.mode=='train' else evaluate(a)
