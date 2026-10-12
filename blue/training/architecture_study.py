"""Frozen full-horizon architecture screen; native reward, fresh MAPPO data.

All arms share scorer, critic, protected rules, optimizer recipe, episode
seeds and budget. Training samples from the on-policy mixture; evaluation
records greedy ranking and sampled execution separately. GRU uses complete
causal sequences in every PPO epoch. Graph/self-graph are parameter-matched
controls. All model artifacts are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
from blue.core.baselines import MaskedRandomBaseline, SleepBaseline
from blue.harness.architectures import StudyActor, marginal, observed_edges
from blue.harness.learning import decision_credit, tensors
from blue.harness.numerics import stable_pick
from blue.harness.policy import HarnessPolicy
from blue.harness.redesign import CentralValue, RedesignHook, central_context, FEATURE_NAMES
from blue.harness.scoring import MLScorer, sha256
from blue.policies.ordered import LancerValues, OrderedPolicy, argmax_pick
from blue.training.harness_redesign import runtime
from blue.training.harness_rl import check_seeds
from blue.training.mappo_guide import ENV_KW

ARMS = {'mlp64': ('mlp',64), 'mlp128': ('mlp',128), 'mlp256': ('mlp',256),
        'mlp512': ('mlp',512), 'attention128': ('attention',128),
        'attention256': ('attention',256), 'gru128': ('gru',128),
        'graph128': ('graph',128), 'graph_self128': ('graph',128)}
SOURCES = ['blue/harness/architectures.py','blue/training/architecture_study.py',
           'blue/harness/redesign.py','blue/harness/learning.py','blue/harness/policy.py',
           'blue/harness/features.py','blue/harness/scoring.py','blue/core/wrapper.py',
           'blue/core/masking.py','blue/core/obs_features.py','blue/policies/ordered.py',
           'blue/training/architecture_preflight.py','blue/training/architecture_aggregate.py',
           'ops/kaggle/architecture/runner_template.py']


class StudyHook(RedesignHook):
    def __init__(self, actor, critic, scorer, config, policy_seed, self_graph=False,
                 execution='sample'):
        self.critic, self.config, self.self_graph = critic, config, self_graph
        super().__init__(actor, scorer, config['bonus'], 'risk', config['temperature'],
                         config['explore'], policy_seed, execution)

    def reset(self):
        super().reset()
        self.histories, self.hidden_states, self.seen = {}, {}, {}
        self.rows, self.cache, self.per_agent = [], {}, {}
        self.capturing = False
        self.graph_edges = 0

    def observe(self, env, agent):
        super().observe(env, agent)
        if self.actor.kind != 'gru' or self.capturing or self.seen.get(agent) == env._tick:
            return
        self.capturing = True
        try:
            full = super().features(env, agent, env.hostnames[agent])
        finally:
            self.capturing = False
        summary = np.concatenate((full.mean(0), full.max(0))).astype(np.float32)
        history = self.histories.setdefault(agent, [])
        if len(history) != env._tick:
            raise ValueError('recurrent history omitted a simulator tick')
        history.append(summary)
        with torch.no_grad():
            self.hidden_states[agent] = self.actor.memory_step(torch.from_numpy(summary),
                                                              self.hidden_states.get(agent))
        self.seen[agent] = env._tick

    def __call__(self, env, agent, cands, scored):
        with torch.no_grad():
            F = torch.from_numpy(self.features(env,agent,cands))
            B = torch.tensor([s for s,_ in scored],dtype=torch.float32)
            M = torch.ones(len(cands)); teacher = cands.index(argmax_pick(scored))
            A = None
            if self.actor.kind == 'graph':
                A = torch.from_numpy(observed_edges(env,agent,cands))
                if self.self_graph: A = torch.eye(len(cands))
                self.graph_edges += int(A.sum())-len(cands)
            logits = self.actor(F,B,M,adjacency=A,memory=self.hidden_states.get(agent))
            p = marginal(logits,B,M,teacher,self.bonus,self.temperature,self.explore)
            if self.execution == 'sample':
                idx = int(torch.multinomial(p,1,generator=self.generator))
            else:
                residual = torch.tanh(logits)
                selected = stable_pick([(s + self.bonus * float(residual[i]), h)
                                        for i, (s, h) in enumerate(scored)])
                idx = cands.index(selected)
            J = central_context(env,self.scorer,agent,self.cache)
            value = float(self.critic(torch.from_numpy(J)))
        self.per_agent[agent] = self.per_agent.get(agent,0)+1
        self.rows.append({'feats':F.numpy(),'base':B.numpy(),'mask':M.numpy(),
            'old_probs':p.numpy(),'joint':J,'choice':idx,'teacher':teacher,'tick':env._tick,
            'agent':agent,'value':value,'adjacency':None if A is None else A.numpy()})
        return cands[idx]


def episode(policy, seed, config, hook=None):
    env = CC4MARLEnv(seed=seed,steps=config['steps'],**config['env_kw'])
    env.reset(seed=seed); policy.reset()
    rewards=[]; analyses=set(); confirmations=0; repairs=0
    previous={a:(dict(env.trackers[a].last_analysis),dict(env.trackers[a].last_remediation)) for a in BLUE_AGENTS}
    import time
    start=time.monotonic()
    while True:
        actions={a:int(policy.select(env,a)) for a in BLUE_AGENTS}
        _,rew,term,trunc,_=env.step(actions);rewards.append(float(rew[0]))
        for a in BLUE_AGENTS:
            tracker=env.trackers[a]; old_an,old_rem=previous[a]
            for h in env.hostnames[a]:
                if tracker.last_analysis.get(h)!=old_an.get(h):
                    analyses.add((a,h))
                    confirmations+=int(tracker.state[h]=='CONFIRMED')
                repairs+=int(tracker.last_remediation.get(h)!=old_rem.get(h))
            previous[a]=(dict(tracker.last_analysis),dict(tracker.last_remediation))
            if hook is not None: hook.observe(env,a)
        if term or trunc:break
    if hook is not None:
        for a in BLUE_AGENTS:
            rows=[r for r in hook.rows if r['agent']==a]
            adv,targets,durations=decision_credit([r['tick'] for r in rows],rewards,
                [r['value'] for r in rows],config['gamma'],1.)
            for r,d,t,dt in zip(rows,adv,targets,durations):
                r.update(advantage=float(d),target=float(t),duration=int(dt))
    result={'seed':seed,'return':sum(rewards),'ticks':len(rewards),
        'completed_analysis_coverage':len(analyses)/sum(len(env.hostnames[a]) for a in BLUE_AGENTS),
        'completed_remediations':repairs,'analysis_confirmations':confirmations,
        'wall_seconds':time.monotonic()-start}
    if isinstance(policy,HarnessPolicy):result['requests']=policy.request_digest.result()
    if hook is not None:
        result['learned_decisions']=len(hook.rows);result['visible_graph_nonself_edges']=hook.graph_edges
    env.close();return result


def actor_distribution(actor, rows, histories, config):
    x=tensors(rows); memory=None;adjacency=None
    if actor.kind=='gru':
        states={key:actor.history_states(torch.tensor(np.stack(sequence),dtype=torch.float32))
                for key,sequence in histories.items()}
        memory=torch.stack([states[r['history_key']][r['tick']] for r in rows])
    if actor.kind=='graph':
        cap=x['M'].shape[1]
        adjacency=torch.tensor(np.stack([np.pad(r['adjacency'],((0,cap-len(r['mask'])),)*2)
                                         for r in rows]),dtype=torch.float32)
    logits=actor(x['F'],x['B'],x['M'],adjacency=adjacency,memory=memory)
    return marginal(logits,x['B'],x['M'],x['teacher'],config['bonus'],config['temperature'],config['explore'])


def train_update(actor, critic, actor_opt, critic_opt, rows, histories, config):
    x=tensors(rows);old=x['P'].clamp(min=1e-12).log()
    with torch.no_grad():
        if not torch.allclose(actor_distribution(actor,rows,histories,config),x['P'],atol=1e-6,rtol=1e-5):
            raise ValueError('replayed behavior mismatch, including recurrent state')
    adv=x['ADV'];adv=(adv-adv.mean())/adv.std(unbiased=False).clamp(min=1e-6)
    count=0;norms=[]
    for _ in range(config['epochs']):
        p=actor_distribution(actor,rows,histories,config);logp=p.clamp(min=1e-12).log()
        kl=(x['P']*(old-logp)).sum(-1).mean()
        if float(kl.detach())>config['target_kl']:break
        ratio=(logp.gather(1,x['CH'][:,None])-old.gather(1,x['CH'][:,None])).squeeze(-1).exp()
        clip=config['ppo_clip']
        loss=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
        actor_opt.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),config['max_grad_norm'])
        if not torch.isfinite(norm):raise ValueError('nonfinite actor update')
        norms.append(float(norm));actor_opt.step();count+=1
    for _ in range(config['epochs']):
        loss=torch.nn.functional.smooth_l1_loss(critic(x['J'])/critic.value_scale,x['RET']/critic.value_scale)
        critic_opt.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),config['max_grad_norm'])
        if not torch.isfinite(norm):raise ValueError('nonfinite critic update')
        critic_opt.step()
    with torch.no_grad():
        p=actor_distribution(actor,rows,histories,config)
        kl=float((x['P']*(old-p.clamp(min=1e-12).log())).sum(-1).mean())
    return {'policy_steps':count,'kl_exact':kl,'actor_gradient_norm_mean':float(np.mean(norms)) if norms else 0.}


def freeze(config, scorer_path):
    source={p:sha256(p) for p in SOURCES}
    simulator={str(p):sha256(p) for p in sorted(Path('cage-challenge-4/CybORG').rglob('*.py'))}
    return {'config':config,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_files':source,'simulator_sha256':hashlib.sha256(json.dumps(simulator,sort_keys=True).encode()).hexdigest(),
        'runtime':runtime(),'scorer_sha256':sha256(scorer_path),'features':list(FEATURE_NAMES),
        'tests':{'capacity':['mlp64','mlp128','mlp256'],'host_comparison':['mlp128','attention128'],
                 'memory':['mlp128','gru128'],'topology':['graph_self128','graph128']},
        'claim':f"Longer development architecture comparison; {len(config['training_rng_cohort'])} independent training RNG seeds; reused evaluation cohort, not held-out performance evidence",
        'uncertainty':'Average policy replicas within episode, then episodes within training seed; paired training-seed t intervals',
        'graph':'Visible local process connections only, symmetric; no hidden simulator topology',
        'limitations':['Original scorer held fixed in locked legacy runtime','Training environment seeds are a previously used development pool',
                       'Evaluation seeds are reused development episodes and support no final/held-out claim',
                       'Training episode seeds 7730-7800 were clear in the structured seed audit; reserve them for this study',
                       'Full-batch PPO common to all arms; not directly comparable to previous minibatch runs',
                       'Sampled and greedy execution answer different deployment questions and are reported separately'],
        'success':'Report all models and tails; superiority needs independent replication and positive interval against both strong controls'}


def run(args):
    torch.set_num_threads(1)
    config={'steps':args.steps,'iterations':args.iters,'eps_per_iter':args.eps_per_iter,
        'training_rng':args.training_rng,'training_rng_cohort':args.cohort_seeds,
        'train_seeds':args.train_seeds,
        'eval_seeds':args.eval_seeds,'policy_replicas':args.policy_replicas,
        'arms':args.arms,
        'evaluation_execution_modes':['greedy-diagnostic','sample'],
        'epochs':4,'target_kl':.02,'ppo_clip':.2,'max_grad_norm':1.,'entropy_coef':0.,
        'gamma':.99,'bonus':.25,'temperature':.5,'explore':.2,'max_age':80.,
        'actor_lr':3e-4,'critic_lr':3e-4,'critic_hidden':64,'value_scale':20.,
        'env_kw':{**ENV_KW,'red_agent':args.red_agent}}
    if args.cohort_sha256:
        config['cohort_sha256']=args.cohort_sha256
    check_seeds(config['train_seeds']);check_seeds(config['eval_seeds'])
    if set(config['train_seeds'])&set(config['eval_seeds']):raise ValueError('train/evaluation overlap')
    if len(set(args.training_rng))!=len(args.training_rng):raise ValueError('duplicate training RNG seed')
    if not args.arms or len(set(args.arms)) != len(args.arms) or not set(args.arms).issubset(ARMS):raise ValueError('invalid architecture shard')
    if not set(args.training_rng).issubset(args.cohort_seeds):raise ValueError('shard RNG outside frozen cohort')
    if len(set(args.eval_seeds))!=len(args.eval_seeds):raise ValueError('duplicate evaluation episode seed')
    if len(set(args.train_seeds))!=len(args.train_seeds):raise ValueError('duplicate training episode seed')
    if len(set(args.cohort_seeds))!=len(args.cohort_seeds):raise ValueError('duplicate cohort RNG seed')
    if min(args.steps,args.iters,args.eps_per_iter,args.policy_replicas)<1:raise ValueError('empty budget')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    plan=freeze(config,args.scorer)
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise ValueError('frozen study drift')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result_path=out/'results.json'
    results=json.loads(result_path.read_text()) if result_path.exists() else {'claim':plan['claim'],'baselines':{},'models':{},'untrained':{}}
    def save():result_path.write_text(json.dumps(results,indent=2)+'\n')
    for seed in args.eval_seeds:
        for name,control in [('lancer',OrderedPolicy(scorer=LancerValues(fruitless_decay=.5))),
            ('guard',HarnessPolicy(max_age=80)),('sleep',SleepBaseline()),('random',MaskedRandomBaseline(seed=seed))]:
            key=f'{name}/{seed}'
            if key not in results['baselines']:
                results['baselines'][key]=episode(control,seed,config);save()
    # Zero output heads give exactly the same untrained marginal in every arm.
    # Cache by configuration and scorer hash in the frozen plan, not across studies.
    results.setdefault('untrained', {})
    for seed in args.eval_seeds:
        modes = {'sample': range(args.policy_replicas), 'greedy-diagnostic': range(1)}
        for execution, replicas in modes.items():
            results['untrained'].setdefault(execution, {})
            for rep in replicas:
                key=f'{seed}/{rep}'
                if key not in results['untrained'][execution]:
                    hook=StudyHook(StudyActor('mlp',64),CentralValue(config['critic_hidden'],config['value_scale']),MLScorer(args.scorer),config,seed*997+rep,execution=execution)
                    results['untrained'][execution][key]=episode(HarnessPolicy(hook,80,record_requests=True),seed,config,hook)
                    save()
    for arm in config['arms']:
        kind,hidden=ARMS[arm]
        for rng in args.training_rng:
            key=f'{arm}/s{rng}';d=out/arm/f's{rng}';d.mkdir(parents=True,exist_ok=True)
            if key in results['models']:continue
            torch.manual_seed(rng);np.random.seed(rng);random.seed(rng)
            actor=StudyActor(kind,hidden)
            # Keep the central value initializer matched across architectures.
            # Actor widths otherwise consume a different number of torch draws.
            torch.manual_seed(1_000_003+rng)
            critic=CentralValue(config['critic_hidden'],config['value_scale'])
            scorer=MLScorer(args.scorer)
            aopt=torch.optim.Adam(actor.parameters(),lr=config['actor_lr'])
            vopt=torch.optim.Adam(critic.parameters(),lr=config['critic_lr']);hist=[]
            if (d/'checkpoint.pt').exists():
                ck=torch.load(d/'checkpoint.pt',weights_only=False)
                actor.load_state_dict(ck['actor']);critic.load_state_dict(ck['critic'])
                aopt.load_state_dict(ck['actor_optimizer']);vopt.load_state_dict(ck['critic_optimizer'])
                torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['random_rng']);hist=ck['history']
            untrained=results['untrained']
            for iteration in range(len(hist),args.iters):
                rows=[];histories={};episodes=[]
                for e in range(args.eps_per_iter):
                    seed=config['train_seeds'][(iteration*args.eps_per_iter+e)%len(config['train_seeds'])]
                    hook=StudyHook(actor,critic,scorer,config,rng*100003+iteration*1009+e+1,arm=='graph_self128')
                    episodes.append(episode(HarnessPolicy(hook,80,record_requests=True),seed,config,hook))
                    for agent,sequence in hook.histories.items():histories[(e,agent)]=sequence
                    for row in hook.rows:row['history_key']=(e,row['agent'])
                    rows.extend(hook.rows)
                stats=train_update(actor,critic,aopt,vopt,rows,histories,config)
                hist.append({'iteration':iteration+1,'episodes':episodes,**stats})
                torch.save({'actor':actor.state_dict(),'critic':critic.state_dict(),
                    'actor_optimizer':aopt.state_dict(),'critic_optimizer':vopt.state_dict(),
                    'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'random_rng':random.getstate(),
                    'history':hist},d/'checkpoint.tmp')
                os.replace(d/'checkpoint.tmp',d/'checkpoint.pt')
                print(json.dumps({'model':key,'iteration':iteration+1,'mean_training_return':np.mean([r['return'] for r in episodes]),**stats}),flush=True)
            cells={'sample':{},'greedy-diagnostic':{}}
            modes = {'sample': range(args.policy_replicas), 'greedy-diagnostic': range(1)}
            for execution, replicas in modes.items():
                for seed in args.eval_seeds:
                    for rep in replicas:
                        hook=StudyHook(actor,critic,scorer,config,seed*997+rep,arm=='graph_self128',execution)
                        cells[execution][f'{seed}/{rep}']=episode(HarnessPolicy(hook,80,record_requests=True),seed,config,hook)
            results['models'][key]={'parameters':sum(p.numel() for p in actor.parameters()),
                'checkpoint_sha256':sha256(d/'checkpoint.pt'),'training':hist,'cells':cells,'untrained':untrained}
            save();print('MODEL COMPLETE '+key,flush=True)
    analyze(results,config,out)


def analyze(results,config,out):
    from scipy.stats import t
    estimates={}
    def summarize(values):
        values=np.asarray(values,dtype=float);n=len(values)
        se=values.std(ddof=1)/n**.5 if n>=3 else None
        return {'training_seed_differences':values.tolist(),'mean':float(values.mean()),'n_training_seeds':n,
                't95':None if n<3 else [float(values.mean()-t.ppf(.975,n-1)*se),float(values.mean()+t.ppf(.975,n-1)*se)]}
    def per_seed(cells):
        grouped={}
        for key,cell in cells.items():
            seed=key.split('/')[0]
            grouped.setdefault(seed,[]).append(cell['return'])
        return {seed:float(np.mean(values)) for seed,values in grouped.items()}
    for execution in ('greedy-diagnostic','sample'):
        controls=('lancer','guard','untrained') + (('mlp64',) if 'mlp64' in config['arms'] else ())
        for arm in config['arms']:
            for control in controls:
                ds=[]
                for rng in config['training_rng']:
                    model=results['models'][f'{arm}/s{rng}']
                    arm_means=per_seed(model['cells'][execution])
                    differences=[]
                    for seed,mean in arm_means.items():
                        if control=='untrained':baseline=model['untrained'][execution][f'{seed}/0']['return']
                        elif control=='mlp64':baseline=per_seed(results['models'][f'mlp64/s{rng}']['cells'][execution])[seed]
                        else:baseline=results['baselines'][f'{control}/{seed}']['return']
                        differences.append(mean-baseline)
                    ds.append(np.mean(differences))
                estimates[f'{execution}:{arm}-minus-{control}']=summarize(ds)
        if {'graph128','graph_self128'}.issubset(config['arms']):
            ds=[np.mean([c['return']-results['models'][f'graph_self128/s{rng}']['cells'][execution][key]['return']
                for key,c in results['models'][f'graph128/s{rng}']['cells'][execution].items()]) for rng in config['training_rng']]
            estimates[f'{execution}:graph128-minus-graph_self128']=summarize(ds)
    (out/'analysis.json').write_text(json.dumps({
        'claim':results['claim'],
        'analysis_source_sha256':sha256(__file__),
        'frozen_training_plan_sha256':sha256(out/'plan.json'),
        'execution_modes':config['evaluation_execution_modes'],
        'interval_policy':'No inferential interval for fewer than three training seeds. Evaluation episodes are shared, not independent training replications.',
        'contrasts':estimates},indent=2)+'\n')
    print('STUDY COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scorer',required=True);p.add_argument('--out',required=True)
    p.add_argument('--iters',type=int,default=2);p.add_argument('--eps-per-iter',type=int,default=2)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--training-rng',type=int,nargs='+',default=[0,1])
    p.add_argument('--cohort-seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--cohort-sha256',default='')
    p.add_argument('--arms',choices=tuple(ARMS),nargs='+',default=list(ARMS))
    p.add_argument('--train-seeds',type=int,nargs='+',default=list(range(7730,7801)))
    p.add_argument('--eval-seeds',type=int,nargs='+',default=list(range(8221,8253)))
    p.add_argument('--policy-replicas',type=int,default=2)
    p.add_argument('--red-agent',choices=('discovery','finite'),default='finite')
    run(p.parse_args())
