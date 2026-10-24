"""Decision-time credit and independent actor/critic optimization for v2.

Elapsed simulator ticks, including forced/busy/guarded work, remain in returns.
V(s) is a baseline, never an action selector. A2C uses one full-batch policy
step on fresh data; PPO uses clipped ratios on that same marginal distribution.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as fn

from blue.harness.redesign import probabilities


def decision_credit(ticks, rewards, values, gamma=.99, gae_lambda=1.):
    """Finite-task returns/GAE at one agent's learned decision times.

    The whole configured task horizon is terminal, not a rollout cutoff. A
    partial rollout would require explicit bootstrap state, which this API
    deliberately does not accept. lambda=1 gives exact discounted MC - V.
    """
    t = np.asarray(ticks, dtype=int); R = np.asarray(rewards, dtype=float)
    V = np.asarray(values, dtype=float)
    if not np.isfinite([gamma, gae_lambda]).all() or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError('invalid credit settings')
    if len(t) != len(V) or not np.isfinite(R).all() or not np.isfinite(V).all():
        raise ValueError('invalid credit values')
    if len(t) and (t[0] < 0 or t[-1] >= len(R) or (np.diff(t) <= 0).any()):
        raise ValueError('decision ticks must increase within task horizon')
    adv = np.zeros(len(t)); targets = np.zeros(len(t)); durations = np.zeros(len(t), dtype=int)
    carry = 0.
    for i in range(len(t)-1, -1, -1):
        end = t[i+1] if i+1 < len(t) else len(R)
        dt = end-t[i]; durations[i] = dt
        reward = np.sum(R[t[i]:end] * gamma**np.arange(dt))
        next_v = V[i+1] if i+1 < len(t) else 0.
        delta = reward + gamma**dt*next_v-V[i]
        carry = delta + (gamma*gae_lambda)**dt*carry
        adv[i] = carry; targets[i] = carry+V[i]
    return adv, targets, durations


def tensors(rows):
    if not rows:
        raise ValueError('empty fresh rollout')
    cap = max(len(r['mask']) for r in rows)
    def pad(key, dims):
        return torch.from_numpy(np.stack([np.pad(np.asarray(r[key], dtype=np.float32),
            ((0, cap-len(r['mask'])), (0,0)) if dims == 2 else (0, cap-len(r['mask']))) for r in rows]))
    return {'F': pad('feats',2), 'B': pad('base',1), 'M':pad('mask',1), 'P':pad('old_probs',1),
            'J':torch.tensor(np.stack([r['joint'] for r in rows]),dtype=torch.float32),
            'CH':torch.tensor([r['choice'] for r in rows]),
            'teacher':torch.tensor([r['teacher'] for r in rows]),
            'ADV':torch.tensor([r['advantage'] for r in rows],dtype=torch.float32),
            'RET':torch.tensor([r['target'] for r in rows],dtype=torch.float32)}


def update(actor, critic, actor_opt, critic_opt, rows, algorithm='mappo', bonus=.25,
           temperature=.5, explore=.2, epochs=4, minibatch=256, clip=.2,
           entropy_coef=0., target_kl=.02, max_grad_norm=1., critic_epochs=4):
    if algorithm not in ('mappo','a2c'):
        raise ValueError('unsupported learner')
    if epochs < 1 or critic_epochs < 1 or minibatch < 1 or not np.isfinite(
        [clip,entropy_coef,target_kl,max_grad_norm]).all() or clip <= 0 or entropy_coef < 0 or target_kl <= 0 or max_grad_norm <= 0:
        raise ValueError('invalid optimizer settings')
    x=tensors(rows); n=len(rows)
    if not all(torch.isfinite(v).all() for v in x.values()):
        raise ValueError('nonfinite fresh rollout')
    def distribution(ix):
        return probabilities(actor,x['F'][ix],x['B'][ix],x['M'][ix],x['teacher'][ix],
                             bonus,temperature,explore)
    log_old=x['P'].clamp(min=1e-12).log()
    with torch.no_grad():
        initial=distribution(np.arange(n))
        if not torch.allclose(initial,x['P'],atol=1e-6,rtol=1e-5):
            raise ValueError('behavior policy drift before update')
    adv=x['ADV']; sd=adv.std(unbiased=False)
    adv=(adv-adv.mean())/sd.clamp(min=1e-6) if n>1 and sd>1e-6 else torch.zeros_like(adv)
    count=0; norms=[]; stop=False
    for _ in range(epochs if algorithm=='mappo' else 1):
        order=np.random.permutation(n) if algorithm=='mappo' else np.arange(n)
        batch=minibatch if algorithm=='mappo' else n
        for start in range(0,n,batch):
            ix=order[start:start+batch]; p=distribution(ix); logp=p.clamp(min=1e-12).log()
            kl=(x['P'][ix]*(log_old[ix]-logp)).sum(-1).mean()
            if algorithm=='mappo' and float(kl.detach())>target_kl:
                stop=True;break
            chosen=logp.gather(1,x['CH'][ix,None]).squeeze(1)
            if algorithm=='mappo':
                old_chosen=log_old[ix].gather(1,x['CH'][ix,None]).squeeze(1)
                ratio=(chosen-old_chosen).exp()
                policy=-torch.minimum(ratio*adv[ix],ratio.clamp(1-clip,1+clip)*adv[ix]).mean()
            else:
                policy=-(chosen*adv[ix]).mean()
            entropy=-(p*logp).sum(-1).mean()
            actor_opt.zero_grad();(policy-entropy_coef*entropy).backward()
            norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),max_grad_norm)
            if not torch.isfinite(norm):raise ValueError('nonfinite actor gradient')
            norms.append(float(norm));actor_opt.step();count+=1
        if stop:break
    value_norms=[];value_loss=0.
    for _ in range(critic_epochs):
        order=np.random.permutation(n)
        for start in range(0,n,minibatch):
            ix=order[start:start+minibatch]
            prediction=critic(x['J'][ix])
            value_loss=fn.smooth_l1_loss(prediction/critic.value_scale,x['RET'][ix]/critic.value_scale)
            critic_opt.zero_grad();value_loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),max_grad_norm)
            if not torch.isfinite(norm):raise ValueError('nonfinite critic gradient')
            value_norms.append(float(norm));critic_opt.step()
    with torch.no_grad():
        p=distribution(np.arange(n));logp=p.clamp(min=1e-12).log()
        exact_kl=float((x['P']*(log_old-logp)).sum(-1).mean())
        oldchosen=log_old.gather(1,x['CH'][:,None]).squeeze(1)
        ratio=(logp.gather(1,x['CH'][:,None]).squeeze(1)-oldchosen).exp()
        errors=x['RET']-critic(x['J'])
        variance=float(x['RET'].var(unbiased=False))
        explained=1-float(errors.var(unbiased=False))/variance if variance>1e-8 else None
    stats={'policy_steps':count,'kl_stopped':stop,'kl_exact':max(exact_kl,0.),
           'clip_fraction':float(((ratio-1).abs()>clip).float().mean()),
           'entropy':float(-(p*logp).sum(-1).mean()),
           'actor_gradient_norm_mean':float(np.mean(norms)) if norms else 0.,
           'critic_gradient_norm_mean':float(np.mean(value_norms)),
           'value_loss_scaled':float(value_loss.detach()),'explained_variance':explained,
           'target_mean':float(x['RET'].mean()),'target_std':float(x['RET'].std(unbiased=False)),
           'mean_decision_duration':float(np.mean([r['duration'] for r in rows]))}
    if not np.isfinite([v for v in stats.values() if v is not None]).all():
        raise ValueError('nonfinite update diagnostics')
    return stats
