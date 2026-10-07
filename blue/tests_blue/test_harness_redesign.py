"""Mechanism checks: variable-time credit, actual behavior likelihood, no leakage."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from blue.harness.redesign import SetActor,CentralValue,EvidenceMemory,FEATURE_NAMES,CRITIC_DIM,probabilities,central_context
from blue.harness.learning import decision_credit,update
from blue.harness.calibration import grouped_weights
from blue.harness.features import NAMES,ACTOR_NAMES


def test_elapsed_rewards_include_busy_and_forced_intervals_and_mc_limit():
    rewards=np.array([1.,-2.,3.,-4.,5.,6.,-7.,8.])
    ticks=[0,3,7];values=[.2,.3,-.1]
    adv,target,dt=decision_credit(ticks,rewards,values,gamma=.9,gae_lambda=1.)
    np.testing.assert_array_equal(dt,[3,4,1])
    expected=[sum(rewards[t:]*.9**np.arange(len(rewards)-t)) for t in ticks]
    np.testing.assert_allclose(target,expected)
    np.testing.assert_allclose(adv,target-values)
    with pytest.raises(ValueError,match='increase'):decision_credit([3,0],rewards,[0.,0.])


def test_variable_time_one_step_bootstrap_and_terminal_zero():
    adv,target,dt=decision_credit([0,3],[1,2,4,8,16],[10.,20.],gamma=.5,gae_lambda=0.)
    assert target[0]==pytest.approx(1+.5*2+.25*4+.5**3*20)
    assert target[1]==pytest.approx(8+.5*16)


def test_set_ranking_permutation_padding_and_context():
    torch.manual_seed(1);actor=SetActor(hidden=8)
    torch.nn.init.normal_(actor.head.weight,std=.1)
    F=torch.randn(4,len(FEATURE_NAMES));B=torch.randn(4);M=torch.ones(4)
    perm=torch.tensor([2,0,3,1])
    torch.testing.assert_close(actor(F[perm],B[perm],M),actor(F,B,M)[perm])
    padded=torch.cat((F,torch.full((2,len(FEATURE_NAMES)),100.)))
    scores=actor(padded,torch.cat((B,torch.tensor([100.,100.]))),torch.tensor([1,1,1,1,0,0]))
    torch.testing.assert_close(scores[:4],actor(F,B,M))
    changed=F.clone();changed[1]+=5
    assert not torch.allclose(actor(F,B,M)[0],actor(changed,B,M)[0])


def test_marginal_mixture_masks_teacher_probability_and_no_hidden_rng():
    torch.manual_seed(2);actor=SetActor(8);F=torch.randn(4,len(FEATURE_NAMES));B=torch.zeros(4);mask=torch.tensor([1,1,0,1])
    before=torch.get_rng_state().clone()
    p=probabilities(actor,F,B,mask,teacher=1,explore=.2)
    assert p[2]==0 and float(p.sum().detach())==pytest.approx(1.)
    assert p[1]>=.8
    torch.testing.assert_close(p,torch.tensor([.2/3,.8+.2/3,0.,.2/3]))
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(ValueError,match='masked teacher'):probabilities(actor,F,B,mask,2)


def test_evidence_survives_busy_ticks_and_clears_only_after_completed_analysis():
    host='host';env=SimpleNamespace(_tick=0,episode_limit=400,hostnames={'a':[host]},
        trackers={'a':SimpleNamespace(last_analysis={host:None})})
    raw=np.zeros((1,len(NAMES)));memory=EvidenceMemory();memory.rows(env,'a',raw)
    env._tick=1;raw[0,NAMES.index('delta_unknown_files')]=1
    first=memory.rows(env,'a',raw)
    env._tick=5;raw[:]=0
    later=memory.rows(env,'a',raw)
    assert first[0,0]==later[0,0]==1
    assert later[0,4]==pytest.approx(4/400)
    env._tick=6;env.trackers['a'].last_analysis[host]=6
    assert memory.rows(env,'a',raw)[0,0]==0
    env._tick=0
    with pytest.raises(ValueError,match='reset'):memory.rows(env,'a',raw)
    memory.reset();assert memory.rows(env,'a',raw)[0,0]==0


def test_host_episode_weights_prevent_longer_host_traces_dominating():
    episode=[1,1,1,1,2,2];agent=['a']*6;hosts=['x','x','x','y','z','z']
    w=grouped_weights(episode,agent,hosts)
    assert sum(w[:4])==pytest.approx(sum(w[4:]))
    assert sum(w[:3])==pytest.approx(w[3])
    assert w.mean()==pytest.approx(1.)


def fresh_rows(actor,count=8):
    rng=np.random.default_rng(4);rows=[]
    for i in range(count):
        F=rng.normal(size=(3,len(FEATURE_NAMES))).astype(np.float32);B=np.array([.2,.4,.1],dtype=np.float32);mask=np.ones(3,dtype=np.float32)
        with torch.no_grad():p=probabilities(actor,torch.from_numpy(F),torch.from_numpy(B),torch.from_numpy(mask),teacher=1).numpy()
        rows.append({'feats':F,'base':B,'mask':mask,'old_probs':p,'joint':rng.normal(size=CRITIC_DIM).astype(np.float32),
                     'choice':i%3,'teacher':1,'advantage':float(i-3),'target':float(100+i),'duration':2})
    return rows


@pytest.mark.parametrize('algorithm',['mappo','a2c'])
def test_both_learners_update_actor_with_matching_behavior_and_independent_gradients(algorithm):
    torch.manual_seed(5);actor=SetActor(8);critic=CentralValue(8)
    rows=fresh_rows(actor);initial=deepcopy(actor.state_dict())
    ao=torch.optim.Adam(actor.parameters(),lr=.001);co=torch.optim.Adam(critic.parameters(),lr=.001)
    s=update(actor,critic,ao,co,rows,algorithm=algorithm,epochs=2,minibatch=4,critic_epochs=1)
    assert any(not torch.equal(initial[k],v) for k,v in actor.state_dict().items())
    assert s['kl_exact']>=0 and np.isfinite(s['actor_gradient_norm_mean'])
    assert s['policy_steps']==(4 if algorithm=='mappo' else 1)


def test_critic_magnitude_cannot_shrink_actor_update():
    torch.manual_seed(6);original=SetActor(8);rows=fresh_rows(original)
    actors=[deepcopy(original),deepcopy(original)]
    critics=[CentralValue(8),CentralValue(8)]
    with torch.no_grad():critics[1].net[-1].bias.fill_(1e4)
    for a,c in zip(actors,critics):
        np.random.seed(2)
        update(a,c,torch.optim.Adam(a.parameters(),lr=.001),torch.optim.Adam(c.parameters(),lr=.001),rows,
            algorithm='a2c',critic_epochs=1)
    for k,v in actors[0].state_dict().items():torch.testing.assert_close(v,actors[1].state_dict()[k],rtol=0,atol=0)


def test_behavior_drift_rejected_and_busy_central_state_not_zeroed():
    actor=SetActor(8);critic=CentralValue(8);rows=fresh_rows(actor)
    rows[0]['old_probs']=np.array([.2,.7,.1],dtype=np.float32)
    with pytest.raises(ValueError,match='behavior policy drift'):
        update(actor,critic,torch.optim.Adam(actor.parameters()),torch.optim.Adam(critic.parameters()),rows)
    from blue.core.wrapper import BLUE_AGENTS
    class Env:
        _tick=10;episode_limit=400;_awaiting={a:('h','Analyse') for a in BLUE_AGENTS}
        @property
        def state(self):raise AssertionError('privileged read')
    class Scorer:
        bundle={'mean':np.zeros(len(NAMES)),'scale':np.ones(len(NAMES))}
        def observe(self,env,agent):return np.ones((2,len(NAMES)),dtype=np.float32)
    c=central_context(Env(),Scorer(),BLUE_AGENTS[0])
    assert c.shape==(CRITIC_DIM,) and c[0]==1


def test_ranking_resolution_handles_numerical_ties_without_hiding_real_gaps():
    from blue.harness.numerics import stable_pick
    assert stable_pick([(1.1106417179107666,'host_1'),(1.110641710460186,'host_3')]) == 'host_3'
    assert stable_pick([(1.1106417179107666,'host_1'),(1.1106417179107666,'host_3')]) == 'host_3'
    assert stable_pick([(1.1,'host_1'),(1.09999,'host_3')]) == 'host_1'
    with pytest.raises(ValueError):stable_pick([(float('nan'),'host_1')])
