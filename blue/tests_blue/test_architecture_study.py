import types
import numpy as np
import pytest
import torch
from blue.harness.architectures import StudyActor, WIDTH, marginal, observed_edges
from blue.training.architecture_study import actor_distribution, train_update
from blue.harness.redesign import CentralValue, CRITIC_DIM


@pytest.mark.parametrize('kind',['mlp','attention','gru','graph'])
def test_mask_and_host_permutation(kind):
    torch.manual_seed(5)
    actor=StudyActor(kind,32)
    torch.nn.init.normal_(actor.head.weight,std=.1)
    features=torch.randn(1,5,WIDTH);base=torch.randn(1,5);mask=torch.tensor([[1.,1.,1.,0.,0.]])
    adjacency=torch.eye(5)[None];adjacency[:,0,1]=1.;adjacency[:,1,0]=1.
    memory=torch.randn(1,32)
    out=actor(features,base,mask,adjacency,memory)
    order=torch.tensor([2,0,4,1,3])
    permuted=actor(features[:,order],base[:,order],mask[:,order],adjacency[:,order][:,:,order],memory)
    assert torch.allclose(permuted[:,[0,1,3]],out[:,order][:,[0,1,3]],atol=1e-6)
    poisoned=features.clone();poisoned[:,3:]=1e6
    assert torch.allclose(actor(poisoned,base,mask,adjacency,memory)[:,:3],out[:,:3],atol=1e-6)
    p=marginal(out,base,mask,torch.tensor([1]))
    assert torch.all(p[:,3:]==0) and torch.allclose(p.sum(-1),torch.ones(1))


def test_gru_causal_sequence_and_online_state_match_with_gradient():
    torch.manual_seed(1);actor=StudyActor('gru',32)
    summaries=torch.randn(7,2*WIDTH,requires_grad=True)
    states=actor.history_states(summaries)
    online=None
    for row in summaries:
        online=actor.memory_step(row,online)
    assert torch.allclose(online,states[-1],atol=1e-6)
    perturbed=summaries.detach().clone();perturbed[4:]+=100
    assert torch.allclose(actor.history_states(perturbed)[:4],states[:4],atol=1e-6)
    states[3].square().sum().backward()
    assert summaries.grad[0].abs().sum()>0
    assert summaries.grad[4:].abs().sum()==0
    assert actor.memory.weight_hh_l0.grad.abs().sum()>0


def test_visible_graph_uses_connections_and_no_hidden_state():
    class VisibleOnly:
        views={'blue_agent_0':{
            'a':{'Interface':[{'ip_address':'10.0.0.1'}],
                 'Processes':[{'Connections':[{'remote_address':'10.0.0.2'},
                                               {'remote_address':'192.168.0.5'}]}]},
            'b':{'Interface':[{'ip_address':'10.0.0.2'}]},
            'c':{'Interface':[{'ip_address':'10.0.0.3'}]}}}
        def __getattr__(self,key):raise AssertionError('privileged attribute access '+key)
    a=observed_edges(VisibleOnly(),'blue_agent_0',['a','b','c'])
    np.testing.assert_array_equal(a,[[1,1,0],[1,1,0],[0,0,1]])


def test_graph_has_gradients_through_nonself_messages():
    actor=StudyActor('graph',32);torch.nn.init.normal_(actor.head.weight,std=.2)
    f=torch.randn(3,WIDTH,requires_grad=True);b=torch.zeros(3);m=torch.ones(3)
    connected=torch.ones(3,3);self_edges=torch.eye(3)
    a=actor(f,b,m,connected);s=actor(f,b,m,self_edges)
    assert not torch.allclose(a,s)
    a.square().sum().backward()
    assert actor.messages[0].weight.grad.abs().sum()>0


def test_recurrent_ppo_reconstructs_behavior_and_updates_memory():
    torch.manual_seed(19);actor=StudyActor('gru',32);critic=CentralValue(64,20.)
    torch.nn.init.normal_(actor.head.weight,std=.2)
    history={('ep','a'):np.random.default_rng(4).normal(size=(5,2*WIDTH)).astype(np.float32)}
    config={'bonus':.25,'temperature':.5,'explore':.2,'epochs':2,'target_kl':.02,
            'ppo_clip':.2,'max_grad_norm':1.}
    rows=[]
    for i in range(3):
        rows.append({'feats':np.ones((3,WIDTH),np.float32)*(i+1), 'base':np.array([1.,.3,.1]),
            'mask':np.ones(3),'old_probs':np.ones(3)/3,'joint':np.ones(CRITIC_DIM,np.float32),
            'choice':i,'teacher':0,'advantage':float(i-1),'target':float(i),'duration':1,
            'history_key':('ep','a'),'tick':i+1,'adjacency':None})
    with torch.no_grad():
        p=actor_distribution(actor,rows,history,config).numpy()
    for row,probs in zip(rows,p):row['old_probs']=probs
    before=actor.memory.weight_hh_l0.detach().clone()
    result=train_update(actor,critic,torch.optim.Adam(actor.parameters(),lr=.001),
        torch.optim.Adam(critic.parameters(),lr=.001),rows,history,config)
    assert result['policy_steps']==2
    assert not torch.equal(before,actor.memory.weight_hh_l0)


def test_invalid_inputs_fail_closed():
    actor=StudyActor()
    with pytest.raises(ValueError,match='empty'):
        actor(torch.zeros(2,WIDTH),torch.zeros(2),torch.zeros(2))


@pytest.mark.parametrize(('kind','width'), [('mlp',512), ('attention',256)])
def test_larger_capacity_candidates_forward_finite_masked_scores(kind,width):
    actor=StudyActor(kind,width)
    features=torch.randn(2,19,WIDTH);base=torch.randn(2,19)
    mask=torch.ones(2,19);mask[0,-2:]=0
    scores=actor(features,base,mask)
    assert scores.shape==(2,19)
    assert torch.isfinite(scores).all()
