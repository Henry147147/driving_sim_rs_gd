import copy
import numpy as np
import pytest
import torch
from altd_rl.rollout import gae,Collector
from altd_rl.policy import ActorCritic
from altd_rl.vec_env import VecEnv,load_config
from altd_rl.ppo import update
from altd_rl import checkpoint

def test_gae_terminal_truncated_and_reset():
    r=torch.tensor([[1.,1.,1.]]);v=torch.tensor([[2.,2.,2.]]);nv=torch.tensor([[10.,10.,10.]])
    term=torch.tensor([[False,False,True]]);trunc=torch.tensor([[False,True,False]])
    a,_=gae(r,v,nv,term,trunc,.9,.95);torch.testing.assert_close(a,torch.tensor([[8.,8.,-1.]]))
    a,_=gae(torch.ones(2,1),torch.zeros(2,1),torch.zeros(2,1),torch.tensor([[False],[False]]),torch.tensor([[True],[False]]))
    torch.testing.assert_close(a,torch.ones(2,1))

def test_ratio_one_and_saturated_gradients():
    p=ActorCritic(32);o=torch.randn(16,32);_,u,lp,_=p.action(o)
    torch.testing.assert_close((p.log_prob(o,u)-lp).exp(),torch.ones(16))
    u=torch.full((16,5),100.);loss=-p.log_prob(o,u).mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in p.actor.parameters())

def test_rollout_update_checkpoint_resume_and_actor(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(12)
    c=load_config('configs/rl/segment_ppo.yaml');c['environment']['num_envs']=2
    e=VecEnv(c);p=ActorCritic(e.d);opt=torch.optim.Adam(p.parameters(),lr=.0003)
    b,ticks=Collector(e).collect(p,8);assert ticks==64
    before=copy.deepcopy(p.state_dict());m=update(p,opt,b,c['learner']);assert np.isfinite(m['value_loss'])
    assert any(not torch.equal(v,before[k]) for k,v in p.state_dict().items())
    checkpoint.save(tmp_path/'c.pt',p,opt,c,e.provenance,dict(update=1));loaded=checkpoint.load(tmp_path/'c.pt')
    q=ActorCritic(**loaded['architecture']);q.load_state_dict(loaded['policy']);o=torch.randn(2,32)
    torch.testing.assert_close(p.action(o,True)[0],q.action(o,True)[0])
    checkpoint.restore_rng(loaded);a=torch.rand(5);checkpoint.restore_rng(loaded);torch.testing.assert_close(a,torch.rand(5))
    checkpoint.export_actor(tmp_path/'actor.pt',p,e.provenance)
    from altd_rl.evaluate import driver
    _,actor=driver(tmp_path/'actor.pt');torch.testing.assert_close(p.action(o,True)[0],actor.action(o,True)[0])

def test_final_observations_before_reset_and_trace(tmp_path):
    from altd_rl.trace import ChunkWriter
    c=load_config('configs/rl/segment_ppo.yaml');c['environment']['num_envs']=1;c['scenario']['deadline_physics_ticks']=2
    e=VecEnv(c);initial=e.observe();b=e.step(np.array([[0.,1.,0.,0.,0.]]));assert b['terminated'][0]
    assert not np.array_equal(initial,b['next_observations'])
    with pytest.raises(ValueError):e.step(np.zeros((1,5)))
    np.testing.assert_array_equal(initial,e.reset([0]))
    w=ChunkWriter(tmp_path,e.provenance,1);w.append(b);w.close();data=np.load(tmp_path/'chunk-000000.npz');np.testing.assert_array_equal(data['next_observation'][0],b['next_observations'][0])


def test_explicit_reset_options_and_reward_validation():
    c=load_config('configs/rl/segment_ppo.yaml');c['environment']['num_envs']=1
    e=VecEnv(c);e.reset([0],seeds=[123]);b=e.step(np.zeros((1,5)));assert b['info'][0]['episode_seed']==123
    with pytest.raises(ValueError):e.reset([0],seeds=[1,2])
    c['reward']['gamma']=.99
    with pytest.raises(ValueError):VecEnv(c)


def test_transformed_entropy_uses_pathwise_gradients():
    p=ActorCritic(32);obs=torch.zeros(4096,32)
    u=p.distribution(obs).rsample();entropy=-p.log_prob(obs,u).mean();entropy.backward()
    assert torch.isfinite(p.log_std.grad).all()
    assert p.log_std.grad.abs().max()>.1


def test_native_legacy_adapter_maps_named_outputs_and_restricts_brake():
    import json
    from pathlib import Path
    c=load_config('configs/rl/segment_ppo.yaml');c['environment']['num_envs']=1;e=VecEnv(c)
    network=dict(shape=[20,5],weights=[[[0.]*5 for _ in range(20)]],biases=[[0.,0.,0.,-1.,0.]])
    raw=e.native.legacy_actions(json.dumps(network),Path('assets/networks/rally.json').read_text(encoding='utf-8-sig'),Path('assets/models/rally.json').read_text(encoding='utf-8-sig'))
    np.testing.assert_array_equal(np.frombuffer(raw,dtype='<f8'),np.zeros(5))


def test_epoch_previews_do_not_change_policy_rng_or_training_env(tmp_path):
    import json
    from altd_rl.preview import record_example
    torch.set_num_threads(1)
    c=load_config('configs/rl/segment_ppo.yaml');c['environment']['num_envs']=2
    c['training'].update(preview_every=1,preview_ticks=24)
    e=VecEnv(c);p=ActorCritic(e.d);opt=torch.optim.Adam(p.parameters(),lr=.0003)
    b,_=Collector(e).collect(p,4)
    callbacks=[]
    def preview(epoch):
        before=copy.deepcopy(p.state_dict());rng=torch.get_rng_state();obs=e.observe().copy()
        path=record_example(p,c,tmp_path,1,epoch);v=json.loads(path.read_text())
        assert v['epoch']==epoch and v['cells'] and len(v['frames'])>1
        assert v['frames'][-1]['tick_end']>=24 or v['frames'][-1]['reason'] is not None
        assert all(torch.equal(before[k],v) for k,v in p.state_dict().items())
        assert torch.equal(rng,torch.get_rng_state())
        np.testing.assert_array_equal(obs,e.observe());callbacks.append(epoch)
    m=update(p,opt,b,c['learner'],epoch_callback=preview)
    assert callbacks==list(range(1,m['epochs_completed']+1))
