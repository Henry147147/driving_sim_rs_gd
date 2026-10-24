"""Run the pinned official FlashSAC agent against the same native transition contract."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/rl/segment_ppo.yaml');p.add_argument('--research-config',default='configs/rl/flashsac.json');p.add_argument('--source',default='target/research/FlashSAC');p.add_argument('--steps',type=int,default=256);p.add_argument('--seconds',type=float);p.add_argument('--updates-per-step',type=int,default=1);p.add_argument('--seed',type=int,default=1729);p.add_argument('--episodes',type=int,default=8);p.add_argument('--out',default='runs/flashsac');a=p.parse_args()
    spec=json.loads(Path(a.research_config).read_text(encoding='utf-8-sig'));source=Path(a.source).resolve()
    if not source.exists():
        subprocess.run(['git','clone',spec['repository'],str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout',spec['revision']],check=True)
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=spec['revision']:raise ValueError('research checkout does not match pinned revision')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():raise ValueError('research checkout must be unmodified')
    sys.path.insert(0,str(source))
    # Official agent has unconditional @torch.compile helpers. Disable compilation on this CPU baseline,
    # preserving their operations and numerical algorithm, without patching the research checkout.
    os.environ['TORCH_COMPILE_DISABLE']='1'
    import numpy as np
    import torch
    import gymnasium as gym
    from flash_rl.agents.flashSAC.agent import FlashSACAgent,FlashSACConfig
    from .vec_env import VecEnv,load_config
    from .evaluate import evaluate
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    cfg=load_config(a.config);env=VecEnv(cfg,a.seed);params=spec['config'];params['seed']=a.seed
    agent=FlashSACAgent(gym.spaces.Box(-np.inf,np.inf,(env.d,),np.float32),gym.spaces.Box(-1.,1.,(5,),np.float32),{},FlashSACConfig(**params))
    mid=np.array([0,0,.5,.5,.5],np.float32);scale=np.array([1,1,.5,.5,.5],np.float32)
    obs=env.observe();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);updates=steps=ticks=0;start=time.perf_counter()
    with (out/'episodes.jsonl').open('w') as log:
        while steps<a.steps and (a.seconds is None or time.perf_counter()-start<a.seconds):
            raw=agent.sample_actions(steps,{'next_observation':obs},True)
            b=env.step(mid+scale*raw)
            agent.process_transition(dict(observation=obs,action=raw,next_observation=b['next_observations'],reward=b['rewards'].astype(np.float32),terminated=b['terminated'],truncated=b['truncated']))
            if agent.can_start_training():
                for _ in range(a.updates_per_step):agent.update();updates+=1
            ticks+=sum(i['tick_end']-i['tick_start'] for i in b['info']);steps+=1;obs=b['next_observations'].copy()
            ids=np.flatnonzero(b['terminated']|b['truncated'])
            for i in ids:log.write(json.dumps(b['info'][i])+'\n')
            if len(ids):obs[ids]=env.reset(ids)
    wall=time.perf_counter()-start;agent.save(str(out/'checkpoint'))
    class Driver:
        def action(self,obs,deterministic=True):
            x=agent.sample_actions(0,{'next_observation':obs.numpy()},not deterministic)
            return torch.from_numpy(mid+scale*x),None,None,None
    result=evaluate(Driver(),cfg,a.episodes,9001,expected=env.provenance)
    report=dict(schema='altd_rl_comparison_v1',algorithm='official FlashSAC',revision=revision,resolved_agent_config=params,
                adaptation='native finite-horizon gamma=1; affine action mapping; CPU eager helpers; original reward normalization, categorical double critic, weight normalization and noise persistence retained',
                transitions=steps*env.n,physics_ticks=ticks,wall_seconds=wall,gradient_updates=updates,utd_gradient_steps=updates/(steps*env.n),sampled_items_per_transition=updates*params['sample_batch_size']/(steps*env.n),
                evaluation=result,provenance=env.provenance,versions=dict(torch=torch.__version__,python=sys.version),training_config=cfg)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('evaluation','provenance','training_config','resolved_agent_config')},indent=2))
if __name__=='__main__':main()
