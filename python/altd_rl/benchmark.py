import argparse
import copy
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .vec_env import VecEnv,load_config
from .policy import ActorCritic

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/rl/rally_ppo.yaml');p.add_argument('--sizes',type=int,nargs='+',default=[64,256,1024,4096]);p.add_argument('--steps',type=int,default=32);p.add_argument('--out',default='runs/benchmark.json');a=p.parse_args()
    cfg=load_config(a.config);torch.set_num_threads(4);reports=[]
    for n in a.sizes:
        c=copy.deepcopy(cfg);c['environment']['num_envs']=n;env=VecEnv(c);policy=ActorCritic(env.d);obs=env.observe();inference=simulation=reset=0.;ticks=0
        for _ in range(a.steps):
            t=time.perf_counter()
            with torch.no_grad():actions=policy.action(torch.from_numpy(obs))[0].numpy()
            t1=time.perf_counter();b=env.step(actions);t2=time.perf_counter();obs=b['next_observations'];ids=np.flatnonzero(b['terminated']|b['truncated'])
            if len(ids):obs[ids]=env.reset(ids)
            inference+=t1-t;simulation+=t2-t1;reset+=time.perf_counter()-t2;ticks+=sum(i['tick_end']-i['tick_start'] for i in b['info'])
        reports.append(dict(n=n,transitions=n*a.steps,physics_ticks=ticks,inference_seconds=inference,environment_bridge_seconds=simulation,reset_seconds=reset,transitions_per_second=n*a.steps/(inference+simulation+reset)))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(hardware=platform.platform(),processor=platform.processor(),torch=torch.__version__,cuda=torch.cuda.is_available(),reports=reports),indent=2));print(out.read_text(encoding='utf-8-sig'))
if __name__=='__main__':main()
