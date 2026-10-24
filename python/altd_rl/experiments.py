import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from .vec_env import VecEnv,load_config
from .policy import ActorCritic
from .evaluate import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/rl/segment_ppo.yaml');p.add_argument('--seeds',type=int,nargs='+',default=[1729,1730,1731]);p.add_argument('--updates',type=int,default=4);p.add_argument('--out',default='runs/comparison');p.add_argument('--flashsac',action='store_true');p.add_argument('--wall-seconds',type=float);a=p.parse_args()
    cfg=load_config(a.config);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];torch.set_num_threads(1)
    for seed in a.seeds:
        torch.manual_seed(seed);env=VecEnv(cfg,seed);initial=ActorCritic(env.d,cfg['learner']['hidden_sizes'])
        before=evaluate(initial,cfg,4,9001)
        d=out/str(seed);t=time.perf_counter()
        args=[sys.executable,'-m','altd_rl.train','--config',a.config,'--seed',str(seed),'--updates',str(a.updates),'--out',str(d)]
        if a.wall_seconds:args+=['--seconds',str(a.wall_seconds)]
        subprocess.run(args,check=True,stdout=(out/f'ppo-{seed}.log').open('w'))
        wall=time.perf_counter()-t
        from .evaluate import driver
        ckpt,policy=driver(d/'checkpoint.pt');after=evaluate(policy,cfg,4,9001)
        row=dict(seed=seed,algorithm='PPO',transitions=ckpt['counters']['transitions'],compute_seconds=ckpt['counters']['wall_seconds'],process_wall_seconds=wall,before=before,evaluation=after)
        rows.append(row)
        if a.flashsac:
            steps=a.updates*cfg['learner']['rollout_steps'];fd=d/'flashsac';args=[sys.executable,'-m','altd_rl.compare','--config',a.config,'--seed',str(seed),'--steps',str(steps),'--episodes','4','--out',str(fd)]
            if a.wall_seconds:args+=['--seconds',str(a.wall_seconds)]
            subprocess.run(args,check=True,stdout=(out/f'flashsac-{seed}.log').open('w'))
            rows.append(json.loads((fd/'report.json').read_text(encoding='utf-8-sig')))
        (out/'report.json').write_text(json.dumps(dict(schema='altd_rl_experiments_v1',budget='wall_seconds' if a.wall_seconds else 'transitions',results=rows),indent=2))
    print(json.dumps([dict(algorithm=r['algorithm'],seed=r.get('seed',r.get('resolved_agent_config',{}).get('seed')),completions=r['evaluation']['completions'],attempts=r['evaluation']['attempts'],median=r['evaluation']['median_valid_time'],transitions=r['transitions']) for r in rows],indent=2))
if __name__=='__main__':main()
