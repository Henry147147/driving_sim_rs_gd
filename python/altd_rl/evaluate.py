import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import numpy as np
import torch
from . import checkpoint
from .policy import ActorCritic
from .vec_env import VecEnv,load_config

def driver(path):
    c=torch.load(path,map_location='cpu',weights_only=False)
    if c['schema']==checkpoint.SCHEMA:
        policy=ActorCritic(**c['architecture']);policy.load_state_dict(c['policy'])
    elif c['schema']=='altd_native_actor_v1':
        policy=ActorCritic(**c['architecture']);policy.load_state_dict(c['actor'],strict=False)
    else:raise ValueError('unsupported driver artifact')
    policy.eval();return c,policy

@torch.no_grad()
def evaluate(policy,cfg,episodes=32,seed=9001,trace=None,expected=None):
    cfg=copy.deepcopy(cfg);cfg['environment']['num_envs']=1;env=VecEnv(cfg,seed)
    if expected:
        for k in ('observation_hash','action_hash','scene_hash','route_hash','math_profile'):
            if expected[k]!=env.provenance[k]:raise ValueError(f'evaluation contract mismatch: {k}')
    obs=env.observe();summaries=[]
    device=next(policy.parameters()).device if isinstance(policy,torch.nn.Module) else torch.device('cpu')
    while len(summaries)<episodes:
        tensor=torch.as_tensor(obs,device=device)
        a,u,lp,v=policy.action(tensor,True);b=env.step(a.cpu().numpy())
        if trace:trace.append(b,dict(u=u,old_log_prob=lp,values=v,normalized_obs=policy.transform(tensor)),0)
        obs=b['next_observations']
        if b['terminated'][0] or b['truncated'][0]:
            summaries.append(b['info'][0]);obs=env.reset([0])
    valid=[s['tick_end']/60 for s in summaries if s['valid_completion']]
    count=len(valid);rate=count/episodes;z=1.96;den=1+z*z/episodes
    center=(rate+z*z/(2*episodes))/den;half=z*((rate*(1-rate)/episodes+z*z/(4*episodes**2))**.5)/den
    return dict(attempts=episodes,completions=count,completion_rate=rate,completion_rate_ci95=None,unique_scenario_starts=1,uncertainty_note="Fixed deterministic start repeated; estimate uncertainty across independent scenarios/training seeds, not repetitions",
                median_valid_time=float(np.median(valid)) if valid else None,p90_valid_time=float(np.quantile(valid,.9)) if valid else None,
                best_valid_time=min(valid) if valid else None,task_kind=cfg['scenario']['task_kind'],
                failures=dict(Counter(s['reason'] for s in summaries if not s['valid_completion'])),
                mean_failed_progress=float(np.mean([s['progress_after'] for s in summaries if not s['valid_completion']])) if count<episodes else None,
                seed=seed,time_resolution_seconds=1/60,episodes=summaries)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--suite');p.add_argument('--config');p.add_argument('--episodes',type=int);p.add_argument('--out');a=p.parse_args()
    c,policy=driver(a.checkpoint);torch.set_num_threads(1)
    suite=json.loads(Path(a.suite).read_text(encoding='utf-8-sig')) if a.suite else {}
    cfg=load_config(a.config or suite['config']) if a.config or suite.get('config') else c.get('config')
    if cfg is None:p.error('actor artifact requires --config or --suite')
    reports=[evaluate(policy,cfg,a.episodes or suite.get('episodes',32),seed,expected=c['provenance']) for seed in suite.get('seeds',[9001])]
    result=dict(schema='altd_rl_evaluation_v1',selection_rule='meet reliability threshold, then minimize median valid time',reliability_threshold=suite.get('reliability_threshold',.9),reports=reports)
    out=Path(a.out or str(Path(a.checkpoint).parent/'evaluation.json'));out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
