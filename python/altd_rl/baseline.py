"""Re-evaluate GA/ARS native network artifacts under the authoritative RL validator."""
import argparse
import json
from pathlib import Path
import numpy as np
from .vec_env import VecEnv,load_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/rl/rally_ppo.yaml');p.add_argument('--network',required=True);p.add_argument('--metadata',default='assets/networks/rally.json');p.add_argument('--model',default='assets/models/rally.json');p.add_argument('--episodes',type=int,default=8);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=load_config(a.config);cfg['environment']['num_envs']=1;env=VecEnv(cfg,9001)
    payload=[Path(x).read_text(encoding='utf-8-sig') for x in (a.network,a.metadata,a.model)];summaries=[]
    while len(summaries)<a.episodes:
        raw=env.native.legacy_actions(*payload);actions=np.frombuffer(raw,dtype='<f8').reshape(1,5);b=env.step(actions)
        if b['terminated'][0]:summaries.append(b['info'][0]);env.reset([0])
    valid=[r['tick_end']/60 for r in summaries if r['valid_completion']]
    report=dict(schema='altd_rl_legacy_evaluation_v1',action_contract='legacy network native inference, original named sensors; brake clipped to [0,1], new action repeat; new clean-lap rules',attempts=len(summaries),completions=len(valid),median_valid_time=float(np.median(valid)) if valid else None,episodes=summaries,provenance=env.provenance)
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','provenance')},indent=2))
if __name__=='__main__':main()
