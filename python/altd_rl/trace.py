"""Compressed NPZ chunks with JSON sidecars. Writes once per chunk, outside physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

class ChunkWriter:
    def __init__(self,path,provenance,chunk_size=256,env_ids=(0,)):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.chunk_size=chunk_size;self.env_ids=set(env_ids);self.rows=[];self.index=len(list(self.path.glob('chunk-*.npz')))
        manifest=self.path/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8-sig'))['provenance']!=provenance:raise ValueError('trace provenance mismatch')
        manifest.write_text(json.dumps(dict(schema='altd_rl_trace_npz_v1',format='numpy compressed NPZ + JSON sidecar',provenance=provenance),indent=2))
    def append(self,b,learner=None,policy_version=0):
        for i,info in enumerate(b['info']):
            if i not in self.env_ids:continue
            row=dict(observation=b['observations'][i],next_observation=b['next_observations'][i],requested_action=np.array(info['requested_action']),applied_action=np.array(info['applied_action']),
                     reward=np.array(b['rewards'][i]),terminated=np.array(b['terminated'][i]),truncated=np.array(b['truncated'][i]),policy_version=np.array(policy_version),info=info)
            if learner:
                for k,v in learner.items():
                    if k in ('u','old_log_prob','values','next_values','normalized_obs','episode_start','valid_transition_mask'):
                        row[k]=v[i].detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v[i])
            self.rows.append(row)
            if len(self.rows)>=self.chunk_size:self.flush()
    def flush(self):
        if not self.rows:return
        stem=self.path/f'chunk-{self.index:06d}'
        np.savez_compressed(str(stem)+'.npz',**{k:np.stack([r[k] for r in self.rows]) for k in self.rows[0] if k!='info'})
        Path(str(stem)+'.json').write_text(json.dumps([r['info'] for r in self.rows],allow_nan=False))
        self.rows=[];self.index+=1
    def close(self):self.flush()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--config');p.add_argument('--episodes',type=int,default=3);p.add_argument('--out',required=True);a=p.parse_args()
    from .evaluate import driver,evaluate
    from .vec_env import VecEnv,load_config
    from .policy import ActorCritic
    if a.checkpoint:
        c,policy=driver(a.checkpoint);cfg=load_config(a.config) if a.config else c['config']
    else:
        if not a.config:p.error('--config required for scripted trace')
        cfg=load_config(a.config);env=VecEnv(cfg);policy=ActorCritic(env.d,cfg['learner']['hidden_sizes'])
        # Reproducible open-loop controller, not a trained policy.
        with torch.no_grad():
            for p_ in policy.actor.parameters():p_.zero_()
            policy.actor[-1].bias.copy_(torch.tensor([0.,2.,-10.,-10.,-10.]))
    env=VecEnv(cfg,9001);writer=ChunkWriter(a.out,env.provenance)
    try:
        result=evaluate(policy,cfg,a.episodes,trace=writer)
        (Path(a.out)/'episodes.json').write_text(json.dumps(result,indent=2))
    finally:writer.close()
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
if __name__=='__main__':main()
