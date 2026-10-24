"""Bounded deterministic driving examples, independent of training rollouts."""
import copy
import json
import os
from pathlib import Path
import torch
from .vec_env import VecEnv

@torch.no_grad()
def record_example(policy,config,out,update,epoch):
    cfg=copy.deepcopy(config)
    cfg['environment']['backend']='cpu';cfg['environment']['num_envs']=1
    env=VecEnv(cfg,9001);obs=env.observe();device=next(policy.parameters()).device
    limit=int(cfg['training'].get('preview_ticks',240));frames=[]
    while True:
        actions=policy.action(torch.as_tensor(obs,device=device),True)[0].cpu().numpy()
        batch=env.step(actions);info=batch['info'][0]
        frames.append(dict(position=info['position'],tick_end=info['tick_end'],episode_id=0,reason=info['reason']))
        obs=batch['next_observations']
        if info['tick_end']>=limit or batch['terminated'][0] or batch['truncated'][0]:break
    folder=Path(out)/'examples';folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'u{update:06d}_e{epoch:02d}.json'
    payload=dict(schema='altd_training_example_v1',kind='ppo_epoch',update=update,epoch=epoch,
                 label=f'Update {update} / epoch {epoch}',frames=frames,
                 cells=[c['polygon'] for c in env.provenance['native_config']['route']['cells']],
                 note='Deterministic CPU reference preview of the actor after this epoch; excluded from PPO data',
                 valid_completion=frames[-1]['reason']=='valid_completion',limited=frames[-1]['reason'] is None)
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(payload),encoding='utf-8');os.replace(tmp,target)
    return target
