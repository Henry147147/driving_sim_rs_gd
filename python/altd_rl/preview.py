"""Complete deterministic driving attempts, independent of training rollouts."""
import copy
import json
import os
import struct
from pathlib import Path
import torch
from .vec_env import VecEnv

@torch.no_grad()
def record_example(policy,config,out,update,epoch):
    cfg=copy.deepcopy(config)
    cfg['environment']['backend']='cpu';cfg['environment']['num_envs']=1
    env=VecEnv(cfg,9001);obs=env.observe();device=next(policy.parameters()).device
    frames=[dict(position=env.provenance['native_config']['route']['spawn'],tick_end=0,episode_id=0,reason=None,rotation=env.provenance['native_config']['route']['rotation'])]
    while True:
        actions=policy.action(torch.as_tensor(obs,device=device),True)[0].cpu().numpy()
        batch=env.step(actions);info=batch['info'][0]
        frames.append(dict(position=info['position'],tick_end=info['tick_end'],episode_id=0,reason=info['reason'],rotation=struct.unpack('<d',struct.pack('<Q',info['raw_state_bits'][4]))[0]))
        obs=batch['next_observations']
        if batch['terminated'][0] or batch['truncated'][0]:break
    folder=Path(out)/'examples';folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'u{update:06d}_e{epoch:02d}.json'
    payload=dict(schema='altd_training_example_v1',kind='ppo_epoch',update=update,epoch=epoch,
                 label=f'Update {update} / epoch {epoch}',frames=frames,
                 cells=[c['polygon'] for c in env.provenance['native_config']['route']['cells']],
                 note=f"Full attempt: {frames[-1]['reason']} at {frames[-1]['tick_end']/60:.2f}s; spawn to terminal, CPU reference, excluded from PPO data",
                 start_tick=0,end_tick=frames[-1]['tick_end'],terminal_reason=frames[-1]['reason'],
                 centerline=json.loads(Path(cfg['scenario']['scene']).read_text(encoding='utf-8-sig'))['track']['path'],
                 walls=json.loads(Path(cfg['scenario']['scene']).read_text(encoding='utf-8-sig'))['track'].get('walls',[]),
                 valid_completion=frames[-1]['reason']=='valid_completion',limited=frames[-1]['reason'] is None)
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(payload),encoding='utf-8');os.replace(tmp,target)
    return target
