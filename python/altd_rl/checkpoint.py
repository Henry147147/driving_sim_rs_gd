import os
import platform
import random
from pathlib import Path
import numpy as np
import torch

SCHEMA='altd_ppo_checkpoint_v1'
def save(path,policy,optimizer,config,provenance,counters):
    obj=dict(schema=SCHEMA,policy=policy.state_dict(),optimizer=optimizer.state_dict(),config=config,provenance=provenance,counters=counters,
             architecture=dict(obs_dim=policy.obs_dim,hidden=policy.hidden),rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
             resume_semantics='completed update; restore learning state, reset environments, new rollout')
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,p)

def load(path,device='cpu'):
    # Checkpoints are trusted local pickle artifacts; never load untrusted checkpoints.
    c=torch.load(path,map_location=device,weights_only=False)
    if c.get('schema')!=SCHEMA:raise ValueError('unsupported checkpoint schema')
    return c

def restore_rng(c):
    torch.set_rng_state(c['rng']['torch'].cpu());np.random.set_state(c['rng']['numpy']);random.setstate(c['rng']['python'])
    if c['rng']['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(c['rng']['cuda'])

def export_actor(path,policy,provenance):
    torch.save(dict(schema='altd_native_actor_v1',architecture=dict(obs_dim=policy.obs_dim,hidden=policy.hidden),
                    actor={k:v for k,v in policy.state_dict().items() if not k.startswith('critic.')},provenance=provenance),path)
