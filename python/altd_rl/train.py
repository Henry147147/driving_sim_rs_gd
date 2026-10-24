import argparse
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .vec_env import VecEnv,load_config
from .policy import ActorCritic
from .rollout import Collector
from .ppo import update
from . import checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--resume');p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--updates',type=int);p.add_argument('--seconds',type=float);p.add_argument('--out');p.add_argument('--device',default='cpu');a=p.parse_args()
    c=checkpoint.load(a.resume,a.device) if a.resume else None
    if not c and not a.config:p.error('--config or --resume required')
    cfg=c['config'] if c else load_config(a.config)
    if cfg['learner']['algorithm']!='ppo':raise ValueError('train supports ppo; use compare for pinned research methods')
    seed=c['counters']['seed'] if c else a.seed
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(cfg['training'].get('torch_threads',4))
    out=Path(a.out or cfg['training']['output']);out.mkdir(parents=True,exist_ok=True)
    random_tracks=cfg['scenario'].get('track_mode','fixed')=='random'
    if random_tracks:
        from .random_tracks import resolve_track
        if not c:cfg=resolve_track(cfg,out,seed,0)
    validation_cfg=resolve_track(cfg,out,seed,0) if random_tracks else copy.deepcopy(cfg)
    validation_cfg['environment']['backend']='cpu'
    env=VecEnv(cfg,seed);policy=ActorCritic(env.d,cfg['learner']['hidden_sizes']).to(a.device)
    opt=torch.optim.Adam(policy.parameters(),lr=cfg['learner']['learning_rate'],eps=1e-5)
    counters=dict(update=0,policy_version=0,transitions=0,physics_ticks=0,seed=seed,wall_seconds=0.)
    if c:
        for key in ('observation_hash','action_hash','scene_hash','route_hash','reward_hash','math_profile'):
            if c['provenance'][key]!=env.provenance[key]:raise ValueError(f'resume provenance changed: {key}')
        policy.load_state_dict(c['policy']);opt.load_state_dict(c['optimizer']);counters=c['counters'];checkpoint.restore_rng(c)
    (out/'resolved_config.json').write_text(json.dumps(cfg,indent=2));(out/'provenance.json').write_text(json.dumps(env.provenance,indent=2))
    collector=Collector(env,a.device)
    from .trace import ChunkWriter
    trace=ChunkWriter(out/'trace'/f"track-{cfg['scenario']['track_index']:06d}" if random_tracks else out/'trace',env.provenance,chunk_size=cfg['logging'].get('trace_chunk_size',256),env_ids=cfg['logging'].get('trace_env_ids',[0])) if cfg['logging']['trace_mode']=='sampled_chunks' else None
    with (out/'episodes.jsonl').open('a',buffering=1) as episodes,(out/'updates.jsonl').open('a',buffering=1) as log:
        def summary(info):episodes.write(json.dumps(dict(info,policy_version=counters['policy_version'],track_index=cfg['scenario'].get('track_index',0)))+'\n')
        training_started=time.perf_counter()
        try:
            for _ in range(a.updates if a.updates is not None else cfg['training']['updates']):
                if a.seconds is not None and time.perf_counter()-training_started>=a.seconds:break
                iteration_start=time.perf_counter()
                track_every=cfg['training'].get('track_every',1)
                if random_tracks and counters['update'] and counters['update']%track_every==0:
                    if trace:trace.close()
                    collector=None;env=None  # Drop the previous GPU session before allocating another.
                    cfg=resolve_track(cfg,out,seed,counters['update']//track_every)
                    env=VecEnv(cfg,seed);collector=Collector(env,a.device)
                    (out/'resolved_config.json').write_text(json.dumps(cfg,indent=2))
                    (out/'provenance.json').write_text(json.dumps(env.provenance,indent=2))
                    if trace is not None:trace=ChunkWriter(out/'trace'/f"track-{cfg['scenario']['track_index']:06d}",env.provenance,chunk_size=cfg['logging'].get('trace_chunk_size',256),env_ids=cfg['logging'].get('trace_env_ids',[0]))
                start=time.perf_counter();b,ticks=collector.collect(policy,cfg['learner']['rollout_steps'],summary,trace,counters['policy_version']);collected=time.perf_counter()
                preview_every=int(cfg['training'].get('preview_every',0))
                callback=None;preview_seconds=0.
                if preview_every and (counters['update']+1)%preview_every==0:
                    from .preview import record_example
                    def callback(epoch):
                        nonlocal preview_seconds
                        preview_start=time.perf_counter();record_example(policy,cfg,out,counters['update']+1,epoch)
                        preview_seconds+=time.perf_counter()-preview_start
                m=update(policy,opt,b,cfg['learner'],cfg['reward']['gamma'],epoch_callback=callback);end=time.perf_counter()
                if trace:trace.flush()
                counters['update']+=1;counters['policy_version']+=1;counters['transitions']+=b['rewards'].numel();counters['physics_ticks']+=ticks;counters['wall_seconds']+=end-iteration_start
                m.update(counters);m.update(track_index=cfg['scenario'].get('track_index',0),route_hash=env.provenance['route_hash'],learner_device=str(next(policy.parameters()).device),environment_backend=cfg['environment']['backend'],track_setup_seconds=start-iteration_start,preview_seconds=preview_seconds,collection_seconds=collected-start,learner_seconds=end-collected-preview_seconds,transitions_per_second=b['rewards'].numel()/(end-iteration_start))
                log.write(json.dumps(m)+'\n');print(json.dumps(m),flush=True)
                checkpoint.save(out/'checkpoint.pt',policy,opt,cfg,env.provenance,counters)
                checkpoint.export_actor(out/'actor.pt',policy,env.provenance)
                every=cfg['training'].get('validation_every',0)
                if every and counters['update']%every==0:
                    from .evaluate import evaluate
                    expected=env.provenance.copy()
                    if random_tracks:
                        from .vec_env import digest
                        expected.update(scene_hash=digest(json.loads(Path(validation_cfg['scenario']['scene']).read_text(encoding='utf-8'))),route_hash=digest(json.loads(Path(validation_cfg['scenario']['track_metadata']).read_text(encoding='utf-8'))))
                    result=evaluate(policy,validation_cfg,cfg['training'].get('validation_episodes',4),9001,expected=expected)
                    if random_tracks:result.update(validation_track_index=0,validation_note='Fixed generated track 0 across all updates')
                    threshold=cfg['training'].get('reliability_threshold',.9)
                    # Every qualifying checkpoint outranks every nonqualifying checkpoint.
                    # Below threshold maximize reliability, then progress; above it minimize valid time.
                    score=[int(result['completion_rate']>=threshold),result['completion_rate'] if result['completion_rate']<threshold else -(result['median_valid_time'] or float('inf'))]
                    if score>counters.get('best_validation_score',[-1,-float('inf')]):
                        counters['best_validation_score']=score
                        checkpoint.save(out/'best.pt',policy,opt,cfg,env.provenance,counters)
                        (out/'best_validation.json').write_text(json.dumps(result,indent=2))
                    checkpoint.save(out/'checkpoint.pt',policy,opt,cfg,env.provenance,counters)

        except KeyboardInterrupt:
            print('Stopped. Last completed update is saved; unfinished rollout/update is discarded.',flush=True)
        finally:
            if trace:trace.close()
if __name__=='__main__':main()
