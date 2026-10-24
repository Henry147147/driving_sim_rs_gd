"""Whole-batch native calls; no Python physics or implicit resets."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import yaml
from . import _native
from ._native import NativeEnv

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def load_config(path):
    c = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if c['schema'] != 'altd_rl_config_v1':
        raise ValueError('unsupported config schema')
    learner=c['learner']
    for key in ('rollout_steps','minibatch_size','epochs'):
        if not isinstance(learner[key],int) or isinstance(learner[key],bool) or learner[key]<=0:raise ValueError(f'invalid {key}')
    if len(learner['hidden_sizes'])==0 or any(not isinstance(x,int) or x<=0 for x in learner['hidden_sizes']):raise ValueError('invalid hidden_sizes')
    if not 0<=learner['gae_lambda']<=1 or not 0<learner['clip_epsilon']<1 or learner['learning_rate']<=0 or learner['target_kl']<=0 or learner['max_grad_norm']<=0:raise ValueError('invalid PPO hyperparameters')
    if c['logging']['trace_mode'] not in ('off','sampled_chunks'):raise ValueError('unsupported trace_mode')
    return c

class VecEnv:
    def __init__(self, config, seed=1729):
        self.config = config
        e, s, r = (config[k] for k in ('environment', 'scenario', 'reward'))
        if e['backend'] not in ('cpu','gpu'):
            raise ValueError('backend must be cpu or gpu')
        if e['physics_hz'] != 60 or e.get('autoreset', False):
            raise ValueError('physics_hz must be 60; autoreset must be false')
        if not isinstance(s['deadline_physics_ticks'], int) or isinstance(s['deadline_physics_ticks'], bool):
            raise ValueError('deadline must be an integer number of physics ticks')
        self.n = e['num_envs']
        route = json.loads(Path(s['track_metadata']).read_text(encoding='utf-8-sig'))
        scene = json.loads(Path(s['scene']).read_text(encoding='utf-8-sig'))
        layout = json.loads(Path(s['sensors']).read_text(encoding='utf-8-sig'))
        native = dict(schema='altd_rl_env_v1', num_envs=self.n, action_repeat_ticks=e['action_repeat_ticks'],
                      deadline_physics_ticks=s['deadline_physics_ticks'], target_gates=s['target_gates'],
                      task_kind=s['task_kind'], reward_profile=r['profile'], gamma=r['gamma'],
                      run_seed=int(seed), math_profile=e['math_profile'], route=route)
        self.native = NativeEnv(json.dumps(scene), json.dumps(layout), json.dumps(native))
        self.schema = json.loads(self.native.schema())
        self.d = len(self.schema['names'])
        if e['backend'] == 'gpu':self.native.enable_gpu()
        self.action_schema = dict(schema='steering_accel_handbrake_brake_boost_v1',
                                  names=['Steering','Acceleration','Handbrake','Brake','Boost'],
                                  low=[-1,-1,0,0,0], high=[1]*5)
        try:
            revision = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
            dirty = bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = 'unknown', None
        self.provenance = dict(native_binary_hash=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(), observation_hash=digest(self.schema), action_hash=digest(self.action_schema),
                               scene_hash=digest(scene), vehicle_hash=digest(scene['vehicle']), route_hash=digest(route),
                               reward_hash=digest(r), simulator_revision=revision, dirty=dirty,
                               math_profile=e['math_profile'], native_config=native)
    def _array(self, raw, n):
        return np.frombuffer(raw, dtype='<f4').reshape(n, self.d).copy()
    def observe(self, ids=None):
        ids = list(range(self.n)) if ids is None else list(map(int, ids))
        return self._array(self.native.observe(ids), len(ids))
    def reset(self, ids=None, seeds=None, spawn_specs=None):
        ids = list(range(self.n)) if ids is None else list(map(int, ids))
        if seeds is not None or spawn_specs is not None:
            raw=self.native.reset_options(ids,json.dumps(dict(seeds=seeds,spawn_specs=spawn_specs)))
        else:raw=self.native.reset(ids)
        return self._array(raw,len(ids))
    def step(self, actions):
        a = np.asarray(actions, dtype='<f8', order='C')
        if a.shape != (self.n, 5):
            raise ValueError('action shape must be [N,5]')
        raw, nxt, meta = self.native.step(a.tobytes())
        b = json.loads(meta)
        b.update(observations=self._array(raw,self.n), next_observations=self._array(nxt,self.n))
        for key in ('rewards','terminated','truncated'):
            b[key] = np.asarray(b[key])
        return b
