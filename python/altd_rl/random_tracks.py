"""PPO adapters for the existing native game-tile generator.

Physics, surface and wall geometry are exported unchanged by the generator.
The RL objective uses a conservative 64-unit half-width centerline corridor;
these corridor boundaries are training rules, not claims about the road edge.
"""
import copy
import json
from pathlib import Path
import numpy as np
from . import _native
from .vec_env import digest


def route_from_scene(scene,identity,half_width=64.):
    raw=np.asarray(scene['track']['path'],dtype=np.float64)
    distances=np.linalg.norm(np.diff(raw,axis=0),axis=1)
    offsets=np.r_[0.,np.cumsum(distances)]
    stations=np.linspace(0.,offsets[-1],max(4,int(np.ceil(offsets[-1]/80.))),endpoint=False)
    p=np.stack([np.interp(stations,offsets,raw[:,axis]) for axis in (0,1)],axis=1)
    delta=np.roll(p,-1,axis=0)-p
    lengths=np.linalg.norm(delta,axis=1)
    if not np.isfinite(p).all() or (lengths<1.).any():raise ValueError('degenerate game-track stations')
    direction=delta/lengths[:,None]
    tangent=direction+np.roll(direction,1,axis=0)
    tangent_lengths=np.linalg.norm(tangent,axis=1)
    if (tangent_lengths<1e-6).any():raise ValueError('reversing game-track tangent')
    tangent/=tangent_lengths[:,None]
    normal=np.stack([-tangent[:,1],tangent[:,0]],axis=1)*half_width
    left,right=p+normal,p-normal
    cells=[]
    for i in range(len(p)):
        j=(i+1)%len(p)
        polygon=np.array([left[i],right[i],right[j],left[j]])
        edges=np.roll(polygon,-1,axis=0)-polygon
        cross=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
        if (cross<=0).any():raise ValueError(f'nonconvex game-track corridor at station {i}')
        cells.append(dict(polygon=polygon.tolist(),start=p[i].tolist(),end=p[j].tolist(),gate=[left[j].tolist(),right[j].tolist()]))
    return dict(schema='altd_rl_route_v1',id=identity,closed=True,spawn=p[0].tolist(),rotation=scene['reset_rotation'],cells=cells)


def resolve_track(config,out,seed,index):
    cfg=copy.deepcopy(config);s=cfg['scenario']
    template=json.loads(Path(s.get('template_scene','assets/scenes/rally_template.json')).read_text(encoding='utf-8-sig'))
    settings=json.loads(Path(s.get('random_track_settings','assets/random_track_settings.json')).read_text(encoding='utf-8-sig'))
    contract=digest(dict(template=template,settings=settings,half_width=64.,adapter='game_curve_corridor_v1',candidate_stride=32,station_spacing=80.))
    if s.get('generator_contract',contract)!=contract:raise ValueError('random track template/settings changed since checkpoint; start a new run')
    rejected=[]
    for attempt in range(32):
        generation=index*32+attempt
        generated=json.loads(_native.random_track_scene(json.dumps(template),json.dumps(settings),int(seed),int(generation),cfg['environment']['math_profile']))
        scene=generated['scene'];identity=f'game_{seed}_{index}'
        try:route=route_from_scene(scene,identity)
        except ValueError as e:
            rejected.append(str(e));continue
        break
    else:raise ValueError(f'No valid game-track RL corridor after 32 deterministic candidates: {rejected[-1]}')
    generated['track']['rl_adapter']=dict(track_index=index,native_generation=generation,rejected_candidates=rejected,half_width=64.,station_spacing=80.)
    folder=Path(out).resolve()/'tracks';folder.mkdir(parents=True,exist_ok=True)
    for suffix,obj in [('scene',scene),('rules',route),('track',generated['track'])]:
        (folder/f't{index:06d}.{suffix}.json').write_text(json.dumps(obj),encoding='utf-8')
    s.update(id=identity,scene=str(folder/f't{index:06d}.scene.json'),track_metadata=str(folder/f't{index:06d}.rules.json'),
             track_index=index,generator_contract=contract,target_gates=len(route['cells']) if s['task_kind']=='full_lap' else min(s['target_gates'],len(route['cells'])))
    return cfg
