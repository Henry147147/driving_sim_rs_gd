"""Generate the explicit, versioned RL polygon-ring road and native physics scenes."""
import argparse
import json
import math
from pathlib import Path

def generate(root):
    n,radius,half=64,1000.,130.
    def ring(r):return [[r*math.cos(i*2*math.pi/n),r*math.sin(i*2*math.pi/n)] for i in range(n+1)]
    centers,inner,outer=ring(radius),ring(radius-half),ring(radius+half)
    cells=[];shapes=[];walls=[]
    for i in range(n):
        cells.append(dict(polygon=[inner[i],outer[i],outer[i+1],inner[i+1]],start=centers[i],end=centers[i+1],gate=[inner[i+1],outer[i+1]]))
        for edge,sgn in [(inner,-1),(outer,1)]:
            a,b=edge[i],edge[i+1];factor=1+sgn*8/(radius+sgn*half)
            aa,bb=[x*factor for x in a],[x*factor for x in b];q=[a,b,bb,aa]
            if sum(q[j][0]*q[(j+1)%4][1]-q[j][1]*q[(j+1)%4][0] for j in range(4))<0:q.reverse()
            shapes.append(dict(points=q,origin=[sum(x[0] for x in q)/4,sum(x[1] for x in q)/4]));walls.append([a,b])
    route=dict(schema='altd_rl_route_v1',id='rally_ring_verified',closed=True,spawn=centers[0],rotation=math.pi,cells=cells)
    scene=json.loads((root/'assets/scenes/rally_template.json').read_text(encoding='utf-8-sig'));scene['track'].update(name='RL verified polygon ring',path=centers,walls=walls,physics_shapes=shapes)
    # Native RL owns its spawn. Legacy CLI uses the separate explicit spawn trace.
    result={'rally_ring.rules.json':route,'rally_ring.scene.json':scene}
    gpu=json.loads(json.dumps(scene));gpu['track']['polygons']=[dict(points=x['points']) for x in shapes]
    gpu['track']['curve']=dict(bake_interval=100,points=[dict(position=x,incoming=[0,0],outgoing=[0,0]) for x in centers])
    result['rally_ring_gpu.scene.json']=gpu
    result['rally_ring.spawn.json']=dict(frames=[dict(position=[1000,0],rotation=math.pi)])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1]
    for name,obj in generate(root).items():
        path=root/'configs/rl/tracks'/name
        if a.check:
            if json.loads(path.read_text(encoding='utf-8-sig'))!=obj:raise SystemExit(f'generated asset differs: {path}')
        else:path.write_text(json.dumps(obj,indent=2))
    print('verified RL ring geometry' if a.check else 'generated RL ring geometry')
if __name__=='__main__':main()
