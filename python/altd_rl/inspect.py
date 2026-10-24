import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('trace');a=p.parse_args();root=Path(a.trace)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'));count=0;episodes={};max_error=0.
    for file in sorted(root.glob('chunk-*.npz')):
        arrays=np.load(file,allow_pickle=False);info=json.loads(file.with_suffix('.json').read_text(encoding='utf-8-sig'))
        if len(info)!=len(arrays['reward']):raise ValueError('chunk row count mismatch')
        if not np.isfinite(arrays['observation']).all() or not np.isfinite(arrays['next_observation']).all():raise ValueError('nonfinite learner observation')
        for i,row in enumerate(info):
            max_error=max(max_error,abs(sum(row['reward'][k] for k in ('progress','time','failure','strict'))-float(arrays['reward'][i])))
            count+=1
            if arrays['terminated'][i] or arrays['truncated'][i]:episodes[(row['env_id'],row['episode_id'])]=row
    result=dict(schema=manifest['schema'],transitions=count,completed_episodes=len(episodes),reward_component_max_error=max_error,
                outcomes=[dict(env_id=r['env_id'],episode_id=r['episode_id'],reason=r['reason'],return_=r['cumulative_return'],ticks=r['tick_end']) for r in episodes.values()],provenance=manifest['provenance'])
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
