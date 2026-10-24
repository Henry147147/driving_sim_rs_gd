import json
from pathlib import Path
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app

@pytest.fixture
def setup(tmp_path):
    repo=tmp_path/'sim';(repo/'python/altd_rl').mkdir(parents=True)
    (repo/'Cargo.toml').write_text('[package]\nname="test"\n')
    (repo/'python/altd_rl/__init__.py').write_text('')
    (repo/'python/altd_rl/_native-test.pyd').touch()
    (repo/'python/altd_rl/train.py').write_text('''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--updates',type=int,default=1);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
try:
 for i in range(a.updates):
  row=dict(update=i+1,transitions=(i+1)*32,value_loss=1/(i+1),transitions_per_second=100)
  with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
  (out/'checkpoint.pt').write_bytes(b'last completed update')
  print(json.dumps(row),flush=True)
  time.sleep(.1)
except KeyboardInterrupt:print('Stopped at saved checkpoint',flush=True)
''')
    (repo/'runs').mkdir()
    app=create_app(repo,data=tmp_path/'data')
    with TestClient(app) as client:
        bootstrap=client.get('/api/bootstrap').json()
        yield client,app.state.dashboard,{'X-CSRF-Token':bootstrap['csrf']},repo


def wait(client,ident):
    for _ in range(100):
        r=client.get('/api/jobs/'+ident).json()
        if r['status'] not in ('running','stopping'):return r
        time.sleep(.1)
    pytest.fail('Job did not finish')


def test_host_origin_csrf_and_access_token(setup,tmp_path):
    client,state,headers,repo=setup
    assert client.get('/api/bootstrap',headers={'Host':'evil.example'}).status_code==403
    assert client.get('/api/bootstrap',headers={'Origin':'https://evil.example'}).status_code==403
    assert client.post('/api/jobs',json={'tool':'train','arguments':[]}).status_code==403
    app=create_app(repo,data=tmp_path/'auth',token='secret')
    with TestClient(app) as c:
        assert c.get('/api/bootstrap').status_code==401
        assert c.get('/api/bootstrap',headers={'Authorization':'Bearer secret'}).status_code==200


def test_real_subprocess_metrics_artifacts_and_shell_isolation(setup):
    client,state,headers,repo=setup
    result=client.post('/api/jobs',headers=headers,json={'tool':'train','arguments':['--out','runs/smoke','--updates','3']})
    assert result.status_code==200
    done=wait(client,result.json()['id'])
    assert done['status']=='completed',done['log']
    assert done['summary']['updates'][-1]['update']==3
    checkpoint=next(f for f in done['summary']['files'] if f['name']=='checkpoint.pt')
    assert client.get('/api/file',params={'path':checkpoint['path'],'download':True}).content==b'last completed update'
    assert client.post('/api/jobs',headers=headers,json={'tool':'shell','arguments':['echo','bad']}).status_code==400
    assert client.post('/api/jobs',headers=headers,json={'tool':'train','arguments':'--out x'}).status_code==400
    assert client.post('/api/jobs',headers=headers,json={'tool':'train','arguments':['x'+chr(0)]}).status_code==400
    assert client.post('/api/jobs',headers=headers,json={'tool':{}}).status_code==400


def test_stop_keeps_checkpoint_and_serializes_jobs(setup):
    client,state,headers,repo=setup
    r=client.post('/api/jobs',headers=headers,json={'tool':'train','arguments':['--out','runs/stop','--updates','500']}).json()
    for _ in range(100):
        if (repo/'runs/stop/checkpoint.pt').exists():break
        time.sleep(.05)
    assert client.post('/api/jobs',headers=headers,json={'tool':'train','arguments':[]}).status_code==409
    assert client.post('/api/jobs/'+r['id']+'/stop',headers=headers,json={'force':False}).status_code==200
    done=wait(client,r['id']);assert done['status']=='stopped'
    assert (repo/'runs/stop/checkpoint.pt').read_bytes()==b'last completed update'


def test_file_containment_and_config_copy(setup):
    client,state,headers,repo=setup
    assert client.get('/api/file',params={'path':'../secret.json'}).status_code==403
    assert client.get('/api/file',params={'path':'x'+chr(0)+'.json'}).status_code==400
    assert client.get('/api/file',params={'path':'x\x00.json'}).status_code==400
    r=client.post('/api/config',headers=headers,json={'text':'schema: example\nnum_envs: 32\n'})
    assert r.status_code==200
    saved=Path(r.json()['path']);assert saved.is_relative_to(state.data)
    assert 'num_envs' in client.get('/api/file',params={'path':str(saved)}).json()['text']
    assert client.post('/api/config',headers=headers,json={'text':'- invalid\n'}).status_code==400
    assert client.post('/api/upload?name=../x.pt',headers=headers,content=b'x').status_code==400
    upload=client.post('/api/upload?name=driver.pt',headers=headers,content=b'checkpoint')
    assert upload.status_code==200 and Path(upload.json()['path']).read_bytes()==b'checkpoint'


def test_trace_preview_and_partial_jsonl(setup):
    client,state,headers,repo=setup
    folder=repo/'runs/trace';folder.mkdir()
    route=dict(cells=[dict(polygon=[[0,0],[10,0],[10,10],[0,10]])])
    (folder/'manifest.json').write_text(json.dumps(dict(provenance=dict(native_config=dict(route=route)))))
    (folder/'chunk-000000.json').write_text(json.dumps([dict(position=[1,2],episode_id=0,tick_end=4),dict(position=[3,4],episode_id=0,tick_end=8)]))
    (folder/'updates.jsonl').write_text('{"update":1,"value_loss":2}\n{"update":')
    data=client.get('/api/preview',params={'path':str(folder)}).json()
    assert data['recorded'] and len(data['points'])==2 and len(data['cells'])==1
    result=client.get('/api/run',params={'path':str(folder)}).json()
    assert result['updates']==[dict(update=1,value_loss=2)]
    assert client.get('/').status_code==200
    assert client.get('/app.js').status_code==200


def test_stop_comparison_also_interrupts_python_children(setup):
    import psutil
    client,state,headers,repo=setup
    (repo/'python/altd_rl/experiments.py').write_text('''import subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-m','altd_rl.train','--out','runs/child','--updates','1000'])
Path('runs/child.pid').write_text(str(p.pid))
p.wait()
''')
    r=client.post('/api/jobs',headers=headers,json={'tool':'experiments','arguments':[]}).json()
    checkpoint=repo/'runs/child/checkpoint.pt'
    for _ in range(100):
        if checkpoint.exists():break
        time.sleep(.05)
    assert checkpoint.exists()
    child_pid=int((repo/'runs/child.pid').read_text())
    client.post('/api/jobs/'+r['id']+'/stop',headers=headers,json={'force':False})
    done=wait(client,r['id']);assert done['status']=='stopped'
    for _ in range(30):
        if not psutil.pid_exists(child_pid):break
        time.sleep(.05)
    assert not psutil.pid_exists(child_pid)


def test_inspect_selects_trace_and_restart_retains_history(setup,tmp_path):
    client,state,headers,repo=setup
    folder=repo/'runs/trace';folder.mkdir()
    assert state.output_for('inspect',['runs/trace'])==str(folder)
    state.jobs['a'*32]=dict(id='a'*32,tool='train',status='completed',started=1,finished=2,pid=0,exit_code=0)
    state.persist(state.jobs['a'*32])
    from app import Dashboard
    restored=Dashboard(repo,data=state.data)
    assert restored.jobs['a'*32]['status']=='completed'


def test_example_selection_and_config_copy(setup):
    client,state,headers,repo=setup
    folder=repo/'runs/examples';(folder/'examples').mkdir(parents=True)
    example=dict(schema='altd_training_example_v1',label='Update 1 / epoch 2',cells=[[[0,0],[1,0],[1,1]]],frames=[dict(position=[1,2],tick_end=4,episode_id=0)])
    (folder/'examples/u000001_e02.json').write_text(json.dumps(example))
    items=client.get('/api/examples',params={'path':'runs/examples'}).json()
    assert items[0]['label']==example['label']
    assert client.get('/api/examples',params={'path':'runs/examples','since':time.time()+1}).json()==[]
    result=client.get('/api/preview',params={'path':'runs/examples','example':items[0]['name']}).json()
    assert result['points'][0]['tick']==4 and result['recorded']
    assert client.get('/api/preview',params={'path':'runs/examples','example':'../../secret.json'}).status_code==400
    saved=client.post('/api/config',headers=headers,json={'text':'{"length":{"min":12,"max":40}}','format':'json'}).json()
    assert saved['path'].endswith('.json')
    assert client.get('/api/config-data',params={'path':saved['path']}).json()['value']['length']['min']==12


def test_fresh_run_reusing_output_gets_its_own_history(setup):
    client,state,headers,repo=setup
    payload=dict(tool='train',arguments=['--out','runs/reused','--updates','3'])
    first=wait(client,client.post('/api/jobs',headers=headers,json=payload).json()['id'])
    payload['arguments'][-1]='1'
    second=wait(client,client.post('/api/jobs',headers=headers,json=payload).json()['id'])
    assert first['output']!=second['output']
    assert len(second['summary']['updates'])==1
    assert len(client.get('/api/run',params={'path':first['output']}).json()['updates'])==3


def test_job_summary_offsets_hide_preexisting_rows_and_results(setup):
    client,state,headers,repo=setup
    folder=repo/'runs/offsets';folder.mkdir()
    old='{"update":9}\n';(folder/'updates.jsonl').write_text(old)
    (folder/'evaluation.json').write_text('{"attempts":999}')
    (folder/'updates.jsonl').write_text(old+'{"update":10}\n')
    summary=state.summary(folder,{'updates.jsonl':len(old)},since=time.time()+1)
    assert summary['updates']==[{'update':10}] and summary['report'] is None and not summary['files']


def test_track_length_limit_rejected_before_saving_or_launching(setup):
    client,state,headers,repo=setup
    bad={'length':{'min':12,'max':100}}
    assert client.post('/api/config',headers=headers,json={'text':json.dumps(bad),'format':'json'}).status_code==400
    path=repo/'tracks.json';path.write_text(json.dumps(bad))
    cfg=repo/'config.yaml';cfg.write_text(json.dumps({'scenario':{'track_mode':'random','random_track_settings':'tracks.json'}}))
    result=client.post('/api/jobs',headers=headers,json={'tool':'train','arguments':['--config','config.yaml']})
    assert result.status_code==400 and '76' in result.json()['detail'] and not state.jobs
    good=client.post('/api/config',headers=headers,json={'text':json.dumps({'length':{'min':12,'max':76}}),'format':'json'})
    assert good.status_code==200
