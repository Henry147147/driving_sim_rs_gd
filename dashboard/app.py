"""Self-hosted CLI dashboard. Start with python app.py --repo PATH."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import psutil
import uvicorn
import yaml

ROOT = Path(__file__).resolve().parent
PYTHON_TOOLS = {
    'train': ('PPO training', 'train'),
    'compare': ('FlashSAC training', 'compare'),
    'evaluate': ('Evaluate PPO / actor', 'evaluate'),
    'trace': ('Record driver trace', 'trace'),
    'inspect': ('Inspect saved trace', 'inspect'),
    'experiments': ('Compare PPO and FlashSAC', 'experiments'),
    'benchmark': ('Benchmark RL throughput', 'benchmark'),
    'baseline': ('Evaluate GA / ARS network', 'baseline'),
}
NATIVE_TOOLS = ('bench', 'train', 'train-scratch', 'evaluate', 'compare-trace',
                'compare-one-step', 'compare-closed-loop', 'compare-network',
                'compare-sensors', 'compare-score', 'serve', 'gpu-info')
ACTIVE = {'running', 'stopping'}


def read_json(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return fallback


def tail(path: Path, limit=120_000):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-limit))
            return stream.read(limit).decode('utf-8', errors='replace')
    except OSError:
        return ''


def json_lines(path: Path, limit=500, offset=0):
    rows = []
    if offset:
        try:
            with path.open('rb') as stream:
                stream.seek(max(offset,path.stat().st_size-700_000));source=stream.read(700_000).decode('utf-8',errors='replace')
        except OSError:source=''
    else:source=tail(path,700_000)
    for line in source.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except ValueError:
            pass
    return rows[-limit:]


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def validate_track_length(settings):
    length=settings.get('length',{'min':12,'max':40})
    if isinstance(length,int) and not isinstance(length,bool):lo=hi=length
    elif isinstance(length,dict):lo,hi=length.get('min'),length.get('max')
    else:raise ValueError('Track length must be an integer or a min/max range')
    if any(not isinstance(x,int) or isinstance(x,bool) for x in (lo,hi)) or lo<4 or hi>77 or lo>hi or lo+lo%2>hi:
        raise ValueError('Game tracks require an even tile count from 4 to 76; minimum must not exceed maximum (the grid has 77 cells)')


class Dashboard:
    def __init__(self, repo: Path, python=sys.executable, data: Path | None = None,
                 token='', hosts=('127.0.0.1', 'localhost', 'testserver')):
        self.repo = repo.resolve()
        if not (self.repo/'Cargo.toml').is_file() or not (self.repo/'python/altd_rl').is_dir():
            raise ValueError('Choose the driving_sim_rs_gd checkout containing Cargo.toml and python/altd_rl')
        self.python = str(Path(python).resolve())
        self.data = (data or ROOT/'.dashboard').resolve()
        self.data.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.csrf = secrets.token_urlsafe(32)
        self.hosts = set(hosts)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.help_cache = {}
        for file in (self.data/'jobs').glob('*/job.json'):
            job = read_json(file)
            if not isinstance(job, dict) or not re.fullmatch(r'[a-f0-9]{32}', str(job.get('id', ''))):
                continue
            self.jobs[job['id']] = job
            if job.get('status') in ACTIVE:
                if self.process(job) is None:
                    job['status'] = 'interrupted'
                else:
                    threading.Thread(target=self.recover, args=(job,), daemon=True).start()
            self.jobs[job['id']] = job

    def path(self, value: str, directory=False):
        if not isinstance(value, str) or not value or chr(0) in value:
            raise HTTPException(400, 'Invalid path')
        p = Path(value)
        p = (p if p.is_absolute() else self.repo/p).resolve()
        if not any(p.is_relative_to(base) for base in (self.repo, self.data)):
            raise HTTPException(403, 'Files must be inside the simulator checkout or dashboard data directory')
        if directory and not p.is_dir():
            raise HTTPException(404, 'Directory not found')
        return p

    def native_binary(self):
        name = 'altd-sim.exe' if os.name == 'nt' else 'altd-sim'
        for profile in ('release', 'debug'):
            p = self.repo/'target'/profile/name
            if p.is_file():
                return str(p)
        raise HTTPException(400, 'Build the native CLI first: cargo +1.95 build --release')

    def command(self, tool: str, arguments: list[str]):
        if tool in PYTHON_TOOLS:
            return ['--module', 'altd_rl.'+PYTHON_TOOLS[tool][1], '--', *arguments]
        if tool == 'native':
            return ['--', self.native_binary(), *arguments]
        if tool.startswith('native:') and tool[7:] in NATIVE_TOOLS:
            return ['--', self.native_binary(), tool[7:], *arguments]
        if tool == 'track':
            return ['--script', str(self.repo/'tools/rl_track.py'), '--', *arguments]
        raise HTTPException(400, 'Unknown command')

    def environment(self):
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONUTF8='1')
        library=self.repo/'target/gpu'/('altd_gpu.dll' if os.name=='nt' else 'libaltd_gpu.so')
        if library.is_file(): env.setdefault('ALTD_GPU_LIB',str(library))
        native = self.repo/'python/altd_rl'
        if any(native.glob('_native*.pyd')) or any(native.glob('_native*.so')):
            env['PYTHONPATH'] = str(self.repo/'python')+os.pathsep+env.get('PYTHONPATH', '')
        return env

    def gpu_python(self):
        local=self.repo/('.venv-gpu/Scripts/python.exe' if os.name=='nt' else '.venv-gpu/bin/python')
        return str(local) if local.is_file() else self.python

    def gpu_status(self):
        code="""import json,torch
r=dict(torch=torch.__version__,cuda=torch.cuda.is_available())
if r['cuda']:
 x=torch.ones(4,device='cuda',requires_grad=True);x.sum().backward()
 r.update(device=torch.cuda.get_device_name(0),backward=bool((x.grad==1).all()))
print(json.dumps(r))"""
        try:
            p=subprocess.run([self.gpu_python(),'-c',code],cwd=self.repo,env=self.environment(),capture_output=True,
                             text=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            result=json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode==0 else dict(cuda=False,error=p.stderr[-1500:])
        except (OSError,ValueError,IndexError,subprocess.TimeoutExpired) as e: result=dict(cuda=False,error=str(e))
        library=Path(self.environment().get('ALTD_GPU_LIB',''))
        result.update(python=self.gpu_python(),simulator_library=str(library) if library.is_file() else None)
        return result

    def examples(self,folder,since=None):
        result=[]
        for p in sorted((folder/'examples').glob('*.json'))[-2000:]:
            if since is not None and p.stat().st_mtime<since:continue
            v=read_json(p,{})
            if v.get('schema')=='altd_training_example_v1':
                result.append(dict(name=p.name,label=v.get('label',p.stem)))
        return result

    @staticmethod
    def process(job):
        try:
            p = psutil.Process(job['pid'])
            return p if abs(p.create_time()-job['process_created']) < .01 and p.is_running() else None
        except (KeyError, psutil.Error):
            return None

    def persist(self, job):
        folder = self.data/'jobs'/job['id']
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder/'job.tmp'
        tmp.write_text(json.dumps(job, indent=2), encoding='utf-8')
        os.replace(tmp, folder/'job.json')

    def output_for(self, tool, args):
        if tool == 'inspect' and args and not args[0].startswith('-'):
            return str(self.path(args[0]))
        options = ('--out-dir', '--out') if tool == 'native' or tool.startswith('native:') else ('--out',)
        for option in options:
            if option in args:
                i = args.index(option)
                if i+1 < len(args):
                    return str(self.path(args[i+1]))
        if tool == 'train' and '--config' in args:
            i = args.index('--config')
            if i+1 < len(args):
                try:
                    cfg = yaml.safe_load(self.path(args[i+1]).read_text(encoding='utf-8-sig'))
                    return str(self.path(cfg['training']['output']))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        if '--checkpoint' in args:
            i = args.index('--checkpoint')
            if i+1 < len(args):
                return str(self.path(args[i+1]).parent)
        return None

    def start(self, payload):
        if not isinstance(payload,dict): raise HTTPException(400,'Job must be an object')
        tool, args = payload.get('tool'), payload.get('arguments', [])
        if not isinstance(tool,str): raise HTTPException(400,'Choose a command')
        if not isinstance(args, list) or len(args)>2048 or any(not isinstance(x, str) or chr(0) in x or len(x)>100_000 for x in args):
            raise HTTPException(400, 'Arguments must be a JSON array of strings')
        args=list(args)
        if '--random-track-settings' in args:
            settings=yaml.safe_load(self.path(args[args.index('--random-track-settings')+1]).read_text(encoding='utf-8-sig'))
            validate_track_length(settings)
        if tool=='train' and '--config' in args:
            config=yaml.safe_load(self.path(args[args.index('--config')+1]).read_text(encoding='utf-8-sig'))
            if config.get('scenario',{}).get('track_mode')=='random':
                settings=yaml.safe_load(self.path(config['scenario'].get('random_track_settings','assets/random_track_settings.json')).read_text(encoding='utf-8-sig'))
                validate_track_length(settings)
        output = self.output_for(tool, args)
        training_tool=tool in ('train','compare','experiments','trace','native:train-scratch') or (tool=='native' and 'train-scratch' in args)
        resume=any(x=='--resume' or x.startswith('--resume=') for x in args)
        if training_tool and output and not resume and Path(output).is_dir() and any(Path(output).iterdir()):
            old=Path(output);output=str(self.path(str(old.with_name(old.name+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]))))
            flag='--out-dir' if tool=='native:train-scratch' or (tool=='native' and 'train-scratch' in args) else '--out'
            if flag in args:args[args.index(flag)+1]=output
            else:args.extend([flag,output])
        command = self.command(tool, args)
        with self.lock:
            if any(j['status'] in ACTIVE for j in self.jobs.values()):
                raise HTTPException(409, 'A job is already running. Stop it or wait for completion.')
            ident = uuid.uuid4().hex
            folder = self.data/'jobs'/ident
            folder.mkdir(parents=True)
            stop = folder/'stop'
            wants_cuda=any(x.startswith('--device=cuda') or (x=='--device' and i+1<len(args) and args[i+1].startswith('cuda')) for i,x in enumerate(args))
            interpreter=self.gpu_python() if wants_cuda else self.python
            argv = [interpreter, '-u', str(ROOT/'runner.py'), '--stop-file', str(stop), *command]
            flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            env=self.environment()
            env['RACING_LAB_STOP_FILE']=str(stop)
            env['PYTHONPATH']=str(ROOT/'runtime')+os.pathsep+env.get('PYTHONPATH','')
            offsets={name:(Path(output)/name).stat().st_size for name in ('updates.jsonl','log.jsonl','episodes.jsonl') if output and (Path(output)/name).is_file()}
            generation_start=None
            if resume and (tool=='native:train-scratch' or (tool=='native' and 'train-scratch' in args)):
                generation_start=(read_json(Path(output)/'checkpoint.json',{}) or {}).get('generation') if output else None
                if generation_start is not None:offsets.pop('log.jsonl',None)
            started=time.time()
            with (folder/'console.log').open('wb') as stream:
                proc = subprocess.Popen(argv, cwd=self.repo, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
            job = dict(id=ident, tool=tool, arguments=args, output=output, status='running',
                       python=interpreter, offsets=offsets, generation_start=generation_start, started=started, finished=None, pid=proc.pid,
                       process_created=psutil.Process(proc.pid).create_time(), exit_code=None)
            self.jobs[ident] = job
            self.persist(job)
            threading.Thread(target=self.watch, args=(job, proc), daemon=True).start()
            return job.copy()

    def finish(self, job, code):
        with self.lock:
            job.update(exit_code=code, finished=time.time(),
                       status='stopped' if job['status']=='stopping' else ('completed' if code in (0, None) else 'failed'))
            self.persist(job)

    def watch(self, job, proc):
        descendants = {}
        while proc.poll() is None:
            try:
                for child in psutil.Process(proc.pid).children(recursive=True):
                    descendants[(child.pid, child.create_time())] = child
            except psutil.Error:
                pass
            time.sleep(.1)
        for child in reversed(list(descendants.values())):
            try:
                if child.is_running(): child.kill()
            except psutil.Error:
                pass
        self.finish(job, proc.returncode)

    def recover(self, job):
        proc = self.process(job)
        if proc:
            try:
                proc.wait()
            except psutil.Error:
                pass
        self.finish(job, None)

    def get(self, ident):
        with self.lock:
            if ident not in self.jobs:
                raise HTTPException(404, 'Job not found')
            return self.jobs[ident]

    def stop(self, ident, force=False):
        with self.lock:
            job = self.get(ident)
            if job['status'] not in ACTIVE:
                return job.copy()
            job['status'] = 'stopping'
            (self.data/'jobs'/ident/'stop').touch()
            self.persist(job)
            if force:
                proc = self.process(job)
                if proc:
                    children = proc.children(recursive=True)
                    for child in reversed(children):
                        try: child.kill()
                        except psutil.Error: pass
                    try: proc.kill()
                    except psutil.Error: pass
            return job.copy()

    def summary(self, folder, offsets=None, since=None,generation_start=None):
        offsets=offsets or {}
        updates = json_lines(folder/'updates.jsonl',offset=offsets.get('updates.jsonl',0)) or json_lines(folder/'log.jsonl',offset=offsets.get('log.jsonl',0))
        if generation_start is not None:updates=[r for r in updates if r.get('generation',-1)>=generation_start]
        episodes = json_lines(folder/'episodes.jsonl',100,offset=offsets.get('episodes.jsonl',0))
        rates = []
        for i in range(0, len(episodes), 10):
            group = episodes[i:i+10]
            returns = [x['cumulative_return'] for x in group if isinstance(x.get('cumulative_return'), (int,float))]
            rates.append(dict(episode=i+len(group), mean_return=sum(returns)/len(returns) if returns else None,
                              completion_rate=sum(bool(x.get('valid_completion')) for x in group)/len(group)))
        files = []
        if folder.is_dir():
            for p in sorted(folder.iterdir(), key=lambda x:x.name):
                if p.is_file() and (since is None or p.stat().st_mtime>=since) and p.suffix.lower() in ('.pt','.json','.jsonl','.npz','.bin','.yaml','.yml'):
                    files.append(dict(name=p.name, path=str(p), size=p.stat().st_size))
        report = None
        for name in ('evaluation.json','report.json','best_validation.json','episodes.json','inspection.json'):
            if (folder/name).is_file() and (since is None or (folder/name).stat().st_mtime>=since) and (folder/name).stat().st_size <= 5_000_000:
                report = read_json(folder/name)
                if report is not None: break
        return dict(updates=updates, episodes=episodes, episode_trend=rates, files=files, report=report)

    def preview(self, folder, example=''):
        if example:
            if not re.fullmatch(r'[a-zA-Z0-9_-]+\.json',example): raise HTTPException(400,'Invalid example name')
            v=read_json(folder/'examples'/example)
            if not v or v.get('schema')!='altd_training_example_v1': raise HTTPException(404,'Example not found')
            points=[dict(position=x['position'],tick=x.get('tick_end'),episode=x.get('episode_id'),reason=x.get('reason'),rotation=x.get('rotation')) for x in v['frames']]
            return dict(cells=v.get('cells',[]),centerline=v.get('centerline',[]),walls=v.get('walls',[]),points=points,recorded=True,note=v.get('note',''),label=v.get('label',''))
        config = read_json(folder/'resolved_config.json')
        if not config:
            manifest = read_json(folder/'manifest.json') or {}
            native = manifest.get('provenance', {}).get('native_config', {})
            route = native.get('route')
        else:
            try: route = read_json(self.path(config['scenario']['track_metadata']))
            except (KeyError, HTTPException): route = None
        route = route or {}
        points = []
        source = sorted(folder.glob('chunk-*.json'))[:100]
        for file in source:
            if file.stat().st_size > 5_000_000: continue
            for row in read_json(file, []) or []:
                p = row.get('position')
                if isinstance(p, list) and len(p)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in p):
                    points.append(dict(position=p, episode=row.get('episode_id'), tick=row.get('tick_end'), reason=row.get('reason')))
                if len(points)>=20_000: break
            if len(points)>=20_000: break
        if not points:
            report = read_json(folder/'evaluation.json') or read_json(folder/'episodes.json') or {}
            reports = report.get('reports', [report])
            for result in reports:
                for row in result.get('episodes', []):
                    p=row.get('position')
                    if isinstance(p,list) and len(p)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in p):
                        points.append(dict(position=p,episode=row.get('episode_id'),tick=row.get('tick_end'),reason=row.get('reason')))
        return dict(cells=[x['polygon'] for x in route.get('cells',[])], points=points,
                    recorded=bool(source), note='Recorded physics decisions' if source else 'Final episode positions; record a trace to replay the full path')


def create_app(repo, python=sys.executable, data=None, token='', hosts=('127.0.0.1','localhost','testserver')):
    state = Dashboard(Path(repo), python, Path(data) if data else None, token, hosts)

    @asynccontextmanager
    async def lifespan(app):
        yield
        running = [j['id'] for j in state.jobs.values() if j['status'] in ACTIVE]
        for ident in running: state.stop(ident)
        deadline = time.time()+8
        while time.time()<deadline and any(state.get(i)['status'] in ACTIVE for i in running):
            import asyncio
            await asyncio.sleep(.2)
        for ident in running:
            if state.get(ident)['status'] in ACTIVE: state.stop(ident, True)

    app = FastAPI(title='Racing Lab', lifespan=lifespan)
    app.state.dashboard = state

    @app.middleware('http')
    async def guard(request: Request, call_next):
        if request.url.hostname not in state.hosts:
            return JSONResponse({'detail':'Host not allowed'}, 403)
        if request.url.path.startswith('/api/'):
            origin = request.headers.get('origin')
            if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'Origin not allowed'},403)
            if state.token and not hmac.compare_digest(request.headers.get('authorization',''), 'Bearer '+state.token):
                return JSONResponse({'detail':'Enter the dashboard access token'},401)
            if request.method not in ('GET','HEAD') and not hmac.compare_digest(request.headers.get('x-csrf-token',''),state.csrf):
                return JSONResponse({'detail':'Refresh the dashboard before submitting changes'},403)
        response=await call_next(request)
        if request.url.path in ('/','/app.js','/style.css'):response.headers['Cache-Control']='no-store'
        return response

    @app.get('/api/bootstrap')
    def bootstrap():
        tools = [dict(id=k,label=v[0],kind='rl') for k,v in PYTHON_TOOLS.items()]
        tools += [dict(id='native',label='Native CLI · root / global options',kind='native')]
        tools += [dict(id='native:'+k,label='Native - '+k,kind='native') for k in NATIVE_TOOLS]
        tools += [dict(id='track',label='Generate / verify RL track',kind='utility')]
        configs = [str(p.relative_to(state.repo)).replace('\\','/') for p in sorted((state.repo/'configs/rl').glob('*.yaml'))]
        return dict(csrf=state.csrf,repo=str(state.repo),python=state.python,tools=tools,configs=configs)

    @app.get('/api/gpu')
    def gpu(): return state.gpu_status()

    @app.get('/api/config-data')
    def config_data(path: str):
        try: return dict(value=yaml.safe_load(state.path(path).read_text(encoding='utf-8-sig')))
        except (OSError,yaml.YAMLError) as e: raise HTTPException(400,str(e))

    @app.get('/api/examples')
    def examples(path: str,since: float | None=None): return state.examples(state.path(path,directory=True),since)

    @app.get('/api/jobs')
    def jobs():
        with state.lock: return sorted((x.copy() for x in state.jobs.values()),key=lambda x:x['started'],reverse=True)

    @app.post('/api/jobs')
    async def start(request: Request):
        try: return state.start(await request.json())
        except (OSError, ValueError, AttributeError) as e: raise HTTPException(400,str(e))

    @app.get('/api/jobs/{ident}')
    def job(ident: str):
        result=state.get(ident).copy()
        result['log']=tail(state.data/'jobs'/ident/'console.log')
        out=Path(result['output']) if result.get('output') else state.data/'jobs'/ident
        result['summary']=state.summary(out if out.is_dir() else out.parent,result.get('offsets'),result['started'],result.get('generation_start'))
        if out.suffix=='.json':
            current=out.is_file() and out.stat().st_mtime>=result['started']
            result['summary'].update(updates=[],episodes=[],episode_trend=[],report=read_json(out) if current else None,files=[dict(name=out.name,path=str(out),size=out.stat().st_size)] if current else [])
        if not result['summary']['updates']:
            result['summary']['updates']=[v for v in json_lines(state.data/'jobs'/ident/'console.log') if 'update' in v or 'generation' in v]
        return clean(result)

    @app.post('/api/jobs/{ident}/stop')
    async def stop(ident: str, request: Request):
        payload=await request.json()
        return state.stop(ident,payload.get('force',False) is True)

    @app.get('/api/runs')
    def runs():
        base=state.repo/'runs'
        if not base.is_dir(): return []
        folders=[p for p in base.iterdir() if p.is_dir()]
        return [dict(name=p.name,path=str(p),modified=p.stat().st_mtime) for p in sorted(folders,key=lambda p:p.stat().st_mtime,reverse=True)[:200]]

    @app.get('/api/run')
    def run(path: str): return clean(state.summary(state.path(path,directory=True)))

    @app.get('/api/preview')
    def preview(path: str,example: str=''): return clean(state.preview(state.path(path,directory=True),example))

    @app.get('/api/file')
    def file(path: str, download: bool=False):
        p=state.path(path)
        if not p.is_file(): raise HTTPException(404,'File not found')
        if download: return FileResponse(p,filename=p.name,media_type='application/octet-stream')
        if p.suffix.lower() not in ('.json','.jsonl','.yaml','.yml','.txt','.log','.md'):
            raise HTTPException(400,'Use Download for binary artifacts')
        return dict(path=str(p),text=tail(p,300_000))

    @app.post('/api/upload')
    async def upload(request: Request, name: str):
        if not re.fullmatch(r'[\w .-]{1,150}',name) or name in ('.','..'):
            raise HTTPException(400,'Invalid filename')
        if Path(name).suffix.lower() not in ('.pt','.json','.yaml','.yml','.bin','.evolution'):
            raise HTTPException(400,'Choose a checkpoint, config, scene or network file')
        folder=state.data/'uploads';folder.mkdir(exist_ok=True)
        p=folder/(uuid.uuid4().hex[:10]+'-'+name)
        size=0
        try:
            with p.open('wb') as stream:
                async for chunk in request.stream():
                    size+=len(chunk)
                    if size>128*1024*1024: raise HTTPException(413,'Upload exceeds 128 MiB')
                    stream.write(chunk)
        except BaseException:
            p.unlink(missing_ok=True)
            raise
        return dict(path=str(p),name=name,size=size)

    @app.post('/api/config')
    async def config(request: Request):
        payload=await request.json();text=payload.get('text','')
        if not isinstance(text,str) or len(text)>200_000: raise HTTPException(400,'Configuration is too large')
        try:
            value=yaml.safe_load(text)
            if not isinstance(value,dict): raise ValueError('Configuration must be an object')
            if 'length' in value:validate_track_length(value)
        except (ValueError,yaml.YAMLError) as e: raise HTTPException(400,str(e))
        folder=state.data/'configs';folder.mkdir(exist_ok=True)
        suffix='.json' if payload.get('format')=='json' else '.yaml'
        p=folder/(uuid.uuid4().hex[:10]+suffix);p.write_text(json.dumps(value,indent=2) if suffix=='.json' else text,encoding='utf-8')
        return dict(path=str(p))

    @app.get('/api/help')
    def help(tool: str):
        if tool not in state.help_cache:
            command=state.command(tool,['--help'])
            if command[0]=='--module': argv=[state.python,'-m',command[1],*command[3:]]
            elif command[0]=='--script': argv=[state.python,command[1],*command[3:]]
            else: argv=command[1:]
            try:
                result=subprocess.run(argv,cwd=state.repo,env=state.environment(),capture_output=True,
                                      text=True,encoding='utf-8',errors='replace',timeout=20,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                state.help_cache[tool]=result.stdout+result.stderr
            except (OSError,subprocess.TimeoutExpired) as e: raise HTTPException(400,str(e))
        return dict(text=state.help_cache[tool])

    app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default_repo = Path(os.environ.get('ALTD_REPO', ROOT.parent))
    parser.add_argument('--repo',type=Path,default=default_repo)
    parser.add_argument('--python',default=sys.executable,help='Python interpreter with altd-rl installed')
    parser.add_argument('--data',type=Path,default=ROOT/'.dashboard')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--token',default=os.environ.get('RACING_LAB_TOKEN',''))
    parser.add_argument('--allow-host',action='append',default=[])
    args=parser.parse_args()
    if args.host not in ('127.0.0.1','localhost','::1') and not args.token:
        parser.error('Set RACING_LAB_TOKEN or --token when listening beyond loopback')
    hosts=set(args.allow_host)|{'localhost','127.0.0.1','::1'}
    if args.host not in ('0.0.0.0','::'): hosts.add(args.host)
    app=create_app(args.repo,args.python,args.data,args.token,hosts)
    print(f'Racing Lab: http://127.0.0.1:{args.port}',flush=True)
    uvicorn.run(app,host=args.host,port=args.port,log_level='info')


if __name__=='__main__': main()
