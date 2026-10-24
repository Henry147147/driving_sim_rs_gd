"use strict";
const $=id=>document.getElementById(id), state={csrf:"",tools:[],jobs:[],job:null,folder:null,summary:{},preview:null,busy:false,play:null,lastPreview:0,viewVersion:0,previewRequest:0,quickConfig:null,configRequest:0};
let auth=sessionStorage.getItem("racingLabToken")||"";
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num=n=>typeof n==="number"&&Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:2}):"—";
function error(message){$("error").textContent=message;$("error").hidden=!message;}
async function api(path,options={}){
 const headers={...options.headers};if(auth)headers.Authorization="Bearer "+auth;
 if(options.method&&options.method!=="GET")headers["X-CSRF-Token"]=state.csrf;
 if(options.body&&typeof options.body!=="string"&&!(options.body instanceof Blob)){headers["Content-Type"]="application/json";options.body=JSON.stringify(options.body);}
 const r=await fetch(path,{...options,headers});
 if(r.status===401){$("login").hidden=false;$("workspace").hidden=true;throw Error("Enter your access token.");}
 if(!r.ok){let e;try{e=await r.json();}catch{e={detail:r.statusText};}throw Error(e.detail||"Request failed");}return r.json();
}
const task=()=>$("tool").value;
function positive(id){const v=Number($(id).value);if(!Number.isInteger(v)||v<1)throw Error("Enter a positive integer for "+id);return String(v);}
function simpleArgs(){
 const t=task(),cfg=$("config").value.trim(),out=$("output").value.trim(),ckpt=$("checkpoint").value.trim(),a=[];
 const opt=(k,v)=>{if(v)a.push(k,v);};
 if(t==='native:train-scratch')return ['--scene','assets/scenes/rally_template.json','--track-mode','random','--random-track-settings',$('trackSettings').value,'--population',positive('population'),'--generations',positive('count'),'--ticks',positive('nativeTicks'),'--seed',$('seed').value,'--out-dir',out,'--preview-every',$('nativeExamples').checked?'1':'0',...($('nativeGpu').checked?['--gpu']:[]),...(state.cliExtra||[])];
 if(t==="inspect")return [$("tracePath").value.trim()];
 if(t==="train"&&$("resume").checked){if(!ckpt)throw Error("Choose a checkpoint to resume.");opt("--resume",ckpt);}else opt("--config",cfg);
 if(["train","compare"].includes(t)){const seed=Number($("seed").value);if(!Number.isSafeInteger(seed)||seed<0)throw Error("Seed must be non-negative.");opt("--seed",String(seed));}
 if(t==="train"){opt("--updates",positive("count"));opt("--device",$("device").value);}
 if(["compare","benchmark"].includes(t))opt("--steps",positive("count"));
 if(t==="experiments"){opt("--updates",positive("count"));a.push("--flashsac","--seeds");const seeds=$("seeds").value.trim().split(/\s+/);if(seeds.some(s=>!/^\d+$/.test(s)))throw Error("Separate integer seeds with spaces.");a.push(...seeds);}
 if(["evaluate","trace","baseline"].includes(t)){if(!ckpt&&t!=="trace")throw Error("Choose a checkpoint or network file.");opt(t==="baseline"?"--network":"--checkpoint",ckpt);opt("--episodes",positive("count"));}
 opt("--out",out);return a.concat(state.cliExtra||[]);
}
function nativeDefaults(t){return t==="native:train-scratch"?["--scene","assets/scenes/rally_template.json","--track-mode","random","--random-track-settings","assets/random_track_settings.json","--population","256","--generations","3","--ticks","600","--out-dir","runs/dashboard_native"]:["--help"];}
function configure(){
 state.cliExtra=[];const t=task(),native=t==='native'||t.startsWith("native:");let visible=["config","count","seed","output","device"];
 const defaults={train:["Updates",20,"runs/dashboard_ppo"],compare:["Vector steps",256,"runs/dashboard_flashsac"],evaluate:["Episodes",8,"runs/dashboard_evaluation.json"],trace:["Episodes",3,"runs/dashboard_trace"],experiments:["PPO updates",4,"runs/dashboard_comparison"],benchmark:["Vector steps",32,"runs/dashboard_benchmark.json"],baseline:["Episodes",8,"runs/dashboard_baseline.json"]};
 if(t==="train")visible.push("checkpoint","resume");if(t==="compare")visible=["config","count","seed","output"];
 if(["evaluate","trace","baseline"].includes(t))visible=["config","count","checkpoint","output"];
 if(t==="experiments")visible=["config","count","seeds","output"];if(t==="benchmark")visible=["config","count","output"];if(t==="inspect")visible=["trace"];if(native||t==="track")visible=[];if(t==="native:train-scratch"){visible=["count","seed","output"];$("countLabel").textContent="Generations";$("count").value=3;$("output").value="runs/dashboard_native";}$("nativeControls").hidden=t!=="native:train-scratch";$("examplesField").hidden=t!=="train";
 ["config","count","seed","checkpoint","resume","trace","output","device","seeds"].forEach(id=>$(id+"Field").hidden=!visible.includes(id));
 if(defaults[t]){$("countLabel").textContent=defaults[t][0];$("count").value=defaults[t][1];$("output").value=defaults[t][2];}
 $("resume").checked=false;$("useAdvanced").checked=(native&&t!=="native:train-scratch")||t==="track";$("advanced").open=$("useAdvanced").checked;
 let a=[];try{a=native?nativeDefaults(t):t==="track"?["--check"]:simpleArgs();}catch{}
 $("arguments").value=JSON.stringify(a,null,2);$("help").hidden=true;loadQuickSettings();loadCliHelpers();
 $("formNote").textContent=native?"All native options are available in the argument list.":t==="compare"?"Pinned official FlashSAC. Optional research dependencies required.":"Headless jobs save results and checkpoints on disk.";
}
function updateArgs(){if(!$("useAdvanced").checked){try{$("arguments").value=JSON.stringify(simpleArgs(),null,2);}catch{}}}
function tab(name){document.querySelectorAll(".panel").forEach(p=>p.hidden=p.id!==name);document.querySelectorAll("[data-tab]").forEach(b=>b.classList.toggle("active",b.dataset.tab===name));requestAnimationFrame(drawAll);}
async function bootstrap(){try{
 const d=await api("/api/bootstrap");state.csrf=d.csrf;state.tools=d.tools;$("repo").textContent=d.repo;$("python").textContent=d.python;
 $("tool").innerHTML=d.tools.map(t=>`<option value="${esc(t.id)}">${esc(t.label)}</option>`).join("");$("configs").innerHTML=d.configs.map(p=>`<option value="${esc(p)}"></option>`).join("");
 $("workspace").hidden=false;$("login").hidden=true;$("connection").textContent="● Connected locally";configure();await refresh();tab('configuration');
 }catch(e){error(e.message);$("connection").textContent="Not connected";}}
async function refresh(){
 const version=state.viewVersion;const [jobs,runs]=await Promise.all([api("/api/jobs"),api("/api/runs")]);if(version!==state.viewVersion||state.launching)return;state.jobs=jobs;
 $("jobs").innerHTML=jobs.length?jobs.slice(0,25).map(j=>`<div class="list-row"><div><strong>${esc(state.tools.find(t=>t.id===j.tool)?.label||j.tool)}</strong><small>${esc(new Date(j.started*1000).toLocaleString())}</small></div><div class="actions"><span class="badge">${esc(j.status)}</span><button data-job="${j.id}">Open</button></div></div>`).join(""):'<p class="hint">No dashboard jobs yet.</p>';
 $("runList").innerHTML=runs.length?runs.map(r=>`<div class="list-row"><div><strong>${esc(r.name)}</strong><small>${esc(new Date(r.modified*1000).toLocaleString())}</small></div><button data-folder="${esc(r.path)}">Open run</button></div>`).join(""):'<p class="hint">No saved runs yet.</p>';
 const active=jobs.find(j=>["running","stopping"].includes(j.status));$("startBtn").disabled=!!active||state.configLoading||state.launching;$("stopBtn").disabled=!active;$("forceBtn").hidden=!active||active.status!=="stopping";
 if(!state.job&&!state.folder&&jobs.length)state.job=active?.id||jobs[0].id;if(state.job)await loadJob(state.job);
}
async function loadJob(id){
 if(state.job!==id){$("followExamples").checked=true;state.lastPreview=0;}state.job=id;state.folder=null;const version=state.viewVersion,j=await api("/api/jobs/"+id);if(version!==state.viewVersion||state.job!==id||state.folder)return;$("viewTitle").textContent=state.tools.find(t=>t.id===j.tool)?.label||j.tool;
 $("status").textContent=j.status[0].toUpperCase()+j.status.slice(1);$("status").dataset.state=j.status;$("jobTime").textContent=Math.round((j.finished||Date.now()/1000)-j.started)+" seconds · exit "+(j.exit_code??"pending");
 $("log").textContent=j.log||"Waiting for process output…";if($("followLog").checked)$("log").scrollTop=$("log").scrollHeight;state.previewSince=j.started;state.previewTool=j.tool;renderSummary(j.summary);
 let folder=j.output;if(folder&&/\.(json|pt)$/i.test(folder))folder=folder.replace(/[\\/][^\\/]+$/,"");
 if(folder&&Date.now()-state.lastPreview>6000){state.lastPreview=Date.now();await loadPreview(folder);}
}
function renderSummary(s){
 state.summary=s;const last=(s.updates||[]).at(-1)||{};$("update").textContent=num(last.update??last.generation);$("transitions").textContent=last.transitions?num(last.transitions)+" transitions":"Waiting for metrics";$("throughput").textContent=num(last.transitions_per_second??last.car_seconds_per_second);$("throughputUnit").textContent=last.car_seconds_per_second!=null?"car-seconds simulated per second":"transitions per second";
 $("completionTitle").textContent=last.lapped_cars!=null?"LEGACY LAP COUNT":"VALID COMPLETIONS";
 const report=s.report;let r=report?.reports?.[0]||report?.evaluation||report;
 if(!r?.attempts&&(s.episodes||[]).length)r={attempts:s.episodes.length,completions:s.episodes.filter(x=>x.valid_completion).length};
 $("completion").textContent=r?.attempts?`${r.completions??0} / ${r.attempts}`:num(last.lapped_cars);$("lapTime").textContent=r?.median_valid_time!=null?"Median valid time: "+r.median_valid_time.toFixed(3)+" s":"Recent completed attempts";
 const files=s.files||[];$("artifacts").innerHTML=files.length?files.map(f=>`<div class="file"><span title="${esc(f.path)}">${esc(f.name)}</span>${f.name.endsWith('.pt')?`<button data-checkpoint="${esc(f.path)}">Use</button>`:/\.(json|jsonl|yaml|yml|txt|log|md)$/i.test(f.name)?`<button data-view="${esc(f.path)}">View</button>`:""}<button data-download="${esc(f.path)}" title="Download">↓</button></div>`).join(""):'<p class="hint">Artifacts appear as the command writes them.</p>';
 $("checkpoints").innerHTML=files.filter(f=>f.name.endsWith('.pt')).map(f=>`<option value="${esc(f.path)}"></option>`).join("");$("recordTrace").disabled=!files.some(f=>["best.pt","checkpoint.pt"].includes(f.name));
 $("report").hidden=!report;if(report){const light={...report};delete light.episodes;delete light.provenance;if(light.reports)light.reports=light.reports.map(({episodes,...r})=>r);$("report").textContent=JSON.stringify(light,null,2).slice(0,20000);}drawAll();
}
async function openFolder(path){state.viewVersion++;state.previewSince=null;state.previewTool=null;$("followExamples").checked=true;state.job=null;state.folder=path;const summary=await api('/api/run?path='+encodeURIComponent(path));if(state.folder!==path)return;renderSummary(summary);$("viewTitle").textContent=path.split(/[\\/]/).filter(Boolean).at(-1);$("status").textContent="Saved run";$("jobTime").textContent="History from disk";await loadPreview(path);tab('monitor');}
async function loadPreview(path){
 const request=++state.previewRequest,version=state.viewVersion;
 try{
 const examples=await api('/api/examples?path='+encodeURIComponent(path)+(state.previewSince?'&since='+state.previewSince:''));
 if(request!==state.previewRequest||version!==state.viewVersion)return;
 state.previewPath=path;
 const previous=$('exampleSelect').value;
 $('exampleSelect').innerHTML='<option value="">Recorded trace / evaluation</option>'+examples.map(x=>`<option value="${esc(x.name)}">${esc(x.label)}</option>`).join('');
 if(state.job&&['train','compare','experiments','native:train-scratch'].includes(state.previewTool)&&!examples.length){state.preview=null;$('playBtn').disabled=true;$('scrub').disabled=true;$('previewNote').textContent='Waiting for a complete attempt from this job.';$('replayLabel').textContent='';drawTrack();return;}
 const choice=$('followExamples').checked&&examples.length?examples.at(-1).name:examples.some(x=>x.name===previous)?previous:'';
 $('exampleSelect').value=choice;
 const preview=await api('/api/preview?path='+encodeURIComponent(path)+'&example='+encodeURIComponent(choice));
 if(request!==state.previewRequest||version!==state.viewVersion)return;
 const changed=state.preview?.label!==preview.label||state.preview?.points?.length!==preview.points.length||state.loadedPreviewPath!==path;
 state.preview=preview;state.loadedPreviewPath=path;$('previewNote').textContent=preview.note;
 $('scrub').max=Math.max(0,preview.points.length-1);
 if(changed&&!state.play)$('scrub').value=0;
 $('scrub').disabled=preview.points.length<2;$('playBtn').disabled=preview.points.length<2||!preview.recorded;drawTrack();
 }catch(e){if(request===state.previewRequest&&version===state.viewVersion){state.preview=null;drawTrack();}}
}
function clearProgress(){
 state.viewVersion++;state.previewRequest++;state.job=null;state.folder=null;state.summary={};state.preview=null;state.previewPath=null;state.previewSince=null;state.loadedPreviewPath=null;state.lastPreview=0;
 if(state.play)clearInterval(state.play);state.play=null;
 $('playBtn').textContent='Play replay';$('playBtn').disabled=true;$('scrub').value=0;$('scrub').max=0;$('scrub').disabled=true;
 $('exampleSelect').innerHTML='<option value="">Waiting for this run</option>';$('followExamples').checked=true;
 $('previewNote').textContent='Waiting for a complete driving attempt from this run.';$('replayLabel').textContent='';
 $('log').textContent='Starting process...';$('report').textContent='';$('report').hidden=true;$('completionTitle').textContent='VALID COMPLETIONS';
 $('viewTitle').textContent=state.tools.find(t=>t.id===task())?.label||task();$('metric').value='value_loss';
 renderSummary({});$('status').textContent='Starting';$('status').dataset.state='running';$('jobTime').textContent='Starting process...';drawAll();
}
function canvas(id){const el=$(id),r=el.getBoundingClientRect();if(!r.width)return null;const d=window.devicePixelRatio||1;el.width=r.width*d;el.height=r.height*d;const ctx=el.getContext('2d');ctx.scale(d,d);return {ctx,w:r.width,h:r.height};}
function chart(id,data,color,empty){
 data=data.filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));$(empty).hidden=data.length>0;const c=canvas(id);if(!c)return;const {ctx,w,h}=c;ctx.font="10px Segoe UI";ctx.strokeStyle="#2b373e";ctx.fillStyle="#98abb6";
 const ys=data.map(x=>x[1]),xs=data.map(x=>x[0]);let lo=data.length?Math.min(...ys):0,hi=data.length?Math.max(...ys):1;if(lo===hi){lo-=.5;hi+=.5;}
 for(let i=0;i<4;i++){const y=10+i*(h-34)/3;ctx.beginPath();ctx.moveTo(46,y);ctx.lineTo(w-8,y);ctx.stroke();ctx.fillText((hi-(hi-lo)*i/3).toFixed(Math.abs(hi)>100?0:2),0,y+4);}
 if(!data.length)return;const a=Math.min(...xs),b=Math.max(...xs),px=x=>46+(x-a)/(b-a||1)*(w-56),py=y=>10+(hi-y)/(hi-lo)*(h-34);ctx.beginPath();data.forEach((p,i)=>i?ctx.lineTo(px(p[0]),py(p[1])):ctx.moveTo(px(p[0]),py(p[1])));ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
 if(data.length===1){ctx.beginPath();ctx.arc(px(data[0][0]),py(data[0][1]),3,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}ctx.fillStyle="#98abb6";ctx.fillText(String(a),46,h-3);ctx.fillText(String(b),w-35,h-3);
}
function drawTrack(){
 const c=canvas('trackCanvas');if(!c)return;const {ctx,w,h}=c,p=state.preview;
 if(!p||(!(p.cells||[]).length&&!(p.centerline||[]).length)){ctx.fillStyle="#98abb6";ctx.font="13px Segoe UI";ctx.textAlign="center";ctx.fillText("A recorded trace will appear here.",w/2,h/2);return;}
 const all=[...(p.cells||[]).flat(),...(p.centerline||[]),...(p.walls||[]).flat()],xs=all.map(v=>v[0]),ys=all.map(v=>v[1]),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),s=Math.min((w-60)/(maxX-minX||1),(h-35)/(maxY-minY||1));
 const xy=([x,y])=>[(x-(minX+maxX)/2)*s+w/2,(y-(minY+maxY)/2)*s+h/2];ctx.lineWidth=1;ctx.strokeStyle="#526670";ctx.fillStyle="#243139";
 (p.cells||[]).forEach(cell=>{ctx.beginPath();cell.forEach((v,i)=>{const [x,y]=xy(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();ctx.fill();ctx.stroke();});
 if(p.centerline?.length){ctx.strokeStyle='#526670';ctx.lineWidth=12;ctx.beginPath();p.centerline.forEach((v,i)=>{const [x,y]=xy(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();ctx.stroke();}
 ctx.strokeStyle='#98abb6';ctx.lineWidth=1;(p.walls||[]).forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(...xy(a));ctx.lineTo(...xy(b));ctx.stroke();});
 const points=p.points.slice(0,Number($("scrub").value)+1);ctx.strokeStyle="#b5f36b";ctx.lineWidth=2;ctx.beginPath();points.forEach((v,i)=>{const [x,y]=xy(v.position);i&&v.episode===points[i-1].episode?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
 const last=points.at(-1);if(last){const [x,y]=xy(last.position);ctx.save();ctx.translate(x,y);const prior=points.at(-2);ctx.rotate(last.rotation??(prior?Math.atan2(last.position[1]-prior.position[1],last.position[0]-prior.position[0])+Math.PI/2:0));ctx.fillStyle='#64d5da';ctx.fillRect(-4,-7,8,14);ctx.fillStyle='#101518';ctx.fillRect(-3,-4,6,3);ctx.restore();$("replayLabel").textContent=`Episode ${last.episode??"—"} · tick ${last.tick??"—"}${last.reason?" · "+last.reason:""}`;}else $("replayLabel").textContent="No recorded positions yet.";
}
function drawAll(){
 const s=state.summary,rows=s.updates||[],current=$('metric').value;
 if(rows.length&&!rows.some(r=>Number.isFinite(r[current]))){const candidate=['value_loss','best_score','mean_score'].find(k=>rows.some(r=>Number.isFinite(r[k])));if(candidate)$('metric').value=candidate;}
 const native=!(s.episode_trend||[]).length&&rows.some(r=>Number.isFinite(r.mean_score));
 $('returnTitle').textContent=native?'Mean population score':'Episode return';
 chart('returnChart',native?rows.map(x=>[x.generation,x.mean_score]):(s.episode_trend||[]).map(x=>[x.episode,x.mean_return]),'#b5f36b','returnEmpty');
 chart('learningChart',rows.map((x,i)=>[x.update??x.generation??i,x[$('metric').value]]),'#64d5da','learningEmpty');drawTrack();
}

async function download(path){const r=await fetch('/api/file?download=true&path='+encodeURIComponent(path),{headers:auth?{Authorization:'Bearer '+auth}:{}});if(!r.ok)throw Error('Download failed');const url=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=url;a.download=path.split(/[\\/]/).at(-1);a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
$('tool').onchange=configure;$('launchForm').oninput=e=>{updateArgs();const id=e.target.id;if(['population','nativeTicks','count'].includes(id)){const slider=document.querySelector('#quickControls [data-native="'+id+'"]');if(slider){slider.max=Math.max(Number(slider.max),Number(e.target.value));slider.value=e.target.value;slider.previousElementSibling.textContent=e.target.value;}}};
$('launchForm').onsubmit=async e=>{e.preventDefault();try{error('');if(!$('useAdvanced').checked&&['train','native:train-scratch'].includes(task())&&!$('resume').checked){if(state.configLoading)throw Error('Configuration is still loading.');if(state.configError)throw Error(state.configError);}const a=$('useAdvanced').checked?JSON.parse($('arguments').value):simpleArgs();if(!Array.isArray(a)||a.some(x=>typeof x!=='string'))throw Error('Arguments must be a JSON array of strings.');state.launching=true;clearProgress();$('startBtn').disabled=true;tab('monitor');if(task()==='train'&&!$('useAdvanced').checked&&!$('resume').checked){
 const cfg=state.quickConfig&&state.quickConfigPath===$('config').value?structuredClone(state.quickConfig):(await api('/api/config-data?path='+encodeURIComponent($('config').value))).value;
 cfg.learner.hidden_sizes=hiddenShape();cfg.scenario.sensors=await saveSensorLayout();
 cfg.training.preview_every=$('enableExamples').checked?1:0;delete cfg.training.preview_ticks;if(cfg.scenario.track_mode==='random')cfg.scenario.random_track_settings=$('ppoTrackPath').value;
 const saved=await api('/api/config',{method:'POST',body:{text:JSON.stringify(cfg,null,2)}});a[a.indexOf('--config')+1]=saved.path;
 }
 if(task()==='native:train-scratch'&&!$('useAdvanced').checked){
 const layout=selectedSensorLayout(),network=(await api('/api/config-data?path=assets/networks/rally.json')).value,model=(await api('/api/config-data?path=assets/models/rally.json')).value;
 network.inputs=layout.names;model.sensor_layout=layout;
 const n=await api('/api/config',{method:'POST',body:{text:JSON.stringify(network),format:'json'}}),m=await api('/api/config',{method:'POST',body:{text:JSON.stringify(model),format:'json'}});
 a.push('--network',n.path,'--model',m.path,'--shape',[layout.names.length,...hiddenShape(),network.outputs.length].join(','));
 }
 const j=await api('/api/jobs',{method:'POST',body:{tool:task(),arguments:a}});state.launching=false;state.job=j.id;state.folder=null;state.preview=null;state.lastPreview=0;$('followExamples').checked=true;renderSummary({});$('status').textContent='Running';$('status').dataset.state='running';$('jobTime').textContent='Starting process…';$('log').textContent='Starting process…';tab('monitor');await refresh();}catch(e){state.launching=false;$('startBtn').disabled=false;error(e.message);}};
$('refresh').onclick=()=>refresh().catch(e=>error(e.message));
async function stop(force){if(force&&!confirm('Force stop this job and its child processes? Work since the last saved checkpoint will be discarded.'))return;try{const j=state.jobs.find(j=>['running','stopping'].includes(j.status));if(j)await api(`/api/jobs/${j.id}/stop`,{method:'POST',body:{force}});await refresh();}catch(e){error(e.message);}}
$('stopBtn').onclick=()=>stop(false);$('forceBtn').onclick=()=>stop(true);
$('showHelp').onclick=async()=>{try{$('help').hidden=false;$('help').textContent='Loading…';$('help').textContent=(await api('/api/help?tool='+encodeURIComponent(task()))).text;}catch(e){error(e.message);}};
let sensorRows=[],networkRequest=0;
function hiddenShape(){const widths=$('hiddenSizes').value.split(/[ ,]+/).filter(Boolean).map(Number);if(!widths.length||widths.some(n=>!Number.isInteger(n)||n<1||n>8192))throw Error('Hidden layers must contain positive widths up to 8192.');return widths;}
function renderShape(){try{const h=hiddenShape();$('layerCount').value=h.length;$('layerCountValue').textContent=h.length;
 $('layerControls').innerHTML=h.map((v,i)=>`<label>Hidden layer ${i+1}<output>${v}</output><input type="range" data-layer="${i}" min="1" max="${Math.max(1024,v)}" value="${v}"></label>`).join('');
 const count=sensorRows.filter(s=>s.enabled).length,extras=task()==='train'?12:0;
 $('networkShape').textContent=[count+extras,...h,5].join(' → ')+(extras?' (actor; value head: 1)':'');
 $('networkNote').textContent=`${count} selected sensors${extras?' + 12 required RL state/history inputs':''}. `+($('resume').checked?'Resume uses the checkpoint sensor layout and architecture.':'A fresh run uses this layout and shape.');
 }catch(e){$('networkShape').textContent=e.message;}}
async function loadNetworkControls(cfg,request){const n=++networkRequest;
 try{const data=(await api('/api/config-data?path='+encodeURIComponent(cfg?.scenario.sensors||'configs/rl/rally.sensors.json'))).value;
 if(n!==networkRequest||request!==state.configRequest)return;
 sensorRows=data.sensors.map((descriptor,i)=>({name:data.names[i],descriptor,enabled:true}));$('hiddenSizes').value=(cfg?.learner.hidden_sizes||[16,16,16,16,12,12,8]).join(',');renderSensors();renderShape();$('resume').onchange();
 }catch(e){error(e.message);}}
function renderSensors(){
 $('sensorControls').innerHTML=sensorRows.map((row,i)=>{const d=row.descriptor;
 const params=Object.entries(d).filter(([k,v])=>k!=='$type'&&typeof v==='number').map(([key,v])=>`<label>${esc(key)}<input type="number" data-sensor="${i}" data-param="${key}" step="any" value="${v}"></label>`).join('');
 const offsets=Array.isArray(d.PositionOffset)?d.PositionOffset.map((v,axis)=>`<label>Offset ${axis?'Y':'X'}<input type="number" data-sensor="${i}" data-axis="${axis}" step="any" value="${v}"></label>`).join(''):'';
 return `<div class="sensor-row"><label class="check"><input type="checkbox" data-sensor="${i}" data-enabled="1" ${row.enabled?'checked':''}>${esc(row.name)} <small>${esc(d.$type)}</small></label><div class="row">${params}${offsets}</div><button type="button" data-move="${i}" data-direction="-1" aria-label="Move sensor up">↑</button> <button type="button" data-move="${i}" data-direction="1" aria-label="Move sensor down">↓</button></div>`;}).join('');
}
function selectedSensorLayout(){const rows=sensorRows.filter(r=>r.enabled);if(!rows.length)throw Error('Enable at least one sensor.');for(const r of rows){for(const [k,v] of Object.entries(r.descriptor)){if(typeof v==='number'&&!Number.isFinite(v))throw Error('Sensor values must be finite.');if(['Length','MaxAcceleration','MaxDistance'].includes(k)&&v<=0)throw Error(k+' must be positive.');}}
 return {names:rows.map(r=>r.name),sensors:rows.map(r=>r.descriptor)};}
async function saveSensorLayout(){return (await api('/api/config',{method:'POST',body:{text:JSON.stringify(selectedSensorLayout()),format:'json'}})).path;}
$('hiddenSizes').onchange=()=>{try{if(state.quickConfig)state.quickConfig.learner.hidden_sizes=hiddenShape();renderShape();}catch(e){error(e.message);}};
$('layerCount').oninput=()=>{let sizes=[];try{sizes=hiddenShape();}catch{}const n=Number($('layerCount').value);while(sizes.length<n)sizes.push(64);$('hiddenSizes').value=sizes.slice(0,n).join(',');$('hiddenSizes').onchange();};
$('layerControls').oninput=e=>{if(e.target.dataset.layer==null)return;const h=hiddenShape();h[Number(e.target.dataset.layer)]=Number(e.target.value);$('hiddenSizes').value=h.join(',');if(state.quickConfig)state.quickConfig.learner.hidden_sizes=h;e.target.previousElementSibling.textContent=e.target.value;const count=sensorRows.filter(s=>s.enabled).length+(task()==='train'?12:0);$('networkShape').textContent=[count,...h,5].join(' → ');};
$('sensorControls').oninput=e=>{const el=e.target;if(el.dataset.sensor==null)return;const row=sensorRows[Number(el.dataset.sensor)];if(el.dataset.enabled)row.enabled=el.checked;else if(el.dataset.axis!=null)row.descriptor.PositionOffset[Number(el.dataset.axis)]=Number(el.value);else if(el.dataset.param)row.descriptor[el.dataset.param]=Number(el.value);renderShape();};
$('sensorControls').onclick=e=>{if(e.target.dataset.move==null||$('resume').checked)return;const i=Number(e.target.dataset.move),j=i+Number(e.target.dataset.direction);if(j<0||j>=sensorRows.length)return;[sensorRows[i],sensorRows[j]]=[sensorRows[j],sensorRows[i]];renderSensors();renderShape();};
$('addSensor').onclick=()=>{const type=$('addSensorType').value,d={$type:type};if(type==='raycast')Object.assign(d,{Degrees:0,Length:800});if(type==='trackCurvature')Object.assign(d,{MinLookaheadDistance:256,MaxLookaheadDistance:512});if(type==='grip')d.PositionOffset=[0,0];if(type.startsWith('acceleration'))d.MaxAcceleration=1000;if(type==='distanceFromWall')d.MaxDistance=800;sensorRows.push({name:type+'_'+sensorRows.length,descriptor:d,enabled:true});renderSensors();renderShape();};
$('resume').onchange=()=>{renderShape();$('sensorControls').querySelectorAll('input').forEach(x=>x.disabled=$('resume').checked);$('hiddenSizes').disabled=$('resume').checked;$('layerCount').disabled=$('resume').checked;$('layerControls').querySelectorAll('input').forEach(x=>x.disabled=$('resume').checked);$('addSensor').disabled=$('resume').checked;};
let quickDefinitions=[];
async function loadQuickSettings(){
 const request=++state.configRequest,t=task();state.configLoading=true;state.configError='';$('startBtn').disabled=true;$('quickSettings').hidden=!['train','compare','benchmark','native:train-scratch'].includes(t);$('networkControls').hidden=!['train','native:train-scratch'].includes(t);
 if($('quickSettings').hidden){state.configLoading=false;$('startBtn').disabled=state.jobs.some(j=>['running','stopping'].includes(j.status));return;}
 if(t==='native:train-scratch'){
 $('ppoTrackSettings').hidden=true;$('moreSettings').hidden=true;
 const networkReady=loadNetworkControls(null,request);$('quickControls').innerHTML=[['population','Population',2,32768,2],['nativeTicks','Generation duration (ticks)',60,14400,60],['count','Generations',1,1000,1]].map(([id,label,min,max,step])=>`<label>${label}<output>${esc($(id).value)}</output><input type="range" data-native="${id}" min="${min}" max="${max}" step="${step}" value="${esc($(id).value)}"></label>`).join('');await networkReady;if(request===state.configRequest){state.configLoading=false;$('startBtn').disabled=state.jobs.some(j=>['running','stopping'].includes(j.status));}return;
 }
 try{
 const path=$('config').value,cfg=(await api('/api/config-data?path='+encodeURIComponent(path))).value;
 if(request!==state.configRequest)return;
 state.quickConfig=cfg;state.quickConfigPath=path;const networkReady=loadNetworkControls(cfg,request);cfg.scenario.track_mode??='fixed';cfg.training.track_every??=1;
 $('moreSettings').hidden=false;$('ppoTrackSettings').hidden=t!=='train';$('ppoTrackPath').value=cfg.scenario.random_track_settings||'assets/random_track_settings.json';
 quickDefinitions=configFields.filter(x=>['environment.num_envs','environment.action_repeat_ticks','environment.backend','scenario.task_kind','scenario.deadline_physics_ticks','learner.rollout_steps','learner.minibatch_size','learner.epochs','learner.learning_rate','training.validation_every'].includes(x[0]));
 if(t!=='train')quickDefinitions=quickDefinitions.filter(x=>!x[0].startsWith('learner.'));
 if(t==='train')quickDefinitions.unshift(['scenario.track_mode','Track source',['random','fixed']],['training.track_every','New game track every N rollouts',1,100,1]);
 $('quickControls').innerHTML=quickDefinitions.map(([key,label,lo,hi,step])=>{const v=getField(cfg,key);if(Array.isArray(lo))return `<label>${label}<select data-quick="${key}">${lo.map(x=>`<option value="${x}" ${x===v?'selected':''}>${x==='random'?'Random game tracks':x==='fixed'?'Fixed teaching ring':x}</option>`).join('')}</select></label>`;
 return `<label>${label}<output>${esc(v)}</output><div class="slider-number"><input type="range" data-quick="${key}" min="${Math.min(lo,v)}" max="${Math.max(hi,v)}" step="${step}" value="${v}"><input type="number" data-quick="${key}" min="${lo}" step="${step}" value="${v}" aria-label="${label} value"></div></label>`;}).join('');await networkReady;if(request===state.configRequest){state.configLoading=false;$('startBtn').disabled=state.jobs.some(j=>['running','stopping'].includes(j.status));}
 }catch(e){state.configLoading=false;state.configError=e.message;error(e.message);}
}
$('quickControls').oninput=e=>{const el=e.target;if(el.dataset.native){$(el.dataset.native).value=el.value;el.previousElementSibling.textContent=el.value;updateArgs();return;}const key=el.dataset.quick;if(!key)return;
 const value=el.tagName==='SELECT'?el.value:Number(el.value);setField(state.quickConfig,key,value);
 if(key==='environment.backend'&&/rally_ring(?:_gpu)?\.scene\.json$/.test(state.quickConfig.scenario.scene))state.quickConfig.scenario.scene=value==='gpu'?'configs/rl/tracks/rally_ring_gpu.scene.json':'configs/rl/tracks/rally_ring.scene.json';
 if(key==='scenario.task_kind'&&value==='full_lap')state.quickConfig.scenario.target_gates=64;
 if(key==='scenario.track_mode'&&value==='fixed'){state.quickConfig.scenario.scene=state.quickConfig.environment.backend==='gpu'?'configs/rl/tracks/rally_ring_gpu.scene.json':'configs/rl/tracks/rally_ring.scene.json';state.quickConfig.scenario.track_metadata='configs/rl/tracks/rally_ring.rules.json';}
 const label=el.closest('label');if(label.querySelector('output'))label.querySelector('output').textContent=String(value);label.querySelectorAll('input').forEach(x=>{if(x!==el)x.value=value;});
};
$('config').onchange=loadQuickSettings;$('moreSettings').onclick=()=>editConfiguration(false);
$('editPpoTracks').onclick=()=>editConfiguration('ppoTracks');
async function loadCliHelpers(){
 const tool=task();$('cliHelpers').textContent='Loading CLI option helpers...';
 try{const {text}=await api('/api/help?tool='+encodeURIComponent(tool));if(task()!==tool)return;
 const lines=text.split('\n'),fields=[];
 lines.forEach((line,index)=>{const m=line.match(/^\s+(?:-\w,\s+)?(--[\w-]+)(?:[ =]([<A-Z_][^\s]*))?/);if(!m||m[1]==='--help')return;const covered=task()==='native:train-scratch'?['--scene','--track-mode','--random-track-settings','--population','--generations','--ticks','--seed','--out-dir','--preview-every','--gpu','--shape','--network','--model']:['--config','--out','--seed','--updates','--device','--resume','--checkpoint','--episodes','--steps','--seeds'];if(covered.includes(m[1]))return;
 let end=index+1;while(end<lines.length&&!/^\s+(?:-\w,\s+)?--[\w-]+/.test(lines[end]))end++;const flag=m[1],takes=!!m[2],near=lines.slice(index,end).join('\n');let choices=near.match(/\[possible values: ([^\]]+)\]/)?.[1].split(',').map(x=>x.trim());if(!choices&&/Possible values:/.test(near))choices=[...near.matchAll(/^\s+-\s+([\w-]+)(?::|\s|$)/gm)].map(x=>x[1]);if(flag==='--math-profile')choices=['proton','win10-fma3','win11-fma3'];
 let args=[];try{args=JSON.parse($('arguments').value);}catch{}const i=args.indexOf(flag),value=i>=0&&takes?args.slice(i+1).filter((x,j,arr)=>arr.slice(0,j+1).every(y=>!y.startsWith('--'))).join(' '):'';
 const checked=i>=0?'checked':'';
 let input=!takes?`<input type="checkbox" data-cli="${flag}" ${checked}>`:choices?`<select data-cli="${flag}"><option value="">CLI default</option>${choices.map(x=>`<option ${x===value?'selected':''}>${esc(x)}</option>`).join('')}</select>`:`<input data-cli="${flag}" value="${esc(value)}" placeholder="${esc(m[2])}">`;
 if(takes&&!choices&&/seconds|mutation|threads|batch-count|checkpoint-every|tracks-per-generation|track-buffer|stop-|limit|chunk/.test(flag)){const fraction=/mutation|percent/.test(flag),max=/mutation/.test(flag)?1:/percent/.test(flag)?100:/tracks-per/.test(flag)?32:10000;input=`<input type="range" data-cli-range="${flag}" min="0" max="${max}" step="${fraction?0.001:1}" value="${value||0}"><input type="number" step="any" data-cli="${flag}" value="${esc(value)}" placeholder="CLI default">`;}
 fields.push(`<label class="${takes?'':'check'}" title="${esc(line.trim())}">${takes?esc(flag):''}${input}${takes?'':esc(flag)}</label>`);
 });$('cliHelpers').innerHTML='<p class="hint">CLI option helpers. These helpers add CLI options to the form. The complete argument list below stays available for manual overrides.</p>'+fields.join('');
 }catch(e){$('cliHelpers').textContent=e.message;}
}
$('cliHelpers').oninput=e=>{if(e.target.dataset.cliRange){const input=e.target.closest('label').querySelector('[data-cli]');input.value=e.target.value;$('cliHelpers').onchange({target:input});}};
$('cliHelpers').onchange=e=>{if(!e.target.dataset.cli)return;const extras=[];$('cliHelpers').querySelectorAll('[data-cli]').forEach(el=>{if(el.type==='checkbox'){if(el.checked)extras.push(el.dataset.cli);}else if(el.value.trim())extras.push(el.dataset.cli,...(el.dataset.cli==='--seeds'?el.value.trim().split(/\s+/):[el.value]));});state.cliExtra=extras;updateArgs();};
let editing={},editingTracks=false;
const configFields=[
 ['environment.num_envs','Parallel cars',1,2048,1],['environment.action_repeat_ticks','Physics ticks per action',1,16,1],
 ['environment.backend','Simulation',['cpu','gpu']],['environment.math_profile','Math',['proton','win10-fma3','win11-fma3']],
 ['scenario.task_kind','Task',['segment','full_lap']],
 ['reward.profile','Reward',['dense_progress_time_v1','strict_time_v1']],
 ['scenario.target_gates','Target gates',1,64,1],['scenario.deadline_physics_ticks','Deadline ticks',60,14400,60],
 ['learner.rollout_steps','Rollout steps',8,2048,8],['learner.minibatch_size','Minibatch size',8,8192,8],
 ['learner.epochs','PPO epochs',1,16,1],['learner.learning_rate','Learning rate',0.00001,0.003,0.00001],
 ['learner.clip_epsilon','Policy clip',0.01,0.4,0.01],['learner.gae_lambda','GAE lambda',0.8,1,0.01],
 ['learner.value_coefficient','Value weight',0,2,0.05],['learner.entropy_coefficient','Entropy weight',0,0.1,0.001],
 ['learner.max_grad_norm','Gradient norm',0.1,5,0.1],['learner.target_kl','KL stop',0.001,0.1,0.001],
 ['training.validation_every','Validate every N updates (0 off)',0,100,1],['training.validation_episodes','Validation episodes',1,64,1],
 ['training.preview_every','Examples every N updates (0 off)',0,100,1],
 ['logging.trace_mode','Record training traces',['off','sampled_chunks']],['logging.episode_summaries','Episode summaries','boolean']
];
const trackFields=[['length.min','Minimum tiles',4,100,2],['length.max','Maximum tiles',4,100,2],['allow_double','Allow double tiles','doubles'],['surfaces.pool','Road surfaces','surfaces']];
function getField(obj,path){return path.split('.').reduce((o,k)=>o?.[k],obj);}
function setField(obj,path,value){const parts=path.split('.');let o=obj;for(const k of parts.slice(0,-1))o=o[k]??={};o[parts.at(-1)]=value;}
function controls(){
 $('configControls').innerHTML=(editingTracks?trackFields:configFields).map(([key,label,lo,hi,step])=>{
 let v=getField(editing,key);if(v==null&&key==='training.preview_every')v=0;if(v==null&&key==='training.preview_ticks')v=240;if(v==null)return '';
 if(Array.isArray(lo))return `<label>${esc(label)}<select data-key="${key}">${lo.map(x=>`<option ${x===v?'selected':''}>${x}</option>`).join('')}</select></label>`;
 if(typeof lo==='string')return `<label class="check"><input type="checkbox" data-key="${key}" data-kind="${lo}" ${(lo==='doubles'?v.includes(true):lo==='surfaces'?v.length>1:v)?'checked':''}>${esc(label)}${lo==='surfaces'?' (off = asphalt only)':''}</label>`;
 return `<label>${esc(label)}<output>${esc(v)}</output><input type="range" data-key="${key}" min="${Math.min(lo,v)}" max="${Math.max(hi,v)}" step="${step}" value="${v}"></label>`;
 }).join('');
}
async function editConfiguration(tracks){try{
 editingTracks=tracks;editing=tracks==='ppoTracks'?(await api('/api/config-data?path='+encodeURIComponent($('ppoTrackPath').value))).value:!tracks&&state.quickConfig&&state.quickConfigPath===$('config').value?structuredClone(state.quickConfig):(await api('/api/config-data?path='+encodeURIComponent($(tracks?'trackSettings':'config').value))).value;
 if(!editing||typeof editing!=='object')throw Error('Configuration must be an object');
 $('configText').value=JSON.stringify(editing,null,2);controls();$('editor').showModal();
 }catch(e){error(e.message);}}
$('editConfig').onclick=()=>editConfiguration(false);$('editTracks').onclick=()=>editConfiguration(true);
$('configControls').oninput=e=>{const el=e.target;if(!el.dataset.key)return;let v=el.type==='range'?Number(el.value):el.type==='checkbox'?el.checked:el.value;
 if(el.dataset.kind==='doubles')v=v?[false,true]:[false];if(el.dataset.kind==='surfaces'){v=v?[0,1,2]:[0];editing.surfaces.count=v.length>1?[1,2,3]:[1];}
 if(el.dataset.key==='environment.backend'&&['configs/rl/tracks/rally_ring.scene.json','configs/rl/tracks/rally_ring_gpu.scene.json'].includes(editing.scenario?.scene)){editing.scenario.scene=v==='gpu'?'configs/rl/tracks/rally_ring_gpu.scene.json':'configs/rl/tracks/rally_ring.scene.json';}
 setField(editing,el.dataset.key,v);if(el.dataset.key==='scenario.task_kind'&&v==='full_lap'){editing.scenario.target_gates=64;controls();}if(el.type==='range')el.previousElementSibling.textContent=el.value;$('configText').value=JSON.stringify(editing,null,2);
};
$('configText').onchange=()=>{try{editing=JSON.parse($('configText').value);controls();}catch{/* Backend also accepts YAML; do not discard raw edits. */}};
$('saveConfig').onclick=async()=>{try{const saved=await api('/api/config',{method:'POST',body:{text:$('configText').value,format:editingTracks?'json':'yaml'}});$(editingTracks==='ppoTracks'?'ppoTrackPath':editingTracks?'trackSettings':'config').value=saved.path;$('editor').close();if(!editingTracks)loadQuickSettings();updateArgs();}catch(e){error(e.message);}};
$('checkGpu').onclick=async()=>{const b=$('checkGpu');b.disabled=true;$('gpuStatus').textContent='Testing CUDA and backward pass...';try{const g=await api('/api/gpu');$('gpuStatus').textContent=g.cuda?`${g.device} / PyTorch ${g.torch} / backward pass ${g.backward?'passed':'failed'}. Simulator library: ${g.simulator_library||'missing; run setup'}. GPU Python: ${g.python}`:`CUDA unavailable: ${g.error||g.torch}. Run the setup command below.`;}catch(e){$('gpuStatus').textContent=e.message;}finally{b.disabled=false;}};
$('exampleSelect').onchange=()=>{$('followExamples').checked=false;if(state.previewPath)loadPreview(state.previewPath);};
$('upload').onclick=()=>$('fileUpload').click();$('fileUpload').onchange=async e=>{try{const f=e.target.files[0];if(f){$('checkpoint').value=(await api('/api/upload?name='+encodeURIComponent(f.name),{method:'POST',body:f})).path;updateArgs();}}catch(e){error(e.message);}};
$('openRun').onclick=()=>openFolder($('openPath').value.trim()).catch(e=>error(e.message));$('metric').onchange=drawAll;$('scrub').oninput=drawTrack;
$('playBtn').onclick=()=>{if(state.play){clearInterval(state.play);state.play=null;$('playBtn').textContent='Play replay';return;}$('scrub').value=0;$('playBtn').textContent='Pause replay';state.play=setInterval(()=>{const n=Number($('scrub').value)+1;if(n>Number($('scrub').max)){$('playBtn').click();return;}$('scrub').value=n;drawTrack();},66);};
$('recordTrace').onclick=()=>{const f=state.summary.files?.find(f=>f.name==='best.pt')||state.summary.files?.find(f=>f.name==='checkpoint.pt');if(!f)return;$('tool').value='trace';configure();$('checkpoint').value=f.path;$('output').value=f.path.replace(/[\\/][^\\/]+$/,'')+'/trace-'+Date.now();updateArgs();window.scrollTo({top:0,behavior:'smooth'});};
document.addEventListener('click',async e=>{const b=e.target.closest('button');if(!b)return;try{if(b.dataset.tab)tab(b.dataset.tab);if(b.dataset.job){await loadJob(b.dataset.job);tab('monitor');}if(b.dataset.folder)await openFolder(b.dataset.folder);if(b.dataset.checkpoint){$('checkpoint').value=b.dataset.checkpoint;updateArgs();}if(b.dataset.download)await download(b.dataset.download);if(b.dataset.view){$('fileText').textContent=(await api('/api/file?path='+encodeURIComponent(b.dataset.view))).text;$('fileTitle').textContent=b.dataset.view.split(/[\\/]/).at(-1);$('fileViewer').showModal();}}catch(e){error(e.message);}});
$('loginBtn').onclick=()=>{auth=$('token').value;sessionStorage.setItem('racingLabToken',auth);bootstrap();};window.addEventListener('resize',drawAll);
bootstrap();setInterval(async()=>{if(state.busy||!state.csrf)return;state.busy=true;try{await refresh();$('connection').textContent='● Connected locally';}catch(e){$('connection').textContent='Connection interrupted';error(e.message);}finally{state.busy=false;}},1500);
