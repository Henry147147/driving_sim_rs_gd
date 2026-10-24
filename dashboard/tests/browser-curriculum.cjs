const {chromium}=require('../.dashboard/browser/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DASHBOARD_URL||'http://127.0.0.1:8765');
 await page.waitForFunction(()=>state.quickConfig&&sensorRows.length===20);
 assert.equal(await page.locator('#configuration').isVisible(),true);
 assert.ok(await page.locator('#quickControls input[type=range]').count()>=8);
 await page.selectOption('[data-quick="environment.backend"]','gpu');await page.selectOption('#device','cuda');
 async function number(key,value){await page.locator(`#quickControls input[type=number][data-quick="${key}"]`).fill(String(value));}
 await number('environment.num_envs',4);await number('learner.rollout_steps',16);await number('learner.minibatch_size',32);await number('learner.epochs',2);await number('scenario.deadline_physics_ticks',360);await number('training.validation_every',0);
 await page.fill('#hiddenSizes','32,16');await page.locator('#hiddenSizes').dispatchEvent('change');
 const toggles=page.locator('#sensorControls input[type=checkbox]');for(let i=0;i<20;i++)if(i!==6&&i!==17)await toggles.nth(i).uncheck();
 await page.selectOption('#addSensorType','speed');await page.click('#addSensor');
 await page.locator('[data-sensor="6"][data-param="Length"]').fill('600');
 assert.ok((await page.locator('#networkShape').textContent()).startsWith('15 → 32 → 16 → 5'));
 const stamp=Date.now(),run=`runs/browser-random-${stamp}`;await page.fill('#count','2');await page.fill('#output',run);
 async function launch(){const pending=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');await page.click('#startBtn');const reply=await pending;assert.equal(reply.status(),200,await reply.text());const job=await reply.json();await page.waitForFunction(id=>state.job===id&&['Completed','Failed'].includes(document.querySelector('#status').textContent),job.id,{timeout:120000});assert.equal(await page.locator('#status').textContent(),'Completed',await page.locator('#log').textContent());return job;}
 await launch();const rows=await page.evaluate(()=>state.summary.updates);assert.equal(rows.length,2);assert.notEqual(rows[0].route_hash,rows[1].route_hash);assert.equal(rows[1].track_index,1);
 await page.waitForFunction(()=>state.preview?.points?.[0]?.tick===0&&state.preview?.points?.at(-1)?.reason);
 const attempt=await page.evaluate(()=>state.preview);assert.ok(attempt.points.length>2);assert.ok(attempt.centerline.length>100);
 const artifacts=await page.evaluate(()=>state.summary.files);const resolved=artifacts.find(x=>x.name==='resolved_config.json');const cfg=await page.evaluate(async path=>(await api('/api/config-data?path='+encodeURIComponent(path))).value,resolved.path);assert.deepEqual(cfg.learner.hidden_sizes,[32,16]);
 const layout=await page.evaluate(async path=>(await api('/api/config-data?path='+encodeURIComponent(path))).value,cfg.scenario.sensors);assert.equal(layout.names.length,3);
 await page.screenshot({path:'.dashboard/random-complete-replay.png',fullPage:true});
 // A late replay response from the previous run must not restore stale data during launch.
 let delayJobs=true;await page.route('**/api/jobs',async route=>{if(route.request().method()==='POST'&&delayJobs)await new Promise(r=>setTimeout(r,3000));await route.continue();});
 await page.route('**/api/preview?**',async route=>{await new Promise(r=>setTimeout(r,900));await route.continue();});
 await page.evaluate(path=>{void loadPreview(path);},run);await page.waitForTimeout(150);
 await page.fill('#count','1');await page.fill('#output',run+'-new');const started=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');await page.click('#startBtn');
 await page.waitForFunction(()=>state.launching===true);await page.waitForTimeout(1400);
 assert.equal(await page.evaluate(()=>state.preview),null);assert.deepEqual(await page.evaluate(()=>state.summary),{});assert.equal(await page.locator('#update').textContent(),'—');assert.equal(await page.locator('#replayLabel').textContent(),'');assert.equal(await page.locator('#report').isVisible(),false);assert.equal(await page.locator('#artifacts .file').count(),0);
 const fresh=await (await started).json();await page.waitForFunction(id=>state.job===id&&document.querySelector('#status').textContent==='Completed',fresh.id,{timeout:120000});delayJobs=false;
 await page.unroute('**/api/preview?**');await page.unroute('**/api/jobs');
 await page.selectOption('#tool','native:train-scratch');await page.click('[data-tab="configuration"]');await page.waitForFunction(()=>document.querySelector('#hiddenSizes').value==='16,16,16,16,12,12,8');
 await page.fill('#hiddenSizes','8');await page.locator('#hiddenSizes').dispatchEvent('change');await page.fill('#population','16');await page.fill('#count','1');await page.fill('#nativeTicks','600');await page.fill('#output',run+'-native');
 const nativeToggles=page.locator('#sensorControls input[type=checkbox]');for(let i=0;i<20;i++)if(i!==17)await nativeToggles.nth(i).uncheck();
 await page.locator('#advanced > summary').click();await page.waitForFunction(()=>document.querySelector('[data-cli="--settings"]'));
 await page.locator('[data-cli="--settings"]').fill('configs/rl/legacy_ars.json');await page.locator('[data-cli="--settings"]').dispatchEvent('change');assert.equal(await page.locator('#useAdvanced').isChecked(),false);
 const native=await launch();assert.ok(native.arguments.includes('--shape'));assert.ok(native.arguments.includes('1,8,5'));
 await page.waitForFunction(()=>state.preview?.points?.[0]?.tick===0&&state.preview?.points?.at(-1)?.reason,{},{timeout:15000});
 await page.click('[data-tab="configuration"]');await page.screenshot({path:'.dashboard/network-sensor-controls.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);assert.deepEqual(errors,[]);
 console.log('Curriculum browser passed: game tracks change, sensor/shape edits used by PPO and ARS, full terminal replays, progress reset survives late replies, CLI helpers and mobile layout.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
