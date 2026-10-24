// Requires the local NVIDIA GPU setup; runs real CUDA PPO and native training.
const {chromium}=require('../.dashboard/browser/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1050}});const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DASHBOARD_URL||'http://127.0.0.1:8765');
 await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('Connected'));
 const stamp=Date.now();
 await page.click('#editConfig');await page.locator('#editor').waitFor({state:'visible'});
 assert.ok(await page.locator('#configControls input[type=range]').count()>12);
 await page.locator('[data-key="learner.epochs"]').fill('2');
 await page.locator('[data-key="environment.backend"]').selectOption('gpu');
 const config=JSON.parse(await page.locator('#configText').inputValue());
 assert.equal(config.environment.backend,'gpu');assert.ok(config.scenario.scene.endsWith('_gpu.scene.json'));
 config.environment.num_envs=4;config.learner.rollout_steps=16;config.learner.minibatch_size=32;
 config.training.validation_every=1;config.training.validation_episodes=1;config.training.preview_ticks=48;
 await page.click('#rawConfig > summary');await page.fill('#configText',JSON.stringify(config));await page.click('#saveConfig');await page.locator('#editor').waitFor({state:'hidden'});
 assert.ok((await page.locator('#config').inputValue()).includes('.dashboard'));
 await page.selectOption('#device','cuda');await page.fill('#count','2');await page.fill('#output',`runs/dashboard-feature-gpu-${stamp}`);
 async function launch(){
  const response=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
  await page.click('#startBtn');const reply=await response;assert.equal(reply.status(),200,await reply.text());const job=await reply.json();
  await page.waitForFunction(id=>state.job===id&&['Completed','Failed'].includes(document.querySelector('#status').textContent),job.id,{timeout:120000});
  assert.equal(await page.locator('#status').textContent(),'Completed',await page.locator('#log').textContent());return job;
 }
 const job=await launch();assert.ok(job.python.includes('.venv-gpu'));
 const last=await page.evaluate(()=>state.summary.updates.at(-1));assert.equal(last.learner_device,'cuda:0');assert.equal(last.environment_backend,'gpu');
 await page.waitForFunction(()=>document.querySelector('#exampleSelect').options.length>=5,{},{timeout:15000});
 await page.selectOption('#exampleSelect','u000001_e01.json');await page.waitForFunction(()=>state.preview?.label==='Update 1 / epoch 1');
 await page.click('#playBtn');await page.waitForTimeout(150);await page.click('#playBtn');
 assert.ok((await page.locator('#replayLabel').textContent()).includes('tick'));
 await page.screenshot({path:'.dashboard/gpu-examples.png',fullPage:true});
 await page.selectOption('#tool','native:train-scratch');await page.fill('#count','2');await page.fill('#population','16');await page.fill('#nativeTicks','120');await page.fill('#output',`runs/dashboard-feature-native-${stamp}`);await page.check('#nativeGpu');
 await page.click('#editTracks');await page.locator('#editor').waitFor({state:'visible'});await page.locator('[data-key="length.min"]').fill('12');await page.click('#saveConfig');await page.locator('#editor').waitFor({state:'hidden'});
 await launch();await page.waitForFunction(()=>[...document.querySelector('#exampleSelect').options].some(x=>x.textContent.includes('Generation')),{},{timeout:15000});
 await page.waitForFunction(()=>state.preview?.centerline?.length>0&&state.preview?.points?.length>0,{},{timeout:15000});
 await page.screenshot({path:'.dashboard/native-examples.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 assert.deepEqual(errors,[]);console.log('Feature browser passed: config sliders, CUDA learner + GPU simulator, per-epoch playback, GPU native generations and track editor.');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
