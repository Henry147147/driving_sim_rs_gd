// Optional browser QA: npm install --prefix .dashboard/browser playwright@1.56.0
const {chromium}=require('../.dashboard/browser/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1050}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DASHBOARD_URL||'http://127.0.0.1:8765');
 await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('Connected'));
 const stamp=Date.now(),train=`runs/dashboard-browser-${stamp}`,trace=`runs/dashboard-trace-${stamp}`;
 async function launch(){
  const response=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
  await page.click('#startBtn');const reply=await response;assert.equal(reply.status(),200,await reply.text());
  const job=await reply.json();await page.waitForFunction(id=>state.job===id&&document.querySelector('#status').textContent==='Completed',job.id,{timeout:90000});return job;
 }
 await page.selectOption('#tool','train');await page.fill('#config','configs/rl/segment_ppo.yaml');await page.locator('#config').dispatchEvent('change');await page.waitForFunction(()=>state.quickConfigPath==='configs/rl/segment_ppo.yaml');await page.fill('#count','12');await page.fill('#output',train);await launch();
 assert.equal(await page.locator('#update').textContent(),'12');await page.screenshot({path:'.dashboard/progress.png',fullPage:true});
 await page.selectOption('#tool','evaluate');await page.fill('#checkpoint',train+'/best.pt');await page.fill('#count','2');await page.fill('#output',train+'/evaluation.json');await launch();
 assert.equal(await page.locator('#completion').textContent(),'2 / 2');
 await page.selectOption('#tool','trace');await page.fill('#checkpoint',train+'/best.pt');await page.fill('#count','2');await page.fill('#output',trace);await launch();
 await page.click('[data-tab="runs"]');await page.fill('#openPath',trace);await page.click('#openRun');await page.waitForFunction(()=>!document.querySelector('#playBtn').disabled);
 await page.click('#playBtn');await page.waitForTimeout(350);await page.click('#playBtn');assert.ok((await page.locator('#replayLabel').textContent()).includes('tick'));
 await page.screenshot({path:'.dashboard/replay.png',fullPage:true});
 await page.selectOption('#tool','native');assert.ok(await page.locator('#useAdvanced').isChecked());await page.click('#showHelp');await page.waitForFunction(()=>document.querySelector('#help').textContent.includes('Usage:'));
 await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.screenshot({path:'.dashboard/mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);console.log('Browser passed: real PPO, evaluation, trace playback, native help, mobile layout, no JS errors.');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
