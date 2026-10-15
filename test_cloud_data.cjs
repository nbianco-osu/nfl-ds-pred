const {chromium}=require('playwright');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
(async()=>{
 const root=path.join(__dirname,'public_site');
 const snapshot=fs.readFileSync(path.join(root,'data/live_snapshot.json'),'utf8');
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
  for(const failed of [false,true]){
   const page=await browser.newPage();let calls=0;const errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.route('https://nbianco-osu.github.io/**',async route=>{calls++;await route.fulfill({status:failed?503:200,contentType:'application/json',body:failed?'Unavailable':snapshot});});
   await page.route('https://nfl-ds-pred.nbianco0918.chatgpt.site/**',async route=>{
    const url=new URL(route.request().url()),name=url.pathname==='/'?'index.html':url.pathname.slice(1),file=path.resolve(root,name);
    if(!file.startsWith(root+path.sep)||!fs.existsSync(file))return route.fulfill({status:404,body:'Not found'});
    await route.fulfill({path:file,contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.json')?'application/json':file.endsWith('.css')?'text/css':'text/html'});
   });
   await page.goto('https://nfl-ds-pred.nbianco0918.chatgpt.site/');
   await page.waitForFunction(()=>document.querySelectorAll('.matchup-game').length>0);
   assert.equal(calls,1,'All panels share one data snapshot request');
   assert.match(await page.locator('#cloudDataStatus').innerText(),failed?/Cloud data unavailable/:/Cloud data revision/);
   await page.click('#evidence49Tab');
   assert.equal(await page.locator('#strictProvenance').inputValue(),'pregame');
   await page.click('#matchEvidenceTab');
   await page.waitForFunction(()=>document.querySelectorAll('#evidenceWeek option').length===18);
   assert.deepEqual(errors,[]);await page.close();
  }
  console.log('PASS: single cloud snapshot, all panels, explicit bundled fallback.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
