import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash,webcrypto} from 'node:crypto';
import vm from 'node:vm';
const source=await readFile(new URL('./data_source.js',import.meta.url),'utf8');
function setup({hosted=false,offline=false,corrupt=false,failOnce=false}={}){
  const payload='[]',sha=createHash('sha256').update(payload).digest('hex');
  const files=Object.fromEntries(['predictions.json','metrics.json'].map(name=>[name,{sha256:sha,path:`revisions/${sha}.json`,bytes:2}]));
  const manifest={schema_version:2,revision:'a'.repeat(64),files},calls=[],status={setAttribute(){}};
  const location=new URL(hosted?'https://nfl-ds-pred.nbianco0918.chatgpt.site/':'https://example.test/');
  const window={dispatchEvent(){},fetch:async url=>{
    url=String(url);calls.push(url);if(offline&&url.includes('github.io'))throw Error('offline');
    if(failOnce && !url.includes('dashboard_manifest')) { failOnce=false; throw Error('temporary failure'); }
    return new Response(url.includes('dashboard_manifest')?JSON.stringify(manifest):corrupt?'{}':payload);
  }};
  const document={createElement:()=>status,querySelector:()=>({append(){}}),addEventListener(){}};
  vm.runInNewContext(source,{window,document,location,URL,Response,AbortSignal,TextDecoder,crypto:webcrypto,CustomEvent:class{},setTimeout});
  return {window,calls,status};
}
test('one manifest and one pinned dataset request for concurrent readers',async()=>{
  const {window,calls}=setup();
  const rows=await Promise.all([window.dashboardFetch('./data/predictions.json'),window.dashboardFetch('./data/predictions.json')]);
  assert.deepEqual(await rows[0].json(),[]);assert.equal(calls.length,2);
  assert.equal(window.dashboardDelivery.loaded,1);
});
test('corruption fails closed without fallback or mixed revisions',async()=>{
  const {window,calls}=setup({corrupt:true});
  await assert.rejects(window.dashboardFetch('./data/predictions.json'),/integrity/);
  assert.equal(calls.length,2);assert.deepEqual([...window.dashboardDelivery.failures],['predictions.json']);
});
test('cloud manifest failure pins an explicitly labelled bundled revision',async()=>{
  const {window}=setup({hosted:true,offline:true});
  await window.dashboardFetch('./data/predictions.json');assert.equal(window.dashboardDelivery.source,'Bundled fallback');
});
test('missing dataset cannot silently load a different revision',async()=>{
  const {window,calls}=setup();await assert.rejects(window.dashboardFetch('./data/unknown.json'),/absent/);
  assert.equal(calls.length,1);
});

test('failed requests can recover on the same pinned revision',async()=>{
  const {window,calls}=setup({failOnce:true});
  await assert.rejects(window.dashboardFetch('./data/predictions.json'),/temporary/);
  assert.equal(window.dashboardDelivery.loaded,0);
  await window.dashboardFetch('./data/predictions.json');
  assert.equal(window.dashboardDelivery.loaded,1);
  assert.deepEqual([...window.dashboardDelivery.failures],[]);
  assert.equal(calls.filter(url=>url.includes('dashboard_manifest')).length,1);
  assert.equal(calls[1],calls[2]);
});

test('repeated corruption never counts as a successful load',async()=>{
  const {window,calls}=setup({corrupt:true});
  for(let attempt=0;attempt<2;attempt++) await assert.rejects(window.dashboardFetch('./data/predictions.json'),/integrity/);
  assert.equal(window.dashboardDelivery.loaded,0);
  assert.equal(window.dashboardDelivery.failures.length,1);
  assert.equal(calls.length,3);
});
