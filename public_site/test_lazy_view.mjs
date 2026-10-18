import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const source=await readFile(new URL('./lazy_view.js',import.meta.url),'utf8');
const settle=()=>new Promise(resolve=>setImmediate(resolve));
function setup({fail=false,broken=false}={}) {
  let observe,requests=0,initializations=0;
  const panel={hidden:true,attrs:{},prepend(...children){this.children=children;},setAttribute(k,v){this.attrs[k]=v;}};
  const window={dashboardFetch:async()=>{requests++;if(fail)throw Error('offline');return {ok:true,json:async()=>({})};}};
  const document={getElementById:()=>panel,createElement:()=>({events:{},setAttribute(){},addEventListener(k,fn){this.events[k]=fn;}})};
  class MutationObserver {constructor(fn){observe=fn;}observe(){}}
  vm.runInNewContext(source,{window,document,MutationObserver});
  window.registerDashboardView({panelId:'test',files:['one','two'],initialize:async()=>{initializations++;if(broken)throw Error('render');}});
  return {panel,open(){panel.hidden=false;observe();},recover(){fail=false;},get requests(){return requests;},get initializations(){return initializations;}};
}
test('hidden views do not fetch; repeated activation initializes only once',async()=>{
  const view=setup();assert.equal(view.requests,0);
  view.open();view.open();await settle();view.open();await settle();
  assert.equal(view.requests,2);assert.equal(view.initializations,1);
  assert.equal(view.panel.attrs['aria-busy'],'false');
});
test('failed input load is retryable before any controls are initialized',async()=>{
  const view=setup({fail:true});view.open();await settle();
  assert.equal(view.initializations,0);assert.equal(view.panel.children[1].hidden,false);
  view.recover();view.panel.children[1].events.click();await settle();
  assert.equal(view.initializations,1);assert.equal(view.panel.children[1].hidden,true);
});
test('renderer failures require reload rather than duplicating partially attached controls',async()=>{
  const view=setup({broken:true});view.open();await settle();view.open();await settle();
  assert.equal(view.initializations,1);assert.match(view.panel.children[0].textContent,/Reload/);
  assert.equal(view.panel.attrs['aria-busy'],'false');
});
