import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const source=await readFile(new URL('./season_chart.js',import.meta.url),'utf8');

test('navigation initializes without chart data and arrows stay in the visible group',()=>{
  const elements=new Map();
  function element(id='') {
    const e={id,hidden:false,children:[],attrs:{},events:{},value:'',
      append(child){this.children.push(child);},before(child){elements.set(child.id,child);},
      setAttribute(k,v){this.attrs[k]=v;},getAttribute(k){return this.attrs[k];},
      addEventListener(k,fn){this.events[k]=fn;},focus(){this.focused=true;}};
    if(id) elements.set(id,e);return e;
  }
  const tabs=['dashboard','scores','season','profiles','research','monitor'].map(name=>{
    const tab=element(name+'Tab');tab.attrs={'aria-controls':name+'Panel','aria-selected':String(name==='dashboard')};
    element(name+'Panel');return tab;
  });
  for(const id of ['chartTeam','chartMeasure','teamCharts','libraryLink']) element(id);
  const bar=element('bar');
  let requests=0;
  const window={dashboardFetch(){requests++;return new Promise(()=>{});}};
  const document={getElementById:id=>elements.get(id),querySelectorAll:()=>tabs,querySelector:()=>bar,createElement:()=>element()};
  vm.runInNewContext(source,{window,document});
  window.setupSeasonCharts({teams:[]});
  assert.equal(requests,0);
  assert.equal(window.dashboardTabsReady,true);
  assert.deepEqual(tabs.filter(t=>!t.hidden).map(t=>t.id),['dashboardTab','scoresTab']);
  tabs[0].events.keydown({key:'ArrowLeft',preventDefault(){}});
  assert.equal(tabs[1].attrs['aria-selected'],'true');
  const group=elements.get('viewGroup');group.value='Teams';group.events.change();
  assert.equal(requests,1);
  assert.deepEqual(tabs.filter(t=>!t.hidden).map(t=>t.id),['seasonTab','profilesTab']);
  elements.get('seasonTab').events.keydown({key:'End',preventDefault(){}});
  assert.equal(elements.get('profilesTab').focused,true);
  elements.get('libraryLink').events.click();
  assert.equal(group.value,'Evaluation');
  assert.equal(elements.get('researchTab').hidden,false);
  window.setupSeasonCharts({teams:[]});
  assert.equal(requests,1,'setup is idempotent');
});
