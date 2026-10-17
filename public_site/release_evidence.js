/* Read-only release evidence, separate from production and prospective grades. */
(() => {
  const el=id=>document.getElementById(id),esc=v=>String(v??'Unavailable').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=v=>Number.isFinite(v)?v.toFixed(3):'Unavailable';
  const tab=document.createElement('button');tab.id='releaseEvidenceTab';tab.setAttribute('role','tab');tab.setAttribute('aria-selected','false');tab.setAttribute('aria-controls','releaseEvidencePanel');tab.tabIndex=-1;tab.textContent='Release Evidence';document.querySelector('.view-tabs').append(tab);
  const panel=document.createElement('section');panel.id='releaseEvidencePanel';panel.hidden=true;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',tab.id);
  panel.innerHTML='<h2>Release Evidence</h2><p id="releaseEvidenceStatus" role="status">Research candidates only. Production remains unchanged.</p><div id="releaseEvidenceBody"></div><h3>Data Delivery</h3><p id="deliveryEvidence" role="status"></p><p><a href="./data/iteration_r61.json" download>Download evidence report</a></p>';
  document.querySelector('.disclaimer').before(panel);
  const table=(heads,rows)=>`<div class="table-wrap" tabindex="0"><table><thead><tr>${heads.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(v=>`<td>${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  function delivery(){const d=window.dashboardDelivery;if(d)el('deliveryEvidence').textContent=`${d.source||'Loading'} | Revision ${d.revision?.slice(0,12)||'pending'} | ${d.loaded} verified datasets | ${(d.bytes/1048576).toFixed(2)} MiB loaded | ${d.failures.length?`Unavailable: ${d.failures.join(', ')}`:'No reported delivery failures'}`;}
  window.addEventListener('deliverychange',delivery);delivery();
  let loaded=false;
  async function load(){
    if(loaded)return;loaded=true;
    try{
      const r=await window.dashboardFetch('./data/iteration_r61.json');if(!r.ok)throw Error();const d=await r.json();
      const v=d.training_vintage||{},c=d.canary||{},f=d.fixed_review||{},p=d.possession_research||{},m=p.mean_metrics||{};
      el('releaseEvidenceStatus').textContent=`${d.generation}. ${f.ready_for_review?'Ready for human review; not approved.':'Not ready for promotion review.'} Production unchanged.`;
      el('releaseEvidenceBody').innerHTML=`<h3>Training Snapshot</h3>${table(['Measure','Value'],[['Completed 2026 games',v.current_season_games],['Snapshot through',v.training_through],['Newly reconciled games',(v.new_completed_games||[]).join(', ')||'None'],['Missing completed games',(v.missing_completed_games||[]).length],['Prospectively matched results',f.matched_games]])}<p>${esc(v.note)}</p>
        <h3>Missing Input Counts</h3>${table(['Feature','Games with missing values'],Object.entries(v.missing_fields||{}).filter(([,n])=>n>0).map(([key,n])=>[window.readableFeature?window.readableFeature(key):key.replaceAll('_',' '),n]))}
        <p>Zero missing values can include historical fallbacks; it does not establish current source availability.</p>
        <h3>Canary Rehearsal</h3>${table(['Check','Result'],[['Single-game p95',`${number(c.single_game_p95_ms)} ms / ${c.proposed_latency_budget_ms??'?'} ms proposed budget`],['Latency budget',c.within_proposed_latency_budget?'Met':'Not met or unavailable'],['Disposable rollback',c.selector_rollback_passed?'Passed':'Unavailable / failed'],['Finite predictions',c.finite_outputs?'Passed':'Unavailable / failed'],['Live activation','Not performed']])}<p>${esc(c.note)}</p>
        <h3>Scoring-Event Research</h3><p>${p.games??0} matched retrospective games. ${d.research_is_stale?'Saved research uses an older source vintage.':'Research and training snapshot use the same vintage.'} Lower CRPS is better; these are not prospective results.</p>${table(['Target','Possession CRPS','Independent Poisson CRPS'],[['Combined score',number(m.possession_total_crps),number(m.poisson_total_crps)],['Home margin',number(m.possession_margin_crps),number(m.poisson_margin_crps)]])}<p>${esc(p.note)}</p>
        <h3>Prospective Review Policy</h3><p>${esc(f.note)}</p>${table(['Completed week','Minimum matched games'],(f.policy?.review_weeks||[]).map((w,i)=>[w,f.policy.minimum_games[i]]))}
        <h3>Pruning Recommendations</h3>${table(['Target','Selected representatives'],Object.entries(d.pruning||{}).map(([task,item])=>[task,item.selected.join(', ')]))}<p>Training-only analysis of six representatives per target, not the full library. Serving weights are unchanged.</p>`;
    }catch{loaded=false;el('releaseEvidenceStatus').textContent='Release evidence unavailable. Select this tab again to retry. Production forecasts are unaffected.';}
  }
  tab.addEventListener('click',load);
  new MutationObserver(()=>{if(!panel.hidden)load();}).observe(panel,{attributes:true,attributeFilter:['hidden']});
})();
