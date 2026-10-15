/* Strict prospective performance never silently falls back to legacy results. */
(() => {
 const el=id=>document.getElementById(id);
 const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const num=v=>Number.isFinite(v)?v.toFixed(3):'Pending';
 const pct=v=>Number.isFinite(v)?(100*v).toFixed(1)+'%':'Pending';
 const table=(heads,id)=>`<div class="table-wrap" tabindex="0"><table><thead><tr>${heads.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody id="${id}"></tbody></table></div>`;
 const tab=document.createElement('button');tab.id='evidence49Tab';tab.setAttribute('role','tab');tab.setAttribute('aria-selected','false');tab.setAttribute('aria-controls','evidence49Panel');tab.tabIndex=-1;tab.textContent='Forecast Accuracy';document.querySelector('.view-tabs').append(tab);
 const panel=document.createElement('section');panel.id='evidence49Panel';panel.hidden=true;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby','evidence49Tab');
 panel.innerHTML=`<h2>Forecast Accuracy</h2><p id="strictNote" role="status">Loading saved evidence.</p>
 <div class="comparison-controls"><label>Evidence<select id="strictProvenance"><option value="pregame">Verified pregame timestamps</option><option value="unverified">Legacy / unverified timestamps</option></select></label><label>Prediction target<select id="strictTarget"><option value="winner">Winner</option><option value="score">Final score</option></select></label><label>Model version<select id="strictVersion"></select></label></div>
 <p id="strictSummary"></p><div style="height:260px;max-width:100%"><canvas id="strictChart" role="img" aria-label="Weekly saved forecast performance"></canvas></div>
 ${table(['Season / week','Graded games','Accuracy / score MAE','Log loss / margin MAE','Brier / total MAE','Spread correct / graded','Total correct / graded'],'strictRows')}
 <h2>Release Readiness</h2><p id="strictRelease">R58 is deferred until sufficient matched prospective results accumulate. No model is approved or promoted automatically.</p>
 <p><a href="./data/forecast_scorecard.json" download>Download timestamp-gated scorecard</a></p>`;
 document.querySelector('.disclaimer').before(panel);
 window.dashboardFetch('./data/forecast_scorecard.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(d=>{
   el('strictNote').textContent=d.note;let chart;
   const groups=()=>d.groups.filter(g=>g.target===el('strictTarget').value&&g.provenance===el('strictProvenance').value);
   const render=()=>{
     const g=groups().find(g=>g.version===el('strictVersion').value),score=el('strictTarget').value==='score';
     const s=g?.summary,rows=g?.weeks??[];
     el('strictSummary').textContent=!s?'No observations in this cohort.':score?`${s.score_n} graded score forecasts. Team score MAE ${num(s.team_score_mae)} points; ${s.spread.pushes} spread pushes and ${s.total.pushes} total pushes excluded.`:`${s.correct} correct of ${s.winner_n} decisive games (${pct(s.accuracy)}). Log loss ${num(s.log_loss)}; ${s.ties} ties excluded.`;
     el('strictRows').innerHTML=rows.map(w=>`<tr>${[`${w.season} / ${w.week}`,score?w.score_n:w.winner_n,score?num(w.team_score_mae):pct(w.accuracy),num(score?w.margin_mae:w.log_loss),num(score?w.total_mae:w.brier),score?`${w.spread.correct} / ${w.spread.n}`:'Not applicable',score?`${w.total.correct} / ${w.total.n}`:'Not applicable'].map(v=>`<td>${esc(v)}</td>`).join('')}</tr>`).join('')||'<tr><td colspan="7">No completed games with this provenance and model version.</td></tr>';
     if(chart)chart.destroy();
     chart=new Chart(el('strictChart'),{type:'line',data:{labels:rows.map(w=>`${w.season} W${w.week}`),datasets:[{label:score?'Team score MAE (points)':'Winner accuracy (%)',data:rows.map(w=>score?w.team_score_mae:w.accuracy==null?null:w.accuracy*100),borderColor:'#087443',backgroundColor:'#087443',tension:0,spanGaps:false}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{beginAtZero:true,...(!score?{max:100}:{})}}}});
   };
   const versions=()=>{el('strictVersion').replaceChildren();groups().forEach(g=>el('strictVersion').add(new Option(g.version,g.version)));render();};
   ['strictTarget','strictProvenance'].forEach(id=>el(id).addEventListener('change',versions));el('strictVersion').addEventListener('change',render);versions();
 }).catch(()=>el('strictNote').textContent='Accuracy report unavailable. No fallback to retrospective or legacy results.');
})();
