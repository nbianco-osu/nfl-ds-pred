window.setupExperiments = async function () {
  const el=id=>document.getElementById(id);
  const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=value=>Number.isFinite(value)?value.toFixed(3):'Unavailable';
  async function read(name) {const r=await fetch(`./data/${name}.json`);if(!r.ok)throw Error('Unavailable');return r.json();}
  let chart;
  try {
    const data=await read('score_experiments');
    el('scoreExperimentNote').textContent=data.note;
    function render() {
      const year=el('experimentSeason').value,target=el('experimentTarget').value;
      const rows=data.models.map(m=>({...m,summary:year==='all'?m.overall:m.seasons[year]}));
      el('scoreExperimentStatus').textContent=`${rows.length} research distributions; ${rows[0].summary.games} identical games each. Available ties included. Production unchanged.`;
      el('scoreExperimentRows').innerHTML=rows.map(r=>{const s=r.summary,m=s.metrics;return `<tr><td>${esc(r.name)}</td><td>${s.games}</td><td>${num(m[target+'_mae'])}</td><td>${num(m[target+'_crps'])}</td><td>${(100*m[target+'_coverage80']).toFixed(1)}%</td><td>${num(m[target+'_width80'])}</td><td>${num(m[target+'_interval_score80'])}</td><td>${s.line_counts[target]??'Not applicable'}</td><td>${num(m[target+'_brier3'])}</td><td>${num(m[target+'_log_loss3'])}</td></tr>`;}).join('');
      chart?.destroy();
      if(window.Chart)chart=new Chart(el('scoreExperimentChart'),{type:'bar',data:{labels:rows.map(r=>r.name),datasets:[
        {label:'Observed coverage',data:rows.map(r=>100*r.summary.metrics[target+'_coverage80']),backgroundColor:['#175cd3','#b42318','#087443','#667085']},
        {type:'line',label:'Nominal 80%',data:rows.map(()=>80),borderColor:'#101828',borderDash:[5,4],pointRadius:0}]},
        options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{min:0,max:100,title:{display:true,text:'80% interval coverage (%)'}},x:{ticks:{callback:function(v){return this.getLabelForValue(v).split(' ');}}}}}});
    }
    ['experimentSeason','experimentTarget'].forEach(id=>el(id).addEventListener('change',render));render();
    el('experimentsTab').addEventListener('click',()=>requestAnimationFrame(()=>chart?.resize()));
  }catch {el('scoreExperimentStatus').textContent='Score experiments unavailable. Production forecasts are unaffected.';}
  try {
    const data=await read('temporal_stack'),s=data.summary,b=data.baseline;
    el('stackNote').textContent=data.note;
    el('stackStatus').textContent=`${data.base_count} no-market base recipes; meta-fit ${data.meta_fit_season}, calibration ${data.calibration_season}, evaluation ${data.test_season}: ${s.n} decisive games. Research only.`;
    el('stackRows').innerHTML=[['Temporal stack',s],['Equal-weight mean',b]].map(([name,m])=>`<tr><td>${name}</td><td>${m.n}</td><td>${(m.accuracy*100).toFixed(1)}%</td><td>${num(m.log_loss)}</td><td>${num(m.brier)}</td><td>${num(m.delta_log_loss)}</td><td>${m.delta_log_loss_ci95.map(num).join(' to ')}</td></tr>`).join('');
    el('stackReview').textContent=s.delta_log_loss_ci95[0]<=0&&s.delta_log_loss_ci95[1]>=0?'Difference is inconclusive: the paired interval includes zero. No promotion.':'Retrospective difference only; prospective evidence and human review still required.';
  }catch {el('stackStatus').textContent='Temporal stacking results unavailable.';}
  try {
    const data=await read('pregame_inputs');
    el('pregameNote').textContent=data.note;
    el('pregameStatus').textContent=`${data.observations} timestamped supplied observations. Missing inputs remain unavailable.`;
    const weeks=[...new Set(data.games.map(g=>g.week))];
    weeks.forEach(w=>el('pregameWeek').add(new Option(`Week ${w}`,String(w))));
    function render(){
      const rows=data.games.filter(g=>el('pregameWeek').value==='all'||g.week===Number(el('pregameWeek').value));
      const starter=(g,team)=>{const r=g.observations['starter:'+team];return r?`${esc(r.player_name)}<small>Supplied confirmation; observed ${esc(r.observed_at)}</small>`:'Unavailable';};
      el('pregameRows').innerHTML=rows.map(g=>{const w=g.observations.weather;return `<tr><td>${g.week}</td><td>${esc(g.away_team)} at ${esc(g.home_team)}</td><td>${starter(g,g.away_team)}</td><td>${starter(g,g.home_team)}</td><td>${w?`${num(w.temperature_f)} F / ${num(w.wind_mph)} mph<small>Observed ${esc(w.observed_at)}</small>`:'Unavailable'}</td></tr>`;}).join('');
    }
    el('pregameWeek').addEventListener('change',render);render();
  }catch{el('pregameStatus').textContent='Pregame input archive unavailable.';}
  try {
    const data=await read('warehouse_status');
    el('warehouseStatus').textContent=data.note;
    el('warehouseRows').innerHTML=Object.entries(data.counts).map(([name,count])=>`<tr><td>${esc(name.replaceAll('_',' '))}</td><td>${count}</td></tr>`).join('');
    el('warehouseVerified').textContent=`Verified ${data.verified_at}. Private tables; no browser database credentials.`;
  }catch{el('warehouseStatus').textContent='Warehouse verification summary unavailable.';}
};
window.setupExperiments();
