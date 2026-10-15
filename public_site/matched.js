window.setupMatchedEvaluation = async function () {
  const el = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = x => Number.isFinite(x) ? x.toFixed(3) : 'Unavailable';
  const wrapLabel = name => {
    const lines = [''];
    for (const word of String(name).split(' ')) {
      const last = lines.length - 1;
      if (lines[last] && lines[last].length + word.length + 1 > 23) lines.push(word);
      else lines[last] += (lines[last] ? ' ' : '') + word;
    }
    return lines;
  };
  let data;
  try {
    const response = await window.dashboardFetch('./data/matched_comparison.json');
    if (!response.ok) throw new Error('Unavailable');
    data = await response.json();
    if (data.model_count !== 64 || data.models.length !== 64) throw new Error('Incomplete');
  } catch {
    el('matchedStatus').textContent = 'Matched evaluation unavailable. Schedule forecasts are unaffected.';
    el('matchedControls').hidden = true;
    return;
  }
  el('matchedNote').textContent = data.note;
  for (const year of data.seasons) el('matchedSeason').add(new Option(String(year), String(year)));
  let comparisonChart, seasonChart;
  const metric = row => el('matchedSeason').value === 'all' ? row.overall : row.seasons[el('matchedSeason').value];
  const intervalPlugin = {id:'pairedIntervals', afterDatasetsDraw(chart) {
    const {ctx,scales:{x}} = chart;
    const intervals = chart.data.datasets[0].intervals;
    if (!intervals) return;
    ctx.save(); ctx.lineWidth = 2;
    chart.getDatasetMeta(0).data.forEach((point,i) => {
      const [lo,hi] = intervals[i];
      ctx.strokeStyle = hi < 0 ? '#087443' : lo > 0 ? '#b42318' : '#667085';
      const a=x.getPixelForValue(lo),b=x.getPixelForValue(hi);
      ctx.beginPath();ctx.moveTo(a,point.y);ctx.lineTo(b,point.y);
      ctx.moveTo(a,point.y-4);ctx.lineTo(a,point.y+4);
      ctx.moveTo(b,point.y-4);ctx.lineTo(b,point.y+4);ctx.stroke();
    });ctx.restore();
  }};
  function details() {
    const row=data.models.find(m=>m.id===el('matchedModel').value);
    if(!row) return;
    el('matchedDetailName').textContent=row.name;
    el('matchedFoldRows').innerHTML=data.folds.map(f=>{
      const r=f.recipes.find(m=>m.id===row.id),m=row.seasons[String(f.season)];
      return `<tr><td>${f.season}</td><td>${m.n}</td><td>${esc(f.final_fit_through)}</td><td>${esc(f.final_calibration_through)}</td><td>${esc(f.test_start)}</td><td>${r.selection_losses.length}</td><td>${r.feature_count}</td><td>${(100*r.missing_test_fraction).toFixed(1)}%</td><td>${number(m.log_loss)}</td><td>${number(m.brier)}</td></tr>`;
    }).join('');
    el('matchedSettings').textContent=data.folds.map(f=>`${f.season}: ${f.recipes.find(m=>m.id===row.id).selected_estimator}`).join('\n\n');
    seasonChart?.destroy();
    if(window.Chart) seasonChart=new Chart(el('matchedSeasonChart'),{type:'bar',data:{labels:data.seasons,datasets:[
      {label:row.name,data:data.seasons.map(y=>row.seasons[y].log_loss),backgroundColor:'#175cd3'},
      {label:data.baseline_name,data:data.seasons.map(y=>data.baseline_seasons[y].log_loss),backgroundColor:'#087443'}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{beginAtZero:true,title:{display:true,text:'Log loss (lower is better)'}}}}});
  }
  function render() {
    const market=el('matchedMarket').value,group=el('matchedGroup').value,key=el('matchedMetric').value;
    const rows=data.models.filter(r=>(market==='all'||r.market===(market==='market'))&&(group==='all'||r.group===group))
      .sort((a,b)=>metric(a)[key]-metric(b)[key]||a.name.localeCompare(b.name));
    const baseline=el('matchedSeason').value==='all'?data.baseline:data.baseline_seasons[el('matchedSeason').value];
    el('matchedStatus').textContent=`${rows.length} research recipes · ${baseline.n} identical decisive games each · ${baseline.weeks} season-week blocks · Annual rolling folds · Production unchanged`;
    el('matchedBaseline').textContent=`${data.baseline_name}: log loss ${number(baseline.log_loss)}, Brier ${number(baseline.brier)}, accuracy ${(baseline.accuracy*100).toFixed(1)}%.`;
    el('matchedRows').innerHTML=rows.map(r=>{const m=metric(r);return `<tr><td>${esc(r.name)}</td><td>${esc(r.group)}</td><td>${r.market?'Yes':'No'}</td><td>${m.n}</td><td>${(100*m.accuracy).toFixed(1)}%</td><td>${number(m.log_loss)}</td><td>${number(m.brier)}</td><td>${number(m.delta_log_loss)}</td><td>${m.delta_log_loss_ci95.map(number).join(' to ')}</td><td>${number(m.delta_brier)}</td><td>${m.delta_brier_ci95.map(number).join(' to ')}</td></tr>`;}).join('')||'<tr><td colspan="11">No recipes in this group.</td></tr>';
    const selected=el('matchedModel').value;
    el('matchedModel').replaceChildren(...rows.map(r=>new Option(r.name,r.id)));
    if(rows.some(r=>r.id===selected))el('matchedModel').value=selected;
    el('matchedDetail').hidden=!rows.length;
    comparisonChart?.destroy();
    const delta=key==='brier'?'delta_brier':'delta_log_loss';
    const top=rows.slice(0,12), intervals=top.map(r=>metric(r)[`${delta}_ci95`]);
    el('matchedChartCaption').textContent=`${top.length} lowest-${key==='brier'?'Brier':'log-loss'} recipes in selected group. Difference from baseline with paired 95% intervals; below zero favors the recipe. Unadjusted for multiple comparisons.`;
    if(window.Chart&&top.length) comparisonChart=new Chart(el('matchedComparisonChart'),{type:'bar',plugins:[intervalPlugin],
      data:{labels:top.map(r=>r.name),datasets:[{label:'Difference vs baseline',data:top.map(r=>metric(r)[delta]),intervals,backgroundColor:'#d0d5dd',borderWidth:0}]},
      options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},
        scales:{x:{min:Math.min(0,...intervals.map(a=>a[0]))-.01,max:Math.max(0,...intervals.map(a=>a[1]))+.01,title:{display:true,text:delta==='delta_brier'?'Brier difference':'Log-loss difference'}},y:{ticks:{font:{size:11},callback:function(value){return wrapLabel(this.getLabelForValue(value));}}}}}});
    details();
  }
  ['matchedSeason','matchedMarket','matchedGroup','matchedMetric'].forEach(id=>el(id).addEventListener('change',render));
  el('matchedModel').addEventListener('change',details);
  el('matchedTab').addEventListener('click',()=>requestAnimationFrame(()=>{comparisonChart?.resize();seasonChart?.resize();}));
  render();
};
window.setupMatchedEvaluation();
