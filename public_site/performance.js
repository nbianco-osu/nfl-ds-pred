window.setupPerformance = async function () {
  const el = id => document.getElementById(id);
  const number = (n, digits = 3) => n == null ? 'Not available' : n.toFixed(digits);
  const rate = n => n == null ? 'Not available' : `${(100*n).toFixed(1)}%`;
  const cell = value => `<td>${value}</td>`;
  const card = (label, value) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
  try {
    const response = await fetch('./data/performance.json');
    if (!response.ok) throw new Error('Saved performance unavailable');
    const data = await response.json();
    el('performanceNote').textContent = data.note;
    for (const group of data.groups) {
      const option = new Option(group.version === 'all' ? 'All versions (mixed)' : group.version === 'legacy' ? 'Legacy (incomplete provenance)' : group.version, group.version);
      el('performanceVersion').add(option);
    }
    el('performanceVersion').value = data.default_version;
    let calibrationChart;
    function render() {
      const group = data.groups.find(g => g.version === el('performanceVersion').value);
      const s = group.summary;
      el('performanceMetrics').innerHTML = card('Winner accuracy', `${rate(s.accuracy)} (${s.correct}/${s.winner_n})`) +
        card('Log loss / Brier', `${number(s.log_loss)} / ${number(s.brier)}`) +
        card(`Score MAE (${s.score_n} games)`, `${number(s.team_score_mae, 1)} pts`) +
        card('Margin / total MAE', `${number(s.margin_mae, 1)} / ${number(s.total_mae, 1)}`);
      el('performanceCounts').textContent = `${s.completed} completed games; ${s.ties} ties excluded from winner metrics. Spread: ${s.spread.correct}/${s.spread.n}, ${s.spread.pushes} pushes, ${s.spread.excluded} ungraded. Total: ${s.total.correct}/${s.total.n}, ${s.total.pushes} pushes, ${s.total.excluded} ungraded. Small samples are not evidence of an advantage.`;
      el('performanceCounts').textContent += ` Timestamped winner forecasts: ${s.timestamped_winner_n??0}/${s.winner_n}. Accuracy 95% interval: ${s.accuracy_ci95?.map(rate).join(' to ')??'Unavailable'}.`;
      el('weeklyPerformance').innerHTML = group.weeks.map(w => `<tr>${[w.week, `${w.correct}/${w.winner_n}`, rate(w.accuracy), number(w.log_loss), number(w.brier), w.score_n, number(w.margin_mae, 1), number(w.total_mae, 1)].map(cell).join('')}</tr>`).join('') || '<tr><td colspan="8">No completed games in this version.</td></tr>';
      el('calibrationRows').innerHTML = s.calibration.map(b => `<tr>${[`${Math.round(b.lower*100)}-${Math.round(b.upper*100)}%`, b.n, rate(b.predicted), rate(b.observed)].map(cell).join('')}</tr>`).join('');
      if(calibrationChart) calibrationChart.destroy();
      if(window.Chart) calibrationChart = new Chart(el('calibrationChart'),{
        type:'scatter',data:{datasets:[
          {label:'Observed (95% Wilson interval)',data:s.calibration.filter(b=>b.n).map(b=>({x:b.predicted*100,y:b.observed*100,lo:b.observed_ci95[0]*100,hi:b.observed_ci95[1]*100,n:b.n})),backgroundColor:'#087443',pointRadius:5},
          {label:'Perfect calibration',data:[{x:0,y:0},{x:100,y:100}],showLine:true,borderColor:'#64748b',borderDash:[4,4],pointRadius:0}
        ]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{min:0,max:100,title:{display:true,text:'Predicted home wins (%)'}},y:{min:0,max:100,title:{display:true,text:'Observed home wins (%)'}}},plugins:{tooltip:{callbacks:{label:item=>item.raw.n?`${item.raw.n} games: ${item.raw.y.toFixed(1)}% (${item.raw.lo.toFixed(1)}-${item.raw.hi.toFixed(1)}%)`: 'Perfect calibration'}}}},
        plugins:[{id:'intervals',afterDatasetsDraw(chart){const ctx=chart.ctx;ctx.save();ctx.strokeStyle='#087443';for(const p of chart.data.datasets[0].data){const x=chart.scales.x.getPixelForValue(p.x),lo=chart.scales.y.getPixelForValue(p.lo),hi=chart.scales.y.getPixelForValue(p.hi);ctx.beginPath();ctx.moveTo(x,lo);ctx.lineTo(x,hi);ctx.moveTo(x-4,lo);ctx.lineTo(x+4,lo);ctx.moveTo(x-4,hi);ctx.lineTo(x+4,hi);ctx.stroke();}ctx.restore();}}]
      });
    }
    el('performanceVersion').addEventListener('change', render);
    render();
    const scoreGroups=data.score_groups||[];
    const names={ensemble:'Selected ensemble',ExtraTreesRegressor:'Extra Trees',RandomForestRegressor:'Random Forest',HistGradientBoostingRegressor:'Histogram Gradient Boosting',Ridge:'Ridge regression'};
    for(const family of [...new Set(scoreGroups.map(g=>g.family))]) el('performanceScoreModel').add(new Option(names[family]||family,family));
    function renderScore() {
      const group=scoreGroups.find(g=>g.family===el('performanceScoreModel').value && g.version===el('performanceScoreVersion').value);
      if(!group) {el('independentScoreSummary').textContent='No score archive available.';return;}
      const s=group.summary;
      el('independentScoreSummary').textContent=`${s.score_n} completed games with archived score forecasts. Spread ${s.spread.correct}/${s.spread.n}, ${s.spread.pushes} pushes; total ${s.total.correct}/${s.total.n}, ${s.total.pushes} pushes. Independent of the winner-version filter. Missing archives and no-edge picks are not graded.`;
      el('independentScoreRows').innerHTML=group.weeks.map(w=>`<tr>${[w.week,w.score_n,number(w.team_score_mae,1),number(w.margin_mae,1),number(w.total_mae,1),`${w.spread.correct}/${w.spread.n}`,`${w.total.correct}/${w.total.n}`].map(cell).join('')}</tr>`).join('');
    }
    function versions() {
      el('performanceScoreVersion').replaceChildren();
      for(const group of scoreGroups.filter(g=>g.family===el('performanceScoreModel').value)) el('performanceScoreVersion').add(new Option(group.version==='all'?'All score versions':group.version,group.version));
      renderScore();
    }
    el('performanceScoreModel').addEventListener('change',versions);
    el('performanceScoreVersion').addEventListener('change',renderScore);
    versions();
  } catch (error) { el('performanceNote').textContent = error.message; }
  try {
    const response = await fetch('./data/backtest.json');
    if (!response.ok) throw new Error('Backtest results unavailable');
    const data = await response.json();
    el('backtestNote').textContent = `${data.start_season}-${data.end_season}; ${data.folds.length} weekly folds, ${data.summary.games} games. ${data.method} ${data.limitations}`;
    el('backtestRows').innerHTML = data.seasons.map(s => `<tr>${[s.season, s.winner_n, rate(s.home_baseline.accuracy), number(s.home_baseline.log_loss), rate(s.form.accuracy), number(s.form.log_loss), number(s.score.margin_mae, 1), number(s.score.total_mae, 1)].map(cell).join('')}</tr>`).join('');
  } catch (error) { el('backtestNote').textContent = error.message; }
};
