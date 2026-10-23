/* Shadow generative evidence; never substitutes for the production schedule. */
(() => {
  const el=id=>document.getElementById(id),pct=n=>(100*n).toFixed(1)+'%',num=n=>Number.isFinite(n)?n.toFixed(3):'Unavailable';
  const tab=document.createElement('button');tab.id='generativeTab';tab.textContent='Generative Forecasts';
  tab.setAttribute('role','tab');tab.setAttribute('aria-selected','false');tab.setAttribute('aria-controls','generativePanel');tab.tabIndex=-1;
  document.querySelector('.view-tabs').append(tab);
  const panel=document.createElement('section');panel.id='generativePanel';panel.hidden=true;
  panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',tab.id);
  panel.innerHTML=`<h2>Generative Forecasts</h2><p>Shadow research only. Production picks are unchanged. Historical replay is not saved pregame performance.</p>
  <p id="genCoverage"></p><p id="genReadiness" role="status"></p>
  <label>Matchup <select id="genGame"></select></label><p id="genSummary"></p><p id="genCutoff"></p>
  <div class="table-wrap" tabindex="0"><table><thead><tr><th>Model</th><th>Home score</th><th>Away score</th><th>Home win</th><th>Tie</th><th>Home covers</th><th>Over</th></tr></thead><tbody id="genScores"></tbody></table></div>
  <p id="genLines"></p><h3>Semi-Markov Regulation Trajectory</h3><div style="height:300px"><canvas id="genChart"></canvas></div>
  <div class="table-wrap"><table><thead><tr><th>Quarter</th><th>Home mean</th><th>Home 80% band</th><th>Away mean</th><th>Away 80% band</th></tr></thead><tbody id="genQuarters"></tbody></table></div>
  <p>Quarter paths belong to the Markov component, not the final-score mixture. Overtime uses tested stopping rules with approximate possession timing, no penalties or onside recoveries.</p>
  <h3>Matched Chronological Evaluation</h3><p>Identical revised-source games; lower loss and CRPS are better. Baselines are chronologically refitted existing model recipes, not the production artifact.</p>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Games</th><th>Winner log loss (3 outcomes)</th><th>Brier (3 outcomes)</th><th>Margin CRPS</th><th>Total CRPS</th></tr></thead><tbody id="genEvaluation"></tbody></table></div>
  <h3>Paired Comparisons</h3><p id="genComparisonNote"></p><div class="table-wrap"><table><thead><tr><th>Baseline</th><th>Metric</th><th>Ensemble difference</th><th>95% interval</th></tr></thead><tbody id="genPaired"></tbody></table></div>
  <h3>Winner Calibration</h3><div class="table-wrap"><table><thead><tr><th>Model</th><th>Games</th><th>Mean predicted home win</th><th>Observed home win</th></tr></thead><tbody id="genCalibration"></tbody></table></div>
  <h3>Season Win Trajectories</h3><label>Team <select id="genTeam"></select></label><div style="height:300px"><canvas id="genSeasonChart"></canvas></div>
  <p id="genSeasonNote"></p><div class="table-wrap"><table><thead><tr><th>Week</th><th>Mean wins</th><th>80% win band</th><th>Mean ties</th></tr></thead><tbody id="genSeasonRows"></tbody></table></div>`;
  document.querySelector('.disclaimer').before(panel);
  const labels={markov:'Semi-Markov',ridge_residual:'Ridge paired residuals',poisson:'Poisson',ensemble:'Generative ensemble'};
  function table(id,rows){el(id).replaceChildren();for(const row of rows){const tr=document.createElement('tr');for(const value of row){const td=document.createElement('td');td.textContent=value;tr.append(td);}el(id).append(tr);}}
  window.registerDashboardView({panelId:panel.id,files:['generative','generative_readiness'],initialize:async()=>{
    const d=await (await window.dashboardFetch('./data/generative.json')).json();
    const readiness=await (await window.dashboardFetch('./data/generative_readiness.json')).json();
    if(readiness.model_version!==d.model_version)throw Error('Readiness generation mismatch');
    const c=d.source_coverage;
    el('genCoverage').textContent=`${d.model_version}. Source observed ${c.observed_at}; ${c.pbp_games}/${c.completed_games} completed games with play-by-play, through week ${c.source_max_week}. Missing: ${c.missing_completed_games.join(', ')||'None'}. ${d.skipped.length} early-season matchups omitted for insufficient earlier transition data.`;
    el('genReadiness').textContent=`Prospective matched results: ${readiness.matched_games}. ${readiness.ready_for_review?'Ready for human review; not approved.':'Not ready for review.'} ${readiness.note||''}`;
    for(const g of d.games){const option=document.createElement('option');option.value=g.game_id;option.textContent=`Week ${g.week}: ${g.away_team} at ${g.home_team}${g.actual?' (replay)':''}`;el('genGame').append(option);}
    const future=d.games.find(g=>!g.actual);if(future)el('genGame').value=future.game_id;
    let chart,seasonChart;
    function game(){const g=d.games.find(g=>g.game_id===el('genGame').value),e=g.distributions.ensemble;
      el('genSummary').textContent=`${g.actual?'Retrospective replay':'Research projection'}: ${g.home_team} ${e.expected_home.toFixed(1)}, ${g.away_team} ${e.expected_away.toFixed(1)}. Final-game home win ${pct(e.home_win)}; tie ${pct(e.tie)}.`;
      el('genCutoff').textContent=`Transition training through week ${g.training_max_week}; ${g.weight_training_games} earlier out-of-fold games for mixture weights. Generated ${g.generated_at}.`;
      el('genLines').textContent=`Reference home margin: ${g.spread_line??'Unavailable'}; total: ${g.total_line??'Unavailable'}. Historical lines are revised schedule values, not timestamp-verified tradable quotes. Pushes remain separate.`;
      table('genScores',Object.entries(g.distributions).map(([k,v])=>[labels[k],v.expected_home.toFixed(1),v.expected_away.toFixed(1),pct(v.home_win),pct(v.tie),v.spread?pct(v.spread.above):'Unavailable',v.total?pct(v.total.above):'Unavailable']));
      const q=g.markov_regulation.quarters;
      table('genQuarters',q.map(r=>[r.quarter,r.home_mean.toFixed(1),r.home_interval80.join(' to '),r.away_mean.toFixed(1),r.away_interval80.join(' to ')]));
      chart?.destroy();if(window.Chart)chart=new Chart(el('genChart'),{type:'line',data:{labels:q.map(r=>'Q'+r.quarter),datasets:[{label:g.home_team,data:q.map(r=>r.home_mean),borderColor:'#175cd3'},{label:g.away_team,data:q.map(r=>r.away_mean),borderColor:'#087443'}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{beginAtZero:true}}}});
    }
    table('genEvaluation',Object.entries(d.matched).map(([k,v])=>[labels[k],v.games,num(v.log_loss3),num(v.brier3),num(v.margin_crps),num(v.total_crps)]));
    el('genComparisonNote').textContent=d.comparison_evidence.note;
    table('genPaired',d.comparison_evidence.paired.map(r=>[labels[r.baseline],r.metric.replaceAll('_',' '),num(r.difference),r.interval95.map(num).join(' to ')]));
    table('genCalibration',Object.entries(d.comparison_evidence.calibration).flatMap(([k,rows])=>rows.map(r=>[labels[k],r.n,pct(r.predicted),pct(r.observed)])));
    for(const team of Object.keys(d.season_trajectories.teams)){const o=document.createElement('option');o.value=team;o.textContent=team;el('genTeam').append(o);}
    el('genSeasonNote').textContent=d.season_trajectories.note;
    function season(){const team=el('genTeam').value,points=d.season_trajectories.teams[team];table('genSeasonRows',points.map(p=>[p.week,p.mean_wins.toFixed(1),`${p.low} to ${p.high}`,p.mean_ties.toFixed(2)]));
      seasonChart?.destroy();if(window.Chart)seasonChart=new Chart(el('genSeasonChart'),{type:'line',data:{labels:points.map(p=>p.week),datasets:[{label:'Lower 80% bound',data:points.map(p=>p.low),borderColor:'#98a2b3',pointRadius:0},{label:'Upper 80% bound',data:points.map(p=>p.high),borderColor:'#98a2b3',backgroundColor:'#e4e7ec',fill:'-1',pointRadius:0},{label:team+' mean wins',data:points.map(p=>p.mean_wins),borderColor:'#175cd3'}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{min:0,max:17},x:{title:{display:true,text:'Week'}}}}});
    }
    el('genGame').addEventListener('change',game);el('genTeam').addEventListener('change',season);game();season();
  }});
})();
