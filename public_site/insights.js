/* Archive-backed insight views; no changes to serving probabilities. */
(() => {
  const el = id => document.getElementById(id);
  const esc = v => String(v ?? 'Unavailable').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = (v,n=2) => v == null ? 'N/A' : Number(v).toFixed(n);
  const pct = v => v == null ? 'N/A' : `${num(v*100,1)}%`;
  const signed = v => v == null ? 'First observation' : `${v>=0?'+':''}${num(v)}`;
  let history = {}, scores = {}, predictions = [], chart;
  let historyAvailable = false;
  const scoreWinner = row => row.predicted_home_score == null || row.predicted_away_score == null ? null :
    row.predicted_home_score === row.predicted_away_score ? 'Equal expected scores' : row.predicted_home_score > row.predicted_away_score ? row.home_team : row.away_team;

  window.matchupSignals = row => {
    const winner = scoreWinner(row);
    const disagreement = winner && winner !== row.predicted_winner;
    const source = window.researchData?.audit?.[row.game_id];
    const gaps = source ? Object.values(source.sources).filter(s=>s.status!=='Current snapshot').length : null;
    const coverage = gaps==null ? 'Source coverage unverified' : `${gaps} source categories with fallback / missing fields`;
    return `<p class="model-agreement ${disagreement?'model-disagreement':''}">${winner ? disagreement ? `Models differ: classifier ${esc(row.predicted_winner)}; score ensemble ${esc(winner)}.` : `Classifier and score ensemble agree: ${esc(winner)}.` : 'Score comparison unavailable.'}</p><p class="coverage-summary">${coverage}${row.status!=='Final'?'; starting QB unconfirmed':''}.</p>`;
  };

  function quality(row) {
    const audit = window.researchData?.audit?.[row.game_id];
    const context = window.researchData?.context?.[row.game_id];
    const exact = audit?.model_version === row.model_version && Math.abs(audit.home_probability-row.home_win_probability)<1e-8;
    let html = '<h4>Data Quality</h4><p>Availability is separate from win probability. Retrieval time does not establish provider freshness.</p><dl class="matchup-details">';
    if(exact) for(const [name,source] of Object.entries(audit.sources)) html += `<dt>${esc(name)}</dt><dd>${esc(source.status)}: ${source.current_fields}/${source.model_fields} fields</dd>`;
    else html += '<dt>Forecast inputs</dt><dd>No matching archived source-quality record.</dd>';
    if(row.status !== 'Final' && context) {
      html += `<dt>Confirmed starting QB</dt><dd>${['away','home'].map(s=>`${esc(row[s+'_team'])}: ${esc(context[s]?.expected_qb || 'Unconfirmed')}`).join('; ')}</dd><dt>Weather</dt><dd>${esc(context.weather?.status)}</dd>`;
      html += `<dt>Same-week injury context</dt><dd>${['away','home'].map(s=>`${esc(row[s+'_team'])}: ${context[s]?.ol_out == null ? 'Unavailable' : `${context[s].ol_out} OL out/doubtful`}`).join('; ')}</dd>`;
      html += '<dt>Roster limitation</dt><dd>Legacy production active-roster status mapping remains under review. Corrected research counts are not production inputs.</dd>';
    }
    if(exact) for(const [name,receipt] of Object.entries(audit.source_receipts || {})) {
      const age = (Date.now()-Date.parse(receipt.retrieved_at))/86400000;
      html += `<dt>${esc(name.replace('load_','').replaceAll('_',' '))} observed</dt><dd>${esc(receipt.retrieved_at)}${row.status!=='Final' && age>8 ? ' - stale observation (>8 days)' : ''}; publication freshness unknown</dd>`;
    }
    return html+'</dl>';
  }

  window.gameInsights = row => {
    const models = Object.entries(scores).map(([name,rows])=>({name,...rows.find(r=>r.game_id===row.game_id)})).filter(r=>r.predicted_home_score!=null && r.predicted_away_score!=null);
    let html = '<div class="game-insights"><h4>Score-Model Disagreement</h4>';
    if(models.length) {
      const margins=models.map(r=>r.predicted_home_score-r.predicted_away_score);
      html+=`<p>${models.length} regression families; home-minus-away margins ${num(Math.min(...margins),1)} to ${num(Math.max(...margins),1)} points. Model spread is not a calibrated uncertainty interval.</p><ul>${models.map(r=>`<li>${esc(r.name)}: ${num(r.predicted_away_score,1)} - ${num(r.predicted_home_score,1)} (away - home); ${esc(r.score_model_version)}</li>`).join('')}</ul>`;
    } else html+='<p>No archived individual score forecasts for this game.</p>';
    html += quality(row)+'<h4>Forecast History</h4>';
    const records = history[row.game_id] || [];
    if(!records.length) html+=`<p>${historyAvailable?'No verified observations. Older forecasts are not reconstructed.':'Forecast history unavailable.'}</p>`;
    else {
      html+='<p>Changes compare the same winner/score version. Inputs that changed are not proven causes.</p><ol class="forecast-timeline">';
      html+=records.map(r=>`<li><time>${esc(r.forecast_at)}</time><p>Home win ${pct(r.probability)}; change ${r.probability_change==null?'First observation':signed(r.probability_change*100)+' pp'}. Away/home score ${num(r.predicted_away_score,1)} / ${num(r.predicted_home_score,1)}; changes ${signed(r.away_score_change)} / ${signed(r.home_score_change)}.</p><p class="version-text">Winner ${esc(r.model_version)}; score ${esc(r.score_model_version || 'Not captured in this observation')}</p>${r.changed_inputs.length?`<details><summary>${r.changed_inputs.length} changed inputs</summary><ul>${r.changed_inputs.map(c=>`<li>${esc(readableFeature(c.feature))}: ${esc(c.before)} to ${esc(c.after)}</li>`).join('')}</ul></details>`:''}${r.changed_sources.length?`<p>Changed source snapshots: ${r.changed_sources.map(esc).join(', ')}</p>`:''}</li>`).join('')+'</ol>';
    }
    const url=new URL(location.href);url.search='';url.searchParams.set('game',row.game_id);
    return html+`<p><a class="game-permalink" href="${esc(url.href)}">Link to this matchup</a></p></div>`;
  };

  window.setupInsights = async (rows, update) => {
    predictions = rows;
    const teams = [...new Map(rows.flatMap(r=>['home','away'].map(s=>[r[s+'_team'],{code:r[s+'_team'],name:r[s+'_team_name'],logo:r[s+'_logo']}]))).values()].sort((a,b)=>a.name.localeCompare(b.name));
    const toolbar=el('teamFilter').closest('.toolbar');
    const preferences=document.createElement('div');preferences.className='preference-controls';
    preferences.innerHTML='<label>Favorite team<select id="favoriteTeam"><option value="">None</option>'+teams.map(t=>`<option value="${esc(t.code)}">${esc(t.name)}</option>`).join('')+'</select></label><button id="resetFilters" type="button">Reset filters</button><a id="shareFilters" href="?">Link to this view</a><span id="preferenceStatus" role="status"></span>';
    toolbar.append(preferences);
    let favorite='';try {favorite=localStorage.getItem('nfl-favorite')||'';}catch{el('preferenceStatus').textContent='Preferences cannot be saved in this browser.';}
    if(!teams.some(t=>t.code===favorite)) favorite='';
    el('favoriteTeam').value=favorite;
    function applyUrl() {
      const params=new URLSearchParams(location.search);
      const game=rows.find(r=>r.game_id===params.get('game'));
      const selected=teams.find(t=>t.code===(params.has('team')?params.get('team'):favorite));
      const week=params.get('week');
      el('weekFilter').value=game?String(game.week):[...el('weekFilter').options].some(o=>o.value===week)?week:String(window.nextMatchupWeek(rows));
      el('teamFilter').value=game?game.home_team_name:selected?.name||'all';
      const confidence=Number(params.get('confidence')||50);
      el('confidenceFilter').value=game?50:Number.isFinite(confidence)?Math.min(85,Math.max(50,confidence)):50;
      update();
      if(game) {const article=[...document.querySelectorAll('.matchup-game')].find(e=>e.dataset.gameId===game.game_id);if(article){article.querySelector('details').open=true;article.scrollIntoView({block:'start'});}}
    }
    function link(replace=false) {
      const url=new URL(location.href);url.search='';
      url.searchParams.set('week',el('weekFilter').value);
      url.searchParams.set('team',teams.find(t=>t.name===el('teamFilter').value)?.code||'all');
      url.searchParams.set('confidence',el('confidenceFilter').value);
      el('shareFilters').href=url.href;
      if(replace) window.history.replaceState(null,'',url);
    }
    applyUrl();link();
    ['weekFilter','teamFilter','confidenceFilter'].forEach(id=>el(id).addEventListener('input',()=>link(true)));
    el('favoriteTeam').addEventListener('change',()=>{
      favorite=el('favoriteTeam').value;
      try {if(favorite)localStorage.setItem('nfl-favorite',favorite);else localStorage.removeItem('nfl-favorite');el('preferenceStatus').textContent=favorite?'Favorite saved on this browser.':'Favorite cleared.';}catch{el('preferenceStatus').textContent='Favorite applies for this visit only.';}
      el('teamFilter').value=teams.find(t=>t.code===favorite)?.name||'all';update();link(true);
    });
    el('resetFilters').addEventListener('click',()=>{el('weekFilter').value=String(window.nextMatchupWeek(rows));el('teamFilter').value='all';el('confidenceFilter').value=50;update();link(true);});
    window.addEventListener('popstate',()=>{applyUrl();link();});
    el('profileTeam').innerHTML=teams.map(t=>`<option value="${esc(t.code)}">${esc(t.name)}</option>`).join('');
    el('profileTeam').value=favorite||teams[0].code;
    const versions=[...new Set(rows.map(r=>r.model_version))].sort().reverse();
    el('profileVersion').innerHTML=versions.map(v=>`<option>${esc(v)}</option>`).join('');
    el('profileVersion').value=rows.find(r=>r.status!=='Final')?.model_version||versions[0];
    function profile() {
      const team=teams.find(t=>t.code===el('profileTeam').value);
      const games=rows.filter(r=>r.home_team===team.code||r.away_team===team.code).sort((a,b)=>a.week-b.week);
      const finals=games.filter(r=>r.status==='Final');
      const wins=finals.filter(r=>r.actual_winner===team.code).length;
      const ties=finals.filter(r=>r.home_score===r.away_score).length;
      const graded=finals.filter(r=>r.model_version===el('profileVersion').value && ['Yes','No'].includes(r.prediction_result));
      const next=games.find(r=>r.status!=='Final');
      const side=next?.home_team===team.code?'home':'away';
      const context=window.researchData?.context?.[next?.game_id]?.[side];
      el('teamProfile').innerHTML=`<h3 class="profile-heading"><img src="${esc(team.logo)}" alt="" width="40" height="40">${esc(team.name)}</h3><p>Record: ${wins}-${finals.length-wins-ties}-${ties}. Saved winner picks correct: ${graded.filter(r=>r.prediction_result==='Yes').length}/${graded.length} decisive games for ${esc(el('profileVersion').value)}. Legacy records may lack original timestamps.</p><p>Next-game research context: opponent-adjusted offensive EPA/play ${num(context?.opponent_adjusted_epa,3)}; prior plays/game ${num(context?.plays_per_game,1)}. Prior five-game averages using earlier weeks; revised source snapshots, not historical live availability.</p>`;
      el('profileGames').innerHTML=games.map(r=>{const s=r.home_team===team.code?'home':'away',o=s==='home'?'away':'home';return `<tr><td>${r.week}</td><td>${esc(r[o+'_team_name'])}</td><td>${pct(r[s+'_win_probability'])}</td><td>${num(r['predicted_'+s+'_score'],1)} - ${num(r['predicted_'+o+'_score'],1)}</td><td>${r.status==='Final'?`${r[s+'_score']} - ${r[o+'_score']}`:'Upcoming'}</td><td>${esc(r.prediction_result)}</td></tr>`;}).join('');
      if(chart)chart.destroy();
      chart=new Chart(el('profileChart'),{type:'line',data:{labels:games.map(r=>`W${r.week}`),datasets:[{label:'Prior opponent-adjusted offensive EPA/play',data:games.map(r=>window.researchData?.context?.[r.game_id]?.[r.home_team===team.code?'home':'away']?.opponent_adjusted_epa??null),borderColor:'#1756a9',pointRadius:3,spanGaps:false}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{title:{display:true,text:'EPA per play'}}}}});
    }
    el('profileTeam').addEventListener('change',profile);el('profileVersion').addEventListener('change',profile);el('profilesTab').addEventListener('click',profile);window.addEventListener('researchready',profile);
    async function get(name){try{const r=await window.dashboardFetch(`./data/${name}.json`);return r.ok?await r.json():null;}catch{return null;}}
    const [h,s,drift]=await Promise.all(['forecast_history','model_score_predictions','drift_monitor'].map(get));
    history=h?.games||{};historyAvailable=!!h;scores=s||{};update();profile();
    el('driftNote').textContent=drift?.note||'Monitoring data unavailable. No inference about performance can be made.';
    const cohorts=drift?.cohorts||[];
    el('driftVersion').innerHTML=[...new Set(cohorts.map(c=>c.model_version))].map(v=>`<option>${esc(v)}</option>`).join('')||'<option>No prospective results yet</option>';
    function renderDrift(){
      const selected=cohorts.filter(c=>c.model_version===el('driftVersion').value);
      el('driftStatus').textContent=selected.length?'Versions are evaluated separately; monitoring never changes production.':'No completed games with verified prospective history yet. Monitoring will populate as archived forecasts settle.';
      el('driftRows').innerHTML=selected.map(c=>`<tr><td>${esc(c.score_model_version||'Winner only')} / ${esc(c.horizon)}</td><td>${c.overall.n} / ${c.overall.ties}</td><td>${num(c.overall.log_loss,3)}</td><td>${num(c.overall.brier,3)}</td><td>${pct(c.overall.calibration_bias)}</td><td>${c.overall.score_n} / ${num(c.overall.score_mae)}</td><td>${c.recent.n} / ${c.baseline.n}</td><td>${esc(c.status)}</td></tr>`).join('');
      el('driftStages').innerHTML=selected.filter(c=>c.horizon==='Latest pregame').flatMap(c=>Object.entries(c.stages).map(([stage,m])=>`<tr><td>${esc(c.score_model_version||'Winner only')} / ${esc(stage)}</td><td>${m.n}</td><td>${num(m.log_loss,3)}</td><td>${num(m.brier,3)}</td><td>${m.score_n} / ${num(m.score_mae)}</td></tr>`)).join('');
    }el('driftVersion').addEventListener('change',renderDrift);renderDrift();window.insightsReady=true;
  };
})();
