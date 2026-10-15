window.researchData = {};
window.setupResearch = async function () {
  const el=id=>document.getElementById(id);
  const pct=n=>n==null?'Unavailable':`${(n*100).toFixed(1)}%`;
  const fixed=n=>n==null?'Unavailable':n.toFixed(3);
  async function get(name) {
    try { const r=await window.dashboardFetch(`./data/${name}.json`); if (!r.ok) return null; return await r.json(); }
    catch { return null; }
  }
  const [audit, context, distributions, metrics, backtest, review, health] = await Promise.all(
    ['game_audit','matchup_context','score_distributions','distribution_metrics','advanced_backtest','challenger_review','refresh_health'].map(get));
  window.researchData={audit:audit||{},context:context||{},distributions:distributions||{}};
  window.dispatchEvent(new Event('researchready'));
  const observed=health?.last_successful_refresh;
  const age=observed?(Date.now()-new Date(observed).getTime())/86400000:null;
  el('healthStatus').textContent=observed?`Last validated refresh: ${new Date(observed).toLocaleString()}. ${age>8?'Stale: more than eight days old.':'Validation passed.'}`:'Refresh health unavailable; freshness is unverified.';
  el('healthStatus').className=age==null||age>8?'health-warning':'health-ok';
  if(backtest) {
    el('advancedNote').textContent=`${backtest.start_season}-${backtest.end_season}: ${backtest.games} games, ${backtest.ties} ties retained (excluded from binary winner metrics). ${backtest.note}`;
    const labels={candidate_probability:'Nested RF / boosting selection',elo_probability:'Elo baseline',market_probability:'Market-informed benchmark',ablation_none:'Ablation reference (logistic)',ablation_epa:'Without EPA',ablation_qb:'Without QB metrics',ablation_rest:'Without rest',ablation_ngs:'Without Next Gen Stats'};
    el('advancedRows').innerHTML=Object.entries(backtest.metrics).map(([name,m])=>`<tr><td>${labels[name]||name}</td><td>${m.n}</td><td>${pct(m.accuracy)}</td><td>${m.accuracy_ci95.map(pct).join(' to ')}</td><td>${fixed(m.log_loss)}</td><td>${fixed(m.brier)}</td></tr>`).join('');
  } else el('advancedNote').textContent='Advanced research results unavailable.';
  if(metrics) {
    el('distributionNote').textContent=`${metrics.evaluation_games} sequentially evaluated games. Model trained through ${metrics.training_through}. ${metrics.note}`;
    el('coverageRows').innerHTML=Object.entries(metrics.coverage80).map(([name,value])=>`<tr><td>${name==='home'?'Home score':name==='away'?'Away score':name==='margin'?'Home minus away margin':'Combined total'}</td><td>80%</td><td>${pct(value)}</td></tr>`).join('');
    if(metrics.line_evaluation) el('distributionLineResults').textContent=Object.entries(metrics.line_evaluation).map(([name,m])=>`${name==='spread'?'Spread':'Total'}: ${m.games} retrospective games, three-class Brier ${fixed(m.three_class_brier)}, log loss ${fixed(m.log_loss)}`).join('. ')+'. These are not live betting results.';
  } else el('distributionNote').textContent='Score-distribution research unavailable.';
  if(review) {
    el('challengerNote').textContent=review.note;
    el('challengerRows').innerHTML=review.cohorts.map(c=>`<tr><td>${c.model_version}</td><td>${c.evaluated_games} / ${c.minimum_games}</td><td>${c.log_loss_difference_ci95?c.log_loss_difference_ci95.map(fixed).join(' to '):'Insufficient data'}</td><td>${c.ready_for_review?'Ready for human review':'Remain in shadow'}</td><td>${c.decision?c.decision.decision:'Not reviewed'}</td></tr>`).join('');
  } else el('challengerNote').textContent='Challenger evidence unavailable; no promotion.';
};

window.gameResearch = function (row) {
  const escape=v=>String(v??'Unavailable').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const data=window.researchData;
  const audit=data.audit?.[row.game_id];
  const context=data.context?.[row.game_id];
  const distribution=data.distributions?.[row.game_id];
  const pct=p=>p==null?'Unavailable':`${(100*p).toFixed(1)}%`;
  let html='<h4>Forecast provenance</h4>';
  if(audit && audit.model_version===row.model_version && Math.abs(audit.home_probability-row.home_win_probability)<1e-8) {
    html+=`<p>Saved ${escape(audit.forecast_at)}. Input fingerprint: <span class="fingerprint">${escape(audit.input_sha256)}</span></p><dl class="matchup-details">`;
    for(const [name,status] of Object.entries(audit.sources)) html+=`<dt>${escape(name)}</dt><dd>${escape(status.status)} (${status.current_fields}/${status.model_fields} fields)</dd>`;
    html+='</dl><h4>Local SHAP: home-win probability</h4>';
    html+=`<p>Reference ${pct(audit.base_probability)}; forecast ${pct(audit.home_probability)}.</p><ul class="local-signals">`;
    for(const signal of audit.top_features) html+=`<li><span>${escape(readableFeature(signal.feature))}</span><strong class="${signal.impact>=0?'signal-positive':'signal-negative'}">${signal.impact>=0?'+':''}${(100*signal.impact).toFixed(2)} pp</strong></li>`;
    html+=`</ul><p>Other features: ${(audit.other_impact*100).toFixed(2)} pp. ${escape(audit.note)}</p>`;
  } else html+='<p>No exact pregame input/explanation archive for this forecast. Historical games are not reconstructed.</p>';
  if(context && row.status!=='Final') {
    html+='<h4>Current research context</h4><dl class="matchup-details">';
    for(const side of ['away','home']) {
      const t=context[side];
      html+=`<dt>${escape(row[side+'_team_name'])}</dt><dd>Latest performance week: ${escape(t.last_performance_week)}. Plays/game: ${t.plays_per_game?.toFixed(1)??'Unavailable'}. Opponent-adjusted EPA/play: ${t.opponent_adjusted_epa?.toFixed(3)??'Unavailable'}. Previous QB: ${escape(t.last_qb)} (${pct(t.last_qb_attempt_share)} of attempts). Starter unconfirmed. OL out/doubtful: ${escape(t.ol_out)}. Roster OL: ${escape(t.active_roster_ol_count)} active / ${escape(t.roster_ol_count)} all statuses.</dd>`;
    }
    html+=`<dt>Weather</dt><dd>${escape(context.weather.status)}${context.weather.temperature_f!=null?`; ${context.weather.temperature_f} F, wind ${context.weather.wind_mph} mph`:''}</dd></dl><p>${escape(context.note)}</p>`;
  }
  if(distribution) {
    const d=distribution;
    html+=`<h4>Experimental score distribution (shadow only)</h4><p>Expected away / home: ${d.expected_away.toFixed(1)} / ${d.expected_home.toFixed(1)}. Central 80% ranges: away ${d.away_interval80.join('-')}, home ${d.home_interval80.join('-')}; total ${d.total_interval80.join('-')}.</p>`;
    if(d.spread) html+=`<p>Archived home handicap ${d.market_home_spread}: home covers ${pct(d.spread.above)}, away covers ${pct(d.spread.below)}, push ${pct(d.spread.push)}.</p>`;
    if(d.total) html+=`<p>Archived total ${d.market_total}: over ${pct(d.total.above)}, under ${pct(d.total.below)}, push ${pct(d.total.push)}.</p>`;
    html+=`<p>Saved ${escape(d.forecast_at)}. ${escape(d.model_version)}. Different model from the production score ensemble; approximate probabilities, not demonstrated market advantage.</p>`;
  }
  return `<div class="game-research">${html}</div>`;
};
