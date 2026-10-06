window.setupScoreLines = function (games) {
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const valid = value => value != null && Number.isFinite(Number(value));
  const points = value => valid(value) ? Number(value).toFixed(1) : 'Not available';
  const signed = value => `${Number(value) > 0 ? '+' : ''}${points(value)}`;
  const modelName = value => ({ExtraTreesRegressor:'Extra Trees', RandomForestRegressor:'Random Forest', HistGradientBoostingRegressor:'Histogram Gradient Boosting', Ridge:'Ridge Regression'})[value] || value;
  const modelForecasts = {};
  const badge = value => `<span class="score-grade ${value === 'Correct' ? 'score-correct' : value === 'Wrong' ? 'score-wrong' : ''}">${esc(value || 'Not available')}</span>`;
  const outcome = (value, spread = false) => !value ? 'Not available' : value === 'Push' ? 'Push' : `${esc(value)}${spread ? ' covers' : ''}`;
  for (const week of [...new Set(games.map(g => g.week))].sort((a,b) => a-b)) el('scoreWeek').add(new Option(`Week ${week}`, week));
  const teams = new Map(games.flatMap(g => [[g.away_team, g.away_team_name], [g.home_team, g.home_team_name]]));
  for (const [code, name] of [...teams].sort((a,b) => a[1].localeCompare(b[1]))) el('scoreTeam').add(new Option(name, code));
  function render() {
    const selected = el('scoreModel').value;
    const source = selected === 'ensemble' ? games : games.map(g => ({...g, ...modelForecasts[selected].get(g.game_id)}));
    const rows = source.filter(g => (el('scoreWeek').value === 'all' || String(g.week) === el('scoreWeek').value) && (el('scoreTeam').value === 'all' || [g.home_team,g.away_team].includes(el('scoreTeam').value)));
    el('scoreCount').textContent = `${rows.length} games`;
    const live = (field, name) => {
      const correct = rows.filter(g => g[field] === 'Correct').length;
      const wrong = rows.filter(g => g[field] === 'Wrong').length;
      const pushes = rows.filter(g => g[field] === 'Push').length;
      return `${name}: ${correct + wrong ? `${correct}/${correct + wrong} correct` : 'no settled pregame picks'}; ${pushes} pushes`;
    };
    el('scoreLiveResults').textContent = `Live archived picks for selected games. ${live('spread_grade','Spread')}. ${live('total_grade','O/U')}.`;
    el('scoreRows').innerHTML = rows.map(g => {
      const predicted = valid(g.predicted_away_score) && valid(g.predicted_home_score);
      const final = valid(g.away_score) && valid(g.home_score);
      const spread = valid(g.market_home_spread) ? `${esc(g.home_team)} ${Number(g.market_home_spread) >= 0 ? '+' : ''}${points(g.market_home_spread)}` : 'Not available';
      const total = predicted ? Number(g.predicted_away_score) + Number(g.predicted_home_score) : null;
      const margin = predicted ? Number(g.predicted_home_score) - Number(g.predicted_away_score) : null;
      const actualTotal = final ? Number(g.away_score) + Number(g.home_score) : null;
      const totalGap = predicted && valid(g.market_total) ? `<small>${signed(total - Number(g.market_total))} vs betting total</small>` : '';
      const actualGap = final && predicted ? `<small>${signed(actualTotal - total)} vs model</small>` : '';
      return `<tr><td>Week ${g.week}<strong>${esc(g.away_team_name)} at ${esc(g.home_team_name)}</strong></td>
        <td data-score="predicted">${predicted ? `${points(g.predicted_away_score)} - ${points(g.predicted_home_score)}` : 'No pregame forecast'}</td>
        <td data-score="total">${points(total)}${totalGap}</td><td>${points(g.market_total)}</td><td>${outcome(g.model_total_side)}</td>
        <td>${predicted ? signed(margin) : 'Not available'}</td><td>${spread}</td><td>${outcome(g.model_spread_side,true)}</td>
        <td>${final ? `${g.away_score} - ${g.home_score}` : 'Pending'}</td><td data-score="actual-total">${final ? points(actualTotal) : 'Pending'}${actualGap}</td>
        <td>${outcome(g.actual_total_side)}</td><td>${badge(g.total_grade)}</td><td>${outcome(g.actual_spread_side,true)}</td><td>${badge(g.spread_grade)}</td>
        <td>${esc(g.line_basis)}<small>${g.line_observed_at ? esc(new Date(g.line_observed_at).toLocaleString()) : 'No archived timestamp'}</small></td></tr>`;
    }).join('');
  }
  el('scoreWeek').addEventListener('change',render);
  el('scoreTeam').addEventListener('change',render);
  el('scoreModel').addEventListener('change',render);
  render();
  fetch('./data/model_score_predictions.json').then(r => {if (!r.ok) throw new Error(); return r.json();}).then(data => {
    for (const [name, rows] of Object.entries(data)) {
      if (rows.length !== games.length || new Set(rows.map(r => r.game_id)).size !== games.length || !games.every(g => rows.some(r => r.game_id === g.game_id))) throw new Error('Incomplete model forecasts');
      modelForecasts[name] = new Map(rows.map(r => [r.game_id,r]));
      el('scoreModel').add(new Option(modelName(name),name));
    }
    el('scoreModelStatus').textContent = 'Individual score forecasts are archived separately from the ensemble. Win probabilities remain those of the production winner classifier, not the selected score regressor.';
  }).catch(() => {el('scoreModelStatus').textContent = 'Individual score forecasts are unavailable; ensemble forecasts remain available.';});
  fetch('./data/score_metrics.json').then(r => {if (!r.ok) throw new Error(); return r.json();}).then(m => {
    el('scoreEvaluation').textContent = `Score ensemble: ${(m.selected_models || []).map(modelName).join(' + ')}. Trained through ${m.training_through}. Retrospective pre-refit evaluation on ${m.holdout_rows} games in 2026: margin mean absolute error ${points(m.margin_mae)} points; total mean absolute error ${points(m.total_mae)} points. Tuning used 2025 only. This small sample is not live forecast accuracy; historical training excludes ties. No cover probabilities are estimated.`;
    const count = value => value ? `${value.correct} / ${value.graded}` : 'Not available';
    el('scoreModelRows').innerHTML = (m.candidates || []).map(c => `<tr><td>${esc(modelName(c.family))}</td><td>${(m.selected_models || []).includes(c.family) ? 'Yes' : 'No'}</td><td>${points(c.validation_objective)}</td><td>${points(c.holdout.margin_mae)}</td><td>${points(c.holdout.total_mae)}</td><td>${count(c.holdout.markets?.spread)}</td><td>${count(c.holdout.markets?.total)}</td></tr>`).join('') + `<tr><td><strong>Selected ensemble</strong></td><td>Serving score forecasts</td><td>Top-two mean</td><td>${points(m.margin_mae)}</td><td>${points(m.total_mae)}</td><td>${count(m.markets?.spread)}</td><td>${count(m.markets?.total)}</td></tr>`;
  }).catch(() => {el('scoreEvaluation').textContent = 'Score model evaluation is temporarily unavailable.';});
};
