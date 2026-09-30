window.setupScoreLines = function (games) {
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const valid = value => value != null && Number.isFinite(Number(value));
  const points = value => valid(value) ? Number(value).toFixed(1) : 'Not available';
  const signed = value => `${Number(value) > 0 ? '+' : ''}${points(value)}`;
  const badge = value => `<span class="score-grade ${value === 'Correct' ? 'score-correct' : value === 'Wrong' ? 'score-wrong' : ''}">${esc(value || 'Not available')}</span>`;
  const outcome = (value, spread = false) => !value ? 'Not available' : value === 'Push' ? 'Push' : `${esc(value)}${spread ? ' covers' : ''}`;
  for (const week of [...new Set(games.map(g => g.week))].sort((a,b) => a-b)) el('scoreWeek').add(new Option(`Week ${week}`, week));
  const teams = new Map(games.flatMap(g => [[g.away_team, g.away_team_name], [g.home_team, g.home_team_name]]));
  for (const [code, name] of [...teams].sort((a,b) => a[1].localeCompare(b[1]))) el('scoreTeam').add(new Option(name, code));
  function render() {
    const rows = games.filter(g => (el('scoreWeek').value === 'all' || String(g.week) === el('scoreWeek').value) && (el('scoreTeam').value === 'all' || [g.home_team,g.away_team].includes(el('scoreTeam').value)));
    el('scoreCount').textContent = `${rows.length} games`;
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
  render();
  fetch('./data/score_metrics.json').then(r => {if (!r.ok) throw new Error(); return r.json();}).then(m => {
    el('scoreEvaluation').textContent = `Trained through ${m.training_through}. Retrospective pre-refit evaluation on ${m.holdout_rows} games in 2026: margin mean absolute error ${points(m.margin_mae)} points; total mean absolute error ${points(m.total_mae)} points. Tuning used 2025 only. This small sample is not live forecast accuracy; historical training excludes ties. No cover probabilities are estimated.`;
  }).catch(() => {el('scoreEvaluation').textContent = 'Score model evaluation is temporarily unavailable.';});
};
