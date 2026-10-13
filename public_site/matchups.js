window.nextMatchupWeek = function (rows) {
  const unfinished = rows.filter(r => r.status !== 'Final');
  return unfinished.length ? Math.min(...unfinished.map(r => r.week)) : Math.max(...rows.map(r => r.week));
};

window.renderMatchups = function (rows) {
  const opened = new Set([...document.querySelectorAll('.matchup-game')].filter(g=>g.querySelector('details')?.open).map(g=>g.dataset.gameId));
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const points = n => n == null ? '--' : Number(n).toFixed(1);
  document.getElementById('matchupList').innerHTML = rows.map(row => {
    const result = row.prediction_result === 'Yes' ? 'Correct' : row.prediction_result === 'No' ? 'Wrong' : row.status === 'Final' ? 'Tie' : 'Pending';
    const tone = result === 'Correct' ? 'prediction-correct' : result === 'Wrong' ? 'prediction-incorrect' : '';
    const team = side => {
      const projected = row.predicted_winner === row[`${side}_team`];
      return `<div class="matchup-team${projected ? ' matchup-projected' : ''}" data-team="${escape(row[`${side}_team`])}"><img src="${escape(row[`${side}_logo`])}" alt="" width="32" height="32"><span>${escape(row[`${side}_team_name`])}${projected ? `<small class="projected-label">${row.status === 'Final' ? 'Pregame pick' : 'Projected winner'}</small>` : ''}<small>${side === 'home' ? 'Home' : 'Away'} &middot; ${(row[`${side}_win_probability`]*100).toFixed(1)}% win chance</small></span><strong>${points(row[`predicted_${side}_score`])}</strong></div>`;
    };
    return `<article class="matchup-game" data-game-id="${escape(row.game_id)}"><div class="matchup-meta"><span>Week ${row.week} &middot; ${escape(row.gameday)}</span><span class="${tone}">${result}</span></div><div class="matchup-meta"><span>Score ensemble</span><span>Expected points</span></div>${team('away')}${team('home')}
      <div class="matchup-pick"><strong>Pick: ${escape(row.predicted_winner_name)}</strong><span>${row.status === 'Final' ? `Final ${row.away_score} - ${row.home_score}` : 'Scheduled'}</span></div>
      ${window.matchupSignals ? window.matchupSignals(row) : ''}
      <details ${opened.has(row.game_id)?'open':''}><summary>Game details</summary><dl class="matchup-details">
        <dt>Expected score (away - home)</dt><dd>${points(row.predicted_away_score)} - ${points(row.predicted_home_score)}; score ensemble</dd>
        <dt>Expected total / source total</dt><dd>${points(row.predicted_total)} / ${points(row.market_total)}</dd>
        <dt>Home handicap / spread pick</dt><dd>${points(row.market_home_spread)} / ${escape(row.model_spread_side || 'No archived pick')}</dd>
        <dt>Total pick / result</dt><dd>${escape(row.model_total_side || 'No archived pick')} / ${escape(row.total_grade)}</dd>
        <dt>Spread result</dt><dd>${escape(row.spread_grade)}</dd>
        <dt>Winner / score version</dt><dd>${escape(row.model_version)} / ${escape(row.score_model_version || 'Not archived')}</dd>
        <dt>Score forecast saved</dt><dd>${escape(row.score_forecast_at || 'No pregame score forecast')}</dd>
        <dt>Line provenance</dt><dd>${escape(row.line_basis)}</dd>
      </dl>${window.gameInsights?window.gameInsights(row):''}${window.gameResearch?window.gameResearch(row):''}</details></article>`;
  }).join('') || '<p>No games match these filters.</p>';
};
