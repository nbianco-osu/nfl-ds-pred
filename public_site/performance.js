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
    function render() {
      const group = data.groups.find(g => g.version === el('performanceVersion').value);
      const s = group.summary;
      el('performanceMetrics').innerHTML = card('Winner accuracy', `${rate(s.accuracy)} (${s.correct}/${s.winner_n})`) +
        card('Log loss / Brier', `${number(s.log_loss)} / ${number(s.brier)}`) +
        card(`Score MAE (${s.score_n} games)`, `${number(s.team_score_mae, 1)} pts`) +
        card('Margin / total MAE', `${number(s.margin_mae, 1)} / ${number(s.total_mae, 1)}`);
      el('performanceCounts').textContent = `${s.completed} completed games; ${s.ties} ties excluded from winner metrics. Spread: ${s.spread.correct}/${s.spread.n}, ${s.spread.pushes} pushes, ${s.spread.excluded} ungraded. Total: ${s.total.correct}/${s.total.n}, ${s.total.pushes} pushes, ${s.total.excluded} ungraded. Small samples are not evidence of an advantage.`;
      el('weeklyPerformance').innerHTML = group.weeks.map(w => `<tr>${[w.week, `${w.correct}/${w.winner_n}`, rate(w.accuracy), number(w.log_loss), number(w.brier), w.score_n, number(w.margin_mae, 1), number(w.total_mae, 1)].map(cell).join('')}</tr>`).join('') || '<tr><td colspan="8">No completed games in this version.</td></tr>';
      el('calibrationRows').innerHTML = s.calibration.map(b => `<tr>${[`${Math.round(b.lower*100)}-${Math.round(b.upper*100)}%`, b.n, rate(b.predicted), rate(b.observed)].map(cell).join('')}</tr>`).join('');
    }
    el('performanceVersion').addEventListener('change', render);
    render();
  } catch (error) { el('performanceNote').textContent = error.message; }
  try {
    const response = await fetch('./data/backtest.json');
    if (!response.ok) throw new Error('Backtest results unavailable');
    const data = await response.json();
    el('backtestNote').textContent = `${data.start_season}-${data.end_season}; ${data.folds.length} weekly folds, ${data.summary.games} games. ${data.method} ${data.limitations}`;
    el('backtestRows').innerHTML = data.seasons.map(s => `<tr>${[s.season, s.winner_n, rate(s.home_baseline.accuracy), number(s.home_baseline.log_loss), rate(s.form.accuracy), number(s.form.log_loss), number(s.score.margin_mae, 1), number(s.score.total_mae, 1)].map(cell).join('')}</tr>`).join('');
  } catch (error) { el('backtestNote').textContent = error.message; }
};
