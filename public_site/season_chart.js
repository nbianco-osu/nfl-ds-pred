/* Shared, self-hosted Chart.js views over immutable exported trajectory data. */
window.setupSeasonCharts = function (data) {
  const teamSelect = document.getElementById('chartTeam');
  const measure = document.getElementById('chartMeasure');
  const container = document.getElementById('teamCharts');
  const tabs = [...document.querySelectorAll('.view-tabs [role="tab"]')];
  let charts = [];
  for (const team of [...data.teams].sort((a, b) => a.name.localeCompare(b.name))) {
    const option = document.createElement('option');
    option.value = team.team;
    option.textContent = team.name;
    teamSelect.append(option);
  }
  function render() {
    charts.forEach(chart => chart.destroy());
    charts = [];
    container.replaceChildren();
    if (!data.teams.length) {
      container.textContent = 'Season chart data is temporarily unavailable. Matchup predictions are unaffected.';
      return;
    }
    const cumulative = measure.value === 'cumulative';
    document.getElementById('chartNote').textContent = cumulative
      ? 'Green: actual wins. Blue: completed wins plus remaining model probabilities. Dashed segments are future projections, not guaranteed wins or the noisy season simulations. Byes leave cumulative wins unchanged.'
      : 'Blue: archived pregame and current future win probabilities. Green/red points: actual wins/losses (ties count as no win). Dashed lines indicate future games. Bye weeks have no probability.';
    for (const team of data.teams.filter(t => teamSelect.value === 'all' || t.team === teamSelect.value)) {
      const card = document.createElement('article');
      card.className = 'team-chart';
      const heading = document.createElement('h3');
      const logo = document.createElement('img');
      logo.src = team.logo;
      logo.alt = '';
      heading.append(logo, document.createTextNode(team.name));
      const frame = document.createElement('div');
      frame.className = 'chart-frame';
      const canvas = document.createElement('canvas');
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', `${team.name}: ${cumulative ? 'cumulative projected wins' : 'weekly win probabilities'}`);
      frame.append(canvas);
      const detail = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = 'Weekly values';
      const table = document.createElement('table');
      table.className = 'chart-values';
      table.innerHTML = '<thead><tr><th>Week</th><th>Opponent</th><th>Win chance</th><th>Projected wins</th></tr></thead>';
      const body = document.createElement('tbody');
      team.points.forEach(p => {
        const row = document.createElement('tr');
        for (const value of [p.week, p.opponent, p.win_probability == null ? 'Bye' : `${(100 * p.win_probability).toFixed(1)}%`, p.projected_wins.toFixed(2)]) {
          const cell = document.createElement('td');
          cell.textContent = value;
          row.append(cell);
        }
        body.append(row);
      });
      table.append(body);
      detail.append(summary, table);
      card.append(heading, frame, detail);
      container.append(card);
      if (!window.Chart) {
        frame.textContent = 'Chart unavailable. Weekly values are available below.';
        continue;
      }
      const values = team.points.map(p => cumulative ? p.projected_wins : p.win_probability == null ? null : p.win_probability * 100);
      charts.push(new Chart(canvas, {
        type: 'line',
        data: {labels: team.points.map(p => p.week), datasets: [
          {label: cumulative ? 'Projected wins' : 'Win probability', data: values, borderColor: '#175cd3', backgroundColor: '#175cd3', borderWidth: 2, pointRadius: 2,
            segment: {borderDash: context => team.points[context.p1DataIndex].status === 'Scheduled' ? [5, 4] : undefined}, spanGaps: false},
          {label: cumulative ? 'Actual wins' : 'Actual outcome', data: team.points.map(p => cumulative ? p.actual_wins : p.actual_win == null ? null : p.actual_win * 100),
            borderColor: '#087443', backgroundColor: team.points.map(p => p.actual_win === 0 ? '#b42318' : '#087443'), borderWidth: 3,
            pointRadius: cumulative ? 3 : 4, showLine: cumulative, spanGaps: false},
        ]},
        options: {responsive: true, maintainAspectRatio: false, animation: false,
          interaction: {intersect: false, mode: 'index'},
          plugins: {legend: {display: false}, tooltip: {callbacks: {
            title: items => `Week ${team.points[items[0].dataIndex].week}: ${team.points[items[0].dataIndex].opponent}`,
            label: item => `${item.dataset.label}: ${item.parsed.y.toFixed(1)}${cumulative ? '' : '%'}`,
          }}},
          scales: {x: {title: {display: true, text: 'Week'}, ticks: {maxTicksLimit: 9}},
            y: {min: 0, max: cumulative ? 17 : 100, title: {display: true, text: cumulative ? 'Wins' : 'Win chance (%)'}}},
        },
      }));
    }
  }
  function activate(tab) {
    tabs.forEach(item => {
      const active = item === tab;
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
      document.getElementById(item.getAttribute('aria-controls')).hidden = !active;
    });
    if (tab.id === 'seasonTab') render();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      activate(tabs[next]);
      tabs[next].focus();
    });
  });
  document.getElementById('libraryLink').addEventListener('click', () => activate(tabs[0]));
  teamSelect.addEventListener('change', render);
  measure.addEventListener('change', render);
};
