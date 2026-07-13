const pct = (value) => `${(value * 100).toFixed(1)}%`;
const pctWhole = (value) => `${value.toFixed(1)}%`;
const byId = (id) => document.getElementById(id);

const featureReplacements = [
  ["team_pbp_prev_season_", "previous season play-by-play "],
  ["team_qb_prev_season_", "previous season QB "],
  ["team_pbp_", "play-by-play "],
  ["team_qb_", "QB "],
  ["team_injury_", "injury report "],
  ["team_roster_", "roster "],
  ["off_epa_per_play", "offensive EPA per play"],
  ["def_epa_allowed_per_play", "defensive EPA allowed per play"],
  ["off_success_rate", "offensive success rate"],
  ["def_success_allowed_rate", "defensive success rate allowed"],
  ["off_explosive_rate", "offensive explosive play rate"],
  ["def_explosive_allowed_rate", "defensive explosive play rate allowed"],
  ["point_diff_per_game", "point differential per game"],
  ["points_for_per_game", "points scored per game"],
  ["last_5_point_diff", "last five game point differential"],
  ["win_pct", "win percentage"],
];

function readableFeature(feature) {
  let name = feature.replace(/^num__|^cat__/, "");
  let prefix = "";
  if (name.startsWith("diff_")) {
    prefix = "Home edge in ";
    name = name.slice(5);
  } else if (name.startsWith("home_")) {
    prefix = "Home team ";
    name = name.slice(5);
  } else if (name.startsWith("away_")) {
    prefix = "Away team ";
    name = name.slice(5);
  }
  for (const [oldValue, newValue] of featureReplacements) {
    name = name.replaceAll(oldValue, newValue);
  }
  name = name.replaceAll("_last_3", ", last 3 games");
  name = name.replaceAll("_last_5", ", last 5 games");
  name = name.replaceAll("_avg", ", season-to-date average");
  name = name.replaceAll("_", " ");
  return `${prefix}${name}`.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metricCard(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderMetrics(predictions, metrics) {
  byId("metrics").innerHTML = [
    metricCard("Predicted Games", predictions.length.toLocaleString()),
    metricCard("Model", "Random Forest"),
    metricCard("Holdout Log Loss", metrics.log_loss.toFixed(3)),
    metricCard("Holdout ROC AUC", metrics.roc_auc.toFixed(3)),
  ].join("");
}

function renderFilters(predictions) {
  const weeks = [...new Set(predictions.map((row) => row.week))].sort((a, b) => a - b);
  const teams = [...new Set(predictions.flatMap((row) => [row.away_team_name, row.home_team_name]))].sort();
  byId("weekFilter").innerHTML = [`<option value="all">All weeks</option>`, ...weeks.map((week) => `<option value="${week}">Week ${week}</option>`)].join("");
  byId("teamFilter").innerHTML = [`<option value="all">All teams</option>`, ...teams.map((team) => `<option value="${team}">${team}</option>`)].join("");
}

function filterPredictions(predictions) {
  const week = byId("weekFilter").value;
  const team = byId("teamFilter").value;
  const confidence = Number(byId("confidenceFilter").value) / 100;
  byId("confidenceValue").textContent = `${byId("confidenceFilter").value}%`;
  return predictions.filter((row) => {
    const weekMatch = week === "all" || row.week === Number(week);
    const teamMatch = team === "all" || row.away_team_name === team || row.home_team_name === team;
    return weekMatch && teamMatch && row.confidence >= confidence;
  });
}

function renderRows(rows) {
  byId("visibleCount").textContent = `${rows.length} games`;
  byId("predictionRows").innerHTML = rows
    .sort((a, b) => a.week - b.week || String(a.gameday).localeCompare(String(b.gameday)))
    .map(
      (row) => `
        <tr>
          <td>${row.week}</td>
          <td><div class="team-cell"><img src="${row.away_logo}" alt="" />${row.away_team_name}</div></td>
          <td><div class="team-cell"><img src="${row.home_logo}" alt="" />${row.home_team_name}</div></td>
          <td><div class="pick-cell"><img src="${row.winner_logo}" alt="" /><strong>${row.predicted_winner_name}</strong></div></td>
          <td>${pct(row.away_win_probability)}</td>
          <td>${pct(row.home_win_probability)}</td>
          <td>${pct(row.confidence)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderWinnerBars(rows) {
  const counts = new Map();
  for (const row of rows) counts.set(row.predicted_winner_name, (counts.get(row.predicted_winner_name) || 0) + 1);
  const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
  const max = Math.max(1, ...entries.map((entry) => entry[1]));
  byId("winnerBars").innerHTML = entries
    .map(([team, count]) => barRow(team, `${count} picks`, (count / max) * 100))
    .join("");
}

function barRow(label, value, width) {
  return `
    <div class="bar-row">
      <div class="bar-label"><span>${label}</span><strong>${value}</strong></div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, width)}%"></div></div>
    </div>
  `;
}

function renderShap(shapRows) {
  const max = Math.max(...shapRows.map((row) => row.mean_abs_shap));
  byId("shapBars").innerHTML = shapRows
    .slice(0, 20)
    .map((row) => barRow(readableFeature(row.feature), row.mean_abs_shap.toFixed(4), (row.mean_abs_shap / max) * 100))
    .join("");
}

async function main() {
  const [predictions, shapRows, metrics] = await Promise.all([
    fetch("./data/predictions.json").then((response) => response.json()),
    fetch("./data/global_shap.json").then((response) => response.json()),
    fetch("./data/metrics.json").then((response) => response.json()),
  ]);

  renderMetrics(predictions, metrics);
  renderFilters(predictions);
  renderShap(shapRows);

  function update() {
    const visible = filterPredictions(predictions);
    renderRows(visible);
    renderWinnerBars(visible);
  }

  for (const id of ["weekFilter", "teamFilter", "confidenceFilter"]) {
    byId(id).addEventListener("input", update);
  }
  update();
}

main().catch((error) => {
  document.body.innerHTML = `<main><section class="panel"><h1>Unable to load dashboard</h1><p>${error.message}</p></section></main>`;
});
