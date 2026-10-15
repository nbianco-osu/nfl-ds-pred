import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const dist = join(root, "dist");

const assets = {
  "/retraining.js": { file: "retraining.js", type: "text/javascript; charset=utf-8" },
  "/data/retraining.json": { file: "data/retraining.json", type: "application/json; charset=utf-8" },
  "/data/retraining_holdout.csv": { file: "data/retraining_holdout.csv", type: "text/csv; charset=utf-8" },
  "/lab.js": { file: "lab.js", type: "text/javascript; charset=utf-8" },
  ...Object.fromEntries(['historical_coverage','feature_value','robustness','checkpoints','error_analysis','data_quality','model_registry','kickoffs','qb_scenarios','reproducibility'].map(name=>[`/data/${name}.json`,{file:`data/${name}.json`,type:'application/json; charset=utf-8'}])),
  "/iteration.js": { file: "iteration.js", type: "text/javascript; charset=utf-8" },
  "/data/reliability.json": { file: "data/reliability.json", type: "application/json; charset=utf-8" },
  "/data/live_challengers.json": { file: "data/live_challengers.json", type: "application/json; charset=utf-8" },
  "/data/season_replay.json": { file: "data/season_replay.json", type: "application/json; charset=utf-8" },
  "/data/uncertainty_research.json": { file: "data/uncertainty_research.json", type: "application/json; charset=utf-8" },
  "/data/opponent_ratings.json": { file: "data/opponent_ratings.json", type: "application/json; charset=utf-8" },
  "/data/input_collection.json": { file: "data/input_collection.json", type: "application/json; charset=utf-8" },
  "/experiments.js": { file: "experiments.js", type: "text/javascript; charset=utf-8" },
  "/data/score_experiments.json": { file: "data/score_experiments.json", type: "application/json; charset=utf-8" },
  "/data/temporal_stack.json": { file: "data/temporal_stack.json", type: "application/json; charset=utf-8" },
  "/data/pregame_inputs.json": { file: "data/pregame_inputs.json", type: "application/json; charset=utf-8" },
  "/data/warehouse_status.json": { file: "data/warehouse_status.json", type: "application/json; charset=utf-8" },
  "/matched.js": { file: "matched.js", type: "text/javascript; charset=utf-8" },
  "/data/matched_comparison.json": { file: "data/matched_comparison.json", type: "application/json; charset=utf-8" },
  "/insights.js": { file: "insights.js", type: "text/javascript; charset=utf-8" },
  "/data/forecast_history.json": { file: "data/forecast_history.json", type: "application/json; charset=utf-8" },
  "/data/drift_monitor.json": { file: "data/drift_monitor.json", type: "application/json; charset=utf-8" },
  "/research.js": { file: "research.js", type: "text/javascript; charset=utf-8" },
  "/data/game_audit.json": { file: "data/game_audit.json", type: "application/json; charset=utf-8" },
  "/data/matchup_context.json": { file: "data/matchup_context.json", type: "application/json; charset=utf-8" },
  "/data/score_distributions.json": { file: "data/score_distributions.json", type: "application/json; charset=utf-8" },
  "/data/distribution_metrics.json": { file: "data/distribution_metrics.json", type: "application/json; charset=utf-8" },
  "/data/advanced_backtest.json": { file: "data/advanced_backtest.json", type: "application/json; charset=utf-8" },
  "/data/challenger_review.json": { file: "data/challenger_review.json", type: "application/json; charset=utf-8" },
  "/data/refresh_health.json": { file: "data/refresh_health.json", type: "application/json; charset=utf-8" },
  "/performance.js": { file: "performance.js", type: "text/javascript; charset=utf-8" },
  "/matchups.js": { file: "matchups.js", type: "text/javascript; charset=utf-8" },
  "/data/performance.json": { file: "data/performance.json", type: "application/json; charset=utf-8" },
  "/data/backtest.json": { file: "data/backtest.json", type: "application/json; charset=utf-8" },
  "/data/model_score_predictions.json": { file: "data/model_score_predictions.json", type: "application/json; charset=utf-8" },
  "/score_lines.js": { file: "score_lines.js", type: "text/javascript; charset=utf-8" },
  "/data/score_metrics.json": { file: "data/score_metrics.json", type: "application/json; charset=utf-8" },
  "/season_chart.js": { file: "season_chart.js", type: "text/javascript; charset=utf-8" },
  "/vendor/chart.umd.min.js": { file: "vendor/chart.umd.min.js", type: "text/javascript; charset=utf-8" },
  "/vendor/Chart.js.LICENSE.md": { file: "vendor/Chart.js.LICENSE.md", type: "text/plain; charset=utf-8" },
  "/data/season_charts.json": { file: "data/season_charts.json", type: "application/json; charset=utf-8" },
  "/data/expanded_coverage.json": { file: "data/expanded_coverage.json", type: "application/json; charset=utf-8" },
  "/data/model_comparison.json": { file: "data/model_comparison.json", type: "application/json; charset=utf-8" },
  "/data/season_simulations.json": { file: "data/season_simulations.json", type: "application/json; charset=utf-8" },
  "/": { file: "index.html", type: "text/html; charset=utf-8" },
  "/index.html": { file: "index.html", type: "text/html; charset=utf-8" },
  "/styles.css": { file: "styles.css", type: "text/css; charset=utf-8" },
  "/app.js": { file: "app.js", type: "text/javascript; charset=utf-8" },
  "/data/predictions.json": { file: "data/predictions.json", type: "application/json; charset=utf-8" },
  "/data/teams.json": { file: "data/teams.json", type: "application/json; charset=utf-8" },
  "/data/global_shap.json": { file: "data/global_shap.json", type: "application/json; charset=utf-8" },
  "/data/metrics.json": { file: "data/metrics.json", type: "application/json; charset=utf-8" },
};

await rm(dist, { recursive: true, force: true });
await mkdir(join(dist, "server"), { recursive: true });

const manifest = {};
for (const [route, asset] of Object.entries(assets)) {
  manifest[route] = {
    content: await readFile(join(root, asset.file), "utf8"),
    type: asset.type,
  };
}

const worker = `const ASSETS = ${JSON.stringify(manifest)};

function responseFor(pathname) {
  const asset = ASSETS[pathname] || ASSETS["/"];
  return new Response(asset.content, {
    headers: {
      "content-type": asset.type,
      "cache-control": pathname.startsWith("/data/")
        ? "public, max-age=300"
        : "public, max-age=3600",
    },
  });
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    return responseFor(url.pathname);
  },
};
`;

await writeFile(join(dist, "server", "index.js"), worker);
