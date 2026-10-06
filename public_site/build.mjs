import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const dist = join(root, "dist");

const assets = {
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
