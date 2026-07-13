import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const dist = join(root, "dist");

const assets = {
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
