const { chromium } = require('playwright');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const root = path.join(__dirname, 'public_site');
  const mime = {'.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css'};
  const server = http.createServer((req, res) => {
    const file = path.resolve(root, '.' + (new URL(req.url, 'http://localhost').pathname === '/' ? '/index.html' : new URL(req.url, 'http://localhost').pathname));
    if (!file.startsWith(root + path.sep) || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
    res.end(fs.readFileSync(file));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({channel: 'msedge', headless: true});
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.waitForFunction(() => document.querySelectorAll('#modelRows tr').length === 32);
    await page.selectOption('#modelGroup', 'Expanded');
    assert.equal(await page.locator('#modelRows tr').count(), 16);
    await page.selectOption('#modelSort', 'accuracy');
    await page.click('#seasonTab');
    assert.equal(await page.locator('#teamCharts canvas').count(), 32);
    await page.waitForFunction(() => [...document.querySelectorAll('#teamCharts canvas')].every(c => c.width > 0 && c.height > 0));
    const ink = await page.locator('#teamCharts canvas').first().evaluate(c => {
      const rgba = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let colored = 0;
      for (let i = 0; i < rgba.length; i += 4) if (rgba[i + 3] && rgba[i + 2] > rgba[i] + 40) colored++;
      return colored;
    });
    assert(ink > 100, 'Forecast line must render');
    await page.screenshot({path: path.join(__dirname, 'data/dashboard_desktop.png')});
    await page.selectOption('#chartTeam', 'SEA');
    await page.selectOption('#chartMeasure', 'cumulative');
    assert.equal(await page.locator('#teamCharts canvas').count(), 1);
    assert.match(await page.locator('#chartNote').innerText(), /actual wins/);
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#seasonPanel').scrollIntoViewIfNeeded();
    await page.screenshot({path: path.join(__dirname, 'data/dashboard_mobile.png')});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Mobile page overflow');
    await page.selectOption('#chartTeam', 'all');
    assert.equal(await page.locator('#teamCharts canvas').count(), 32);
    await page.focus('#seasonTab');
    await page.keyboard.press('ArrowLeft');
    assert.equal(await page.locator('#dashboardTab').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#seasonPanel').isVisible(), false);
    await page.click('#scoresTab');
    await page.waitForFunction(() => document.querySelectorAll('#scoreModelRows tr').length === 5);
    assert.match(await page.locator('#scoreEvaluation').innerText(), /Extra Trees/);
    assert.match(await page.locator('#scoreLiveResults').innerText(), /no settled pregame picks/);
    assert.equal(await page.locator('#scoreRows tr').count(), 272);
    await page.selectOption('#scoreWeek', '4');
    assert.equal(await page.locator('#scoreRows tr').count(), 16);
    assert.match(await page.locator('#scoreRows').innerText(), /covers/);
    const forecasts = JSON.parse(fs.readFileSync(path.join(root, 'data/predictions.json')));
    const firstWeek4 = forecasts.find(g => g.week === 4);
    assert.equal(await page.locator('[data-score="predicted"]').first().innerText(), `${firstWeek4.predicted_away_score.toFixed(1)} - ${firstWeek4.predicted_home_score.toFixed(1)}`);
    assert((await page.locator('[data-score="total"]').first().innerText()).startsWith((firstWeek4.predicted_home_score + firstWeek4.predicted_away_score).toFixed(1)));
    assert.equal(await page.locator('[data-score="actual-total"]').first().innerText(), 'Pending');
    await page.selectOption('#scoreWeek', '6');
    assert.match(await page.locator('#scoreRows').innerText(), /Not available/);
    await page.selectOption('#scoreWeek', '1');
    assert.match(await page.locator('#scoreRows').innerText(), /No pregame forecast/);
    assert.match(await page.locator('#scoreRows').innerText(), /No archived pick/);
    const firstWeek1 = forecasts.find(g => g.week === 1);
    assert.equal(await page.locator('[data-score="actual-total"]').first().innerText(), (firstWeek1.home_score + firstWeek1.away_score).toFixed(1));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Scores mobile overflow');
    await page.screenshot({path: path.join(__dirname, 'data/scores_mobile.png')});
    await page.setViewportSize({width:1440,height:1000});
    await page.selectOption('#scoreWeek', '4');
    await page.screenshot({path: path.join(__dirname, 'data/scores_desktop.png')});
    assert.deepEqual(errors, []);
    console.log('PASS: 32-model table, model filters/sort, 32 nonblank charts, team/mode controls, mobile overflow, keyboard tabs, no page errors.');
  } finally {
    if (browser) await browser.close();
    server.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
