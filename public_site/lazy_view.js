/* Fetch a view's pinned inputs before attaching its renderers and controls. */
window.registerDashboardView = function ({panelId, files, initialize}) {
  const panel = document.getElementById(panelId);
  const status = document.createElement('p');
  status.setAttribute('role', 'status');
  const retry = document.createElement('button');
  retry.textContent = 'Retry view'; retry.hidden = true;
  panel.prepend(status, retry);
  let state = 'idle';
  async function load() {
    if (state === 'loading' || state === 'ready' || state === 'broken') return;
    state = 'loading'; retry.hidden = true;
    status.textContent = 'Loading saved research results...';
    panel.setAttribute('aria-busy', 'true');
    try {
      await Promise.all(files.map(async name => {
        const response = await window.dashboardFetch(`./data/${name}.json`);
        if (!response.ok) throw Error('Unavailable');
        await response.json();
      }));
    } catch {
      state = 'failed'; retry.hidden = false;
      status.textContent = 'Research data unavailable. Production forecasts are unaffected.';
      panel.setAttribute('aria-busy', 'false');
      return;
    }
    try {
      await initialize();
      state = 'ready'; status.textContent = '';
    } catch {
      state = 'broken';
      status.textContent = 'This view could not render. Reload the page to retry. Production forecasts are unaffected.';
    } finally { panel.setAttribute('aria-busy', 'false'); }
  }
  retry.addEventListener('click', load);
  new MutationObserver(() => { if (!panel.hidden) load(); })
    .observe(panel, {attributes: true, attributeFilter: ['hidden']});
  if (!panel.hidden) load();
};
