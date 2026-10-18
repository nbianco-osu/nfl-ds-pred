# R75-R77: On-demand research views

- R75: Matched Evaluation and Experiments & Inputs request their five datasets
  only when opened. Current snapshot defers 257,415 uncompressed bytes (about
  251 KiB); this is a payload reduction, not a measured latency improvement.
- R76: Shared loading state with aria-busy, a live status message and explicit
  Retry view control. Fetch inputs before attaching controls; successful views
  initialize once. Renderer failures request reload rather than risking duplicate
  controls. All retries use the existing pinned and hash-verified data loader.
- R77: Three additional regression tests cover deferred loading, concurrent
  activation, retry recovery and renderer failure. Included in GitHub validation.

Verification: 10 Node tests pass. Static build verifies all 52 datasets. Installed
Chrome through bundled Playwright (Browser plugin not available) passes at 1440px
and 390px: neither research payload requested on initial schedule; injected 503
recovers; revisit does not duplicate options; no JavaScript errors or page-width
overflow. Temporary QA harness and screenshots are outside the repository.

No training, forecast, cohort, scheduler or production-model changes. Other
research panels still preload. Experiments waits for all four inputs before
initializing; independently retryable subsections remain a future improvement.

Publication blocked: GitHub push dry run fails in the Windows credential helper
with sandbox access denied. R71-R77 are not confirmed live. Prior Sites secure
helper input remains blocked; no new Sites version was published in this round.

Next: defer remaining large research payloads, make each experiment section
independently retryable, and run browser fault-injection checks in CI.
