# R71-R74: Dashboard delivery and navigation

- R71: Render schedule rows and bind filters as soon as predictions arrive.
  Optional summaries no longer block the schedule. Load season chart data only
  when its view opens, with explicit loading and retry states.
- R72: Group tabs into Games, Teams, Evaluation and Operations. Preserve existing
  panel IDs, ARIA selection, and model-library links. Arrow/Home/End keys traverse
  only the visible category. Navigation initializes before schedule data arrives.
- R73: Evict failed dataset promises, enabling retry against the same immutable
  revision. Clear recovered failures; failed or corrupt loads never count as
  successful downloads. Do not substitute data from another revision.
- R74: Add regression coverage for retries, repeated corruption, lazy chart
  requests, grouped keyboard navigation and idempotent initialization.

Verification: seven Node tests; static archive integrity build; desktop 1440px
and mobile 390px Playwright checks using installed Chrome (Browser plugin not
available). Both viewports recovered from an injected chart HTTP 503, requested
no chart data at startup, had no page JavaScript errors or page-width overflow.
Screenshots and temporary browser harness remain outside the repository.

Limits: other research panels still preload their datasets. Manifest failures
require a page reload. This is a frontend-only iteration: production models,
forecasts, training data, shadow cohorts and refresh schedules are unchanged.
Publication status is reported separately; local checks do not imply deployment.

Next candidates: defer the remaining research payloads; make per-view loading
and failure states consistent; add a CI browser smoke test with network faults.
