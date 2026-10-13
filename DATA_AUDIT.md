# Data Availability and Leakage Audit

Audit started 2026-10-13. This is a prospective improvement, not certification of
past data or a claim that historical releases have been recovered.

## What Is Preserved

- Raw nflreadpy loader responses are content-addressed Parquet files under
  `audit/sources/`. Raw files remain local and are Git-ignored. Source receipts
  record SHA-256, row count and observation time; they do not invent provider
  publication times. A loader may serve its cache.
- Winner forecasts made after this audit contain creation timestamps. Exact
  feature values, model version, model-file SHA-256 and source hashes are saved
  under `audit/forecasts/<game_id>/`. Local explanations reconstruct the matching
  model probability before being exported.
- Forecasts freeze at kickoff, including games still awaiting final scores.
  Missing kickoff time uses local midnight conservatively. Completed games with
  no original archive are never retrospectively filled.
- Score-distribution experiments preserve first forecasts within independent
  version cohorts under `learning/score_distributions/`. These do not replace
  winner-calibration cohorts or production score forecasts.

## Known Limits

| Input | Cutoff / current behavior | Remaining risk |
| --- | --- | --- |
| PBP and QB performance | Only earlier weeks enter current rollups | Source records may be revised later; historical first-release snapshots unavailable |
| Rest | Schedule-derived pregame context | Schedule revisions must remain tied to the captured response |
| Injury reports | Current-week reports where present; otherwise historical fallback | Missing reports do not mean healthy. Original publication times unknown |
| Weekly rosters | Current-week source snapshot where available | A roster entry is not proof of game-day availability |
| Legacy active-roster feature | Existing production feature definition remains unchanged | Existing extractor searches the word `active`, but current source uses `ACT`/`INA` codes. The legacy feature is unreliable; correcting production requires consistent historical rebuild and model review. New research active-OL counts use exact `ACT`/`ACTIVE` matching |
| Expected QB | Explicitly unconfirmed | Previous leading passer and attempt share are not a confirmed starter |
| Weather | Only a supplied, timestamped forecast can enter research context | Production training includes historical schedule temperature/wind, whose pregame availability is unverified; current missing values remain imputed. No weather provider is configured |
| Market lines | Saved serving snapshots freeze; historical research uses revised schedule lines | Research line metrics are not verified live market results or profitability |
| Advanced historical features | Lagged features selected for research; current-week injury/roster and observed weather excluded | Revised history and missing tie-game features limit as-of claims |

## Research Context

`matchup_context.py` exports prior five-game play volume, a simple opponent-adjusted
EPA proxy, last QB usage and same-week OL information. These inputs do not silently
change the production feature contract. Play volume is not seconds per play; the
opponent adjustment subtracts the opposing defense's average from earlier weeks.
Historical weather and confirmed-starter data are still needed for their ablations.

Optional `data/weather_forecasts.csv` columns:

```text
game_id,issued_at,valid_at,temperature_f,wind_mph,provider
```

Times must be timezone-aware ISO timestamps. Issuance must precede kickoff and
must not be in the future; validity must be within three hours of kickoff. Missing
or invalid coverage stays unavailable. This adapter is not an automatic weather
data feed. Do not use observed postgame weather as a pregame forecast.

## Validation and Operations

`refresh_predictions.py` stages work, validates public data and unchanged completed
forecasts, then atomically replaces individual files. Failure during application
rolls back applied files. This is not a filesystem-wide crash-atomic transaction;
the success manifest is written last. Deploy only validated builds. Raw source
snapshots should be backed up separately because they are not pushed to GitHub.

`publication_changes.py` ignores operational timestamps, but retains changes to
forecast timestamps, model versions, probabilities and input fingerprints.
The weekly automation uses this comparison before publishing. It does not retrain
models, run expensive backtests or approve candidates automatically.

## Sources

- [nflverse data availability](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
- [Schedule and line definitions](https://nflreadr.nflverse.com/articles/dictionary_schedules.html)
- [Injury report loader](https://nflreadr.nflverse.com/reference/load_injuries.html)

Actual fetched coverage and receipts take precedence over assumptions based only
on a general documentation page.
