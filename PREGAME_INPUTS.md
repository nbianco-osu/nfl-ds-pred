# Timestamped Pregame Imports (R20)

R23 enables the non-commercial Open-Meteo forecast API. Confirmed starters still
require supplied source-backed imports. Missing inputs remain unavailable; past
QB usage and schedule QB names are not confirmation. No paid service or key is used.

Import a JSON array or CSV with `python pregame_inputs.py --season 2026 --input PATH`.
The importer records its own UTC observation time. Supplied `observed_at` values
are ignored. Completed games, late observations, future publication times and
timezone-free timestamps are rejected. An invalid batch writes nothing. Exact
duplicates retain their first observation; changed reports become new records.

Required common fields: `kind`, `game_id`, `provider`, `source_url` (public HTTPS),
and `published_at` (ISO 8601 with timezone). These are supplied source assertions,
not independently verified facts. Keep the original source and licensing terms.

For `kind=starter`, also supply `team`, `player_id`, `player_name`, and
`status=confirmed`. Team must participate in the game. Use a source that explicitly
confirms the starter, not a depth-chart projection or injury-status inference.

For `kind=weather`, also supply `stadium_id`, `valid_at`, `temperature_f` and
`wind_mph`. Stadium must match the schedule and validity must be within three
hours of kickoff. Publication time is not retrieval time: never fabricate an
issue timestamp for a provider that only reports when it was retrieved.

Archive: `audit/pregame_inputs.json`. Public summary:
`public_site/data/pregame_inputs.json`. Weekly refresh exports the archive without
backfilling history or retraining production. Store source URLs, not credentials.
Prospective feature-value measurement remains pending until enough independently
observed games accumulate. R23 initially collected 23 weather observations.

Provider research: [Open-Meteo documentation](https://open-meteo.com/en/docs) and
[terms](https://open-meteo.com/en/terms). The free endpoint has non-commercial-use
and attribution conditions. `weather_inputs.py` uses stadium-ID coordinate matches
from greerreNFL/Stadiums. Missing coordinates stay explicit, including international
venues. Closed/dome games are skipped; forecasts for other roofs describe outdoor
conditions, not confirmed roof status. Forecasts are limited to the provider horizon.
API issue time is unavailable: `published_at` is null, and eligibility uses actual
local `observed_at`. Raw responses are private and content-addressed. No weather
history is backfilled. Failures preserve prior observations and appear in monitoring.
