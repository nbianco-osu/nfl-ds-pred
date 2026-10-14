# Timestamped Pregame Imports (R20)

No automatic starter or weather provider is configured. Missing inputs remain
unavailable. Past QB usage and historical schedule QB names are not confirmation.
No paid service, API key, scraping job or external bucket was added.

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
observed games accumulate. Current observations: none at implementation.

Provider research: [Open-Meteo documentation](https://open-meteo.com/en/docs) and
[terms](https://open-meteo.com/en/terms). No integration is activated; its free
endpoint has non-commercial-use and attribution conditions. Venue coordinates
and a suitable issue-time policy must be established before enabling a provider.
