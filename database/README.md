# NFL Research Warehouse

Project: `nfl-ds-pred` (`unzfybzuynezjhcjyptu`), Free, us-west-2.

## Recovery Complete (2026-10-14)

All 10,089 manifest records are present. With user authorization, the connected
Supabase tool inserted the remaining 1,217 rows in 49 conflict-safe batches.
Each batch verified metadata and semantic JSONB equality; existing rows were
not overwritten. Aggregate SHA-256 fingerprints for all seven datasets match
the original local manifest sources after normalizing numeric scale (`7.0` = `7`).
Counts, fingerprints and access checks are recorded in `import_verification.json`.
RLS remains enabled; neither anon nor authenticated has SELECT privileges.
No credentials were collected and no schema or grants were changed.

`warehouse_fingerprints.py` verifies source file checksums and reproduces the
content fingerprints; `verify_snapshot.sql` provides the read-only remote query.
`prepare_connector_recovery.py` stages SQL requests for
the specific recovered source, not an automatic general-purpose sync job.
The public dashboard still uses static exports. R22 remains proposed.

### Previous Import Checkpoint (2026-10-13)

PARTIAL: 8,872 of 10,089 intended records are stored and verified against a full
browser CSV export, including semantic JSON payload comparisons and unique IDs.
Part 1: 4,102 / 5,319 matched; part 2: all 4,770 matched. The missing 1,217 rows
are historical research matchups. Supabase's browser importer repeatedly failed
with `Failed to fetch (api.supabase.com)` while the dashboard showed a service
incident banner. Existing rows were preserved; no delete or overwrite was used.

The user explicitly chose browser import, not a password-based direct connection.
At that checkpoint no upload was running. The recovery plan was to refresh/export the table and
reconcile again before importing. The current local remainder is
`data/database_import/resume_missing_part_01.csv` (1,217 rows), SHA-256
`7ad4626c50e39e7f3b39c64716be35cab17eeef112470f29f23e8a7cffc05662`.
Its export checkpoint SHA-256 is
`7034ee979dacb799b351767ea93baf665c85a7e2d59f7e2f06a7ccf379a66c23`.
Receipts are alongside the CSVs; raw exports/CSV payloads are not committed.
Do not re-import old `recovery_part_*` files. All current forecasts, schedules,
team-game metrics, team metadata, source receipts and 32 new model records are
included in the fully verified second batch.

## Schema and Access

The initial warehouse is a private snapshot table with dataset, season
and game keys plus JSONB payloads. Snapshot hashes and deterministic record IDs
keep different revisions separate. This preserves wide and changing feature
schemas without thousands of sparse SQL columns. Missing payload keys mean null.
It is not yet a normalized application database or a replacement for local files.

`schema.sql` is proposed initial DDL, not an applied or generated CLI migration.
The SQL editor could not initialize during setup. The browser Table Editor
created the table with a text `id` primary key, JSONB `payload`, bigint `season`,
and the other text columns. The Policies page verified RLS enabled, API DISABLED,
and no policies on October 13, 2026. The draft's additional indexes, checks, unique constraint,
`imported_at`, explicit privilege revocations and service-role grants have NOT
been applied or verified remotely. Reconcile the existing table with the draft
before API sync; `CREATE TABLE IF NOT EXISTS` alone will not add missing columns
or constraints. Do not describe the draft as the deployed schema.
The existing public dashboard continues to serve static exports; no secret key
is sent to the browser. Do not disable RLS to troubleshoot access.

```powershell
.\.venv\Scripts\python.exe collect_research_data.py
.\.venv\Scripts\python.exe research_dataset.py
.\.venv\Scripts\python.exe train_research32.py
.\.venv\Scripts\python.exe prepare_database.py
.\.venv\Scripts\python.exe sync_supabase.py
```

Initial CSV import uses only files listed in `data/database_import/manifest.json`.
Each is below the dashboard's 100 MB limit. Do not re-import CSVs blindly: verify
counts first. For repeatable sync, set `SUPABASE_URL` and `SUPABASE_SECRET_KEY`
privately in the process environment, apply and verify the intended grants and
constraints, then run `sync_supabase.py --push`. It uses
50-record batches, immutable-ID conflict handling and read-back verification.
Do not paste keys into chat or commit them. Remote sync is opt-in, not part of
the weekly refresh until credentials and access are configured securely.

For an interrupted browser import, export the existing table as CSV and use
`reconcile_browser_import.py --source <original-part.csv> --export <export.csv>
--output <new-missing.csv>`. This validates matching rows (including JSON payloads)
and rejects duplicate IDs before producing only missing rows. Keep its receipt.
Wait for all imports to finish, export again, and reconcile EVERY manifest part
before declaring completion. A progress toast or matching total count alone is
not enough. Recovery files are valid only against the export used to create them;
never blindly rerun an old recovery file after another interrupted import.

FTN charted metrics: **FTN Data via nflverse**, CC-BY-SA 4.0. Treat derived
charted data as share-alike when redistributing. Snap counts: Pro Football
Reference via nflverse; retain privately pending redistribution-terms review.
Raw 2022-2026 source files remain local under `data/research_sources/` with
checksums and receipts. No paywall or access restriction was bypassed.

## Next Research

R22 update (2026-10-14): `normalized_warehouse.sql` and
`harden_setup_function.sql` are applied. The additive `nfl_private` schema has
36 team identifiers, 38 venues, 42 schedule-QB identities and 272 schedule
observations. These identities are not confirmed starters or complete rosters.
All tables have RLS and no public read grants. Existing 10,089 snapshots remain
intact. See `normalized_verification.json` for the verified receipt.

Weekly refresh stages an insert-only queue with `warehouse_queue.py`. The
connected Supabase tool executes and verifies batches; this is assistant-mediated,
not a standalone background service. The initial 58 batches matched all 580
records and inserted none. Never mark a queue synced before remote verification.
The older secret-key REST path above is separate and is not configured or
validated for the new private normalization trigger. Raw objects remain local
pending license and storage review. See `backtests/NEXT_RESEARCH.md` for R19-R21.

1. R18 now compares all 64 recipes on identical rolling folds; collect prospective evidence before any promotion.
2. Test Poisson/Tweedie score models and football-scoring-process simulations;
   compare interval coverage and margin/total scoring, not just winner accuracy.
3. Collect timestamped confirmed starters and weather forecasts prospectively.
4. Use regularized temporal stacking, trained only on out-of-fold forecasts.
5. Normalize team/player/venue dimensions and add private raw-object storage
   after storage/cost and source-license review.

Sources:
- https://nflreadr.nflverse.com/reference/load_ftn_charting.html
- https://nflreadr.nflverse.com/reference/load_snap_counts.html
- https://scikit-learn.org/1.8/modules/calibration.html
- https://supabase.com/docs/guides/database/import-data
- https://supabase.com/docs/guides/database/postgres/row-level-security
