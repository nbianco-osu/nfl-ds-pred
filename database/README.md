# NFL Research Warehouse

Project: `nfl-ds-pred` (`unzfybzuynezjhcjyptu`), Free, us-west-2.

The initial warehouse is a private snapshot table with indexed dataset, season
and game keys plus JSONB payloads. Snapshot hashes and deterministic record IDs
keep different revisions separate. This preserves wide and changing feature
schemas without thousands of sparse SQL columns. Missing payload keys mean null.
It is not yet a normalized application database or a replacement for local files.

`schema.sql` is the reviewed initial DDL, not a generated CLI migration.
RLS is enabled, anon/authenticated grants revoked, and no public policies added.
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
privately in the process environment and run `sync_supabase.py --push`. It uses
50-record batches, immutable-ID conflict handling and read-back verification.
Do not paste keys into chat or commit them. Remote sync is opt-in, not part of
the weekly refresh until credentials and access are configured securely.

FTN charted metrics: **FTN Data via nflverse**, CC-BY-SA 4.0. Treat derived
charted data as share-alike when redistributing. Snap counts: Pro Football
Reference via nflverse; retain privately pending redistribution-terms review.
Raw 2022-2026 source files remain local under `data/research_sources/` with
checksums and receipts. No paywall or access restriction was bypassed.

## Next Research

1. Evaluate all 64 models on identical rolling folds before ranking them.
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
