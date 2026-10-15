# R57 private runner

Nothing here activates schedules or changes billing. Only `runner/` and
`test_runner.py` are owned by this change. Do not commit runtime assets publicly.

## Bootstrap

From the shared source root, after inspecting the bundle policy:

```powershell
.\.venv\Scripts\python.exe -m unittest test_runner -v
.\.venv\Scripts\python.exe -m pip install pynacl
.\.venv\Scripts\python.exe -m runner.bootstrap --create --deploy-key --upload
```

Git credential manager is queried with captured subprocess output. OAuth stays
in memory, is never logged or stored, and is never installed as an Actions secret.
The only cross-repository writer secret is a new repository-specific deploy key.
Its temporary local private-key file is deleted after sealed-box encryption.
Existing private state is never reset by repeated bootstrap. A partial key install
stops for review instead of silently rotating/revoking another key.

The bundle includes JSON/CSV/Parquet/joblib under models, data, backtests,
predictions, audit, learning, and public_site/data. It excludes archive directories,
notebooks, binaries, images, credentials, dotfiles, locks and Python caches.
No executable code, environment, .git metadata, or deployment archives are bundled.
Every member has a SHA256 in private state.json; the gzip asset has its own digest.
Concurrent local edits invalidate the snapshot. Bootstrap only while local work
is idle. Setup will refuse an existing refresh lock.

## Parent handoff and activation

1. Parent reviews and commits source changes, including runner, publication_gate,
   forecast_scorecard, iteration_evidence and dashboard_snapshot, then pushes main.
2. Dispatch the private workflow with `duecheck`. This restores a fresh runtime,
   verifies asset/member hashes, and prints due windows without refresh, state
   mutation, public publishing, or fetching a new schedule.
3. Dispatch `dry-run` on a fresh runner. This performs a real isolated staged
   refresh, preserves immutable forecasts/models/checkpoints, validates transitions,
   exports scorecard/iteration evidence and the single live_snapshot.json.
   It does not publish, save private state, or approve itself. Save its successful
   run URL, exact source SHA, private state SHA and logs for parent review.
4. Local automation is still authoritative. Parent must take a new idle snapshot
   for switch-over; bootstrap deliberately cannot replace initialized state.
   Do not flip flags against an old bootstrap snapshot: reconcile current local
   state using preservation checks and a CAS commit first. This switch-over import
   is an explicit remaining operator step, not an automatic bootstrap overwrite.
5. Only after explicit parent approval set all three private config gates true:
   automatic_enabled, parent_approved, dry_run_verified. Record verification run
   provenance with the configuration change. Parent disables local automation.
6. Review schedule-plan.json and install its UTC cron entries in the private
   workflow only after approval. The full-season static weekly union covers all
   known kickoff times, including DST, with three checks per 24h/90m window.
   No weekly renewal is needed. Daily 11:17 UTC checks detect flex-schedule drift
   and report uncovered cron coverage; Tuesday 09:00 America/New_York forces a
   results refresh (13:00 and 14:00 UTC cron, runtime DST guard selects one).
   Other checks refresh changed completed results too. Unknown kickoffs are never
   invented. Daily drift proposals never edit or activate the workflow themselves.
   The current 2026 union is 54 cron entries, at most 151 checks/week, about 657/month.
   Unusual flex changes outside this union require parent review, not weekly upkeep.

Immediately before switching run `python -B -m runner.bootstrap --handoff-check`.
It compares every runtime file hash against private state and refuses if local
authority has advanced. Pause local writers during final snapshot, verification,
approval and activation. A test is not an authorization for overlapping writers.

## Persistence and publication

### Verified handoff prerequisites (2026-10-15)

- Fresh Linux dry-run: private Actions run `38800654893`, successful, source
  `a0d5ed7572ba924029cd53f59b50db187f2fe660`, private baseline
  `093559f0802b5818ca418fb18b3b7642f26f54bb`.
- Validation: 272 games, 79 completed/locked forecasts, 192 coverage series,
  no unexplained coverage drops. Full local regression suite: 144 tests passed.
- User explicitly approved the scheduler handoff. `python -B -m runner.handoff`
  reconciles only the three permitted public reports under the refresh lock;
  all model and forecast hashes remain unchanged. It does not activate scheduling.
- Live authority is determined by private `config.json` gates and the installed
  workflow, not this checklist. Keep local writers paused only while completing
  the handoff; restore them if activation fails.

### Completed handoff

User-approved activation commit: private `83b4ed89d6e8be2fe45d3680165a484a241119b1`.
Live Actions run `38814252099` succeeded, persisted private generation 2 and
published public commit `c30fbf3f4c52548280ca2bdf7cf3fbf51bca4414`. Pages run
`38815013255` succeeded. The served 51-file snapshot was hash-verified against
that public commit. Tuesday 09:00 Eastern refreshes are now cloud-owned.

Local pregame automation is PAUSED. The Tuesday local automation is ACTIVE but
read-only: health, schedule coverage and prospective readiness checks only.
Do not run local refreshes against the old private runtime. A public git pull
does not synchronize ignored model/data state: before future training or recovery,
restore the latest verified private bundle into a separate workspace and reconcile
it under the single-writer protocol. Never overwrite newer cloud forecast cohorts.

Actions concurrency serializes runs. A unique content-addressed private release
is uploaded first; private main advances from the exact parent with force=false.
A racing writer causes conflict rather than state regression. Failed uploads or
CAS leave harmless unreferenced assets; do not delete releases referenced by state
history. Public publishing happens only after private persistence. A public push
failure leaves the private checkpoint intact for retry. Never force push publicly.

The public allowlist is captured from existing tracked data/audit/learning exports
and public_site/data, plus three explicit new JSON exports. No `git add -f`, model
publishing, workflow changes, or .gitignore edits. The deploy-key push triggers the
existing public Pages workflow. Parent owns that workflow's source-only validation
compatibility and Hosted Sites remote live_snapshot.json path/fallback handling.
New shadow inputs are permitted only under the pinned generation and exact
64-hex SHA256 JSON filenames, with provenance validated before publication.
No Sites API credentials are needed. Parent must verify the new snapshot is tracked
and served by Pages before switch-over.

## Quotas and latency

Private Actions consume the account's included minutes. Included usage is not a
guarantee of zero charges if the account already permits paid overages. No billing
settings are readjusted here; verify a zero-spend budget before scheduling. Avoid
Actions artifact/cache storage for the model bundle. Releases retain private state;
review storage growth and retention manually. The workflow timeout is 40 minutes.
GitHub cron can be delayed or dropped and is not a precise kickoff timer. Missed
windows remain missed, never fabricated or backfilled. The plan avoids 4,320
monthly unconditional checks, but measured runtime and quota review remain needed.
No-due/no-results-change runs never upload a new runtime bundle. Public publishing
uses publication_changes semantic comparison and excludes the duplicate snapshot
from change detection. The legacy shadow calibration learner is frozen in cloud
runs: grading/export refresh continues, but its training state is not refitted.

References: https://docs.github.com/en/billing/concepts/product-billing/github-actions
and https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
