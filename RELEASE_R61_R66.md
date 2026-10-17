# R61-R66: controlled cloud migration and evidence

## Scope

| ID | Implementation | Boundary |
| --- | --- | --- |
| R61 | Explicit private migration workflow, serialized with scheduled refreshes. Hash-verified 240-model package, corrected differential dependencies, robust ensemble shadow scores/probabilities/ranges, separate generation archives. | No production model promotion. Migration rejects any overwrite of an installed generation or previous private file outside its narrow selector/public-export scope. |
| R62 | Content-addressed 25-core-feature training vintages from authoritative schedules, current snapshots and week-cutoff PBP. Includes completed-game reconciliation and missing-field counts. | Observed-as-of revised sources, not historical original-publication proof. Charted features excluded; original tables preserved. No automatic fitting. |
| R63 | Predeclared 90-minute robust winner versus production review at completed weeks 10/14/18; minimum 50/100/200 matched decisive games and six weeks. Immutable look records and version checks. | Bonferroni-adjusted week-bootstrap intervals are approximate; not an anytime-valid guarantee. No repeated-peeking promotion or mixing production versions. |
| R64 | Training-only 2022-2024 OOF error correlation pruning, at most four members per task; measured inference latency and a proposed 250 ms single-game budget. | Prunes the six representative members per task, not all 240 configurations. Recommendations do not silently refit weights or replace the active artifact. Ten latency samples are a smoke test, not an SLA. |
| R65 | Research shared-pace possession simulator with integer field goals, touchdowns, conversions and defensive scoring. Earlier-week drive calibration, saved parameters, matched CRPS comparison against Poisson. | Two paired overtime opportunities are a bounded approximation, not a complete football clock/field-position/rules engine. No production promotion. |
| R66 | Explicit approval-bound inference hook, champion fallback on rejected candidate inference, disposable selector switch/rollback rehearsal and matched champion probability-shift diagnostics. | No live production selector change. Existing production picks and versions remain authoritative; human review/prospective evidence still required. |

## Cloud Migration

The private `migrate.yml` workflow shares `nfl-private-state` concurrency with the
existing scheduled runner. It restores the latest runtime **inside** that queue,
verifies every old/new file, installs only new generation directories, records
actual-time future forecasts, validates production preservation and coverage,
uploads a durable private bundle, then advances private state with compare-and-swap.
It publishes only approved data/audit/research exports afterward. A failed
pre-persistence migration leaves the previous selector and state untouched.

The payload is content-addressed and lives in the existing private repository.
No private model binary or credential is added to the public source repository.
The local public checkout is not a complete private-runtime restore. Do not use
its older `active_shadow.json` or ignored raw blobs as the cloud's authoritative state.

```powershell
.\.venv\Scripts\python.exe -B -m runner.migrate_candidates prepare --generation expanded240-20261017T220233813195Z --ensemble robust-r60-20261017T220803934180Z
.\.venv\Scripts\python.exe -B -m unittest test_iteration_r61 test_robust_ensemble
```

Preparation creates a new local staging directory and intentionally refuses to
overwrite one. Dispatch the private manual workflow with its returned manifest
asset ID/hash and the reviewed public source commit. It is not a scheduled retraining job.

## Evidence Outputs

- `public_site/data/retraining.json`: full 240-model candidate inventory.
- `public_site/data/shadow_next.json`: actual new-cohort forecasts and grades.
- `public_site/data/iteration_r61.json`: training vintage, fixed-review readiness,
  pruning, possession research and canary/rollback diagnostics.
- Private `data/training_vintages/<hash>/`: immutable core table and manifest.
- Private `audit/fixed_reviews/<generation>/`: immutable completed review looks.
- Private `candidate-migration.json`: installation/preservation receipt.

Normal cloud refreshes update the training vintage, prospective review and canary
coverage without refitting. Possession/pruning research runs only during explicit
migration; later source changes mark its results stale rather than silently rerunning it.
Older shadow inputs with missing differential fields remain unchanged and must
not be treated as corrected prospective evidence. R58 remains deferred.

## Verified Run: October 17, 2026

- Private migration preserved all previous artifacts and 80 completed production
  forecasts. Initial publication stopped at an overly narrow research-export
  allowlist; the repaired normal cloud refresh succeeded (run 39076598607).
- Public validation and Pages deployment succeeded for data commit `89f028e`.
  All 164 Python regression tests passed; dashboard validation reports 272 games,
  including 80 completed games.
- The new vintage reconciles the October 15 SEA-DEN game and has no missing
  completed-game rows. Five opponent metrics remain missing for the 16 first-week
  games. Existing fallback-filled features are not evidence of source availability.
- Canary core inputs were complete for 16 upcoming games. Finite-output and
  disposable rollback checks passed. Single-game p95 was 287 ms across ten samples,
  exceeding the proposed 250 ms budget; maximum matched champion probability
  difference was 17.6 percentage points. No activation occurred.
- Possession research covered 48 matched games: total CRPS 7.687 versus 8.255 for
  independent Poisson; margin CRPS 6.682 versus 6.711 (lower is better). These are
  retrospective, revised-source results without a prospective superiority claim.
  The subsequent normal refresh produced a different vintage, so the saved research
  is explicitly marked stale relative to it rather than silently refitted.
- The prospective fixed-review cohort has zero completed matched games and is not
  ready for promotion review. Pruning recommends two winner members and one score
  member out of the six representative members per task; weights remain unchanged.
