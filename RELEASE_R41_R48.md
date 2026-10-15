# R41-R48 and the 120-Configuration Generation

Implemented October 15, 2026. Production serving models and existing picks are
unchanged. Research is not a betting recommendation or a demonstrated improvement.

## Models

Pinned generation: `expanded120-20261015T191351669323Z`.
The previous 60 configurations are retained byte-for-byte on matching source
hashes. Added 24 winner and 24 score neural-network configurations plus six Ridge
and six Poisson score configurations: 120 total, comprising 80 winner and 40
score configurations. This is not 120 different algorithms. Two supplementary
temporal ensembles are separate from that count.

Small networks have 8, 16 or 32 units in one layer, or 8+4, 16+8 or 32+16 units
in two layers. Two regularization levels and two feature views are compared.
Score-network targets are standardized using training-only transformers. Random
seed 42, no random early-stopping split; convergence warnings are recorded.
No convergence warnings were observed in the 60 added configurations.

Training inputs remain the available 2,454 regular-season games from 2017 through
October 12, 2026. Historical missing ties and revised-source limitations remain.
2026 holdout results cover 78 decisive games or 79 score outcomes. The final
winner artifacts calibrate on completed 2026 games; final score models refit on
all available rows. Those final artifacts are not evaluated on their own labels.

## Upgrade Status

| ID | Delivered | Important Boundary |
| --- | --- | --- |
| R41 | Matched candidate/production-estimator recipe comparisons, week-bootstrap intervals, and prospective pairing against observed production picks | Recipe refits use identical features and cutoffs, not the actual historical live artifact. Both retrospective comparisons are inconclusive. |
| R42 | All 120 candidates plus two ensembles forecast the next eight days | Shadow only; complete input snapshots and artifact hashes are retained. No historical pregame backfill. |
| R43 | Immutable generation/model/game/horizon ledger and scorecard | Initial/24h/90m groups stay separate. Ties excluded for winner metrics, pushes/missing lines/no-edge excluded from correctness. Zero prospective completed games at release. |
| R44 | Frozen convex ensembles, weights learned from out-of-sample 2025 weeks 1-9 | Winner weight is about 46% logistic/54% small neural net; RF gets zero. Score blend selects 100% Ridge, so it is not claimed as a diversification gain. |
| R45 | Home/away/margin/total 80% residual ranges and coverage evaluation | Late-2025 residual calibration is separate from early-2025 weight fitting. Not joint coverage or a guarantee under season drift; reused widths for final refits require prospective validation. |
| R46 | Game-level observation ages, missingness and stale-input warnings | Retrieval is not publication. Charted metrics remain frozen. Confirmed starters/weather are context only and not invented or inserted into models. |
| R47 | Active 10-minute due-window heartbeat with shared process locking | Refresh only in uncaptured windows. Host/scheduler availability and refresh duration can miss windows. Never backfill. Tuesday cadence retained. |
| R48 | Read-only release-review screen, checklist export, rollback inventory and serialized local activation | New candidate feature contracts are incompatible with serving. Integration and explicit human approval still required. No public approval endpoint and no automatic promotion. |

R41 core-view log-loss difference is -0.0119 with 95% interval [-0.0364, 0.0174];
charted-view difference is +0.0050, interval [-0.0315, 0.0351]. Both include zero.
Winner ensemble holdout log loss is 0.6729 versus 0.6817 for the individual
selected on early 2025; this is descriptive, not a significance claim.
Nominal 80% score interval coverage on 79 games: home 82.3%, away 84.8%,
margin 75.9%, total 84.8%. Model/feature selection was designed after inspecting
historical seasons; no multiple-comparison adjustment or prospective skill is claimed.

## Commands and Operations

```powershell
.\.venv\Scripts\python.exe train_extension120.py --source updated-20261015T185719479420Z
.\.venv\Scripts\python.exe next_evidence.py --version GENERATION
.\.venv\Scripts\python.exe export_retraining.py --version GENERATION
.\.venv\Scripts\python.exe shadow_next.py --season 2026
.\.venv\Scripts\python.exe pregame_refresh.py
.\.venv\Scripts\python.exe pregame_refresh.py --run
```

`models/retrained/active_shadow.json` pins the generation for future refreshes.
Local model bytes remain ignored by Git; retain them on the scheduler host.
Training reports and holdout CSVs are tracked in `backtests/retraining/`.
`audit/shadow_next/<generation>/` retains immutable actual forecasts and exact
numeric input archives. The first saved observation in each window remains
unchanged on repeated runs. Updating the pinned generation creates a separate
cohort and never changes an older one. Source fallback can be stale: see the
freshness report instead of assuming all covariates were refreshed this turn.

The pregame heartbeat runs a lightweight due check every ten minutes. Full
staged refresh runs only when a window is due; the process lock prevents overlap
with Tuesday refresh or manual model activation. The lock is excluded from
staging and released after process exit. No automatic retraining is scheduled.
Publication still requires validation and a substantive dashboard change.

The public release screen is deliberately read-only. The existing local
`release_control.py` approval and activation commands remain the only deployment
control for model bundles. The 50-game review threshold is an operational
minimum, not a statistical acceptance test. Private rollback blobs must be
backed up separately; manifests alone cannot restore models on another host.
