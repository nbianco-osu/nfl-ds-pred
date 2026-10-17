# R59-R60: candidate expansion and production ensemble candidate

## Delivered

- **R59:** 240 saved configurations, up from 120: 140 winner and 100 score models.
  The original 120 artifacts retain their exact hashes. The additions include
  regularized linear models, Extra Trees, Random Forest, histogram boosting,
  small neural networks, approximate RBF classification and Poisson regression.
  These are configurations, not 240 different algorithms.
- **R60:** a unified winner/score inference artifact with six diverse members per
  task; chronological out-of-fold weight fitting; nonnegative weights capped at
  35%; shrinkage toward equal weighting and a correlated-error penalty; separate
  sigmoid calibration; paired home/away residual uncertainty; strict input and
  artifact checks. A future-game-only inference CLI does not write production picks.
- Added `robust_features.py` to expand home/away dependencies before deriving
  the new ensemble's differential features. A regression test reproduces the
  legacy diff-only failure and verifies form/EPA differences are populated.
- Production models, cloud activation, existing shadow generations and archived
  picks were not changed. The 240-model library and ensemble are saved locally;
  the cloud's pinned 120-model shadow generation is still active. New artifacts
  have **not** been installed into the private runner or promoted to production.

## Reproducibility

Library: `expanded240-20261017T220233813195Z`.
Ensemble: `robust-r60-20261017T220803934180Z`.
Artifacts live under `models/retrained/<generation>/`; reports and predictions
also live under `backtests/retraining/<generation>/`. Model binaries remain local,
not in the public repository. Both use scikit-learn 1.8.0.

The two research inputs were checked against private cloud state
`d7c9b23a077273ebb5d7ea359a91f7ef53e9f96a`, generation 5, before training:

- `research_matchups.csv`: `19e4691c2c9fc0cbb82d7377d137d223534648672104bdd48ecf0456b48333ed`
- `historical_opponent_features.csv`: `43afde7696fb95d7e4786f887762ece98ea73449e0002f4546a24bc0e638fffe`

Their completed-game cutoff is **October 12, 2026**, not the current calendar
date. No newer game was fabricated or silently assumed present. Historical source
ties remain incompletely covered; the 2026 score sample includes its available tie.

```powershell
.\.venv\Scripts\python.exe -B train_extension240.py --source expanded120-20261015T191351669323Z
.\.venv\Scripts\python.exe -B train_robust_ensemble.py --generation expanded240-20261017T220233813195Z
.\.venv\Scripts\python.exe -B predict_robust.py --generation robust-r60-20261017T220803934180Z --features path-to-verified-features.csv
.\.venv\Scripts\python.exe -B -m unittest test_robust_ensemble
```

Inference requires `game_id`, timezone-aware `kickoff_at` and the 25 named
features in the ensemble report. Inputs with over 40% missing features, infinite
values, duplicate columns/IDs, passed kickoffs or an overlapping training cutoff
are rejected. A returned score is not an archived prospective observation. This
interface and the new live feature adapter must be connected to the existing
exact-input archive and approved fallback path before production release. The
feature builder requires the complete verified private runtime, including raw
PBP source blobs; a public git pull alone is insufficient. Missing values are imputed by fitted
training-only transformers; missing starters/weather are not invented.

The attempted smoke test against the current public shadow input archive was
correctly rejected: its diff-only adapter leaves 20 of the 25 core fields missing.
That is an existing shadow-input defect, not evidence that those raw NFL data are
all unavailable. The new dependency-expanding adapter fixes the construction
logic; it has not been installed in the cloud. Existing shadow results must retain
their original missing-input limitation. Start a separately versioned corrected
cohort rather than retroactively repairing its archived inputs or labels.

## Evaluation

Weights use annual expanding-window predictions for 2022-2024 (813 games per
task); each fold trains on preceding seasons only. Evaluation base models stop
at 2024, calibration uses 2025, and evaluation uses 2026 (78 decisive winner
games, 79 score games). Baselines receive the same independent calibration.
The best individual baseline is selected on OOF predictions, not on 2026.

| Measure | Ensemble | Equal-weight baseline |
| --- | ---: | ---: |
| Winner log loss (lower is better) | 0.67950 | 0.67992 |
| Winner accuracy | 58.97% | 57.69% |
| Mean per-team score absolute error | 7.898 points | 7.898 points |
| Total-score absolute error | 11.844 points | 11.856 points |

The paired week-bootstrap 95% interval for log-loss difference is
[-0.00240, 0.00216]; it includes zero. Per-team score-error difference versus equal
weighting is also inconclusive. Home and away nominal 80% marginal score ranges
each cover 82.28% of this small sample. This is not guaranteed joint coverage,
proof of production superiority, or evidence of betting profitability.

The final candidate refits base models through 2025 and calibrates on the available
2026 rows. Its own 2026 calibration labels are **not** presented as final-candidate
test performance. Evaluation and final candidate artifacts are saved separately.
Repeated research on 2026 means it is not a fresh untouched test. No convergence
warnings occurred in the 120 additions or ensemble fits.

## Next Recommendations

Verification: all 155 Python regression tests passed; all 240 candidate artifact
hashes and both ensemble artifact hashes verified; saved estimators reproduce
their in-memory predictions after reload. No production artifacts were modified.

| ID | Priority | Proposed work and acceptance check |
| --- | --- | --- |
| R61 | First | Install the new generation and corrected differential-feature adapter into the private runner using an append-only, compare-and-swap bundle migration; add robust-ensemble shadow inference without replacing old cohorts. Verify live input coverage, saved forecasts on identical games/horizons and latency before any promotion. |
| R62 | High | Refresh the research training table from authoritative cloud observations, with as-of source vintages and a missing-game reconciliation report. Preserve prior datasets rather than rewriting historical evidence. |
| R63 | High | Add a predeclared sequential review policy: fixed review dates or valid repeated-testing bounds, paired to the champion. A 50-game minimum alone is not statistical evidence. |
| R64 | Medium | Distill/prune the 240 configurations using training-only error correlation and measured inference latency. Target a small production roster with a defined latency budget; retain research artifacts for reproducibility. |
| R65 | Medium | Develop a scoring-event/possession model that represents touchdowns, field goals, safeties and overtime. Evaluate score plausibility, totals, margins and proper distribution scores on matched time folds. |
| R66 | Medium | Implement the production feature adapter and explicit fallback contract, then rehearse an atomic cloud canary/rollback. Verify that neither cutover nor rollback rewrites post-kickoff picks. |

R58 remains deferred. New configurations alone do not satisfy its prospective
evidence requirement. Recommended order: R61, R62, R63, then R64/R66; R65 is a
larger research project.

Method reference: [scikit-learn time-aware cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
