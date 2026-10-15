# Post-R40 Retraining

Completed October 15, 2026, with scikit-learn 1.8.0.

Generation: `updated-20261015T185719479420Z`.

## Inputs and Outputs

- 2,454 available regular-season rows from 2017 through October 12, 2026.
- 79 completed 2026 games: 78 decisive winner labels and one tie.
- Existing core EPA, QB usage, rest, injury and roster features, plus historical
  raw EPA, opponent-adjusted net EPA, sack-rate differential and big-play-rate
  differential. A second feature view also includes lagged charting/snap inputs.
- Missing inputs stay missing in the dataset; preprocessing is fitted only on
  the corresponding training partition. No confirmed starter or weather data
  is fabricated. Revised historical snapshots are not original-time evidence.
- Historical research data omits some tied games. Score evaluation includes
  available ties, but is not a complete historical schedule cohort.

Sixty configurations were saved: 24 classifier families across two feature views,
eight noisy GPR kernel variants, and four home/away score regressors. These are
configurations, not 60 distinct algorithms. Existing estimator factories are
reused; this does not retrain every legacy artifact in place.

Artifacts: `models/retrained/updated-20261015T185719479420Z/` (local, Git-ignored).
Portable results: `backtests/retraining/updated-20261015T185719479420Z/`.
Each directory contains `report.json` and `holdout_predictions.csv`; the model
directory additionally contains 60 serialized estimators. Reports record input
and artifact hashes, selected settings, missingness, cutoffs and evaluation.

## Evaluation

Classifier settings: fit through 2023, calibrate on 2024, select on 2025;
evaluate 2026 using a base fit through 2024 and calibration on 2025. Final
artifacts fit through 2025 and calibrate on completed decisive 2026 games.
GPR uses its existing disjoint week-based calibration and at most 1,000 recent
base-training rows. Noise settings are selected on 2025; 2026 is evaluated
before final fitting. Score settings are selected on 2025, evaluated on 2026
using models fit through 2025, then refitted on all available rows.

The winner configuration selected by 2025 validation log loss was the calibrated
passive-aggressive classifier with charted inputs. On the 78 decisive 2026
holdout games it scored 60.26% accuracy, 0.67775 log loss, and 0.23932 Brier score.
The score configuration selected by 2025 team MAE was Ridge (alpha 1000).
On 79 games its mean team-score MAE was 7.9234 points, margin MAE 10.4714,
and total MAE 12.0296. These results do not establish improvement over the
production model. Multiple configurations were explored; no multiple-comparison
correction or prospective evaluation is claimed. Final refits are not scored
on the same labels they used for training/calibration.

## Status and Reproduction

Shadow-only: no serving artifact, saved pick, learning cohort, release approval,
or dashboard was changed. Production adoption requires a compatible inference
path and a separate review; the new feature contract is not a drop-in replacement.
No automatic weekly retraining was added.

```powershell
.\.venv\Scripts\python.exe retrain_updated.py
.\.venv\Scripts\python.exe -B -m unittest discover -p 'test*.py'
```

Every invocation creates a separate generation. Reuse the pinned environment
and retained input snapshots. All 60 artifacts passed serialization/reload
checks and SHA-256 verification. Holdout model/game keys are unique. All 88
Python tests passed, including four new temporal/imputation/probability tests.
