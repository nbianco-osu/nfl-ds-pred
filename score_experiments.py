"""R19 score-distribution research; no serving models or archived picks modified."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, PoissonRegressor, TweedieRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_io import write_json, write_csv
from rolling_backtest import FEATURES, prepare
from score_distribution import draws as residual_draws, summarize

KINDS = ['ridge_residual', 'poisson', 'tweedie', 'scoring_events']
LABELS = ['Ridge paired residuals', 'Poisson scores', 'Tweedie compound scores', 'Scoring-event simulation']


def estimator(kind):
    model = Ridge(alpha=10) if kind == 'ridge_residual' else MultiOutputRegressor(
        TweedieRegressor(power=1.5, alpha=1, max_iter=1000) if kind == 'tweedie'
        else PoissonRegressor(alpha=1, max_iter=1000))
    return make_pipeline(StandardScaler(), model)


def fit(kind, base, calibration):
    if base.gameday.max() >= calibration.gameday.min():
        raise ValueError('Score calibration must follow base training')
    model = estimator(kind).fit(base[FEATURES], base[['home_score', 'away_score']])
    mean = np.maximum(.01, model.predict(calibration[FEATURES]))
    residual = calibration[['home_score', 'away_score']].to_numpy() - mean
    dispersion = np.maximum(.01, np.mean(residual**2 / mean**1.5, axis=0))
    return {'model': model, 'residuals': residual, 'dispersion': dispersion, 'kind': kind}


def sample(artifact, mean, trials=10000, seed=42):
    mean = np.maximum(.01, np.asarray(mean, dtype=float))
    if mean.shape != (2,) or not np.isfinite(mean).all():
        raise ValueError('Two finite score means required')
    rng = np.random.default_rng(seed)
    kind = artifact['kind']
    if kind == 'ridge_residual':
        return residual_draws(mean, artifact['residuals'], seed, trials)
    if kind == 'poisson':
        return rng.poisson(mean, size=(trials, 2))
    if kind == 'tweedie':
        # Compound Poisson-Gamma at fixed power 1.5, with pre-test dispersion.
        phi = np.asarray(artifact['dispersion'])
        counts = rng.poisson(2 * np.sqrt(mean) / phi, size=(trials, 2))
        return np.rint(rng.gamma(counts, phi * np.sqrt(mean) / 2))
    if kind != 'scoring_events':
        raise ValueError('Unknown experiment')
    # Hypothesis, not fitted event frequencies: safety, FG, TD, TD+1, TD+2.
    marks = np.array([2, 3, 6, 7, 8])
    weights = np.array([.01, .34, .04, .59, .02])
    pace = rng.gamma(20, 1/20, size=(trials, 1, 1))
    rates = mean[None, :, None] * weights[None, None, :] / np.dot(marks, weights)
    return (rng.poisson(pace * rates) * marks).sum(axis=2)


def crps(samples, actual):
    values = np.sort(np.asarray(samples, dtype=float))
    n = len(values)
    return float(np.mean(np.abs(values-actual)) - np.sum((2*np.arange(1,n+1)-n-1)*values)/(n*n))


def evaluate(samples, home, away, spread=None, total=None):
    result = {}
    actual = {'home': home, 'away': away, 'margin': home-away, 'total': home+away}
    values = {'home': samples[:,0], 'away': samples[:,1],
              'margin': samples[:,0]-samples[:,1], 'total': samples.sum(axis=1)}
    for key, value in values.items():
        low, high = np.quantile(value, [.1, .9])
        result[key+'_mae'] = float(abs(value.mean()-actual[key]))
        result[key+'_crps'] = crps(value, actual[key])
        result[key+'_coverage80'] = float(low <= actual[key] <= high)
        result[key+'_width80'] = float(high-low)
        result[key+'_interval_score80'] = float(high-low + 10*max(low-actual[key],0) + 10*max(actual[key]-high,0))
    for key, line in [('margin', spread), ('total', total)]:
        if line is None or not np.isfinite(line):
            continue
        v = values[key]
        # Jeffreys smoothing avoids treating a finite Monte Carlo zero as impossible.
        p = (np.array([(v<line).sum(), (v==line).sum(), (v>line).sum()])+.5)/(len(v)+1.5)
        label = 0 if actual[key]<line else 1 if actual[key]==line else 2
        result[key+'_brier3'] = float(np.sum((p-np.eye(3)[label])**2))
        result[key+'_log_loss3'] = float(-np.log(p[label]))
    return result


def run(root):
    paths = [root/'backtests/source_schedule.csv', root/'data/nfl_2026_completed.csv']
    raw = pd.concat([pd.read_csv(p) for p in paths]).drop_duplicates('game_id', keep='last')
    data = prepare(raw)
    results, folds = [], []
    for year in [2024, 2025, 2026]:
        base = data.loc[data.season.between(2017, year-2)]
        calibration = data.loc[data.season.eq(year-1)]
        test = data.loc[data.season.eq(year)]
        if calibration.gameday.max() >= test.gameday.min():
            raise ValueError('Outer score fold overlap')
        folds.append({'season': year, 'fit_through':str(base.gameday.max().date()),
                      'calibration_through':str(calibration.gameday.max().date()),
                      'test_start':str(test.gameday.min().date()), 'games':len(test)})
        for kind in KINDS:
            artifact = fit(kind, base, calibration)
            means = artifact['model'].predict(test[FEATURES])
            for i, row in enumerate(test.itertuples()):
                scores = evaluate(sample(artifact, means[i]), row.home_score, row.away_score,
                                  getattr(row, 'spread_line', None), getattr(row, 'total_line', None))
                results.append({'model':kind, 'game_id':row.game_id, 'season':year, 'week':int(row.week), **scores})
            print(f'{year} {kind}: {len(test)} games', flush=True)
    frame = pd.DataFrame(results)
    report = {'status':'research_only', 'created_at':datetime.now(timezone.utc).isoformat(), 'folds':folds,
              'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'note':'Annual chronological base fit and previous-season residual/dispersion calibration; fixed settings, no outer-label tuning. 10,000 draws, seed 42. Same source games, including available ties, for every experiment. Revised historical lines are not proven pregame snapshots. CRPS and interval score reward both sharpness and coverage; lower is better. Poisson assumes independent point counts. Tweedie samples independent rounded compound Poisson-Gamma scores. Scoring events use assumed 2/3/6/7/8-point frequencies and shared Gamma pace, not learned drive transitions or an overtime rules engine. These are hypotheses, not calibrated guarantees or serving models.',
              'models':[]}
    for kind, label in zip(KINDS,LABELS):
        subset = frame.loc[frame.model.eq(kind)]
        numeric = [c for c in subset if c not in ['model','game_id','season','week']]
        def summary(rows):
            return {'games':len(rows), 'metrics':{c:float(rows[c].mean()) if rows[c].notna().any() else None for c in numeric},
                    'line_counts':{key:int(rows[key+'_brier3'].notna().sum()) for key in ['margin','total']}}
        report['models'].append({'id':kind, 'name':label, 'overall':summary(subset),
            'seasons':{str(year):summary(g) for year,g in subset.groupby('season')}})
    write_csv(root/'backtests/score_experiments.csv', frame)
    write_json(root/'public_site/data/score_experiments.json', report)
    # Save a separate research generation; never replace the serving score artifacts.
    digest = hashlib.sha256(json.dumps(report['source_hashes'],sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()[:12]
    folder = root/'models/research'/('score-r19-'+digest)
    folder.mkdir(parents=True,exist_ok=True)
    base = data.loc[data.season.between(2017,2024)]
    calibration = data.loc[data.season.ge(2025)]
    for kind in KINDS:
        artifact = fit(kind, base, calibration)
        artifact.update(model_version=folder.name+'-'+kind, status='shadow_only',
                        training_through=str(base.gameday.max().date()), calibration_through=str(calibration.gameday.max().date()))
        joblib.dump(artifact,folder/(kind+'.joblib'))
    return report


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        report = run(Path(__file__).resolve().parent)
    print('R19 complete:', [(f['season'], f['games']) for f in report['folds']])
