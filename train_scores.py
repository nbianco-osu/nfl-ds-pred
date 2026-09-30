"""Fit independent no-market expected home/away score regressors."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from model_utils import build_model_pipeline, get_feature_columns


def score_metrics(actual, predicted):
    return {'home_mae': float(mean_absolute_error(actual[:, 0], predicted[:, 0])),
            'away_mae': float(mean_absolute_error(actual[:, 1], predicted[:, 1])),
            'margin_mae': float(mean_absolute_error(actual[:, 0] - actual[:, 1], predicted[:, 0] - predicted[:, 1])),
            'total_mae': float(mean_absolute_error(actual.sum(axis=1), predicted.sum(axis=1)))}


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / 'data/nfl_matchups_1999_2026_advanced.csv').sort_values(['season', 'week', 'game_id'])
    if data.game_id.duplicated().any():
        raise ValueError('Duplicate games')
    cols = get_feature_columns(data, exclude_market=True)
    assert not {'home_score', 'away_score', 'home_win', 'spread_line', 'total_line'}.intersection(cols)
    targets = ['home_score', 'away_score']
    train = data.loc[data.season.lt(2025)]
    val = data.loc[data.season.eq(2025)]
    history = data.loc[data.season.lt(2026)]
    test = data.loc[data.season.eq(2026)]
    selected, trials = [], []
    for cls in [RandomForestRegressor, ExtraTreesRegressor]:
        best, best_error = None, float('inf')
        for depth in [5, 10]:
            estimator = cls(n_estimators=200, max_depth=depth, min_samples_leaf=15, random_state=42, n_jobs=2)
            model = build_model_pipeline(train, cols)
            model.set_params(model=estimator)
            model.fit(train[cols], train[targets])
            error = float(mean_absolute_error(val[targets], model.predict(val[cols])))
            trials.append({'family': cls.__name__, 'depth': depth, 'validation_score_mae': error})
            if error < best_error:
                best, best_error = clone(model), error
        best.fit(history[cols], history[targets])
        selected.append(best)
    predicted = np.mean([model.predict(test[cols]) for model in selected], axis=0)
    metrics = {'evaluation': '2026 retrospective pre-refit; tuning on 2025 only', 'holdout_rows': len(test),
               'training_rows': len(data), 'training_through': str(data.gameday.max()),
               'trials': trials, **score_metrics(test[targets].to_numpy(), predicted),
               'note': 'Expected points, not exact score outcomes or ATS probabilities. Two equally weighted tree regressors; independent of the winner classifier. Historical dataset excludes ties.'}
    for model in selected:
        model.fit(data[cols], data[targets])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = root / 'models/score_models/score_ensemble.joblib'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        archive = root / 'models/archive' / stamp
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / path.name)
    joblib.dump({'models': selected, 'feature_cols': cols, 'model_version': f'scores_{stamp}', 'metrics': metrics}, path, compress=3)
    path.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
