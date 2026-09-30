"""Fit independent no-market expected home/away score regressors."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from model_utils import build_model_pipeline, get_feature_columns
from score_markets import comparisons, number


def score_metrics(actual, predicted):
    return {'home_mae': float(mean_absolute_error(actual[:, 0], predicted[:, 0])),
            'away_mae': float(mean_absolute_error(actual[:, 1], predicted[:, 1])),
            'margin_mae': float(mean_absolute_error(actual[:, 0] - actual[:, 1], predicted[:, 0] - predicted[:, 1])),
            'total_mae': float(mean_absolute_error(actual.sum(axis=1), predicted.sum(axis=1)))}


def market_evaluation(games, predicted):
    """Retrospective source-line evaluation, never a live pregame archive."""
    rows = []
    for game, scores in zip(games.to_dict('records'), np.round(np.maximum(0, predicted), 1)):
        spread = number(game.get('spread_line'))
        result = comparisons(game, dict(predicted_home_score=scores[0], predicted_away_score=scores[1],
                                        market_home_spread=-spread if spread is not None else None,
                                        market_total=number(game.get('total_line'))))
        rows.append(result)
    summary = {}
    for field, label in [('spread_grade', 'spread'), ('total_grade', 'total')]:
        grades = pd.Series([row[field] for row in rows])
        correct, wrong = int(grades.eq('Correct').sum()), int(grades.eq('Wrong').sum())
        summary[label] = dict(correct=correct, wrong=wrong, graded=correct + wrong,
                              pushes=int(grades.eq('Push').sum()), no_edge=int(grades.eq('No edge').sum()),
                              missing=int(grades.eq('No archived pick').sum()),
                              accuracy=correct / (correct + wrong) if correct + wrong else None)
    return summary, rows


def candidates():
    for cls in [RandomForestRegressor, ExtraTreesRegressor]:
        yield cls.__name__, [(dict(max_depth=depth, max_features=0.5),
                              cls(n_estimators=200, max_depth=depth, max_features=0.5,
                                  min_samples_leaf=15, random_state=42, n_jobs=2)) for depth in [5, 10]]
    yield 'Ridge', [(dict(alpha=alpha), Ridge(alpha=alpha)) for alpha in [100, 1000]]
    yield 'HistGradientBoostingRegressor', [
        (dict(max_leaf_nodes=leaves, l2_regularization=10), MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=leaves, min_samples_leaf=30,
                                          learning_rate=0.05, l2_regularization=10, early_stopping=False,
                                          random_state=42))) for leaves in [7, 15]]


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
    if any(frame.empty for frame in [train, val, history, test]):
        raise ValueError('Training, validation and holdout seasons must all be present')
    selected, trials = [], []
    for family, configurations in candidates():
        best, best_error = None, float('inf')
        for params, estimator in configurations:
            print(f'Tuning {family}: {params}', flush=True)
            model = build_model_pipeline(train, cols)
            model.set_params(model=estimator)
            model.fit(train[cols], train[targets])
            validation = score_metrics(val[targets].to_numpy(), np.maximum(0, model.predict(val[cols])))
            error = (validation['margin_mae'] + validation['total_mae']) / 2
            trials.append({'family': family, 'params': params, 'validation_objective': error, **validation})
            if error < best_error:
                best, best_error = clone(model), error
        print(f'Evaluating {family} on 2026 before refitting', flush=True)
        best.fit(history[cols], history[targets])
        estimate = np.maximum(0, best.predict(test[cols]))
        markets, _ = market_evaluation(test, estimate)
        selected.append(dict(family=family, model=best, validation_objective=best_error,
                             prediction=estimate, holdout={**score_metrics(test[targets].to_numpy(), estimate),
                                                           'markets': markets}))
    # Ensemble membership depends only on 2025, never on the 2026 evaluation.
    selected.sort(key=lambda item: item['validation_objective'])
    members = selected[:2]
    predicted = np.maximum(0, np.mean([item['model'].predict(test[cols]) for item in members], axis=0))
    markets, holdout = market_evaluation(test, predicted)
    metrics = {'evaluation': '2026 retrospective pre-refit; tuning on 2025 only', 'holdout_rows': len(test),
               'training_rows': len(data), 'training_through': str(data.gameday.max()),
               'trials': trials, **score_metrics(test[targets].to_numpy(), predicted),
               'selection': 'Equal-weight top two families by mean of margin and total MAE on 2025',
               'selected_models': [item['family'] for item in members], 'markets': markets,
               'candidates': [{k: item[k] for k in ['family', 'validation_objective', 'holdout']} for item in selected],
               'note': 'Expected points, not exact score outcomes or ATS probabilities. Independent of winner classifier. Historical dataset excludes ties. Market evaluation uses retrospective source lines, not archived pregame picks; missing lines, pushes and no-edge picks excluded from accuracy.'}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = root / 'models/score_models/score_ensemble.joblib'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        archive = root / 'models/archive' / stamp
        archive.mkdir(parents=True, exist_ok=True)
        for old in path.parent.iterdir():
            if old.is_file():
                shutil.copy2(old, archive / old.name)
    for item in selected:
        print(f"Final fit: {item['family']}", flush=True)
        item['model'].fit(data[cols], data[targets])
        joblib.dump({'model': item['model'], 'feature_cols': cols, 'metrics': item['holdout'],
                     'model_version': f"scores_{stamp}_{item['family']}"},
                    path.parent / f"{item['family']}.joblib", compress=3)
    joblib.dump({'models': [item['model'] for item in members], 'feature_cols': cols,
                 'model_version': f'scores_{stamp}', 'metrics': metrics}, path, compress=3)
    report = pd.DataFrame(holdout).drop(columns=['line_basis'])
    report['evaluation'] = 'Retrospective holdout; NOT archived pregame predictions'
    report.to_csv(path.parent / 'holdout_predictions.csv', index=False)
    path.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
