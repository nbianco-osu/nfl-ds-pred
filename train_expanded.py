"""Sixteen research configurations: eight estimator families and two feature views."""
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from gpr_models import FEATURES
from model_utils import build_model_pipeline, get_feature_columns
from train_gpr import metrics


def families():
    return {
        'logistic': [LogisticRegression(C=c, max_iter=3000) for c in [.03, .1, .3]],
        'random_forest': [RandomForestClassifier(n_estimators=300, max_depth=d, min_samples_leaf=12, n_jobs=2, random_state=42) for d in [4, 8, 12]],
        'extra_trees': [ExtraTreesClassifier(n_estimators=300, max_depth=d, min_samples_leaf=12, n_jobs=2, random_state=42) for d in [4, 8, 12]],
        'hist_boost': [HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=n, learning_rate=.03, l2_regularization=3, early_stopping=False, random_state=42) for n in [7, 15, 31]],
        'gradient_boost': [GradientBoostingClassifier(n_estimators=150, max_depth=d, learning_rate=.03, min_samples_leaf=20, random_state=42) for d in [1, 2, 3]],
        'adaboost': [AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=d, min_samples_leaf=20), n_estimators=100, learning_rate=.03, random_state=42) for d in [1, 2, 3]],
        'sgd_logistic': [SGDClassifier(loss='log_loss', alpha=a, max_iter=3000, tol=1e-5, average=True, random_state=42) for a in [.01, .03, .1]],
        'gaussian_nb': [GaussianNB(var_smoothing=s) for s in [.001, .01, .1]],
    }


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / 'data/nfl_matchups_1999_2026_expanded.csv').sort_values(['season', 'week', 'game_id'])
    data = data.loc[data.season.ge(2017) & data.home_score.ne(data.away_score)].copy()
    if data.game_id.duplicated().any():
        raise ValueError('Duplicate games')
    train, validation, history, test = (data.loc[mask] for mask in [data.season.lt(2025), data.season.eq(2025), data.season.lt(2026), data.season.eq(2026)])
    compact = list(FEATURES) + [c for c in data if c.startswith('diff_team_ext_')]
    views = {'contrasts': compact, 'full_context': get_feature_columns(data, exclude_market=True)}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = {'created_at': stamp, 'models': [], 'note': 'Eight estimator families, two feature views; 16 configurations, not 16 distinct algorithms. Three validation trials each. 2017+ history; 2025 tuning; 2026 retrospective pre-refit evaluation. No automatic promotion.'}
    predictions = test[['game_id', 'home_win']].copy()
    for view, cols in views.items():
        for family, candidates in families().items():
            name = f'{family}_{view}'
            trials = []
            best, best_loss = None, float('inf')
            for estimator in candidates:
                model = build_model_pipeline(train, cols)
                model.set_params(model=estimator)
                model.fit(train[cols], train.home_win)
                trial = metrics(validation.home_win, model.predict_proba(validation[cols])[:, 1])
                trials.append({'configuration': str(estimator), **trial})
                if trial['log_loss'] < best_loss:
                    best, best_loss = clone(model), trial['log_loss']
            best.fit(history[cols], history.home_win)
            p = best.predict_proba(test[cols])[:, 1]
            predictions[name] = p
            result = {'name': name, 'family': family, 'view': view, 'feature_count': len(cols),
                      'holdout_season': 2026, 'test_rows': len(test), 'validation_season': 2025,
                      'final_training_rows': len(data), 'final_training_through': str(data.gameday.max()),
                      'selected_configuration': str(best.named_steps['model']), 'trials': trials,
                      **metrics(test.home_win, p)}
            best.fit(data[cols], data.home_win)
            path = root / f'models/home_win_expanded_{name}.joblib'
            if path.exists():
                archive = root / 'models/archive' / stamp
                archive.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, archive / path.name)
            artifact = dict(model=best, feature_cols=get_feature_columns(data, exclude_market=True), selected_feature_cols=cols, metrics=result, target_col='home_win',
                            model_version=f'expanded_{name}_{stamp}', trained_at=stamp,
                            training_input='data/nfl_matchups_1999_2026_expanded.csv', experimental=True,
                            requires_expanded_features=True, training_through_season=2026)
            joblib.dump(artifact, path, compress=3)
            loaded = joblib.load(path)
            np.testing.assert_allclose(loaded['model'].predict_proba(test[cols]), best.predict_proba(test[cols]))
            report['models'].append(result)
            (root / 'models/expanded_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'{name}: pre-refit log loss {result["log_loss"]:.4f}', flush=True)
    predictions.to_csv(root / 'models/expanded_holdout_predictions.csv', index=False)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
