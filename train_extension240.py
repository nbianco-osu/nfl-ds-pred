"""Extend a hash-matched 120-model generation without mutating serving state."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge, PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_io import write_csv, write_json
from retrain_updated import load_data, partitions, pipeline, probability_metrics
from train_research32 import calibrated
from train_scores import score_metrics


def recipes(views):
    for view, cols in views.items():
        for i in range(5):
            trees = dict(n_estimators=100, max_depth=[3, 5, 7, 9, None][i],
                         min_samples_leaf=[30, 24, 18, 12, 20][i], random_state=71, n_jobs=2)
            boost = dict(max_iter=100, max_leaf_nodes=[5, 9, 13, 17, 21][i],
                         l2_regularization=[1, 3, 10, 30, 100][i],
                         learning_rate=.04, early_stopping=False, random_state=71)
            network = dict(hidden_layer_sizes=[(6,), (12,), (24,), (12, 6), (24, 12)][i],
                           alpha=30, max_iter=400, early_stopping=False, random_state=71)
            winners = {
                'logistic': LogisticRegression(C=[.003, .01, .03, .3, 3][i], max_iter=1500),
                'extra': ExtraTreesClassifier(**trees), 'forest': RandomForestClassifier(**trees),
                'hist': HistGradientBoostingClassifier(**boost),
                'mlp': MLPClassifier(**network),
                'rbf': make_pipeline(Nystroem(gamma=[.0003, .001, .003, .01, .03][i],
                    n_components=64, random_state=71), LogisticRegression(C=.1, max_iter=1500)),
            }
            scores = {
                'ridge': Ridge(alpha=[3, 30, 300, 3000, 30000][i]),
                'extra': ExtraTreesRegressor(**trees), 'forest': RandomForestRegressor(**trees),
                'hist': MultiOutputRegressor(HistGradientBoostingRegressor(**boost)),
                'mlp': TransformedTargetRegressor(regressor=MLPRegressor(**network), transformer=StandardScaler()),
                'poisson': MultiOutputRegressor(PoissonRegressor(alpha=[.03, .3, 3, 30, 300][i], max_iter=1000)),
            }
            for task, models in [('winner', winners), ('score', scores)]:
                for family, model in models.items():
                    yield f'r59_{task}_{family}_{i}_{view}', task, model, cols


def run(root, source):
    if sklearn.__version__ != '1.8.0':
        raise ValueError('scikit-learn 1.8.0 required')
    if Path(source).name != source:
        raise ValueError('Generation name required')
    parent = root/'models/retrained'/source
    old = json.loads((parent/'report.json').read_text())
    data, hashes = load_data(root)
    if old['source_hashes'] != hashes or len(old['models']) != 120 or old['status'] != 'completed_shadow_only':
        raise ValueError('Require completed, source-matched 120-model generation')
    version = 'expanded240-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = root/'models/retrained'/version
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(old, version=version, parent_generation=source, status='in_progress',
        production_changed=False, created_at=datetime.now(timezone.utc).isoformat(),
        note='120 byte-identical retained configurations plus 120 newly fitted configurations; not 240 distinct algorithms. '
             'Historical 2026 evaluation is repeatedly inspected research, not an untouched or prospective test. '
             'All final artifacts are shadow-only. No automatic production promotion. '+old['note'])
    for item in old['models']:
        path = parent/(item['name']+'.joblib')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['artifact_sha256']:
            raise ValueError('Parent artifact hash mismatch')
        shutil.copy2(path, folder/path.name)
    frames = [pd.read_csv(parent/'holdout_predictions.csv')]
    decisive = data.loc[data.home_score.ne(data.away_score)].copy()
    decisive['home_win'] = decisive.home_score.gt(decisive.away_score).astype(int)
    for name, task, estimator, cols in recipes(report['features']):
        rows = decisive if task == 'winner' else data
        base, cal, val, test = partitions(rows)
        history = rows.loc[rows.season.lt(2026)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            if task == 'winner':
                validation = calibrated(estimator, base, cal, cols).predict_proba(val[cols])[:, 1]
                evaluated = calibrated(estimator, rows.loc[rows.season.le(2024)], val, cols)
                predicted = evaluated.predict_proba(test[cols])[:, 1]
                final = calibrated(estimator, history, test, cols)
                result = dict(validation_log_loss=probability_metrics(val.home_win, validation)['log_loss'],
                              holdout=probability_metrics(test.home_win, predicted))
            else:
                targets = ['home_score', 'away_score']
                earlier = rows.loc[rows.season.le(2024)]
                validation = np.maximum(0, pipeline(estimator).fit(earlier[cols], earlier[targets]).predict(val[cols]))
                predicted = np.maximum(0, pipeline(estimator).fit(history[cols], history[targets]).predict(test[cols]))
                final = pipeline(estimator).fit(rows[cols], rows[targets])
                result = dict(validation_team_mae=float(np.abs(validation-val[targets].to_numpy()).mean()),
                              holdout=score_metrics(test[targets].to_numpy(), predicted))
        if not np.isfinite(predicted).all():
            raise ValueError('Nonfinite predictions')
        result.update(name=name, task=task, selected_configuration=str(estimator),
                      convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught))
        path = folder/(name+'.joblib')
        joblib.dump(dict(model=final, feature_cols=cols, model_version=version+'-'+name,
            experimental=True, status='shadow_only', metrics=result, source_hashes=hashes,
            training_through=report['training_through']), path, compress=3)
        method = 'predict_proba' if task == 'winner' else 'predict'
        np.testing.assert_allclose(getattr(joblib.load(path)['model'], method)(test[cols].head()),
                                   getattr(final, method)(test[cols].head()))
        result['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['models'].append(result)
        out = test[['game_id', 'season', 'week', 'home_score', 'away_score']].copy()
        out['model'] = name
        if task == 'winner': out['home_probability'] = predicted
        else: out[['predicted_home_score', 'predicted_away_score']] = predicted
        frames.append(out)
        write_json(folder/'report.json', report)
        write_csv(folder/'holdout_predictions.csv', pd.concat(frames, ignore_index=True))
        print(f'{len(report["models"])}/240 {name}', flush=True)
    if len(report['models']) != 240 or len({m['name'] for m in report['models']}) != 240:
        raise ValueError('Incomplete or duplicate model generation')
    report['status'] = 'completed_shadow_only'
    for destination in [folder, root/'backtests/retraining'/version]:
        write_json(destination/'report.json', report)
        write_csv(destination/'holdout_predictions.csv', pd.concat(frames, ignore_index=True))
    print('COMPLETED '+version, flush=True)
    return version


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    with threadpool_limits(limits=2):
        run(Path(__file__).resolve().parent, parser.parse_args().source)
