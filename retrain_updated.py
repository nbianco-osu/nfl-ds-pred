"""Train isolated post-R40 candidates; never overwrite serving models or picks."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_io import write_csv, write_json
from gpr_models import FEATURES, KERNELS, CalibratedGPRClassifier
from historical_metrics import VIEWS
from train_expanded import families as expanded_families
from train_research32 import families as research_families, calibrated
from train_scores import candidates as score_candidates, score_metrics


def load_data(root):
    source = root / 'data/research_matchups.csv'
    ratings = root / 'data/historical_opponent_features.csv'
    data = pd.read_csv(source)
    extra = pd.read_csv(ratings)
    cols = VIEWS['all']
    if data.game_id.duplicated().any() or extra.game_id.duplicated().any():
        raise ValueError('Duplicate source game IDs')
    data = data.loc[data.season.ge(2017)].merge(
        extra[['game_id', 'feature_cutoff'] + cols], on='game_id', how='left',
        validate='one_to_one', indicator=True)
    if not data['_merge'].eq('both').all():
        raise ValueError('Missing historical ratings rows')
    data['gameday'] = pd.to_datetime(data.gameday)
    data['feature_cutoff'] = pd.to_datetime(data.feature_cutoff)
    if data.feature_cutoff.isna().any() or (data.feature_cutoff > data.gameday).any():
        raise ValueError('Invalid pregame feature cutoffs')
    data = data.dropna(subset=['home_score', 'away_score']).sort_values(['gameday', 'game_id'])
    return data, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, ratings]}


def partitions(data):
    base = data.loc[data.season.le(2023)]
    calibration = data.loc[data.season.eq(2024)]
    validation = data.loc[data.season.eq(2025)]
    test = data.loc[data.season.eq(2026)]
    frames = [base, calibration, validation, test]
    if any(f.empty for f in frames):
        raise ValueError('All chronological partitions are required')
    if any(a.gameday.max() >= b.gameday.min() for a, b in zip(frames, frames[1:])):
        raise ValueError('Chronological partition overlap')
    return frames


def probability_metrics(y, p):
    p = np.asarray(p)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    return dict(log_loss=float(log_loss(y, p, labels=[0, 1])),
                accuracy=float(accuracy_score(y, p >= .5)), brier=float(np.mean((p-y)**2)))


def pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         StandardScaler(), clone(estimator))


def run(root):
    if sklearn.__version__ != '1.8.0':
        raise ValueError('Use scikit-learn 1.8.0 for artifact compatibility')
    data, hashes = load_data(root)
    decisive = data.loc[data.home_score.ne(data.away_score)].copy()
    decisive['home_win'] = decisive.home_score.gt(decisive.away_score).astype(int)
    base, cal, val, test = partitions(decisive)
    history = decisive.loc[decisive.season.lt(2026)]
    version = 'updated-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = root / 'models/retrained' / version
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(version=version, status='in_progress', production_changed=False,
                  sklearn_version=sklearn.__version__, source_hashes=hashes,
                  training_through=str(data.gameday.max().date()), games=len(data),
                  current_season_games=len(data.loc[data.season.eq(2026)]),
                  winner_holdout_games=len(test), historical_ties_in_source='Historical research source omits some ties; score evaluation covers available rows only.',
                  note='Shadow-only. Revised-source retrospective evaluation, not prospective evidence. Settings selected on 2025, evaluated on 2026 before final refit. Classifiers exclude ties. No weather or confirmed-starter values invented. Prior usage is not a confirmed starter. GPR uses a bounded recent training window and disjoint week-based calibration. Final fitted artifacts are not evaluated on their own training labels.',
                  features={}, models=[])
    outputs = []

    def save(name, model, cols, result, rows, predicted, kind):
        artifact = dict(model=model, feature_cols=cols, model_version=version+'-'+name,
                        experimental=True, status='shadow_only', metrics=result,
                        source_hashes=hashes, training_through=report['training_through'])
        path = folder / (name+'.joblib')
        joblib.dump(artifact, path, compress=3)
        loaded = joblib.load(path)['model']
        method = 'predict_proba' if kind == 'winner' else 'predict'
        np.testing.assert_allclose(getattr(loaded, method)(rows[cols].head()),
                                   getattr(model, method)(rows[cols].head()), rtol=1e-12)
        out = rows[['game_id', 'season', 'week', 'home_score', 'away_score']].copy()
        out['model'] = name
        if kind == 'winner':
            out['home_probability'] = predicted
        else:
            out[['predicted_home_score', 'predicted_away_score']] = predicted
        outputs.append(out)
        result.update(name=name, task=kind, artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['models'].append(result)
        write_json(folder/'report.json', report)
        write_csv(folder/'holdout_predictions.csv', pd.concat(outputs, ignore_index=True))
        print(f'{len(report["models"]):02d}/60 {name}: {json.dumps(result.get("holdout", {}))}', flush=True)

    core = list(FEATURES) + VIEWS['all']
    views = {'core_updated': core,
             'charted_updated': core + [c for c in data if c.startswith('diff_research_')]}
    report['features'] = views
    for view, cols in views.items():
        for family, trials in {**expanded_families(), **research_families()}.items():
            scores = []
            for trial in trials:
                model = calibrated(trial, base, cal, cols)
                scores.append(probability_metrics(val.home_win, model.predict_proba(val[cols])[:, 1])['log_loss'])
            selected = trials[int(np.argmin(scores))]
            evaluation = calibrated(selected, decisive.loc[decisive.season.le(2024)], val, cols)
            p = evaluation.predict_proba(test[cols])[:, 1]
            result = dict(validation_log_loss=min(scores), trials=scores, selected_configuration=str(selected),
                          holdout=probability_metrics(test.home_win, p),
                          final_fit_through=str(history.gameday.max().date()),
                          final_calibration_through=str(test.gameday.max().date()))
            final = calibrated(selected, history, test, cols)
            save(f'{family}_{view}', final, cols, result, test, p, 'winner')

    cols = ['season', 'week'] + core
    for kernel in KERNELS:
        trials = [CalibratedGPRClassifier(kernel_name=kernel, noise_level=noise,
                                         extra_features=tuple(VIEWS['all'])) for noise in [.1, .3]]
        scores = []
        earlier = decisive.loc[decisive.season.le(2024)]
        for model in trials:
            model.fit(earlier[cols], earlier.home_win)
            scores.append(probability_metrics(val.home_win, model.predict_proba(val[cols])[:, 1])['log_loss'])
        selected = clone(trials[int(np.argmin(scores))])
        evaluation = clone(selected).fit(history[cols], history.home_win)
        p = evaluation.predict_proba(test[cols])[:, 1]
        result = dict(validation_log_loss=min(scores), trials=scores,
                      selected_configuration=str(selected), holdout=probability_metrics(test.home_win, p))
        final = selected.fit(decisive[cols], decisive.home_win)
        result.update(final_base_rows=final.training_rows_, final_calibration_rows=final.calibration_rows_,
                      final_base_end=final.training_end_, final_calibration_start=final.calibration_start_)
        save('gpr_'+kernel, final, cols, result, test, p, 'winner')

    _, _, val, test = partitions(data)
    base = data.loc[data.season.le(2024)]
    history = data.loc[data.season.lt(2026)]
    targets = ['home_score', 'away_score']
    for name, trials in score_candidates():
        scores = []
        for _, estimator in trials:
            model = pipeline(estimator).fit(base[core], base[targets])
            scores.append(float(mean_absolute_error(val[targets], np.maximum(0, model.predict(val[core])))))
        selected = trials[int(np.argmin(scores))][1]
        evaluation = pipeline(selected).fit(history[core], history[targets])
        predicted = np.maximum(0, evaluation.predict(test[core]))
        if not np.isfinite(predicted).all():
            raise ValueError('Invalid score predictions')
        result = dict(validation_team_mae=min(scores), trials=scores, selected_configuration=str(selected),
                      holdout=score_metrics(test[targets].to_numpy(), predicted), final_training_rows=len(data))
        final = pipeline(selected).fit(data[core], data[targets])
        save('score_'+name, final, core, result, test, predicted, 'score')
    if len(report['models']) != 60:
        raise ValueError('Incomplete candidate generation')
    report['status'] = 'completed_shadow_only'
    report['missing_counts'] = {c:int(data[c].isna().sum()) for c in views['charted_updated']}
    write_json(folder/'report.json', report)
    destination = root/'backtests/retraining'/version
    write_json(destination/'report.json', report)
    write_csv(destination/'holdout_predictions.csv', pd.concat(outputs, ignore_index=True))
    print(f'Completed 60 models: {folder}', flush=True)
    return report


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run(Path(__file__).resolve().parent)
