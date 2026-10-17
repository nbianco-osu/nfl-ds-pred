"""R60: chronological cross-fitting, constrained pooling and independent calibration."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from audit_io import write_csv, write_json
from historical_metrics import interval, loss
from retrain_updated import load_data, pipeline, probability_metrics
from robust_ensemble import RobustMatchupEnsemble, fit_weights, logit
from train_extension240 import recipes
from train_scores import score_metrics


def separated(before, after):
    if before.empty or after.empty or before.gameday.max() >= after.gameday.min():
        raise ValueError('Training and evaluation dates overlap or are empty')
    if set(before.game_id) & set(after.game_id):
        raise ValueError('Training and evaluation games overlap')


def run(root, generation):
    if sklearn.__version__ != '1.8.0': raise ValueError('scikit-learn 1.8.0 required')
    if Path(generation).name != generation: raise ValueError('Generation name required')
    source = root/'models/retrained'/generation
    parent = json.loads((source/'report.json').read_text())
    data, hashes = load_data(root)
    if parent['status'] != 'completed_shadow_only' or len(parent['models']) != 240 or hashes != parent['source_hashes']:
        raise ValueError('Complete, hash-matched 240-model generation required')
    cols = parent['features']['core_updated']
    declared = list(recipes({'core_updated': cols}))
    chosen = [r for r in declared if '_1_core_updated' in r[0]]
    winner_recipes = [(name, model) for name, task, model, _ in chosen if task == 'winner']
    score_recipes = [(name, model) for name, task, model, _ in chosen if task == 'score']
    decisive = data.loc[data.home_score.ne(data.away_score)].copy()
    decisive['home_win'] = decisive.home_score.gt(decisive.away_score).astype(int)
    targets = ['home_score', 'away_score']
    version = 'robust-r60-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = root/'models/retrained'/version
    folder.mkdir(parents=True, exist_ok=False)
    folds = []; winner_oof = []; score_oof = []; winner_y = []; score_y = []; oof_rows = []

    def fit_members(rows, task):
        configs = winner_recipes if task == 'winner' else score_recipes
        y = rows.home_win if task == 'winner' else rows[targets]
        return [pipeline(model).fit(rows[cols], y) for _, model in configs]

    def values(models, rows, task):
        if task == 'winner': return np.column_stack([m.predict_proba(rows[cols])[:, 1] for m in models])
        return np.stack([np.maximum(0, m.predict(rows[cols])) for m in models], axis=-1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        for year in [2022, 2023, 2024]:
            for task, rows in [('winner', decisive), ('score', data)]:
                train = rows.loc[rows.season.lt(year)]; test = rows.loc[rows.season.eq(year)]
                separated(train, test)
                matrix = values(fit_members(train, task), test, task)
                if task == 'winner': winner_oof.append(matrix); winner_y.append(test.home_win.to_numpy())
                else: score_oof.append(matrix); score_y.append(test[targets].to_numpy())
                for i, (name, _) in enumerate(winner_recipes if task == 'winner' else score_recipes):
                    out = test[['game_id', 'season', 'week', 'home_score', 'away_score']].copy()
                    out['model'] = name; out['training_through'] = str(train.gameday.max().date())
                    if task == 'winner': out['home_probability'] = matrix[:, i]
                    else: out[['predicted_home_score', 'predicted_away_score']] = matrix[:, :, i]
                    oof_rows.append(out)
                folds.append(dict(task=task, year=year, train_rows=len(train), test_rows=len(test),
                                  train_end=str(train.gameday.max().date()), test_start=str(test.gameday.min().date())))
            print(f'OOF {year} complete', flush=True)
        wp = np.concatenate(winner_oof); wy = np.concatenate(winner_y)
        sp = np.concatenate(score_oof); sy = np.concatenate(score_y)
        ww = fit_weights(wp, wy, True); sw = fit_weights(sp, sy)

        def build(base_end, calibration_year, label):
            wb = decisive.loc[decisive.season.le(base_end)]; wc = decisive.loc[decisive.season.eq(calibration_year)]
            sb = data.loc[data.season.le(base_end)]; sc = data.loc[data.season.eq(calibration_year)]
            separated(wb, wc); separated(sb, sc)
            wm = fit_members(wb, 'winner'); sm = fit_members(sb, 'score')
            pooled = values(wm, wc, 'winner')@ww
            calibrator = LogisticRegression(C=1, max_iter=1500).fit(logit(pooled).reshape(-1, 1), wc.home_win)
            residuals = sc[targets].to_numpy()-values(sm, sc, 'score')@sw
            return RobustMatchupEnsemble(wm, ww, sm, sw, calibrator, residuals, cols, version+'-'+label)

        evaluation = build(2024, 2025, 'evaluation')
        test_w = decisive.loc[decisive.season.eq(2026)]; test_s = data.loc[data.season.eq(2026)]
        separated(data.loc[data.season.le(2025)], test_s)
        p = evaluation.predict_proba(test_w[cols])[:, 1]; score = evaluation.predict(test_s[cols])
        members = values(evaluation.winner_models, test_w, 'winner')
        selected = int(np.argmin([-np.mean(wy*np.log(np.clip(wp[:, i], 1e-6, 1))+(1-wy)*np.log(np.clip(1-wp[:, i], 1e-6, 1))) for i in range(wp.shape[1])]))
        # The baseline gets the identical independent calibration season.
        cal = decisive.loc[decisive.season.eq(2025)]
        cal_members = values(evaluation.winner_models, cal, 'winner')
        baselines = {}
        for name, cal_p, test_p in [('equal_weight', cal_members.mean(axis=1), members.mean(axis=1)),
                                   ('best_oof_individual', cal_members[:, selected], members[:, selected])]:
            calibrator = LogisticRegression(C=1, max_iter=1500).fit(logit(cal_p).reshape(-1, 1), cal.home_win)
            baseline = calibrator.predict_proba(logit(test_p).reshape(-1, 1))[:, 1]
            delta = loss(p, test_w.home_win.to_numpy())-loss(baseline, test_w.home_win.to_numpy())
            baselines[name] = dict(metrics=probability_metrics(test_w.home_win, baseline),
                delta_log_loss=float(delta.mean()), paired_week_ci95=interval(delta, test_w.week.to_numpy()))
        details = evaluation.predict_details(test_s[cols]); actual = test_s[targets].to_numpy()
        coverage = ((actual >= details['score_lower']) & (actual <= details['score_upper'])).mean(axis=0)
        score_selected = int(np.argmin(np.mean((sp-sy[..., None])**2, axis=(0, 1))))
        score_cal = data.loc[data.season.eq(2025)]
        cal_scores = values(evaluation.score_models, score_cal, 'score')
        test_scores = values(evaluation.score_models, test_s, 'score')
        score_baselines = {}
        for name, cs, ts in [('equal_weight', cal_scores.mean(axis=-1), test_scores.mean(axis=-1)),
                             ('best_oof_individual', cal_scores[:, :, score_selected], test_scores[:, :, score_selected])]:
            bias = (score_cal[targets].to_numpy()-cs).mean(axis=0)
            baseline = np.maximum(0, ts+bias)
            delta = np.abs(score-actual).mean(axis=1)-np.abs(baseline-actual).mean(axis=1)
            score_baselines[name] = dict(metrics=score_metrics(actual, baseline),
                delta_team_mae=float(delta.mean()), paired_week_ci95=interval(delta, test_s.week.to_numpy()))
        final = build(2025, 2026, 'candidate')
    report = dict(version=version, parent_generation=generation, status='completed_shadow_only',
        production_changed=False, source_hashes=hashes, sklearn_version=sklearn.__version__,
        training_through=str(data.gameday.max().date()), feature_cols=cols, folds=folds,
        winner_members=[name for name, _ in winner_recipes], score_members=[name for name, _ in score_recipes],
        winner_weights=ww.tolist(), score_weights=sw.tolist(), maximum_member_weight=.35,
        weight_fit_seasons=[2022, 2023, 2024], calibration_season=2025, evaluation_season=2026,
        winner_metrics=probability_metrics(test_w.home_win, p), score_metrics=score_metrics(actual, score),
        winner_evaluation_games=len(test_w), score_evaluation_games=len(test_s), baselines=baselines,
        score_baselines=score_baselines, historical_ties_in_source=parent.get('historical_ties_in_source'),
        score_marginal_80pct_coverage=coverage.tolist(),
        final_base_through_season=2025, final_calibration_games=len(test_s), final_calibration_season=2026,
        convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        release_gate={'approved':False, 'prospective_games':0, 'reasons':[
            'New generation needs separately recorded prospective forecasts and evaluation.',
            'Serving adapter and fallback behavior require integration/release review.',
            'Retrospective 2026 results were inspected in prior research; not a fresh test.']},
        note='Six diverse members per task, fixed independently of 2026 results. Convex weights capped at 35%, '
             'shrunk toward uniform, with correlated-error penalty. Separate chronological calibration. '
             'Reported metrics belong to pre-refit evaluation artifact; final candidate uses 2026 only for calibration. '
             'Paired residuals preserve score dependence, but ranges are not coverage guarantees. '
             'No production state, scheduler, historical pick or shadow cohort changed.')
    for label, model in [('evaluation', evaluation), ('candidate', final)]:
        path = folder/(label+'.joblib')
        joblib.dump(dict(model=model, feature_cols=cols, model_version=model.model_version,
            status='shadow_only', training_through=report['training_through'] if label == 'candidate' else str(data.loc[data.season.le(2025)].gameday.max().date()),
            source_hashes=hashes, experimental=True), path, compress=3)
        loaded = joblib.load(path)['model']
        np.testing.assert_allclose(loaded.predict_proba(test_w[cols].head()), model.predict_proba(test_w[cols].head()))
        np.testing.assert_allclose(loaded.predict(test_s[cols].head()), model.predict(test_s[cols].head()))
        report[label+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    predictions = test_s[['game_id', 'season', 'week', 'home_score', 'away_score']].copy()
    predictions[['predicted_home_score', 'predicted_away_score']] = score
    predictions['home_probability'] = evaluation.predict_proba(test_s[cols])[:, 1]
    for destination in [folder, root/'backtests/retraining'/version]:
        write_json(destination/'report.json', report)
        write_csv(destination/'holdout_predictions.csv', predictions)
        write_csv(destination/'oof_predictions.csv', pd.concat(oof_rows, ignore_index=True))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--generation', required=True)
    with threadpool_limits(limits=2): run(Path(__file__).resolve().parent, parser.parse_args().generation)
