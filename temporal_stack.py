"""R21 fixed regularized stacking of R18 temporal out-of-fold probabilities."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_io import write_json, write_csv
from matched_backtest import summarize


def build(rows, report):
    ids = sorted(m['id'] for m in report['models'] if not m['market'])
    subset = rows.loc[rows.model_id.isin(ids)]
    if subset.duplicated(['model_id','game_id']).any():
        raise ValueError('Duplicate OOF predictions')
    matrix = subset.pivot(index='game_id',columns='model_id',values='probability').reindex(columns=ids)
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all() or ((matrix<0)|(matrix>1)).any().any():
        raise ValueError('Missing or invalid OOF probabilities')
    for key in ['season','week','gameday','home_win']:
        if subset.groupby('game_id')[key].nunique().ne(1).any():
            raise ValueError('OOF label/metadata disagreement')
    meta = subset.drop_duplicates('game_id').set_index('game_id').loc[matrix.index]
    parts = [meta.season.eq(year) for year in [2024,2025,2026]]
    if any(not p.any() for p in parts):
        raise ValueError('Need separate meta-fit, calibration and test seasons')
    dates = pd.to_datetime(meta.gameday)
    if not dates[parts[0]].max()<dates[parts[1]].min() or not dates[parts[1]].max()<dates[parts[2]].min():
        raise ValueError('Stack temporal partitions overlap')
    for fold in report['folds']:
        if fold['final_calibration_through'] >= fold['test_start']:
            raise ValueError('Base predictions are not temporal OOF')
    x = np.log(np.clip(matrix,1e-6,1-1e-6)/np.clip(1-matrix,1e-6,1-1e-6))
    model = make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=3000,random_state=42))
    model.fit(x.loc[parts[0]],meta.loc[parts[0],'home_win'])
    calibrated = CalibratedClassifierCV(FrozenEstimator(model),method='sigmoid')
    calibrated.fit(x.loc[parts[1]],meta.loc[parts[1],'home_win'])
    test = meta.loc[parts[2],['season','week','gameday','home_win']].copy()
    test['probability'] = calibrated.predict_proba(x.loc[parts[2]])[:,1]
    test['baseline_probability'] = matrix.loc[parts[2]].mean(axis=1)
    return calibrated, ids, test.reset_index()


def run(root):
    source = root/'backtests/matched_predictions.csv'
    report = json.loads((root/'public_site/data/matched_comparison.json').read_text())
    model, ids, predictions = build(pd.read_csv(source),report)
    version = 'stack-r21-'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    folder = root/'models/research'/version
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'base_ids':ids,'base_run_id':report['run_id'],
                 'model_version':version,'status':'research_only','meta_fit_season':2024,'calibration_season':2025},folder/'stack.joblib')
    output = {'status':'research_only','model_version':version,'base_run_id':report['run_id'],'base_count':len(ids),
              'meta_fit_season':2024,'calibration_season':2025,'test_season':2026,
              'summary':summarize(predictions),
              'baseline':summarize(predictions.assign(probability=predictions.baseline_probability)),
              'note':'Fixed logistic C=0.01 on logit-transformed no-market R18 OOF forecasts. Meta-fit 2024; sigmoid calibration 2025; untouched evaluation 2026. Equal-weight mean of the same base probabilities is the comparator. No random K-fold stacking or selection of top historical recipes. Recipe-library design is retrospective; the small 2026 sample is not prospective validation. Separate saved artifact, no promotion. Future forecasts are withheld until matching R18 base-fit artifacts and live feature coverage are available; legacy library weights must not be substituted.'}
    write_csv(root/'backtests/stack_predictions.csv',predictions)
    write_json(root/'public_site/data/temporal_stack.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__': run(Path(__file__).resolve().parent)
