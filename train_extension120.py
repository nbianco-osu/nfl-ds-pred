"""Add 60 small-network/regularized configurations to the frozen R40 generation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_io import write_csv, write_json
from retrain_updated import load_data, partitions, pipeline, probability_metrics
from train_research32 import calibrated
from train_scores import score_metrics


def recipes(views):
    architectures = [(8,), (16,), (32,), (8,4), (16,8), (32,16)]
    for view, cols in views.items():
        for layers in architectures:
            for alpha in [1., 10.]:
                label = 'x'.join(map(str,layers))+'_'+str(int(alpha))+'_'+view
                options = dict(hidden_layer_sizes=layers, alpha=alpha, max_iter=600,
                               early_stopping=False, random_state=42, learning_rate_init=.003)
                yield 'nn_winner_'+label, 'winner', MLPClassifier(**options), cols
                yield 'nn_score_'+label, 'score', TransformedTargetRegressor(
                    regressor=MLPRegressor(**options), transformer=StandardScaler()), cols
    for alpha in [.01,.1,1.,10.,100.,1000.]:
        ridge_alpha = 10000. if alpha == 1000. else alpha
        yield 'ridge_score_'+str(ridge_alpha), 'score', Ridge(alpha=ridge_alpha), views['core_updated']
        yield 'poisson_score_'+str(alpha), 'score', MultiOutputRegressor(PoissonRegressor(alpha=alpha,max_iter=1000)), views['core_updated']


def run(root, source):
    if Path(source).name != source: raise ValueError('Generation name required')
    original = root/'models/retrained'/source
    old = json.loads((original/'report.json').read_text())
    data, hashes = load_data(root)
    if old['source_hashes'] != hashes or len(old['models']) != 60 or old['status'] != 'completed_shadow_only':
        raise ValueError('The completed 60-model source must match current input hashes')
    version = 'expanded120-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = root/'models/retrained'/version
    folder.mkdir(parents=True,exist_ok=False)
    report = dict(old,version=version,status='in_progress',parent_generation=source,
                  note=old['note']+' 60 retained configurations reuse byte-identical artifacts on identical source hashes. 60 additions include 24 small winner MLPs, 24 target-standardized score MLPs, six Ridge and six Poisson settings. No new distinct algorithm count is claimed. Network settings are predeclared configurations; no random internal validation split. Convergence warnings are reported.')
    for m in report['models']:
        path=original/(m['name']+'.joblib')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=m['artifact_sha256']:raise ValueError('Parent artifact mismatch')
        shutil.copy2(path,folder/path.name)
    frames=[pd.read_csv(original/'holdout_predictions.csv')]
    decisive=data.loc[data.home_score.ne(data.away_score)].copy()
    decisive['home_win']=decisive.home_score.gt(decisive.away_score).astype(int)
    for name,task,estimator,cols in recipes(report['features']):
        rows=decisive if task=='winner' else data
        base,cal,val,test=partitions(rows)
        history=rows.loc[rows.season.lt(2026)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            if task=='winner':
                validation=calibrated(estimator,base,cal,cols).predict_proba(val[cols])[:,1]
                evaluation=calibrated(estimator,rows.loc[rows.season.le(2024)],val,cols)
                predicted=evaluation.predict_proba(test[cols])[:,1]
                result=dict(validation_log_loss=probability_metrics(val.home_win,validation)['log_loss'],holdout=probability_metrics(test.home_win,predicted))
                final=calibrated(estimator,history,test,cols)
            else:
                targets=['home_score','away_score']
                validation=np.maximum(0,pipeline(estimator).fit(rows.loc[rows.season.le(2024),cols],rows.loc[rows.season.le(2024),targets]).predict(val[cols]))
                predicted=np.maximum(0,pipeline(estimator).fit(history[cols],history[targets]).predict(test[cols]))
                result=dict(validation_team_mae=float(np.abs(validation-val[targets].to_numpy()).mean()),holdout=score_metrics(test[targets].to_numpy(),predicted))
                final=pipeline(estimator).fit(rows[cols],rows[targets])
        if not np.isfinite(predicted).all():raise ValueError('Nonfinite predictions')
        result.update(name=name,task=task,selected_configuration=str(estimator),
                      convergence_warnings=sum(issubclass(w.category,ConvergenceWarning) for w in caught))
        path=folder/(name+'.joblib')
        joblib.dump(dict(model=final,feature_cols=cols,model_version=version+'-'+name,
                         experimental=True,status='shadow_only',metrics=result,
                         source_hashes=hashes,training_through=report['training_through']),path,compress=3)
        loaded=joblib.load(path)['model'];method='predict_proba' if task=='winner' else 'predict'
        np.testing.assert_allclose(getattr(loaded,method)(test[cols].head()),getattr(final,method)(test[cols].head()))
        result['artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        report['models'].append(result)
        out=test[['game_id','season','week','home_score','away_score']].copy();out['model']=name
        if task=='winner':out['home_probability']=predicted
        else:out[['predicted_home_score','predicted_away_score']]=predicted
        frames.append(out)
        write_json(folder/'report.json',report)
        print(f'{len(report["models"]):03d}/120 {name} {result["holdout"]}',flush=True)
    if len(report['models'])!=120:raise ValueError('Incomplete generation')
    report['status']='completed_shadow_only'
    output=pd.concat(frames,ignore_index=True)
    for destination in [folder,root/'backtests/retraining'/version]:
        write_json(destination/'report.json',report);write_csv(destination/'holdout_predictions.csv',output)
    print('COMPLETED '+version,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    with threadpool_limits(limits=2):run(Path(__file__).resolve().parent,p.parse_args().source)
