"""R41/R44/R45: matched recipes, temporal ensembles and interval evaluation."""
import argparse
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from audit_io import write_csv, write_json
from historical_metrics import interval, loss
from retrain_updated import load_data, pipeline, probability_metrics
from train_research32 import calibrated
from train_scores import score_metrics


def weights(values, y, classification=False):
    values=np.asarray(values);y=np.asarray(y)
    def objective(w):
        prediction=np.tensordot(values,w,axes=([-1],[0]))
        return float(loss(prediction,y).mean()) if classification else float(((prediction-y)**2).mean())
    n=values.shape[-1]
    result=minimize(objective,np.ones(n)/n,method='SLSQP',bounds=[(0,1)]*n,
                    constraints={'type':'eq','fun':lambda w:w.sum()-1})
    if not result.success or not np.isfinite(result.x).all():raise ValueError('Ensemble optimization failed')
    w=np.maximum(0,result.x);return w/w.sum()


def radius(errors,coverage=.8):
    errors=np.asarray(errors)
    if not len(errors) or not np.isfinite(errors).all():raise ValueError('Finite calibration errors required')
    rank=min(len(errors),int(np.ceil((len(errors)+1)*coverage)))
    return float(np.sort(np.abs(errors))[rank-1])


def score_values(scores):
    return {'home':scores[:,0],'away':scores[:,1],'margin':scores[:,0]-scores[:,1],'total':scores.sum(axis=1)}


def run(root,version):
    folder=root/'models/retrained'/version
    report=json.loads((folder/'report.json').read_text())
    data,hashes=load_data(root)
    if report['status']!='completed_shadow_only' or hashes!=report['source_hashes']:raise ValueError('Generation/source mismatch')
    decisive=data.loc[data.home_score.ne(data.away_score)].copy()
    decisive['home_win']=decisive.home_score.gt(decisive.away_score).astype(int)
    history=decisive.loc[decisive.season.le(2024)];val=decisive.loc[decisive.season.eq(2025)];test=decisive.loc[decisive.season.eq(2026)]
    production=joblib.load(root/'models/home_win_advanced_automl_no_market_deep.joblib')
    recipe=clone(production['model'].named_steps['model'])
    if 'n_jobs' in recipe.get_params():recipe.set_params(n_jobs=2)
    outputs=[];comparisons=[]
    holdouts=pd.read_csv(folder/'holdout_predictions.csv')
    for view,cols in report['features'].items():
        candidates=[m for m in report['models'] if m['task']=='winner' and m['name'].endswith(view)]
        chosen=min(candidates,key=lambda m:m['validation_log_loss'])
        reference=calibrated(recipe,history,val,cols)
        ref=reference.predict_proba(test[cols])[:,1]
        p=holdouts.loc[holdouts.model.eq(chosen['name'])].set_index('game_id').loc[test.game_id].home_probability.to_numpy()
        delta=loss(p,test.home_win.to_numpy())-loss(ref,test.home_win.to_numpy())
        ci=interval(delta,test.week.to_numpy())
        comparisons.append({'view':view,'candidate':chosen['name'],'games':len(test),
            'candidate_metrics':probability_metrics(test.home_win,p),'reference_metrics':probability_metrics(test.home_win,ref),
            'delta_log_loss':float(delta.mean()),'ci95':ci,'status':'Inconclusive' if ci[0]<=0<=ci[1] else 'Retrospective difference; prospective review required'})
        out=test[['game_id','season','week']].copy();out['view']=view;out['candidate']=p;out['reference']=ref;outputs.append(out)
    # Weights use early 2025; late 2025 remains separate for interval calibration.
    cols=report['features']['core_updated']
    winner_recipes=[LogisticRegression(C=.1,max_iter=2000),
        RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=12,n_jobs=2,random_state=42),
        MLPClassifier(hidden_layer_sizes=(8,),alpha=10,max_iter=600,early_stopping=False,random_state=42)]
    base=decisive.loc[decisive.season.le(2023)];cal=decisive.loc[decisive.season.eq(2024)]
    v=np.column_stack([calibrated(m,base,cal,cols).predict_proba(val[cols])[:,1] for m in winner_recipes])
    early=val.week.le(9).to_numpy();ww=weights(v[early],val.loc[early].home_win,True)
    e=np.column_stack([calibrated(m,history,val,cols).predict_proba(test[cols])[:,1] for m in winner_recipes])
    winner_final=[calibrated(m,decisive.loc[decisive.season.le(2025)],test,cols) for m in winner_recipes]
    winner_metrics=probability_metrics(test.home_win,e@ww)
    val_s=data.loc[data.season.eq(2025)];test_s=data.loc[data.season.eq(2026)]
    targets=['home_score','away_score'];base_s=data.loc[data.season.le(2024)];history_s=data.loc[data.season.le(2025)]
    score_recipes=[Ridge(alpha=1000),ExtraTreesRegressor(n_estimators=200,max_depth=5,min_samples_leaf=15,n_jobs=2,random_state=42)]
    sv=np.stack([np.maximum(0,pipeline(m).fit(base_s[cols],base_s[targets]).predict(val_s[cols])) for m in score_recipes],axis=-1)
    early_s=val_s.week.le(9).to_numpy();sw=weights(sv[early_s],val_s.loc[early_s,targets].to_numpy())
    se=np.stack([np.maximum(0,pipeline(m).fit(history_s[cols],history_s[targets]).predict(test_s[cols])) for m in score_recipes],axis=-1)
    score_final=[pipeline(m).fit(data[cols],data[targets]) for m in score_recipes]
    predicted=se@sw;calibrated_values=score_values((sv@sw)[~early_s]);actual_values=score_values(val_s.loc[~early_s,targets].to_numpy())
    radii={key:radius(actual_values[key]-value) for key,value in calibrated_values.items()}
    actual_test=score_values(test_s[targets].to_numpy());predicted_values=score_values(predicted)
    coverage={key:float((abs(actual_test[key]-value)<=radii[key]).mean()) for key,value in predicted_values.items()}
    chosen_w=int(np.argmin([loss(v[early,i],val.loc[early].home_win.to_numpy()).mean() for i in range(v.shape[1])]))
    chosen_s=int(np.argmin([((sv[early_s,:,i]-val_s.loc[early_s,targets].to_numpy())**2).mean() for i in range(sv.shape[-1])]))
    evidence={'generation':version,'comparisons':comparisons,
        'comparison_note':'Production Random Forest estimator recipe refitted on IDENTICAL candidate features, rows and temporal calibration: base through 2024, calibration 2025, test 2026. Candidates selected using 2025 only. This is not the saved live model performance. GPR is excluded because its fit/calibration protocol differs. Paired 95% intervals resample weeks; small sample, revised sources and post-hoc design; no multiplicity correction or automatic promotion.',
        'ensemble':{'winner_weights':ww.tolist(),'winner_members':['Logistic regression','Random forest','Small neural network (8 units)'],
                    'score_weights':sw.tolist(),'score_members':['Ridge','Extra trees'],
                    'winner_metrics':winner_metrics,'winner_individual_selected_on_early2025':probability_metrics(test.home_win,e[:,chosen_w]),
                    'score_metrics':score_metrics(test_s[targets].to_numpy(),predicted),
                    'score_individual_selected_on_early2025':score_metrics(test_s[targets].to_numpy(),se[:,:,chosen_s]),
                    'note':'Convex weights learned only on 2025 weeks 1-9 out-of-sample predictions, then frozen. Compare with an individual selected on those same early-2025 predictions. No claim that a combination improves accuracy.'},
        'intervals':{'nominal_coverage':.8,'radii':radii,'holdout_coverage':coverage,'games':len(test_s),
                     'calibration_games':int((~early_s).sum()),'calibration_through':str(val_s.gameday.max().date()),
                     'note':'Separate absolute-residual ranges calibrated on 2025 weeks 10 onward, after weight fitting. Finite-sample rank correction; not joint intervals or a guarantee under season drift. Final-refit forecast ranges reuse these historical residual widths and must be checked prospectively.'}}
    artifact=dict(winner_models=winner_final,winner_weights=ww,score_models=score_final,score_weights=sw,
                  radii=radii,feature_cols=cols,model_version=version+'-ensemble',training_through=report['training_through'])
    joblib.dump(artifact,folder/'temporal_ensemble.joblib',compress=3)
    write_json(folder/'next_evidence.json',evidence)
    write_json(root/'public_site/data/next_evidence.json',evidence)
    write_csv(root/'backtests/retraining'/version/'matched_reference.csv',pd.concat(outputs,ignore_index=True))
    print(json.dumps(evidence,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',required=True)
    with threadpool_limits(limits=2):run(Path(__file__).resolve().parent,p.parse_args().version)
