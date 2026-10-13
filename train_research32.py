"""32 additional configurations; disjoint season-based calibration and tuning."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from gpr_models import FEATURES
from train_gpr import metrics
from audit_io import write_json, write_csv


def families():
    return {
        'shrinkage_lda':[LinearDiscriminantAnalysis(solver='lsqr',shrinkage=s) for s in [.5,.9]],
        'regularized_qda':[QuadraticDiscriminantAnalysis(reg_param=s) for s in [.5,.9]],
        'neighbors':[KNeighborsClassifier(n_neighbors=k) for k in [41,81]],
        'rbf_svm':[SVC(C=c,gamma='scale') for c in [.1,1]],
        'linear_svm':[LinearSVC(C=c,max_iter=5000,random_state=42) for c in [.01,.1]],
        'ridge_classifier':[RidgeClassifier(alpha=a) for a in [100,1000]],
        'passive_aggressive':[PassiveAggressiveClassifier(C=c,max_iter=2000,random_state=42) for c in [.001,.01]],
        'neural_network':[MLPClassifier(hidden_layer_sizes=(n,),alpha=10,max_iter=250,random_state=42,early_stopping=False) for n in [8,16]],
        'bagged_trees':[BaggingClassifier(DecisionTreeClassifier(max_depth=d,min_samples_leaf=30),n_estimators=50,random_state=42,n_jobs=2) for d in [2,4]],
        'bagged_logistic':[BaggingClassifier(LogisticRegression(C=c,max_iter=2000),n_estimators=15,random_state=42,n_jobs=2) for c in [.01,.1]],
        'elastic_net':[LogisticRegression(solver='saga',l1_ratio=.5,C=c,max_iter=3000,random_state=42) for c in [.01,.1]],
        'spline_logistic':[make_pipeline(SplineTransformer(n_knots=3),LogisticRegression(C=c,max_iter=3000)) for c in [.01,.1]],
        'rbf_features':[make_pipeline(RBFSampler(gamma=g,n_components=100,random_state=42),LogisticRegression(C=.1,max_iter=2000)) for g in [.01,.1]],
        'nystroem_logistic':[make_pipeline(Nystroem(gamma=g,n_components=100,random_state=42),LogisticRegression(C=.1,max_iter=2000)) for g in [.01,.1]],
        'nearest_centroid':[NearestCentroid(shrink_threshold=s) for s in [None,.5]],
        'polynomial_svm':[SVC(kernel='poly',degree=2,C=c,coef0=1) for c in [.01,.1]],
    }


def calibrated(estimator, base, calibration, cols):
    if base.gameday.max() >= calibration.gameday.min(): raise ValueError('Temporal calibration overlap')
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),clone(estimator))
    model.fit(base[cols],base.home_win)
    result=CalibratedClassifierCV(FrozenEstimator(model),method='sigmoid')
    result.fit(calibration[cols],calibration.home_win)
    return result


def main(root):
    path=root/'data/research_matchups.csv'
    data=pd.read_csv(path).sort_values(['gameday','game_id'])
    data=data.loc[data.season.ge(2017)&data.home_win.notna()&data.home_score.ne(data.away_score)]
    core=[c for c in FEATURES if c in data and pd.api.types.is_numeric_dtype(data[c])]
    views={'core':core,'charted':core+[c for c in data if c.startswith('diff_research_')]}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=root/'models/research32'/stamp;folder.mkdir(parents=True,exist_ok=True)
    report={'created_at':stamp,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'models':[],
        'note':'16 method families x 2 feature views = 32 research configurations. Two settings per configuration. Fit <=2023, calibrate 2024, select on 2025. Evaluate 2026 after fit <=2024/calibrate 2025. Final artifacts fit <=2025/calibrate on completed 2026. Retrospective revised-source results, not independent prospective performance. FTN via nflverse CC-BY-SA 4.0; charted view lacks these inputs before 2022. No production promotion.'}
    test=data.loc[data.season.eq(2026)]
    holdout=test[['game_id','home_win']].copy()
    for view,cols in views.items():
        for family, trials in families().items():
            name=f'research_{family}_{view}'
            print(f'Training {name}',flush=True)
            scores=[]
            for estimator in trials:
                candidate=calibrated(estimator,data.loc[data.season.le(2023)],data.loc[data.season.eq(2024)],cols)
                scores.append(metrics(data.loc[data.season.eq(2025),'home_win'],candidate.predict_proba(data.loc[data.season.eq(2025),cols])[:,1])['log_loss'])
            selected=trials[int(np.argmin(scores))]
            candidate=calibrated(selected,data.loc[data.season.le(2024)],data.loc[data.season.eq(2025)],cols)
            p=candidate.predict_proba(test[cols])[:,1]
            holdout[name]=p
            result={'name':name,'family':'Research32','method':family,'view':view,'status':'Shadow research','market_inputs':False,
                'holdout_season':2026,'test_rows':len(test),'training':f'Temporal calibration; {len(cols)} inputs; evaluated through {test.gameday.max()}',
                'validation_log_loss':min(scores),'trials':scores,'final_training_through':str(test.gameday.max()),**metrics(test.home_win,p)}
            final=calibrated(selected,data.loc[data.season.le(2025)],test,cols)
            artifact={'model':final,'feature_cols':cols,'model_version':name+'_'+stamp,'experimental':True,'metrics':result,'source_sha256':report['source_sha256']}
            target=folder/(name+'.joblib');joblib.dump(artifact,target,compress=3)
            np.testing.assert_allclose(joblib.load(target)['model'].predict_proba(test[cols].head()),final.predict_proba(test[cols].head()))
            report['models'].append(result)
            write_json(root/'models/research32_report.json',report)
            print(f'{name}: log loss {result["log_loss"]:.4f}',flush=True)
    write_csv(root/'models/research32_holdout.csv',holdout)
    print(f'Saved {len(report["models"])} models to {folder}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main(Path(__file__).resolve().parent)
