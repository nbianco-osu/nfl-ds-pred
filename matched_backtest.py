"""R18: nested annual rolling evaluation of 64 research recipes, never production."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits

from audit_io import write_csv, write_json
from export_model_comparison import LABELS, KERNEL_LABELS
from gpr_models import FEATURES, KERNELS, CalibratedGPRClassifier
from model_utils import build_model_pipeline
from train_automl import search_space
from train_expanded import families as expanded_families
from train_research32 import families as research_families

PROTOCOL = 'r18-annual-v1'


def load_data(root):
    paths = [root/'data/research_matchups.csv', root/'data/nfl_matchups_1999_2026_expanded.csv']
    data, expanded = [pd.read_csv(path) for path in paths]
    if data.game_id.duplicated().any() or expanded.game_id.duplicated().any():
        raise ValueError('Duplicate source games')
    extra = [c for c in expanded if c.startswith(('home_team_ext_', 'away_team_ext_', 'diff_team_ext_'))]
    data = data.merge(expanded[['game_id']+extra], on='game_id', how='left', validate='one_to_one')
    data['gameday'] = pd.to_datetime(data.gameday, errors='raise')
    data = data.loc[data.season.ge(2017) & data.home_score.notna() & data.away_score.notna()].copy()
    decisive = data.home_score.ne(data.away_score)
    if not data.loc[decisive, 'home_win'].eq(data.loc[decisive].home_score.gt(data.loc[decisive].away_score).astype(int)).all():
        raise ValueError('Inconsistent target labels')
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return data.sort_values(['gameday','game_id']), hashes


def recipes(root, data):
    library = json.loads((root/'public_site/data/model_comparison.json').read_text())['models']
    labels = {row['id']: row['name'] for row in library}
    result = []
    for filename in LABELS:
        artifact = joblib.load(root/'models'/filename)
        cols = artifact['feature_cols']
        if 'automl' in filename:
            depth = 'deep' if 'deep' in filename else 'standard'
            trials = []
            for settings in ParameterSampler(search_space(42, depth), n_iter=4 if depth=='standard' else 6, random_state=42):
                model = clone(settings['model'])
                model.set_params(**{k.removeprefix('model__'): v for k,v in settings.items() if k!='model'})
                if 'n_jobs' in model.get_params(): model.set_params(n_jobs=2)
                if 'early_stopping' in model.get_params(): model.set_params(early_stopping=False)
                trials.append(model)
        else:
            trials = [LogisticRegression(C=c,max_iter=3000,random_state=42) for c in [.03,.3]]
        result.append(dict(id=filename, name=labels[filename], group='Original', market='no_market' not in filename, cols=cols, trials=trials, mode='calibrated'))
    extra = [c for c in data if c.startswith('diff_team_ext_')]
    for kernel in KERNELS:
        key = f'home_win_gpr_{kernel}.joblib'
        trials = [CalibratedGPRClassifier(kernel,noise,scale,1000,extra_features=tuple(extra)) for noise in [.1,.5] for scale in [3.,8.]]
        result.append(dict(id=key,name=labels[key],group='GPR',market=False,cols=['season','week',*FEATURES,*extra],trials=trials,mode='gpr'))
    for view in ['contrasts','full_context']:
        for family,trials in expanded_families().items():
            key = f'home_win_expanded_{family}_{view}.joblib'
            artifact = joblib.load(root/'models'/key)
            cols = artifact['selected_feature_cols']
            result.append(dict(id=key,name=labels[key],group='Expanded',market=False,cols=cols,trials=trials,mode='calibrated'))
    for view in ['core','charted']:
        cols = list(FEATURES) + ([c for c in data if c.startswith('diff_research_')] if view=='charted' else [])
        for family,trials in research_families().items():
            key = f'research_{family}_{view}'
            result.append(dict(id=key,name=labels[key],group='Research32',market=False,cols=cols,trials=trials,mode='calibrated'))
    if len(result)!=64 or {r['id'] for r in result}!={r['id'] for r in library}:
        raise ValueError('Registry must match the complete 64-model library')
    forbidden = {'game_id','gameday','home_score','away_score','home_win'}
    for recipe in result:
        if forbidden.intersection(recipe['cols']) or not set(recipe['cols']).issubset(data):
            raise ValueError(f'Invalid features: {recipe["id"]}')
        if not recipe['market'] and {'spread_line','total_line'}.intersection(recipe['cols']):
            raise ValueError('Market leakage into no-market recipe')
    return result


def fold_parts(data, season):
    decisive = data.loc[data.home_score.ne(data.away_score)]
    test = decisive.loc[decisive.season.eq(season)]
    base = decisive.loc[decisive.season.lt(season-2)]
    calibration = decisive.loc[decisive.season.eq(season-2)]
    validation = decisive.loc[decisive.season.eq(season-1)]
    if any(x.empty for x in [base,calibration,validation,test]):
        raise ValueError('Empty fold partition')
    if not base.gameday.max() < calibration.gameday.min() or not calibration.gameday.max() < validation.gameday.min() or not validation.gameday.max() < test.gameday.min():
        raise ValueError('Fold dates overlap')
    return base,calibration,validation,test


def fit(recipe, estimator, base, calibration):
    if base.gameday.max() >= calibration.gameday.min():
        raise ValueError('Calibration must be later than training')
    cols = recipe['cols']
    if recipe['mode']=='gpr':
        combined = pd.concat([base,calibration]).sort_values(['season','week','game_id'])
        return clone(estimator).set_params(calibration_rows=len(calibration)).fit(combined[cols],combined.home_win)
    model = build_model_pipeline(base,cols)
    # Preserve all-empty columns so newer feature sources do not change dimensions.
    model.set_params(preprocess__num__imputer=SimpleImputer(strategy='median',keep_empty_features=True),
                     preprocess__cat__imputer=SimpleImputer(strategy='most_frequent',keep_empty_features=True),model=clone(estimator))
    model.fit(base[cols],base.home_win)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model),method='sigmoid')
    return calibrated.fit(calibration[cols],calibration.home_win)


def probability(model, frame, cols):
    p = np.asarray(model.predict_proba(frame[cols])[:,1],dtype=float)
    if len(p)!=len(frame) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probabilities')
    return np.clip(p,1e-6,1-1e-6)


def losses(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return -y*np.log(p)-(1-y)*np.log1p(-p), (p-y)**2


def summarize(rows, iterations=1000):
    """Paired resampling by season-week keeps within-week dependence together."""
    y=rows.home_win.to_numpy()
    ll,br=losses(y,rows.probability.to_numpy())
    bll,bbr=losses(y,rows.baseline_probability.to_numpy())
    blocks=list(rows.groupby(['season','week'],sort=True).indices.values())
    rng=np.random.default_rng(42)
    deltas=np.empty((iterations,2))
    for i in range(iterations):
        selected=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
        deltas[i]=[(ll[selected]-bll[selected]).mean(),(br[selected]-bbr[selected]).mean()]
    return {'n':len(rows),'accuracy':float(((rows.probability.to_numpy()>=.5)==y).mean()),
            'log_loss':float(ll.mean()),'brier':float(br.mean()),
            'delta_log_loss':float((ll-bll).mean()),'delta_brier':float((br-bbr).mean()),
            'delta_log_loss_ci95':np.quantile(deltas[:,0],[.025,.975]).tolist(),
            'delta_brier_ci95':np.quantile(deltas[:,1],[.025,.975]).tolist(),
            'weeks':len(blocks)}


def export(root, all_rows, registry, manifests, hashes, run_id):
    if all_rows.duplicated(['model_id','game_id']).any(): raise ValueError('Duplicate predictions')
    counts=all_rows.groupby('model_id').size()
    if len(counts)!=64 or counts.nunique()!=1: raise ValueError('Incomplete matched cohort')
    expected=set(all_rows.loc[all_rows.model_id.eq(registry[0]['id']),'game_id'])
    if any(set(g.game_id)!=expected for _,g in all_rows.groupby('model_id')): raise ValueError('Different evaluation games')
    models=[]
    for recipe in registry:
        frame=all_rows.loc[all_rows.model_id.eq(recipe['id'])].reset_index(drop=True)
        models.append({k:recipe[k] for k in ['id','name','group','market']} | {'overall':summarize(frame),
            'seasons':{str(int(year)):summarize(g.reset_index(drop=True)) for year,g in frame.groupby('season')}})
    games=all_rows.drop_duplicates('game_id')
    baseline=games.copy();baseline['probability']=baseline.baseline_probability
    note=('Retrospective revised-source research, not prospective evidence or deployed-model accuracy. '
          '64 library recipes are re-tuned on identical annual rolling folds; no saved fitted weights or post-test chosen settings are reused. '
          'Base fit, sigmoid calibration, inner selection and outer evaluation use disjoint chronological seasons. '
          'Original AutoML uses a bounded reproducible 4-trial standard / 6-trial deep search, not the original full search. '
          'Each selected recipe refits on base+calibration and calibrates on the former validation season before testing. '
          'GPR retains its 1,000-row recent regression window. Missing features are imputed using training data; source coverage varies. '
          'Market lines and historical weather are revised snapshots, not proven pregame observations. '
          'Paired 95% intervals use 1,000 season-week bootstrap draws, seed 42; intervals are not adjusted for 64 comparisons. '
          'No automatic promotion and no claim of future improvement. Historical source omits older ties; 2026 ties are excluded explicitly.')
    output={'protocol':PROTOCOL,'run_id':run_id,'status':'research_only','created_at':datetime.now(timezone.utc).isoformat(),
            'source_hashes':hashes,'model_count':64,'games':len(games),'seasons':sorted(games.season.astype(int).unique().tolist()),
            'through':str(games.gameday.max()),'note':note,'baseline':summarize(baseline.reset_index(drop=True)),
            'baseline_name':'Training-only home-win frequency','models':models,'folds':manifests,
            'baseline_seasons':{str(int(year)):summarize(g.assign(probability=g.baseline_probability).reset_index(drop=True)) for year,g in games.groupby('season')}}
    write_json(root/'public_site/data/matched_comparison.json',output)
    write_csv(root/'backtests/matched_predictions.csv',all_rows)
    return output


def run(root, seasons=(2024,2025,2026), limit=None):
    data,hashes=load_data(root)
    registry=recipes(root,data)
    signature=json.dumps({'sources':hashes,'seasons':list(seasons),'protocol':PROTOCOL,
                          'code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True)
    run_id=hashlib.sha256(signature.encode()).hexdigest()[:16]
    folder=root/'backtests/matched'/run_id;folder.mkdir(parents=True,exist_ok=True)
    frames=[];manifests=[]
    for season in seasons:
        base,calibration,validation,test=fold_parts(data,season)
        final_base=pd.concat([base,calibration]).sort_values(['gameday','game_id'])
        manifest={'season':season,'base_through':str(base.gameday.max().date()),'calibration_start':str(calibration.gameday.min().date()),
                  'selection_start':str(validation.gameday.min().date()),'test_start':str(test.gameday.min().date()),
                  'final_fit_through':str(final_base.gameday.max().date()),'final_calibration_through':str(validation.gameday.max().date()),
                  'test_games':len(test),'ties_excluded':int((data.season.eq(season)&data.home_score.eq(data.away_score)).sum()),'recipes':[]}
        baseline=float(pd.concat([final_base,validation]).home_win.mean())
        for recipe in registry[:limit]:
            path=folder/f'{season}_{recipe["id"]}.json'
            if path.exists():
                saved=json.loads(path.read_text())
            else:
                scores=[]
                for estimator in recipe['trials']:
                    model=fit(recipe,estimator,base,calibration)
                    scores.append(float(log_loss(validation.home_win,probability(model,validation,recipe['cols']),labels=[0,1])))
                selected=int(np.argmin(scores))
                model=fit(recipe,recipe['trials'][selected],final_base,validation)
                p=probability(model,test,recipe['cols'])
                saved={'game_ids':test.game_id.tolist(),'probability':p.tolist(),'selection_losses':scores,'selected':selected,
                       'selected_estimator':str(recipe['trials'][selected]),'feature_count':len(recipe['cols']),
                       'missing_test_fraction':float(test[recipe['cols']].isna().to_numpy().mean())}
                write_json(path,saved)
            if saved['game_ids']!=test.game_id.tolist(): raise ValueError('Cached cohort changed')
            out=test[['game_id','season','week','gameday','home_win']].copy()
            out['model_id']=recipe['id'];out['probability']=saved['probability'];out['baseline_probability']=baseline
            frames.append(out)
            manifest['recipes'].append({'id':recipe['id'],**{k:v for k,v in saved.items() if k not in ['game_ids','probability']}})
            print(f'{season} {recipe["id"]}: {len(test)} matched games',flush=True)
        manifests.append(manifest)
    if limit is not None:
        print('Smoke run only; public results not modified',flush=True)
        return
    result=export(root,pd.concat(frames,ignore_index=True),registry,manifests,hashes,run_id)
    print(f'Published research export: {result["games"]} identical games x 64 recipes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons',type=int,nargs='+',default=[2024,2025,2026])
    parser.add_argument('--limit',type=int,help='Smoke-only configuration count; never publishes')
    args=parser.parse_args()
    with threadpool_limits(limits=2): run(Path(__file__).resolve().parent,args.seasons,args.limit)
