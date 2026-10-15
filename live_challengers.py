"""R24/R25: frozen research models, chronological distributions, prospective cohorts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom, rankdata
from threadpoolctl import threadpool_limits

from audit_io import write_json, write_csv
from rolling_backtest import FEATURES, prepare
from score_experiments import estimator, evaluate
from score_distribution import summarize
from pregame_inputs import kickoff, timestamp

KINDS = ['poisson', 'negative_binomial', 'correlated_negative_binomial']


def fit(base, calibration):
    if base.gameday.max() >= calibration.gameday.min():
        raise ValueError('Overlapping fit/calibration dates')
    model = estimator('poisson').fit(base[FEATURES], base[['home_score','away_score']])
    mean = np.maximum(.01, model.predict(calibration[FEATURES]))
    residual = calibration[['home_score','away_score']].to_numpy()-mean
    dispersion = np.maximum(1e-5, np.sum(residual**2-mean, axis=0)/np.sum(mean**2, axis=0))
    ranks = np.column_stack([norm.ppf((rankdata(residual[:,i])-.5)/len(residual)) for i in range(2)])
    rho = float(np.corrcoef(ranks.T)[0,1])
    rho = float(np.clip(rho, -.8, .8)) if np.isfinite(rho) else 0.
    return {'model':model, 'dispersion':dispersion, 'rho':rho,
            'fit_through':base.gameday.max().date().isoformat(),
            'calibration_through':calibration.gameday.max().date().isoformat()}


def sample(artifact, mean, kind, trials=10000, seed=42):
    mean = np.maximum(.01,np.asarray(mean,dtype=float))
    if mean.shape != (2,) or not np.isfinite(mean).all():
        raise ValueError('Finite score means required')
    rng = np.random.default_rng(seed)
    if kind == 'poisson':
        return rng.poisson(mean,size=(trials,2))
    if kind not in KINDS:
        raise ValueError('Unknown challenger')
    rho = artifact['rho'] if kind == 'correlated_negative_binomial' else 0.
    u = np.clip(norm.cdf(rng.multivariate_normal([0,0],[[1,rho],[rho,1]],trials)),1e-9,1-1e-9)
    n = 1/np.asarray(artifact['dispersion'])
    return nbinom.ppf(u,n,n/(n+mean))


def history(root,season):
    paths = [root/'backtests/source_schedule.csv',root/f'data/nfl_{season}_completed.csv']
    raw = pd.concat([pd.read_csv(p) for p in paths]).drop_duplicates('game_id',keep='last')
    return prepare(raw), {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def train(root,season=2026):
    data, hashes = history(root,season)
    rows, folds = [], []
    artifact = None
    for year in range(season-2,season+1):
        base=data.loc[data.season.between(2017,year-2)]
        calibration=data.loc[data.season.eq(year-1)]
        test=data.loc[data.season.eq(year)]
        if test.empty or calibration.empty or base.empty:
            raise ValueError('Missing chronological fold')
        artifact=fit(base,calibration)
        if calibration.gameday.max()>=test.gameday.min():
            raise ValueError('Calibration/test overlap')
        folds.append({'season':year,'games':len(test),'fit_through':artifact['fit_through'],
                      'calibration_through':artifact['calibration_through'], 'dispersion':artifact['dispersion'].tolist(),
                      'copula_rho':artifact['rho']})
        means=artifact['model'].predict(test[FEATURES])
        for kind in KINDS:
            for i,g in enumerate(test.itertuples()):
                values=sample(artifact,means[i],kind)
                rows.append({'kind':kind,'game_id':g.game_id,'season':year,
                    **evaluate(values,g.home_score,g.away_score,getattr(g,'spread_line',None),getattr(g,'total_line',None))})
            print(year,kind,len(test),flush=True)
    frame=pd.DataFrame(rows)
    version='r25-'+hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()[:12]
    artifact.update(model_version=version,status='shadow_only',feature_cols=FEATURES,season=season)
    folder=root/'models/research'/version
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact,folder/'models.joblib')
    write_json(root/'models/research/live_challengers.json',{'model_version':version,'artifact':str((folder/'models.joblib').relative_to(root)), 'season':season})
    metrics=[c for c in frame if c not in ['kind','game_id','season']]
    models=[]
    for kind,group in frame.groupby('kind',sort=False):
        models.append({'id':kind,'games':len(group),'metrics':{c:float(group[c].mean()) for c in metrics},
            'seasons':{str(y):{'games':len(g),'metrics':{c:float(g[c].mean()) for c in metrics}} for y,g in group.groupby('season')}})
    report={'status':'research_only','model_version':version,'folds':folds,'models':models,'source_hashes':hashes,
        'note':'Matched annual chronological folds. Poisson score means fit through test year minus 2; NB2 dispersion and Gaussian-copula rank correlation estimated on the preceding year only. Variance = mean + dispersion * mean squared. 10,000 draws, seed 42. Fixed settings; no outer-label tuning. Point counts are not a football scoring process and ties are not an overtime model. Revised historical source data/lines; not prospective results. Correlation is estimated from calibration residual ranks, not fitted discrete likelihood. No automatic promotion or claim of improvement.'}
    write_csv(root/'backtests/uncertainty_evaluation.csv',frame)
    write_json(root/'public_site/data/uncertainty_research.json',report)
    return artifact


def form_features(game,schedule,data):
    cutoff=pd.to_datetime(schedule.loc[schedule.week.eq(game['week']),'gameday']).min()
    values=[]
    for team in [game['home_team'],game['away_team']]:
        p=data.loc[data.gameday.lt(cutoff)&(data.home_team.eq(team)|data.away_team.eq(team))].sort_values(['gameday','game_id']).tail(8)
        values.extend([float(np.where(p.home_team.eq(team),p.home_score,p.away_score).mean()),
                       float(np.where(p.home_team.eq(team),p.away_score,p.home_score).mean())] if len(p) else [21.,21.])
    return pd.DataFrame([values],columns=FEATURES),cutoff.date().isoformat()


def grade(records,schedule):
    lookup=schedule.set_index('game_id').to_dict('index')
    result=[]
    for kind in KINDS:
        matched=[]
        for record in records:
            g=lookup.get(record['game_id'])
            if record['kind']!=kind or not g or pd.isna(g['home_score']) or pd.isna(g['away_score']):
                continue
            if timestamp(record['forecast_at'])>=kickoff(g) or record.get('production_probability') is None:
                continue
            if g['home_score']==g['away_score']:
                continue
            y=float(g['home_score']>g['away_score'])
            matched.append((record['home_win_probability'],record['production_probability'],y))
        summary={'kind':kind,'games':len(matched),'status':'Awaiting prospective outcomes' if not matched else 'Shadow only; human review required'}
        if matched:
            a=np.array(matched); p=np.clip(a[:,:2],1e-8,1-1e-8); y=a[:,2,None]
            ll=-(y*np.log(p)+(1-y)*np.log(1-p))
            summary.update(challenger_log_loss=float(ll[:,0].mean()),production_log_loss=float(ll[:,1].mean()),
                challenger_brier=float(((p[:,0]-a[:,2])**2).mean()),production_brier=float(((p[:,1]-a[:,2])**2).mean()))
        result.append(summary)
    return result


def export(root,season=2026,now=None):
    now=timestamp(now or datetime.now(timezone.utc))
    pointer=root/'models/research/live_challengers.json'
    if not pointer.exists():
        return
    config=json.loads(pointer.read_text()); artifact=joblib.load(root/config['artifact'])
    if config['season']!=season:
        raise ValueError('Challenger season mismatch; explicit training required')
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    schedule=schedule.loc[schedule.game_type.eq('REG')]
    data,_=history(root,season)
    version=artifact['model_version']
    archive_path=root/'learning/live_challengers'/version/'forecasts.json'
    saved=json.loads(archive_path.read_text()) if archive_path.exists() else {}
    production={r['game_id']:r for r in json.loads((root/'public_site/data/predictions.json').read_text())}
    retrospective=[]
    for game in schedule.to_dict('records'):
        x,cutoff=form_features(game,schedule,data)
        if artifact['calibration_through']>=cutoff:
            raise ValueError('Challenger calibration reaches forecast week')
        mean=artifact['model'].predict(x)[0]
        for kind in KINDS:
            key=game['game_id']+'|'+kind
            record={'game_id':game['game_id'],'week':int(game['week']),'kind':kind,'model_version':version,
                'fit_through':artifact['fit_through'],'calibration_through':artifact['calibration_through'],
                'feature_cutoff':cutoff,'features':x.iloc[0].to_dict(),
                **summarize(sample(artifact,mean,kind))}
            # Winner probability is conditional on a decisive game; ties reported separately.
            record['home_win_probability']=record['home_win']/(record['home_win']+record['away_win'])
            future=now<kickoff(game) and pd.isna(game['home_score']) and pd.isna(game['away_score'])
            retrospective.append(record|{'basis':'Current future projection' if future else 'Retrospective; not an archived pregame pick'})
            if future and key not in saved:
                baseline=production.get(game['game_id'],{})
                saved[key]=record|{'forecast_at':now.isoformat(),'kickoff_at_observation':kickoff(game).isoformat(),
                    'production_probability':baseline.get('home_win_probability'),'production_model_version':baseline.get('model_version'),
                    'production_forecast_at':baseline.get('winner_forecast_at')}
    write_json(archive_path,saved)
    records=list(saved.values())
    # Never pool different production versions in matched evaluations.
    cohorts={v:grade([r for r in records if r.get('production_model_version')==v],schedule)
             for v in sorted({r.get('production_model_version') for r in records if r.get('production_model_version')})}
    write_json(root/'public_site/data/live_challengers.json',{'model_version':version,'status':'shadow_only',
        'forecasts':records,'cohorts':cohorts,'note':'First saved pregame forecast per game/challenger version is immutable. Production comparison uses its probability captured at the same observation. Outcomes exclude ties and require pre-kickoff timestamps. Winner probability conditional on no tie. No automatic promotion; prospective sample starts now.'})
    write_json(root/'public_site/data/season_replay.json',{'season':season,'games':len(schedule),'forecasts':retrospective,
        'note':'Separate retrospective reconstruction for completed games, never substituted for archived picks. Frozen pre-2026 fit/calibration, prior-week form features. Future projections use currently observed history, not invented future results.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true'); parser.add_argument('--season',type=int,default=2026)
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    with threadpool_limits(limits=2):
        if args.train:
            train(root,args.season)
        export(root,args.season)
