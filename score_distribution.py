"""Shadow-only paired-residual score distributions with prequential coverage checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_io import write_json, write_csv
from rolling_backtest import FEATURES, prepare
from score_markets import pregame


def fit(rows):
    return make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(rows[FEATURES],rows[['home_score','away_score']])


def draws(mean, residuals, seed=42, trials=10000):
    rng=np.random.default_rng(seed)
    errors=np.asarray(residuals,dtype=float)
    if len(errors)<128 or errors.shape[1]!=2 or not np.isfinite(errors).all():
        raise ValueError('Need 128 finite paired out-of-sample residuals')
    # Paired residual resampling preserves empirical home-away error dependence.
    return np.maximum(0,np.rint(np.asarray(mean)+errors[rng.integers(0,len(errors),trials)]))


def summarize(samples, spread=None, total=None):
    home,away=samples.T
    margin,points=home-away,home+away
    result={'expected_home':float(home.mean()),'expected_away':float(away.mean()),
        'home_interval80':np.quantile(home,[.1,.9]).tolist(),'away_interval80':np.quantile(away,[.1,.9]).tolist(),
        'margin_interval80':np.quantile(margin,[.1,.9]).tolist(),'total_interval80':np.quantile(points,[.1,.9]).tolist(),
        'home_win':float((margin>0).mean()),'away_win':float((margin<0).mean()),'tie':float((margin==0).mean())}
    for name,values,line in [('spread',margin, -spread if spread is not None and pd.notna(spread) else None),('total',points,total)]:
        result[name]=None if line is None or pd.isna(line) else {'above':float((values>line).mean()),'below':float((values<line).mean()),'push':float((values==line).mean())}
    return result


def train(root):
    source=pd.read_csv(root/'backtests/source_schedule.csv')
    current=pd.read_csv(root/'data/nfl_2026_completed.csv')
    source=pd.concat([source,current]).drop_duplicates('game_id',keep='last')
    data=prepare(source)
    residuals=[]
    evaluations=[]
    for (season,week),test in data.loc[data.season.ge(2022)].groupby(['season','week']):
        training=data.loc[data.gameday.lt(test.gameday.min())]
        means=fit(training).predict(test[FEATURES])
        if len(residuals)>=128:
            for i,game in enumerate(test.itertuples()):
                spread=-float(game.spread_line) if pd.notna(game.spread_line) else None
                total=float(game.total_line) if pd.notna(game.total_line) else None
                distribution=summarize(draws(means[i],residuals[-512:]),spread,total)
                actual={'home':game.home_score,'away':game.away_score,'margin':game.home_score-game.away_score,'total':game.home_score+game.away_score}
                evaluated={'game_id':game.game_id,'season':int(season),'week':int(week),
                    **{key+'_covered80':distribution[key+'_interval80'][0]<=value<=distribution[key+'_interval80'][1] for key,value in actual.items()}}
                for name,value,line in [('spread',actual['margin'],-spread if spread is not None else None),('total',actual['total'],total)]:
                    if line is not None:
                        outcome='above' if value>line else 'below' if value<line else 'push'
                        p=distribution[name]
                        evaluated[name+'_brier3']=sum((prob-float(key==outcome))**2 for key,prob in p.items())
                        evaluated[name+'_log_loss']=-float(np.log(max(p[outcome],1e-12)))
                evaluations.append(evaluated)
        residuals.extend((test[['home_score','away_score']].to_numpy()-means).tolist())
    through=data.gameday.max().date().isoformat()
    artifact={'model':fit(data),'residuals':residuals[-512:],'training_through':through,
        'model_version':'shadow-ridge-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), 'feature_cols':FEATURES,
        'status':'shadow_only','training_rows':len(data)}
    folder=root/'models/research'
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact,folder/'score_distribution.joblib')
    results=pd.DataFrame(evaluations)
    write_csv(root/'backtests/distribution_coverage.csv',results)
    report={'model_version':artifact['model_version'],'training_through':through,'status':'shadow_only',
        'evaluation_games':len(results),'coverage80':{key:float(results[key+'_covered80'].mean()) for key in ['home','away','margin','total']},
        'line_evaluation':{name:{'games':int(results[name+'_brier3'].notna().sum()),'three_class_brier':float(results[name+'_brier3'].mean()),'log_loss':float(results[name+'_log_loss'].mean())} for name in ['spread','total']},
        'note':'Research Ridge score model, not the serving ensemble. Weekly refits; coverage uses only residuals from earlier held-out weeks. Paired empirical residuals, 10,000 draws, seed 42; rounded and clipped to nonnegative scores. Approximate score distribution, not a football scoring-process model. Three-class Brier is the summed squared error for above/below/push; lower is better. Historical market evaluation uses revised source lines, not verified pregame snapshots. Homogeneous errors and integer rounding can miscalibrate cover/over/push probabilities. Prospective evaluation starts only after archiving; never backfilled.'}
    write_json(root/'public_site/data/distribution_metrics.json',report)
    print(json.dumps(report,indent=2))


def export(root,schedule=None):
    path=root/'models/research/score_distribution.joblib'
    if not path.exists():
        return
    artifact=joblib.load(path)
    archive_path=root/'predictions/score_distribution_archive.json'
    archive=json.loads(archive_path.read_text()) if archive_path.exists() else {}
    def preserve_cohorts(records):
        for version in sorted({r['model_version'] for r in records.values()}):
            if not version.replace('-','').isalnum():
                raise ValueError('Invalid score-distribution generation')
            path=root/'learning/score_distributions'/version/'forecasts.json'
            saved=json.loads(path.read_text()) if path.exists() else {}
            for key,value in records.items():
                if value['model_version']==version:
                    saved.setdefault(key,value)
            write_json(path,saved)
    preserve_cohorts(archive)
    schedule=pd.read_csv(root/'data/schedule_2026.csv') if schedule is None else schedule
    history=pd.concat([pd.read_csv(root/'backtests/source_schedule.csv'),pd.read_csv(root/'data/nfl_2026_completed.csv')]).drop_duplicates('game_id',keep='last')
    history['gameday']=pd.to_datetime(history.gameday)
    now=datetime.now(timezone.utc)
    for game in schedule.to_dict('records'):
        if not pregame(game,now):
            continue
        cutoff=pd.to_datetime(schedule.loc[schedule.week.eq(game['week']),'gameday']).min()
        fields=[]
        for team in [game['home_team'],game['away_team']]:
            prior=history.loc[history.gameday.lt(cutoff)&(history.home_team.eq(team)|history.away_team.eq(team))].sort_values('gameday').tail(8)
            pf=np.where(prior.home_team.eq(team),prior.home_score,prior.away_score)
            pa=np.where(prior.home_team.eq(team),prior.away_score,prior.home_score)
            fields.extend([float(np.mean(pf)),float(np.mean(pa))] if len(prior) else [21.,21.])
        mean=artifact['model'].predict(pd.DataFrame([fields],columns=FEATURES))[0]
        spread=-float(game['spread_line']) if pd.notna(game.get('spread_line')) else None
        total=float(game['total_line']) if pd.notna(game.get('total_line')) else None
        result=summarize(draws(mean,artifact['residuals']),spread,total)
        record={**result,'model_version':artifact['model_version'],'training_through':artifact['training_through'],
                'market_home_spread':spread,'market_total':total,'game_id':game['game_id']}
        old=archive.get(game['game_id'],{})
        old_values={k:v for k,v in old.items() if k!='forecast_at'}
        record['forecast_at']=old.get('forecast_at') if old_values==record else now.isoformat()
        archive[game['game_id']]=record
    write_json(archive_path,archive)
    preserve_cohorts(archive)
    write_json(root/'public_site/data/score_distributions.json',archive)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    if args.train:
        train(root)
    export(root)
