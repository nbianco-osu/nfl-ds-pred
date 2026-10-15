"""R28/R29/R38: sourced historical ratings and matched feature-value experiments."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_io import write_csv, write_json
from opponent_ratings import build, prepare as prepare_plays
from rolling_backtest import prepare, FEATURES
from matchup_context import source
from expanded_features import ALIASES

COLUMNS=['game_id','play_id','posteam','defteam','play_type','epa','sack','pass_attempt','rush_attempt','yards_gained','qb_dropback']
VIEWS={'baseline':['home_raw_epa','away_raw_epa'],
       'adjusted':['home_raw_epa','away_raw_epa','diff_net_epa'],
       'sacks':['home_raw_epa','away_raw_epa','diff_sack_rate_differential'],
       'big_plays':['home_raw_epa','away_raw_epa','diff_big_play_rate_differential'],
       'all':['home_raw_epa','away_raw_epa','diff_net_epa','diff_sack_rate_differential','diff_big_play_rate_differential']}


def collect(root,start=2017,end=2026):
    folder=root/'data/historical_metrics';folder.mkdir(parents=True,exist_ok=True)
    schedule_path=folder/'schedule.csv'
    if schedule_path.exists():
        schedule=pd.read_csv(schedule_path)
    else:
        schedule=pd.DataFrame(nfl.load_schedules(list(range(start,end+1))).to_dicts())
        write_csv(schedule_path,schedule)
    # Current local schedule is authoritative for this season's saved analysis snapshot.
    current=root/f'data/schedule_{end}.csv'
    if current.exists():
        schedule=pd.concat([schedule.loc[schedule.season.ne(end)],pd.read_csv(current)],ignore_index=True)
    for side in ['home_team','away_team']:
        schedule[side]=schedule[side].replace(ALIASES)
    blocks=[];coverage=[]
    for year in range(start,end+1):
        cache=folder/f'pbp_{year}.parquet'
        if not cache.exists() or year==end:
            if year==end:
                p=source(root,'load_pbp')
                if p.empty:raise ValueError('Current PBP snapshot missing')
                p[COLUMNS].to_parquet(cache,index=False)
            else:
                for attempt in range(3):
                    try:
                        nfl.load_pbp([year]).select(COLUMNS).write_parquet(cache)
                        break
                    except ConnectionError:
                        if attempt==2:raise
                        time.sleep(2**attempt)
        p=pd.read_parquet(cache)
        p[['posteam','defteam']]=p[['posteam','defteam']].replace(ALIASES)
        games=schedule.loc[schedule.season.eq(year)&schedule.game_type.eq('REG')].copy()
        features,report=build(p,games,year)
        plays=prepare_plays(p,games)
        for side in ['home','away']:
            values=[]
            for game in games.set_index('game_id').loc[features.game_id].to_dict('records'):
                cutoff=pd.to_datetime(games.loc[games.week.eq(game['week']),'gameday']).min()
                prior=plays.loc[plays.week.lt(game['week'])&plays.gameday.lt(cutoff)&plays.posteam.eq(game[side+'_team'])]
                values.append(float(prior.epa.mean()) if prior.epa.notna().any() else np.nan)
            features[side+'_raw_epa']=values
        blocks.append(features)
        write_csv(folder/f'features_{year}.csv',features)
        coverage.append({'season':year,'scheduled_games':len(games),'pbp_games':report['source_games'],
                         'plays':report['source_plays'],'missing_completed_games':report['missing_completed_game_ids'],
                         'pbp_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()})
        print(f'R28 {year}: {report["source_games"]} PBP games',flush=True)
    combined=pd.concat(blocks,ignore_index=True)
    write_csv(root/'data/historical_opponent_features.csv',combined)
    report={'seasons':coverage,'source':'nflverse via nflreadpy','schedule_sha256':hashlib.sha256(schedule.to_csv(index=False).encode()).hexdigest(),
            'note':'Revised historical PBP snapshots; not original publication-time evidence. Regular season, within-season prior-week features. Week 1 is missing, not invented. Fixed 200-play shrinkage, positive defense/net is better. Raw EPA baseline is prior-week offensive EPA. No DVOA data.'}
    write_json(root/'public_site/data/historical_coverage.json',report)
    return schedule,combined


def loss(p,y):
    p=np.clip(p,1e-8,1-1e-8)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def interval(delta,clusters,seed=42):
    groups=[np.asarray(delta)[np.asarray(clusters)==c] for c in sorted(set(clusters))]
    rng=np.random.default_rng(seed)
    samples=[np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(1000)]
    return np.quantile(samples,[.025,.975]).tolist()


def evaluate(root,schedule,features):
    data=prepare(schedule).merge(features.drop(columns=['season','week','feature_cutoff']),on='game_id',validate='one_to_one')
    results=[];folds=[];stress=[]
    for year in [2024,2025,2026]:
        train=data.loc[data.season.between(2017,year-1)]
        test=data.loc[data.season.eq(year)]
        decisive=train.loc[train.home_score.ne(train.away_score)]
        if test.empty or train.gameday.max()>=test.feature_cutoff.min():raise ValueError('Invalid chronological fold')
        y=decisive.home_score.gt(decisive.away_score).astype(int)
        folds.append({'season':year,'train_through':str(train.gameday.max().date()),'test_start':str(test.gameday.min().date()),'games':len(test)})
        for name,extra in VIEWS.items():
            cols=FEATURES+extra
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=1500))
            score=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=10))
            model.fit(decisive[cols],y);score.fit(train[cols],train[['home_score','away_score']])
            p=model.predict_proba(test[cols])[:,1];s=np.maximum(0,score.predict(test[cols]))
            for i,g in enumerate(test.itertuples()):
                results.append({'game_id':g.game_id,'season':year,'week':int(g.week),'view':name,'home_probability':float(p[i]),
                    'home_score':g.home_score,'away_score':g.away_score,'predicted_home_score':float(s[i,0]),'predicted_away_score':float(s[i,1])})
            if name=='all':
                mask=test.home_score.ne(test.away_score);labels=test.loc[mask].home_score.gt(test.loc[mask].away_score).to_numpy()
                for removed in ['none','epa','sacks','big_plays','all_new']:
                    x=test[cols].copy()
                    drop=[] if removed=='none' else [c for c in extra if removed=='all_new' or ('epa' in c if removed=='epa' else 'sack' in c if removed=='sacks' else 'big_play' in c)]
                    # Unavailable fields use training medians, not zero or future values.
                    x[drop]=np.nan
                    ps=model.predict_proba(x)[:,1]
                    stress.append({'season':year,'scenario':removed,'games':int(mask.sum()),
                        'log_loss':float(loss(ps[mask],labels).mean()),'mean_probability_shift':float(abs(ps-p).mean())})
    frame=pd.DataFrame(results);models=[]
    baseline=frame.loc[frame.view.eq('baseline')].set_index('game_id')
    for view,group in frame.groupby('view'):
        d=group.loc[group.home_score.ne(group.away_score)]
        labels=d.home_score.gt(d.away_score).to_numpy();p=d.home_probability.to_numpy()
        l=loss(p,labels);b=loss(baseline.loc[d.game_id].home_probability.to_numpy(),labels)
        clusters=(d.season.astype(str)+'-'+d.week.astype(str)).to_numpy()
        models.append({'view':view,'games':len(group),'decisive_games':len(d),'log_loss':float(l.mean()),
            'brier':float(((p-labels)**2).mean()),'delta_log_loss':float((l-b).mean()),'delta_ci95':interval(l-b,clusters),
            'team_score_mae':float((abs(group.predicted_home_score-group.home_score).sum()+abs(group.predicted_away_score-group.away_score).sum())/(2*len(group)))})
    write_csv(root/'backtests/feature_value_predictions.csv',frame)
    write_json(root/'public_site/data/feature_value.json',{'models':models,'folds':folds,
        'note':'Identical fixed logistic/Ridge pipelines and chronological annual folds; baseline = prior-eight-game score form + raw offensive EPA. Each added view differs only in features. Median imputation/scaling fit only on training. Paired 95% intervals resample season-week clusters (1,000 draws, seed42). No multiple-comparison correction; designed after historical seasons. Retrospective evidence only; no automatic promotion.',
        'features_sha256':hashlib.sha256((root/'data/historical_opponent_features.csv').read_bytes()).hexdigest()})
    write_json(root/'public_site/data/robustness.json',{'scenarios':stress,'note':'Fixed all-feature research classifier, no refitting. Missing groups are replaced through the fitted training imputer. This simulates unavailable fields, not causal injury effects or measured live outages. Same test games within each season.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',type=int,default=2017);parser.add_argument('--end',type=int,default=2026)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    with threadpool_limits(limits=2):
        schedule,features=collect(root,args.start,args.end);evaluate(root,schedule,features)
