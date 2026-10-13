"""Nested chronological candidate research with source-line and Elo benchmarks."""
import argparse
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_io import receipt, write_csv, write_json
from rolling_backtest import prepare, FEATURES


def wilson(correct, n):
    if not n:
        return [None, None]
    p, z = correct/n, 1.96
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-radius), float(center+radius)]


def evaluate(rows, column):
    valid = rows.loc[rows.home_score.ne(rows.away_score) & rows[column].notna()]
    y = valid.home_score.gt(valid.away_score).astype(int).to_numpy()
    p = valid[column].to_numpy()
    if not len(y):
        return {'n': 0, 'accuracy': None, 'log_loss': None, 'brier': None, 'accuracy_ci95': [None, None]}
    correct = int(((p >= .5) == y).sum())
    return {'n': len(y), 'accuracy': correct/len(y), 'log_loss': float(log_loss(y, p, labels=[0,1])),
            'brier': float(np.mean((p-y)**2)), 'accuracy_ci95': wilson(correct, len(y))}


def elo(games):
    ratings, result = {}, {}
    previous_season = None
    for (season, _), week in games.groupby(['season', 'week'], sort=True):
        if season != previous_season:
            ratings = {team: 1500+(value-1500)*.75 for team,value in ratings.items()}
            previous_season = season
        updates = {}
        for g in week.itertuples():
            p = 1/(1+10**((ratings.get(g.away_team,1500)-ratings.get(g.home_team,1500)-55)/400))
            result[g.game_id] = p
            outcome = .5 if g.home_score == g.away_score else float(g.home_score > g.away_score)
            delta = 20*(outcome-p)
            updates[g.home_team] = updates.get(g.home_team,0)+delta
            updates[g.away_team] = updates.get(g.away_team,0)-delta
        for team,delta in updates.items():
            ratings[team] = ratings.get(team,1500)+delta
    return result


def candidate(kind):
    if kind.startswith('rf'):
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
            RandomForestClassifier(n_estimators=64, max_depth=8 if kind=='rf8' else 12,
                min_samples_leaf=8, max_features='sqrt', random_state=42, n_jobs=2))
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
        HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7 if kind=='hgb7' else 15,
            l2_regularization=5, early_stopping=False, random_state=42))


def logistic():
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=.1, max_iter=1000))


def run(root, start=2024, end=2025):
    raw = nfl.load_schedules(list(range(1999, end+1)))
    receipt(root, 'backtest_schedules', raw)
    games = pd.DataFrame(raw.to_dicts())
    games = games.loc[games.game_type.eq('REG') & games.home_score.notna() & games.away_score.notna()].copy()
    write_csv(root/'backtests/source_schedule.csv', games)
    data = prepare(games)
    expanded = pd.read_csv(root/'data/nfl_matchups_1999_2026_expanded.csv')
    groups = {
        'epa': [c for c in expanded if c.startswith('diff_team_pbp_') and c.endswith('last_5')],
        'qb': [c for c in expanded if c.startswith('diff_team_qb_') and c.endswith('last_5')],
        'rest': ['diff_team_rest_days'],
        'ngs': [c for c in expanded if c.startswith('diff_team_ext_ngs_')],
    }
    extra = [c for columns in groups.values() for c in columns]
    data = data.merge(expanded[['game_id']+extra], on='game_id', how='left', validate='one_to_one')
    data['elo_probability'] = data.game_id.map(elo(games))
    features = FEATURES + extra
    forecasts, folds = [], []
    for (season,week), test in data.loc[data.season.between(start,end)].groupby(['season','week']):
        cutoff = test.gameday.min()
        training = data.loc[data.gameday.lt(cutoff) & data.season.ge(max(2017,season-5)) & data.home_score.ne(data.away_score)].copy()
        valid_weeks = training[['season','week']].drop_duplicates().tail(8)
        validation_start = training.merge(valid_weeks, on=['season','week']).gameday.min()
        inner = training.loc[training.gameday.lt(validation_start)]
        validation = training.loc[training.gameday.ge(validation_start)]
        if len(inner)<256 or not len(validation):
            raise ValueError('Insufficient chronological inner training/validation')
        y = lambda frame: frame.home_score.gt(frame.away_score).astype(int)
        losses = {}
        for kind in ['rf8','rf12','hgb7','hgb15']:
            model = candidate(kind).fit(inner[features], y(inner))
            losses[kind] = float(log_loss(y(validation),model.predict_proba(validation[features])[:,1], labels=[0,1]))
        best = min(losses, key=losses.get)
        model = candidate(best).fit(training[features],y(training))
        out = test[['game_id','season','week','home_score','away_score','elo_probability']].copy()
        out['candidate_probability'] = model.predict_proba(test[features])[:,1]
        for omitted in ['none',*groups]:
            cols = [c for c in features if omitted=='none' or c not in groups[omitted]]
            ablation = logistic().fit(training[cols],y(training))
            out['ablation_'+omitted] = ablation.predict_proba(test[cols])[:,1]
        market_train = training.dropna(subset=['spread_line','total_line'])
        market_test = test[['spread_line','total_line']].notna().all(axis=1)
        out['market_probability'] = np.nan
        if len(market_train)>=256 and market_test.any():
            market = logistic().fit(market_train[['spread_line','total_line']],y(market_train))
            out.loc[market_test,'market_probability'] = market.predict_proba(test.loc[market_test,['spread_line','total_line']])[:,1]
        folds.append({'season':int(season),'week':int(week),'test_cutoff':str(cutoff.date()),
            'train_through':str(training.gameday.max().date()), 'inner_train_through':str(inner.gameday.max().date()),
            'validation_start':str(validation_start.date()),'selected':best,'validation_log_loss':losses,
            'training_rows':len(training),'test_rows':len(test)})
        forecasts.append(out)
        print(f'{season} week {week}: {best}', flush=True)
    combined = pd.concat(forecasts,ignore_index=True)
    write_csv(root/'backtests/advanced_weekly_predictions.csv', combined)
    columns = ['candidate_probability','elo_probability','market_probability','ablation_none',*[f'ablation_{g}' for g in groups]]
    report = {'status':'research_only', 'start_season':start,'end_season':end,'folds':folds,
        'features':features,'ablations':groups,'games':len(combined), 'ties':int(combined.home_score.eq(combined.away_score).sum()),
        'metrics':{c:evaluate(combined,c) for c in columns},
        'note':'Nested weekly evaluation: four RF/HGB settings selected on eight earlier weeks, then refit on all earlier data (up to five seasons). Not the production 700-tree model. Ablations hold logistic settings fixed. Elo uses K=20, home advantage 55, offseason retention 75%. These are fixed assumptions. Market benchmark uses revised source lines, not proven pregame lines. Advanced history is revised; missing tie-game features are imputed within training folds. Wilson 95% accuracy intervals assume independent games and do not account for weekly dependence. No automatic promotion.'}
    write_json(root/'public_site/data/advanced_backtest.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-season',type=int,default=2024)
    parser.add_argument('--end-season',type=int,default=2025)
    args=parser.parse_args()
    run(Path(__file__).resolve().parent,args.start_season,args.end_season)
