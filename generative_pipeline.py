"""R79-R84 research builder. Produces a frozen shadow artifact, never production picks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from markov_research import extract_drives, fit, simulate_batch as simulate, summarize as path_summary
from generative_overtime import finalize
from generative_season import simulate as season_simulate
from rolling_backtest import prepare, FEATURES
from score_experiments import fit as fit_score, sample, evaluate
from score_distribution import summarize

KINDS = ['markov', 'ridge_residual', 'poisson']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources(folder):
    folder = Path(folder)
    meta = json.loads((folder/'manifest.json').read_text())
    frames = {}
    for name, item in meta['sources'].items():
        path = folder/item['file']
        if not path.resolve().is_relative_to(folder.resolve()) or digest(path) != item['sha256']:
            raise ValueError('Source integrity failure')
        frames[name] = pd.read_parquet(path)
    schedule = frames['schedule']
    if schedule.game_id.duplicated().any():
        raise ValueError('Duplicate schedule games')
    completed = schedule.loc[schedule.home_score.notna() & schedule.away_score.notna() & schedule.game_type.eq('REG')]
    pbp = frames['pbp'].loc[frames['pbp'].game_id.isin(completed.game_id)].copy()
    coverage = dict(completed_games=len(completed),pbp_games=int(pbp.game_id.nunique()),
                    missing_completed_games=sorted(set(completed.game_id)-set(pbp.game_id)),
                    source_max_week=int(pbp.week.max()), observed_at=meta['sources']['pbp']['observed_at'])
    return pbp, schedule, meta, coverage


def features(history, games):
    """Weekly-frozen recent scores; neither current-week nor future labels enter."""
    out = games.copy()
    for side in ['home','away']:
        values = []
        for g in out.itertuples():
            prior = history.loc[((history.season < g.season) | ((history.season == g.season) & (history.week < g.week))) & history.home_score.notna() & history.away_score.notna()].sort_values(['season','week','gameday','game_id'])
            team = getattr(g, side+'_team')
            prior = prior.loc[prior.home_team.eq(team) | prior.away_team.eq(team)].tail(8)
            points = [(r.home_score,r.away_score) if r.home_team==team else (r.away_score,r.home_score) for r in prior.itertuples()]
            values.append(np.mean(points,axis=0) if points else [21.,21.])
        out[[side+'_for',side+'_against']] = values
    return out


def probs(scores):
    margin = scores[:,0]-scores[:,1]
    return (np.array([(margin>0).sum(),(margin==0).sum(),(margin<0).sum()])+.5)/(len(scores)+1.5)


def weights(prior):
    if len(prior) < 16:
        return np.full(3,1/3)
    p = np.asarray([r['probabilities'] for r in prior])
    y = np.eye(3)[[r['outcome'] for r in prior]]
    result = minimize(lambda w: np.mean(np.sum((np.einsum('nmc,m->nc',p,w)-y)**2,axis=1))+.01*np.sum((w-1/3)**2),
                      np.full(3,1/3),bounds=[(0,1)]*3,constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP')
    if not result.success:
        raise ValueError('Mixture weight optimization failed')
    w = np.maximum(0,result.x)
    return w/w.sum()


def forecast(artifact, game, feature, trials=1000):
    seed = int(hashlib.sha256(game['game_id'].encode()).hexdigest()[:8],16)
    paths = simulate(artifact['markov'],game['home_team'],game['away_team'],trials,seed)
    scores = {'markov':finalize(paths,artifact['markov'],game['home_team'],game['away_team'],seed+1)}
    for i, kind in enumerate(KINDS[1:]):
        model = artifact[kind]
        mean = model['model'].predict(feature[FEATURES])[0]
        scores[kind] = sample(model,mean,trials,seed+10+i).astype(int)
    rng = np.random.default_rng(seed+20)
    component = rng.choice(3,size=trials,p=artifact['weights'])
    mixture = np.stack([scores[k] for k in KINDS],axis=1)[np.arange(trials),component]
    scores['ensemble'] = mixture
    # nflverse spread_line is the expected HOME margin, not a signed home handicap.
    spread = game.get('spread_line'); total = game.get('total_line')
    row = dict(game_id=game['game_id'],home_team=game['home_team'],away_team=game['away_team'],week=int(game['week']),
               weights=dict(zip(KINDS,map(float,artifact['weights']))),trials=trials,
               distributions={k:summarize(v,-spread if pd.notna(spread) else None,total) for k,v in scores.items()},
               markov_regulation=path_summary(paths),spread_line=float(spread) if pd.notna(spread) else None,
               total_line=float(total) if pd.notna(total) else None)
    return row, scores


def matched_summary(rows):
    output = {}
    for name in KINDS+['ensemble']:
        metrics = [r['evaluation'][name] for r in rows if 'evaluation' in r]
        if not metrics:
            continue
        keys = set.union(*(set(m) for m in metrics))
        output[name] = dict(games=len(metrics),**{k:float(np.mean([m[k] for m in metrics if k in m])) for k in sorted(keys)})
    return output


def build(root, source_dir, output, season, trials):
    root, output = Path(root),Path(output)
    if output.exists():
        raise ValueError('Run folder already exists; never overwrite a generation')
    pbp, schedule, manifest, coverage = load_sources(source_dir)
    historical_path = root/'backtests/source_schedule.csv'
    historical = pd.read_csv(historical_path)
    history = pd.concat([historical.loc[historical.season.lt(season)],schedule],ignore_index=True).drop_duplicates('game_id',keep='last')
    data = prepare(history)
    base = data.loc[data.season.between(2017,season-2)]
    calibration = data.loc[data.season.eq(season-1)]
    if len(base)<256 or len(calibration)<128:
        raise ValueError('Insufficient chronological baseline training/calibration')
    with threadpool_limits(limits=1):
        baselines = {kind:fit_score(kind,base,calibration) for kind in KINDS[1:]}
    drives = extract_drives(pbp)
    prior, rows, skipped = [],[],[]
    for week, batch in schedule.loc[schedule.game_type.eq('REG')].sort_values(['week','game_id']).groupby('week'):
        try:
            model = fit(drives,season,week,.35)
        except ValueError:
            skipped.extend(batch.game_id.tolist()); continue
        artifact = dict(markov=model,weights=weights(prior),**baselines)
        inputs = features(history,batch)
        batch_results = []
        for i,g in enumerate(batch.to_dict('records')):
            row,scores = forecast(artifact,g,inputs.iloc[[i]],trials)
            row.update(training_max_week=int(model['drives'].week.max()),weight_training_games=len(prior),
                       generated_at=datetime.now(timezone.utc).isoformat(),kind='revised_source_replay' if pd.notna(g['home_score']) else 'research_projection')
            if pd.notna(g['home_score']) and pd.notna(g['away_score']):
                actual = [g['home_score'],g['away_score']]
                outcome = 0 if actual[0]>actual[1] else 1 if actual[0]==actual[1] else 2
                batch_results.append(dict(probabilities=[probs(scores[k]).tolist() for k in KINDS],outcome=outcome,week=int(week)))
                row['actual'] = actual
                row['evaluation'] = {}
                for kind, draws in scores.items():
                    p = probs(draws)
                    row['evaluation'][kind] = dict(evaluate(draws,*actual,g.get('spread_line'),g.get('total_line')),
                        brier3=float(np.sum((p-np.eye(3)[outcome])**2)),log_loss3=float(-np.log(p[outcome])),
                        decisive_log_loss=None if outcome==1 else float(-np.log(p[outcome]/(p[0]+p[2]))))
                # Missing values are excluded per metric in summaries, not treated as zero.
            rows.append(row)
        prior.extend(batch_results)
        print(f'Week {int(week)} complete: {len(rows)} games',flush=True)
    for r in rows:
        for m in r.get('evaluation',{}).values():
            if m.get('decisive_log_loss') is None:m.pop('decisive_log_loss',None)
    sources = {k:v['sha256'] for k,v in manifest['sources'].items()}
    sources['historical_schedule'] = digest(historical_path)
    sources['implementation'] = hashlib.sha256(b''.join((root/name).read_bytes() for name in ['markov_research.py','generative_overtime.py','generative_pipeline.py','generative_season.py'])).hexdigest()
    version = 'generative-'+hashlib.sha256(json.dumps(dict(sources=sources,trials=trials),sort_keys=True).encode()).hexdigest()[:16]
    # Pin the fitted transition bank and mixture. Cloud inference never refits this artifact.
    next_week = int(schedule.loc[schedule.home_score.isna(),'week'].min()) if schedule.home_score.isna().any() else 19
    pinned = fit(drives,season,next_week,.35)
    frozen = dict(markov=pinned,weights=weights([r for r in prior if r['week']<next_week]),**baselines,
                  model_version=version,source_hashes=sources,training_max_week=int(pinned['drives'].week.max()),season=season)
    future = {r['game_id']:r['distributions']['ensemble'] for r in rows}
    season_result = season_simulate(schedule.loc[schedule.game_type.eq('REG')],future)
    report = dict(model_version=version,status='shadow_only',season=season,source_coverage=coverage,source_hashes=sources,
                  created_at=datetime.now(timezone.utc).isoformat(),games=rows,skipped=skipped,
                  matched=matched_summary(rows),season_trajectories=season_result,
                  baseline_fit_through=str(base.gameday.max().date()),baseline_calibration_through=str(calibration.gameday.max().date()),
                  note='Revised-source retrospective evaluation; not prospective. Three models: semi-Markov, existing Ridge paired-residual and Poisson recipes refitted chronologically. OT stopping rules are tested; possession timing, conversions and field transitions remain approximations. No production promotion.')
    output.mkdir(parents=True)
    joblib.dump(frozen,output/'candidate.joblib')
    report['artifact_sha256'] = digest(output/'candidate.joblib')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    drives.to_csv(output/'transitions.csv',index=False)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--season',type=int,default=2026);p.add_argument('--trials',type=int,default=1000)
    a = p.parse_args()
    result = build(Path(__file__).parent,a.sources,a.output,a.season,a.trials)
    print(json.dumps({'model_version':result['model_version'],'games':len(result['games']),'coverage':result['source_coverage']}))
