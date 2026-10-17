"""R62-R66 immutable training vintages, fixed reviews, pruning and release rehearsal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
import pandas as pd

from audit_io import write_json,write_csv
from robust_features import build_live_features
from retrain_updated import load_data
from score_markets import pregame


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def vintage(root,season,cols):
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    completed=schedule.loc[schedule.game_type.eq('REG')&schedule.home_score.notna()&schedule.away_score.notna()].copy()
    if completed.game_id.duplicated().any():raise ValueError('Duplicate completed games')
    original,source_hashes=load_data(root)
    features=build_live_features(root,schedule,completed,cols,season)
    new=completed[['game_id','season','week','gameday','home_score','away_score']].set_index('game_id').join(features).reset_index()
    old=original.loc[original.season.lt(season),new.columns]
    table=pd.concat([old,new],ignore_index=True)
    table['gameday']=pd.to_datetime(table.gameday)
    table=table.sort_values(['gameday','game_id'])
    if table.game_id.duplicated().any():raise ValueError('Duplicate training vintage rows')
    sources={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/f'data/current_features_{season}.csv',root/f'data/opponent_features_{season}.csv',root/f'data/schedule_{season}.csv',root/'audit/sources/load_pbp/latest.json']}
    sources.update(source_hashes)
    sources['audit/sources/load_pbp/latest.json']=read(root/'audit/sources/load_pbp/latest.json')['sha256']
    csv=table.to_csv(index=False).encode();identity=hashlib.sha256(csv+json.dumps(sources,sort_keys=True).encode()).hexdigest()
    folder=root/'data/training_vintages'/identity
    report=dict(vintage=identity,observed_as_of=datetime.now(timezone.utc).isoformat(),sources=sources,
        rows=len(table),current_season_games=len(new),new_completed_games=sorted(set(new.game_id)-set(original.game_id)),
        missing_completed_games=sorted(set(completed.game_id)-set(new.game_id)),training_through=str(pd.to_datetime(new.gameday).max().date()),
        missing_fields={c:int(new[c].isna().sum()) for c in cols},production_retrained=False,
        note='Immutable observed-as-of revised-source snapshot, not proof of original pregame publication. '
             'Performance uses earlier weeks; injury/roster sources may be revised. Charted features excluded from this 25-core-feature table. '
             'Historical source tie gaps retained. Does not overwrite original research tables or fit any model.')
    if not folder.exists():
        write_csv(folder/'core.csv',table);write_json(folder/'manifest.json',report)
    else:
        if (folder/'core.csv').read_bytes()!=csv:raise ValueError('Vintage collision')
        report=read(folder/'manifest.json')
    return report,table


def review(root,version,schedule):
    policy=read(root/'models/retrained'/version/'review_policy.json')
    if not policy:return {'status':'No predeclared policy'}
    archive=read(root/'audit/shadow_next'/version/'forecasts.json',{})
    records=list(archive.values());games=schedule.set_index('game_id').to_dict('index');pairs=[]
    for candidate in records:
        if candidate.get('model')!=policy['candidate'] or candidate.get('horizon')!=policy['horizon']:continue
        if candidate.get('generation')!=version:raise ValueError('Mixed candidate generation')
        if policy.get('declared_at') and pd.Timestamp(candidate['forecast_at'])<pd.Timestamp(policy['declared_at']):raise ValueError('Forecast predates review policy')
        game=games.get(candidate['game_id'],{})
        if pd.isna(game.get('home_score')) or pd.isna(game.get('away_score')) or game.get('home_score')==game.get('away_score'):continue
        refs=[r for r in records if r['game_id']==candidate['game_id'] and r['horizon']==candidate['horizon'] and r['generation']==version and r['model'].startswith('production_')]
        if len(refs)!=1:continue
        reference=refs[0]
        if any(pd.Timestamp(r['forecast_at'])>=pd.Timestamp(r['kickoff_at_observation']) for r in [candidate,reference]):raise ValueError('Late review observation')
        y=int(game['home_score']>game['away_score'])
        def loss(p):
            p=np.clip(p,1e-6,1-1e-6);return -y*np.log(p)-(1-y)*np.log(1-p)
        pairs.append(dict(game_id=candidate['game_id'],week=int(game['week']),delta=float(loss(candidate['home_probability'])-loss(reference['home_probability'])),reference=reference['model']))
    if len({p['game_id'] for p in pairs})!=len(pairs):raise ValueError('Repeated review label')
    looks=[]
    for week,minimum in zip(policy['review_weeks'],policy['minimum_games']):
        path=root/'audit/fixed_reviews'/version/f'week_{week}.json'
        if path.exists():looks.append(read(path));continue
        target=schedule.loc[schedule.week.eq(week)&schedule.game_type.eq('REG')]
        if target.empty or target[['home_score','away_score']].isna().any().any():continue
        cohort=[p for p in pairs if p['week']<=week]
        weeks=sorted({p['week'] for p in cohort});refs={p['reference'] for p in cohort}
        result=dict(review_week=week,games=len(cohort),weeks=len(weeks),status='Insufficient matched evidence',automatic_promotion=False)
        if len(cohort)>=minimum and len(weeks)>=policy['minimum_weeks'] and len(refs)==1:
            rng=np.random.default_rng(42+week);groups=[np.array([p['delta'] for p in cohort if p['week']==w]) for w in weeks]
            means=[float(np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]).mean()) for _ in range(10000)]
            alpha=policy['familywise_alpha']/policy['looks'];ci=np.quantile(means,[alpha/2,1-alpha/2]).tolist()
            result.update(delta_log_loss=float(np.mean([p['delta'] for p in cohort])),adjusted_ci=ci,
                status='Ready for human review' if ci[1]<0 else 'No demonstrated advantage',production_version=next(iter(refs)))
        elif len(refs)>1:result['status']='Blocked: mixed production reference versions'
        write_json(path,result);looks.append(result)
    return dict(policy=policy,matched_games=len(pairs),looks=looks,ready_for_review=any(x['status']=='Ready for human review' for x in looks),
        note='Only predeclared 90-minute ensemble-versus-production pairs. Ties excluded. Fixed completed-week looks; immutable results, no repeated peeking promotion.')


def prune_oof(frame,task,threshold=.98,max_members=4):
    rows=frame.loc[frame.model.str.contains('_'+task+'_')]
    if rows.empty or not rows.season.isin([2022,2023,2024]).all():raise ValueError('Training-only 2022-2024 OOF rows required')
    if rows.duplicated(['model','game_id']).any():raise ValueError('Duplicate OOF rows')
    meta=rows.drop_duplicates('game_id').set_index('game_id')
    key='home_probability' if task=='winner' else 'predicted_home_score'
    matrix=rows.pivot(index='game_id',columns='model',values=key)
    if matrix.isna().any().any():raise ValueError('Unmatched pruning games')
    y=meta.loc[matrix.index].home_score.gt(meta.loc[matrix.index].away_score).astype(int).to_numpy() if task=='winner' else meta.loc[matrix.index].home_score.to_numpy()
    errors=matrix.to_numpy()-y[:,None]
    if task=='score':
        away=rows.pivot(index='game_id',columns='model',values='predicted_away_score').reindex(index=matrix.index,columns=matrix.columns)
        if away.isna().any().any():raise ValueError('Unmatched away scores')
        errors=np.concatenate([errors,away.to_numpy()-meta.loc[matrix.index].away_score.to_numpy()[:,None]])
    risk=(errors**2).mean(axis=0);corr=np.nan_to_num(np.corrcoef(errors,rowvar=False),nan=0.)
    selected=[];excluded=[]
    for i in np.argsort(risk):
        reason='correlated_errors' if any(abs(corr[i,j])>=threshold for j in selected) else 'latency_roster_cap'
        if len(selected)<max_members and not any(abs(corr[i,j])>=threshold for j in selected):selected.append(int(i))
        else:excluded.append({'model':matrix.columns[i],'reason':reason})
    return dict(selected=[matrix.columns[i] for i in selected],excluded=excluded,error_correlation_threshold=threshold,
        training_seasons=[2022,2023,2024],games=len(matrix),note='Prunes the 6 predeclared ensemble members per task, not all 240 configurations. Ranking uses OOF squared error; no 2026 test labels. Recommendation only; serving weights unchanged.')


def rehearsal(candidate,frame,production,version):
    before=hashlib.sha256(json.dumps(production,sort_keys=True).encode()).hexdigest()
    start=time.perf_counter();details=candidate.predict_details(frame);elapsed=(time.perf_counter()-start)*1000
    timings=[]
    for _ in range(10):
        start=time.perf_counter();candidate.predict_details(frame.head(1));timings.append((time.perf_counter()-start)*1000)
    champion={r['game_id']:r for r in production}
    shifts=[abs(float(details['home_probability'][i])-float(champion[game]['home_win_probability'])) for i,game in enumerate(frame.index) if game in champion]
    with tempfile.TemporaryDirectory(prefix='nfl-selector-rehearsal-') as directory:
        path=Path(directory)/'selector.json';write_json(path,{'active':'champion'})
        saved=path.read_bytes();write_json(path,{'active':version,'mode':'canary_simulation'})
        path.write_bytes(saved)
        restored=read(path)=={'active':'champion'}
    after=hashlib.sha256(json.dumps(production,sort_keys=True).encode()).hexdigest()
    return dict(protocol_version=2,status='Dry-run only',games=len(frame),batch_latency_ms=elapsed,per_game_ms=elapsed/max(len(frame),1),
        single_game_p50_ms=float(np.median(timings)),single_game_p95_ms=float(np.quantile(timings,.95)),latency_samples=10,
        proposed_latency_budget_ms=250,within_proposed_latency_budget=bool(np.quantile(timings,.95)<=250),
        matched_champion_games=len(shifts),max_probability_shift=max(shifts) if shifts else None,
        selector_rollback_passed=restored,production_input_unchanged=before==after,
        finite_outputs=bool(np.isfinite(details['scores']).all() and np.isfinite(details['home_probability']).all()),
        production_activation=False,note='Disposable selector switch/rollback rehearsal, not a live production cutover. Explicit approval and prospective evidence remain required.')


def guarded_inference(candidate,champion,artifact_sha256,approval=None):
    """Explicit serving hook; unapproved or failed canaries use the champion."""
    valid=bool(approval and approval.get('artifact_sha256')==artifact_sha256 and
               approval.get('approved') is True and approval.get('reviewer') and approval.get('approved_at'))
    if not valid:return {'route':'champion','reason':'No matching explicit approval','forecast':champion()}
    try:return {'route':'candidate','forecast':candidate()}
    except (ValueError,RuntimeError,KeyError) as exc:
        return {'route':'champion','reason':'Candidate rejected: '+type(exc).__name__,'forecast':champion()}


def export(root,season=2026,run_research=False):
    root=Path(root);active=read(root/'models/retrained/active_shadow.json',{})
    version=active.get('generation');folder=root/'models/retrained'/str(version)
    report=read(folder/'report.json',{})
    if report.get('ensemble_kind')!='robust_r60':return
    artifact=joblib.load(folder/'temporal_ensemble.joblib');cols=artifact['feature_cols']
    manifest,table=vintage(root,season,cols)
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    output=read(root/'public_site/data/iteration_r61.json',{})
    if output.get('generation')!=version:output={}
    previous_vintage=output.get('training_vintage',{}).get('vintage')
    output.update(generation=version,training_vintage=manifest,fixed_review=review(root,version,schedule))
    now=datetime.now(timezone.utc);upcoming=schedule.loc[schedule.apply(lambda g:pregame(g,now),axis=1)].head(16)
    if len(upcoming) and (run_research or previous_vintage!=manifest['vintage'] or output.get('canary',{}).get('protocol_version')!=2):
        live=build_live_features(root,schedule,upcoming,cols,season)
        output['canary']=rehearsal(artifact['model'],live,read(root/'public_site/data/predictions.json',[]),version)
        output['canary']['max_missing_fraction']=float(live.isna().mean(axis=1).max())
    if run_research:
        ensemble=report['ensemble_generation'];efolder=root/'models/retrained'/ensemble
        oof=pd.read_csv(efolder/'oof_predictions.csv')
        output['pruning']={task:prune_oof(oof,task) for task in ['winner','score']}
        from matchup_context import source
        from possession_research import evaluate
        evaluation=joblib.load(efolder/'evaluation.joblib')['model']
        output['possession_research']=evaluate(source(root,'load_pbp'),table.loc[table.season.eq(season)],evaluation,cols)
        output['research_source_vintage']=manifest['vintage']
    output['research_is_stale']=output.get('research_source_vintage')!=manifest['vintage']
    output['note']='R61-R66 research and operational evidence; no production promotion or automatic retraining. Old input archives remain immutable. Model binaries stay private.'
    write_json(root/'public_site/data/iteration_r61.json',output)
    return output
