"""R42/R43/R46/R48: timestamped shadow forecasts, grading, freshness and review."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_io import write_json
from historical_metrics import VIEWS
from matchup_context import source
from opponent_ratings import prepare
from predict_schedule import build_schedule_features
from pregame_inputs import kickoff, timestamp, select
from research_dataset import lagged
from research_operations import read
from score_markets import pregame, comparisons, number
from retrain_updated import probability_metrics
from next_evidence import score_values
from historical_metrics import interval, loss


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_version(version):
    if not version or Path(version).name!=version or version in {'.','..'}:raise ValueError('Generation name required')
    return version


def horizon(game,now):
    remaining=(kickoff(game)-now).total_seconds()/60
    if abs(remaining-1440)<=30:return '24h'
    if abs(remaining-90)<=15:return '90m'
    return 'initial'


def append_forecast(archive,record,game,now):
    if not pregame(game,now) or timestamp(record['forecast_at'])!=now:
        raise ValueError('Only actual pregame observations may be captured')
    if record['horizon']!=horizon(game,now):raise ValueError('Observation is outside the requested horizon')
    if pd.Timestamp(record['training_through']).date()>=kickoff(game).date():
        raise ValueError('Training cutoff overlaps game')
    key='|'.join([record['generation'],record['model'],game['game_id'],record['horizon']])
    if key not in archive:archive[key]=record
    return key


def features(root,schedule,upcoming,report,season):
    history=pd.read_csv(root/'data/nfl_matchups_1999_2025_advanced.csv')
    history=pd.concat([history,schedule.loc[schedule.home_score.notna()&schedule.away_score.notna()]],ignore_index=True)
    core=list(dict.fromkeys(['season','week']+report['features']['charted_updated']))
    if report.get('feature_adapter') == 'expanded_differentials_v1':
        from robust_features import dependency_columns
        core=dependency_columns(core)
    snapshots=pd.read_csv(root/f'data/current_features_{season}.csv').set_index('game_id')
    _,x=build_schedule_features(upcoming,history,{'feature_cols':core},snapshots)
    x.index=upcoming.game_id
    extra=pd.read_csv(root/f'data/opponent_features_{season}.csv').set_index('game_id')
    for c in VIEWS['all']:
        if c in extra:x[c]=extra.reindex(x.index)[c]
    plays=prepare(source(root,'load_pbp'),schedule)
    for g in upcoming.to_dict('records'):
        cutoff=pd.to_datetime(schedule.loc[schedule.week.eq(g['week']),'gameday']).min()
        for side in ['home','away']:
            prior=plays.loc[plays.week.lt(g['week'])&plays.gameday.lt(cutoff)&plays.posteam.eq(g[side+'_team'])]
            x.loc[g['game_id'],side+'_raw_epa']=float(prior.epa.mean()) if len(prior) else np.nan
    research_path=root/'data/research_team_games.csv'
    if research_path.exists():
        historical=pd.read_csv(root/'backtests/source_schedule.csv')
        joined=pd.concat([historical.loc[historical.season.lt(season)],schedule],ignore_index=True).drop_duplicates('game_id',keep='last')
        charted=lagged(joined,pd.read_csv(research_path),upcoming.game_id).set_index('game_id')
        for c in report['features']['charted_updated']:
            if c.startswith('diff_research_'):x[c]=charted.reindex(x.index)[c]
    return x


def freshness(root,games,x,season,now):
    receipt=read(root,f'data/current_features_{season}.json',{}).get('receipts',{})
    current=pd.read_csv(root/f'data/current_features_{season}.csv').set_index('game_id')
    records=list(read(root,'audit/pregame_inputs.json',{'observations':{}})['observations'].values())
    result=[]
    for g in games:
        inputs=[]
        for label,key,fragment,hours in [('EPA','load_pbp','pbp',72),('QB usage','load_player_stats','qb',72),
                                       ('Injuries','load_injuries','injury',24),('Roster','load_rosters_weekly','roster',72)]:
            r=receipt.get(key,{})
            observed=r.get('retrieved_at');age=(now-timestamp(observed)).total_seconds()/3600 if observed else None
            cols=[c for c in current if fragment in c]
            missing=len(cols) if g['game_id'] not in current.index else int(current.loc[g['game_id'],cols].isna().sum())
            inputs.append({'input':label,'observed_at':observed,'provider_published_at':r.get('provider_published_at'),
                           'age_hours':age,'status':'Missing' if not cols or missing==len(cols) else 'Stale receipt' if age is None or age>hours or age<0 else 'Recently retrieved; publication unknown',
                           'missing_fields':missing,'fields':len(cols),'used_by_candidate':True})
        chosen=select(records,g,now)
        for key in ['starter:'+g['home_team'],'starter:'+g['away_team'],'weather']:
            r=chosen.get(key,{});observed=r.get('observed_at')
            age=(now-timestamp(observed)).total_seconds()/3600 if observed else None
            inputs.append({'input':key,'observed_at':observed,'age_hours':age,'provider_published_at':r.get('published_at'),
                           'status':'Missing' if not r else 'Stale observation' if age>24 else 'Observed', 'used_by_candidate':False})
        inputs.append({'input':'Charted/snap metrics','observed_at':None,'age_hours':None,'status':'Frozen research snapshot; freshness unverified','used_by_candidate':True})
        result.append({'game_id':g['game_id'],'inputs':inputs,'model_missing_fields':int(x.loc[g['game_id']].isna().sum())})
    return result


def grade(archive,schedule):
    games=schedule.set_index('game_id').to_dict('index');groups={}
    for r in archive.values():
        g=games.get(r['game_id'])
        if not g or pd.isna(g.get('home_score')) or pd.isna(g.get('away_score')):continue
        if timestamp(r['forecast_at'])>=timestamp(r['kickoff_at_observation']):raise ValueError('Invalid archive timestamp')
        key=(r['generation'],r['model'],r['horizon'],r['task'])
        groups.setdefault(key,[]).append((r,g))
    rows=[]
    for (version,model,h,task),pairs in groups.items():
        row={'generation':version,'model':model,'horizon':h,'task':task,'games':len(pairs),'status':'Descriptive only; no automatic promotion'}
        if task=='winner':
            decisive=[(r,g) for r,g in pairs if g['home_score']!=g['away_score']]
            row['ties_excluded']=len(pairs)-len(decisive);row['graded']=len(decisive)
            if decisive:row.update(probability_metrics(np.array([g['home_score']>g['away_score'] for r,g in decisive]),[r['home_probability'] for r,g in decisive]))
        else:
            predicted=np.array([[r['predicted_home_score'],r['predicted_away_score']] for r,g in pairs])
            actual=np.array([[g['home_score'],g['away_score']] for r,g in pairs])
            row['team_mae']=float(abs(predicted-actual).mean())
            row['margin_mae']=float(abs((predicted[:,0]-predicted[:,1])-(actual[:,0]-actual[:,1])).mean())
            row['total_mae']=float(abs(predicted.sum(axis=1)-actual.sum(axis=1)).mean())
            results=[comparisons(dict(g,game_id=r['game_id']),r) for r,g in pairs]
            for label in ['spread','total']:
                values=[v[label+'_grade'] for v in results]
                row[label+'_correct']=values.count('Correct');row[label+'_wrong']=values.count('Wrong')
                row[label+'_pushes']=values.count('Push');row[label+'_ungraded']=len(values)-sum(values.count(v) for v in ['Correct','Wrong','Push'])
            for key in ['home','away','margin','total']:
                covered=[]
                for r,g in pairs:
                    if key not in r.get('intervals',{}):continue
                    value=score_values(np.array([[g['home_score'],g['away_score']]]))[key][0]
                    low,high=r['intervals'][key];covered.append(low<=value<=high)
                if covered:row[key+'_coverage80']=float(np.mean(covered))
        rows.append(row)
    return rows


def paired_production(archive,schedule):
    games=schedule.set_index('game_id').to_dict('index')
    refs=[r for r in archive.values() if r['model'].startswith('production_')]
    groups={}
    for r in archive.values():
        g=games.get(r['game_id'])
        if r['task']!='winner' or r['model'].startswith('production_') or not g:continue
        if pd.isna(g['home_score']) or pd.isna(g['away_score']) or g['home_score']==g['away_score']:continue
        for ref in refs:
            if all(r[k]==ref[k] for k in ['game_id','horizon','generation']):
                key=(r['generation'],r['model'],ref['model'],r['horizon'])
                y=g['home_score']>g['away_score']
                delta=float(loss(np.array([r['home_probability']]),np.array([y]))[0]-loss(np.array([ref['home_probability']]),np.array([y]))[0])
                groups.setdefault(key,[]).append((g['week'],delta))
    return [{'generation':v,'candidate':m,'production_version':p,'horizon':h,'games':len(items),
             'delta_log_loss':float(np.mean([d for w,d in items])),
             'ci95':interval([d for w,d in items],[w for w,d in items]) if len({w for w,d in items})>=2 else None}
            for (v,m,p,h),items in groups.items()]


def export(root,season=2026,now=None):
    fixed_now=now
    now=timestamp(now or datetime.now(timezone.utc))
    config=read(root,'models/retrained/active_shadow.json')
    if not config:return
    version=safe_version(config['generation']);folder=root/'models/retrained'/version
    report=read(folder,'report.json');ensemble=joblib.load(folder/'temporal_ensemble.joblib')
    production=joblib.load(root/'models/home_win_advanced_automl_no_market_deep.joblib')
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    upcoming=schedule.loc[schedule.apply(lambda g:pregame(g,now) and kickoff(g)<=now+pd.Timedelta(days=8),axis=1)].copy()
    path=root/'audit/shadow_next'/version/'forecasts.json';archive=read(root,path,{})
    live=[];fresh=[]
    if len(upcoming):
        x=features(root,schedule,upcoming,report,season);games=upcoming.to_dict('records')
        fresh=freshness(root,games,x,season,now)
        inputs={}
        for g in games:
            values={c:None if pd.isna(v) else float(v) for c,v in x.loc[g['game_id']].items()}
            digest=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
            inputs[g['game_id']]=digest
            input_path=root/'audit/shadow_next'/version/'inputs'/(digest+'.json')
            if not input_path.exists():write_json(input_path,{'features':values,'observed_at':now.isoformat()})
        predictions=[]
        for m in report['models']:
            artifact_path=folder/(m['name']+'.joblib')
            if sha(artifact_path)!=m['artifact_sha256']:raise ValueError('Shadow artifact integrity failure')
            a=joblib.load(artifact_path);cols=a['feature_cols']
            result=a['model'].predict_proba(x[cols])[:,1] if m['task']=='winner' else np.maximum(0,a['model'].predict(x[cols]))
            predictions.append((m['name'],m['task'],result,m['artifact_sha256']))
        cols=ensemble['feature_cols']
        robust_details=None
        if report.get('ensemble_kind') == 'robust_r60':
            robust_details=ensemble['model'].predict_details(x[cols])
            wp=robust_details['home_probability'];sp=robust_details['scores']
        else:
            wp=np.column_stack([m.predict_proba(x[cols])[:,1] for m in ensemble['winner_models']])@ensemble['winner_weights']
            sp=np.stack([np.maximum(0,m.predict(x[cols])) for m in ensemble['score_models']],axis=-1)@ensemble['score_weights']
        predictions.extend([('ensemble_winner','winner',wp,sha(folder/'temporal_ensemble.joblib')),('ensemble_score','score',sp,sha(folder/'temporal_ensemble.joblib'))])
        for name,task,values,digest in predictions:
            if not np.isfinite(values).all():raise ValueError('Nonfinite shadow prediction')
            if task=='winner' and ((values<0)|(values>1)).any():raise ValueError('Invalid probability')
            for i,g in enumerate(games):
                observed=timestamp(fixed_now or datetime.now(timezone.utc))
                if not pregame(g,observed):continue
                r={'generation':version,'model':name,'task':task,'game_id':g['game_id'],'week':int(g['week']),
                   'home_team':g['home_team'],'away_team':g['away_team'],'horizon':horizon(g,observed),
                   'forecast_at':observed.isoformat(),'kickoff_at_observation':kickoff(g).isoformat(),
                   'training_through':report['training_through'],'artifact_sha256':digest,'input_sha256':inputs[g['game_id']]}
                if task=='winner':r['home_probability']=float(values[i]);r['predicted_winner']=g['home_team'] if values[i]>=.5 else g['away_team']
                else:
                    r.update(predicted_home_score=float(values[i,0]),predicted_away_score=float(values[i,1]),
                             market_home_spread=-float(g['spread_line']) if pd.notna(g.get('spread_line')) else None,
                             market_total=number(g.get('total_line')),line_observed_at=now.isoformat())
                    if name=='ensemble_score':
                        if robust_details is not None:
                            r['intervals']={k:[float(robust_details['score_lower'][i,j]),float(robust_details['score_upper'][i,j])] for j,k in enumerate(['home','away'])}
                            r['intervals'].update({k:[float(robust_details[k+'_lower'][i]),float(robust_details[k+'_upper'][i])] for k in ['total','margin']})
                            r['interval_note']=robust_details['note']
                        else:
                            r['intervals']={k:[max(0,float(v[0]-ensemble['radii'][k])) if k!='margin' else float(v[0]-ensemble['radii'][k]),float(v[0]+ensemble['radii'][k])] for k,v in score_values(values[i:i+1]).items()}
                key=append_forecast(archive,r,g,observed);live.append(archive[key])
        # Freeze contemporaneous production picks for genuinely prospective pairing.
        production_picks={p['game_id']:p for p in read(root,'public_site/data/predictions.json',[])}
        for g in games:
            captured_at=timestamp(fixed_now or datetime.now(timezone.utc))
            if not pregame(g,captured_at):continue
            p=production_picks.get(g['game_id'],{})
            observed=p.get('winner_forecast_at')
            if not observed or timestamp(observed)>=kickoff(g) or timestamp(observed)>now:continue
            r={'generation':version,'model':'production_'+p.get('model_version','legacy'),'task':'winner',
               'game_id':g['game_id'],'week':int(g['week']),'horizon':horizon(g,captured_at),'forecast_at':captured_at.isoformat(),
               'kickoff_at_observation':kickoff(g).isoformat(),'training_through':production['metrics']['final_training_through'],
               'home_probability':float(p['home_win_probability']),'original_forecast_at':observed}
            # This is a copied production observation, not a newly fitted candidate.
            append_forecast(archive,r,g,captured_at)
    write_json(path,archive)
    scorecard=grade(archive,schedule)
    write_json(root/'public_site/data/shadow_next.json',{'generation':version,'forecasts':live,'scorecard':scorecard,
        'prospective_comparisons':paired_production(archive,schedule),
        'archived_observations':len(archive),'note':'Actual pregame forecasts only, next eight days. First observation per model/game/horizon is immutable. Initial, 24h and 90m cohorts are graded separately; model generations never mix. Historical predictions are not backfilled. No automatic promotion. Lines are observed source snapshots, not guaranteed closing lines. Missing lines, pushes and ties are not losses.'})
    write_json(root/'public_site/data/feature_freshness.json',{'games':fresh,'note':'Receipt ages are local observation ages, not provider publication times or proof of current coverage. Stale thresholds: injuries/weather/starters 24h, weekly performance/rosters 72h. Charted inputs remain frozen. Confirmed starters and weather are displayed but NOT features of these models.'})
    candidates=[]
    for task,key in [('winner','validation_log_loss'),('score','validation_team_mae')]:
        m=min((m for m in report['models'] if m['task']==task),key=lambda m:m[key])
        artifact=joblib.load(folder/(m['name']+'.joblib'))
        serving=production if task=='winner' else joblib.load(root/'models/score_models/score_ensemble.joblib')
        compatible=artifact['feature_cols']==serving['feature_cols']
        n=sum(r.get('graded',r['games']) for r in scorecard if r['model']==m['name'] and r['horizon']=='initial')
        candidates.append({'name':m['name'],'task':task,'artifact_sha256':m['artifact_sha256'],'prospective_games':n,
            'feature_contract_compatible':compatible,'review_sample_threshold':50,'approved':False,
            'status':'Blocked: explicit serving adapter and human approval required' if not compatible else 'Human review required',
            'checks':[{'check':'Artifact integrity','passed':sha(folder/(m['name']+'.joblib'))==m['artifact_sha256']},
                      {'check':'At least 50 initial prospective games (review threshold, not statistical proof)','passed':n>=50},
                      {'check':'Production feature contract','passed':compatible}]})
    write_json(root/'public_site/data/release_review.json',{'generation':version,'candidates':candidates,
        'rollback_bundles':[p.parent.name for p in (root/'releases').glob('*/manifest.json')],
        'note':'Read-only public review. No browser approval or activation endpoint. Use the existing local release_control.py snapshot/plan/approve/activate/rollback workflow after evidence review and compatible serving integration. A clone does not include private rollback blobs. No approval has been granted and no live model changed.'})
    print(f'Shadow generation {version}: {len(live)} displayed forecasts, {len(archive)} preserved observations, {len(scorecard)} graded groups',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--season',type=int,default=2026)
    with threadpool_limits(limits=2):export(Path(__file__).resolve().parent,p.parse_args().season)
