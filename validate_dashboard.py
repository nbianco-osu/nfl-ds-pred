"""Dependency-free publication gate for exported forecast data."""
import json
import math
from datetime import datetime
from pathlib import Path


def validate(root):
    folder=Path(root)/'public_site/data'
    for path in folder.glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Invalid JSON number: {value}')))
    rows=json.loads((folder/'predictions.json').read_text())
    if not rows or len({r['game_id'] for r in rows})!=len(rows):
        raise ValueError('Missing or duplicate forecasts')
    for r in rows:
        p,q=r['home_win_probability'],r['away_win_probability']
        if not all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in [p,q]) or abs(p+q-1)>1e-8:
            raise ValueError('Invalid win probability')
        if r['predicted_winner'] not in [r['home_team'],r['away_team']]:
            raise ValueError('Invalid winner team')
        if r['status']=='Final':
            h,a=r['home_score'],r['away_score']
            if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 and v==int(v) for v in (h,a)):
                raise ValueError('Invalid completed actual score')
            expected='Tie (no winner)' if h==a else 'Yes' if r['predicted_winner']==(r['home_team'] if h>a else r['away_team']) else 'No'
            if r['prediction_result']!=expected:
                raise ValueError('Incorrect outcome grading')
        for field in ['predicted_home_score','predicted_away_score']:
            if r.get(field) is not None and (not math.isfinite(r[field]) or r[field]<0):
                raise ValueError('Invalid expected score')
    audit_path=folder/'game_audit.json'
    if audit_path.exists():
        audits=json.loads(audit_path.read_text())
        for r in rows:
            audit=audits.get(r['game_id'])
            if audit and (audit['model_version']!=r['model_version'] or abs(audit['home_probability']-r['home_win_probability'])>1e-8):
                raise ValueError('Explanation is not for the displayed forecast')
    dist_path=folder/'score_distributions.json'
    if dist_path.exists():
        for d in json.loads(dist_path.read_text()).values():
            if abs(d['home_win']+d['away_win']+d['tie']-1)>1e-9:
                raise ValueError('Incoherent score outcome distribution')
            for field in ['spread','total']:
                if d[field] and abs(sum(d[field].values())-1)>1e-9:
                    raise ValueError('Incoherent line probabilities')
    history_path=folder/'forecast_history.json'
    if history_path.exists():
        for game_id, records in json.loads(history_path.read_text())['games'].items():
            if game_id not in {r['game_id'] for r in rows}:
                raise ValueError('Unknown game in forecast history')
            if len({r['id'] for r in records}) != len(records):
                raise ValueError('Duplicate history observations')
            for r in records:
                if not 0 <= r['probability'] <= 1:
                    raise ValueError('Invalid history probability')
                if datetime.fromisoformat(r['forecast_at'].replace('Z','+00:00')).tzinfo is None:
                    raise ValueError('History timestamp lacks timezone')
    matched_path=folder/'matched_comparison.json'
    if matched_path.exists():
        matched=json.loads(matched_path.read_text())
        models=matched['models']
        if matched['status']!='research_only' or len(models)!=64 or len({m['id'] for m in models})!=64:
            raise ValueError('Incomplete matched model registry')
        for model in models:
            if model['overall']['n']!=matched['games']:
                raise ValueError('Mismatched evaluation counts')
            if sum(m['n'] for m in model['seasons'].values())!=matched['games']:
                raise ValueError('Invalid evaluation season totals')
            for year, summary in model['seasons'].items():
                if summary['n']!=matched['baseline_seasons'][year]['n']:
                    raise ValueError('Mismatched season cohort')
            for summary in [model['overall'], *model['seasons'].values()]:
                if not 0<=summary['accuracy']<=1 or not 0<=summary['brier']<=1 or summary['log_loss']<0:
                    raise ValueError('Invalid evaluation metric')
                for key in ['delta_log_loss_ci95','delta_brier_ci95']:
                    low,high=summary[key]
                    if not math.isfinite(low) or not math.isfinite(high) or low>high:
                        raise ValueError('Invalid paired interval')
        for fold in matched['folds']:
            if not fold['final_fit_through']<fold['selection_start'] or not fold['final_calibration_through']<fold['test_start']:
                raise ValueError('Evaluation fold overlap')
    experiments=folder/'score_experiments.json'
    if experiments.exists():
        report=json.loads(experiments.read_text())
        if report['status']!='research_only' or len(report['models'])!=4:
            raise ValueError('Invalid score research registry')
        expected=sum(f['games'] for f in report['folds'])
        for fold in report['folds']:
            if not fold['fit_through']<fold['calibration_through']<fold['test_start']:
                raise ValueError('Score research dates overlap')
        for model in report['models']:
            if model['overall']['games']!=expected:
                raise ValueError('Score research cohorts differ')
            for summary in [model['overall'],*model['seasons'].values()]:
                for key,value in summary['metrics'].items():
                    if value is not None and (not math.isfinite(value) or value<0 or (key.endswith('coverage80') and value>1)):
                        raise ValueError('Invalid score experiment metric')
    inputs=folder/'pregame_inputs.json'
    if inputs.exists():
        records=json.loads(inputs.read_text())
        if len({g['game_id'] for g in records['games']})!=len(records['games']):
            raise ValueError('Duplicate pregame archive game')
        for game in records['games']:
            for record in game['observations'].values():
                observed=datetime.fromisoformat(record['observed_at'])
                if record.get('published_at') is None and not (record.get('provider')=='Open-Meteo' and record.get('timestamp_basis')):
                    raise ValueError('Unknown input publication basis')
                published=datetime.fromisoformat(record.get('published_at') or record['observed_at'])
                kickoff=datetime.fromisoformat(record['kickoff_at_observation'])
                if any(t.tzinfo is None for t in [observed,published,kickoff]) or not published<=observed<kickoff:
                    raise ValueError('Invalid pregame input provenance')
    stack=folder/'temporal_stack.json'
    if stack.exists():
        report=json.loads(stack.read_text())
        if report['status']!='research_only' or not report['meta_fit_season']<report['calibration_season']<report['test_season']:
            raise ValueError('Invalid temporal stacking protocol')
        if report['summary']['n']!=report['baseline']['n']:
            raise ValueError('Stack comparator cohort differs')
    for name in ['live_challengers','season_replay']:
        path=folder/(name+'.json')
        if not path.exists():
            continue
        report=json.loads(path.read_text())
        seen=set()
        for record in report['forecasts']:
            key=(record['game_id'],record['kind'],record['model_version'])
            if key in seen:
                raise ValueError('Repeated challenger forecast')
            seen.add(key)
            if not 0<=record['home_win_probability']<=1 or abs(record['home_win']+record['away_win']+record['tie']-1)>1e-8:
                raise ValueError('Invalid challenger probabilities')
            if not record['fit_through']<record['calibration_through']<record['feature_cutoff']:
                raise ValueError('Invalid challenger chronology')
            if name=='live_challengers' and not datetime.fromisoformat(record['forecast_at'])<datetime.fromisoformat(record['kickoff_at_observation']):
                raise ValueError('Late shadow forecast')
            for target in ['home','away','margin','total']:
                lo,hi=record[target+'_interval80']
                if not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:
                    raise ValueError('Invalid uncertainty interval')
        if name=='season_replay' and {r['game_id'] for r in report['forecasts']}!={r['game_id'] for r in rows}:
            raise ValueError('Incomplete season replay')
    quality=folder/'data_quality.json'
    if quality.exists() and not all(c['passed'] for c in json.loads(quality.read_text())['checks']):
        raise ValueError('Data contract validation failed')
    feature_value=folder/'feature_value.json'
    if feature_value.exists():
        report=json.loads(feature_value.read_text())
        if len({m['decisive_games'] for m in report['models']})!=1:
            raise ValueError('Feature experiments use different cohorts')
        if any(f['train_through']>=f['test_start'] for f in report['folds']):
            raise ValueError('Feature experiment train/test overlap')
    shadow=folder/'shadow_next.json'
    if shadow.exists():
        report=json.loads(shadow.read_text());seen=set()
        for record in report['forecasts']:
            key=(record['generation'],record['model'],record['game_id'],record['horizon'])
            if key in seen:raise ValueError('Duplicate next-generation forecast')
            seen.add(key)
            if record['generation']!=report['generation']:raise ValueError('Mixed shadow generations')
            if not datetime.fromisoformat(record['forecast_at'])<datetime.fromisoformat(record['kickoff_at_observation']):raise ValueError('Late next-generation forecast')
            if record['task']=='winner':
                if not math.isfinite(record['home_probability']) or not 0<=record['home_probability']<=1:raise ValueError('Invalid shadow probability')
            else:
                if any(not math.isfinite(record[k]) or record[k]<0 for k in ['predicted_home_score','predicted_away_score']):raise ValueError('Invalid shadow score')
                for lo,hi in record.get('intervals',{}).values():
                    if not math.isfinite(lo+hi) or lo>hi:raise ValueError('Invalid shadow interval')
    from publication_gate import validate_shadow_provenance
    validate_shadow_provenance(root)
    return {'games':len(rows),'completed':sum(r['status']=='Final' for r in rows)}


if __name__=='__main__':
    print(validate(Path(__file__).resolve().parent))
