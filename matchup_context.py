"""Pregame contextual research inputs; availability is explicit and never inferred."""
import json
import numpy as np
import pandas as pd
from audit_io import write_json, write_csv


def source(root,name):
    path=root/'audit/sources'/name/'latest.json'
    if not path.exists():
        return pd.DataFrame()
    meta=json.loads(path.read_text())
    return pd.read_parquet(path.parent/(meta['sha256']+'.parquet'))


def context(root,season):
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    pbp=source(root,'load_pbp')
    qb=source(root,'load_player_stats')
    injuries=source(root,'load_injuries')
    rosters=source(root,'load_rosters_weekly')
    finals=schedule.loc[schedule.home_score.notna() & schedule.away_score.notna()]
    stats=pd.DataFrame()
    if not pbp.empty:
        p=pbp.loc[pbp.game_id.isin(finals.game_id)&pbp.posteam.notna()&pbp.epa.notna()&pbp.play_type.isin(['pass','run'])]
        stats=p.groupby(['week','game_id','posteam','defteam']).agg(epa=('epa','mean'),plays=('epa','size')).reset_index()
        adjusted=[]
        for r in stats.itertuples():
            previous=stats.loc[stats.week.lt(r.week)&stats.defteam.eq(r.defteam),'epa']
            adjusted.append(float(r.epa-previous.mean()) if len(previous) else np.nan)
        stats['opponent_adjusted_epa']=adjusted
    weather_path=root/'data/weather_forecasts.csv'
    weather=pd.read_csv(weather_path) if weather_path.exists() else pd.DataFrame()
    reports={}
    rows=[]
    for game in schedule.to_dict('records'):
        record={'game_id':game['game_id'],'week':game['week'],'sources':{}}
        numeric={'game_id':game['game_id'],'season':season,'week':game['week']}
        for side in ['home','away']:
            team=game[side+'_team']
            prior=stats.loc[stats.week.lt(game['week'])&stats.posteam.eq(team)].sort_values('week').tail(5) if len(stats) else stats
            team_info={'last_performance_week':int(prior.week.max()) if len(prior) else None,
                'plays_per_game':float(prior.plays.mean()) if len(prior) else None,
                'opponent_adjusted_epa':float(prior.opponent_adjusted_epa.mean()) if len(prior) and prior.opponent_adjusted_epa.notna().any() else None,
                'expected_qb':None,'qb_status':'Unconfirmed; previous usage is not a starter announcement',
                'last_qb':None,'last_qb_attempt_share':None,'ol_out':None,'roster_ol_count':None,'active_roster_ol_count':None}
            if len(qb):
                prior_qb=qb.loc[qb.team.eq(team)&qb.week.lt(game['week'])&qb.position.eq('QB')&qb.attempts.gt(0)]
                if len(prior_qb):
                    last=prior_qb.loc[prior_qb.week.eq(prior_qb.week.max())].sort_values('attempts',ascending=False)
                    first=last.iloc[0]
                    team_info['last_qb']=str(first.get('player_display_name',first.get('player_name',first.get('player_id'))))
                    team_info['last_qb_attempt_share']=float(first.attempts/last.attempts.sum())
            if len(injuries):
                injury=injuries.loc[injuries.team.eq(team)&injuries.week.eq(game['week'])]
                if len(injury):
                    team_info['ol_out']=int((injury.position.isin(['C','G','T','OL','OT','OG']) & injury.report_status.fillna('').str.lower().isin(['out','doubtful'])).sum())
            if len(rosters):
                roster=rosters.loc[rosters.team.eq(team)&rosters.week.eq(game['week'])]
                if len(roster):
                    team_info['roster_ol_count']=int(roster.position.isin(['C','G','T','OL','OT','OG']).sum())
                    team_info['active_roster_ol_count']=int((roster.position.isin(['C','G','T','OL','OT','OG'])&roster.status.fillna('').str.upper().isin(['ACT','ACTIVE'])).sum())
            record[side]=team_info
            for key in ['plays_per_game','opponent_adjusted_epa','last_qb_attempt_share','ol_out','roster_ol_count','active_roster_ol_count']:
                numeric[side+'_'+key]=team_info[key]
        record['weather']={'status':'No timestamped forecast source configured'}
        if len(weather):
            candidates=weather.loc[weather.game_id.eq(game['game_id'])].copy()
            issued=pd.to_datetime(candidates.issued_at,utc=True,errors='coerce')
            kickoff=pd.Timestamp(f"{game['gameday']} {game.get('gametime') if pd.notna(game.get('gametime')) else '00:00'}",tz='America/New_York').tz_convert('UTC')
            candidates=candidates.loc[issued.lt(kickoff)&issued.le(pd.Timestamp.now(tz='UTC'))]
            if len(candidates):
                selected=candidates.sort_values('issued_at').iloc[-1]
                valid=pd.Timestamp(selected.valid_at)
                if valid.tzinfo and abs((valid.tz_convert('UTC')-kickoff).total_seconds())<=10800:
                    record['weather']={'status':'Timestamped supplied forecast','issued_at':selected.issued_at,
                        'provider':selected.provider,'temperature_f':float(selected.temperature_f),'wind_mph':float(selected.wind_mph)}
        record['note']='Research context, not additional production-model inputs. Plays/game is a volume proxy, not seconds/play. Opponent adjustment subtracts the opponent defense mean from earlier weeks, then averages five games. Historical snapshots are revised; no historical availability claim.'
        reports[game['game_id']]=record
        rows.append(numeric)
    write_json(root/'public_site/data/matchup_context.json',reports)
    write_csv(root/f'data/context_features_{season}.csv',pd.DataFrame(rows))
    return reports
