"""Research-only data build with lagged FTN and snap-count inputs."""
from pathlib import Path
import nflreadpy as nfl
import numpy as np
import pandas as pd
from audit_io import write_csv
from current_features import apply_current
from model_utils import get_feature_columns
from prediction_features import make_prediction_row, fill_latest_team_state

ALIASES={'LA':'LAR','STL':'LAR','SD':'LAC','OAK':'LV'}


def lagged(schedule, stats, game_ids=None):
    output=[]
    metrics=[c for c in stats if c not in ['game_id','team']]
    targets = schedule if game_ids is None else schedule.loc[schedule.game_id.isin(game_ids)]
    for game in targets.to_dict('records'):
        prior=schedule.loc[(schedule.season.lt(game['season']) | (schedule.season.eq(game['season']) & schedule.week.lt(game['week']))) & schedule.home_score.notna() & schedule.away_score.notna()]
        out={'game_id':game['game_id']}
        for side in ['home','away']:
            team=ALIASES.get(game[side+'_team'],game[side+'_team'])
            ids=prior.loc[prior.home_team.replace(ALIASES).eq(team)|prior.away_team.replace(ALIASES).eq(team)].sort_values(['season','week']).tail(5).game_id
            sample=stats.loc[stats.game_id.isin(ids)&stats.team.eq(team)]
            for metric in metrics:
                out[f'{side}_research_{metric}']=sample[metric].mean()
        for metric in metrics:
            out['diff_research_'+metric]=out['home_research_'+metric]-out['away_research_'+metric]
        output.append(out)
    return pd.DataFrame(output)


def build(root):
    root=Path(root)
    folder=root/'data/research_sources'
    schedule=nfl.load_schedules(list(range(1999,2027))).to_pandas()
    schedule=schedule.loc[schedule.game_type.eq('REG')].copy()
    blocks=[]
    for year in range(2022,2027):
        ftn=pd.read_parquet(folder/f'ftn_{year}.parquet')
        mapping=folder/f'play_teams_{year}.parquet'
        if not mapping.exists():
            nfl.load_pbp([year]).select(['game_id','play_id','posteam']).write_parquet(mapping)
        plays=pd.read_parquet(mapping).dropna(subset=['posteam']).drop_duplicates(['game_id','play_id'])
        joined=ftn.merge(plays,left_on=['nflverse_game_id','nflverse_play_id'],right_on=['game_id','play_id'],validate='one_to_one')
        fields=['is_motion','is_play_action','is_rpo','is_no_huddle','n_blitzers','n_pass_rushers','is_interception_worthy','is_catchable_ball']
        charted=joined.groupby(['game_id','posteam'])[fields].mean().reset_index().rename(columns={'posteam':'team'})
        snaps=pd.read_parquet(folder/f'snap_counts_{year}.parquet')
        snaps['team']=snaps.team.replace(ALIASES)
        usage=[]
        for (game_id,team), rows in snaps.groupby(['game_id','team']):
            ol=rows.loc[rows.position.isin(['C','G','T','OL','OT','OG'])].sort_values('offense_snaps',ascending=False)
            shares=rows.loc[rows.position.eq('QB'),'offense_pct']
            usage.append({'game_id':game_id,'team':team,'ol_top5_snap_share':ol.head(5).offense_snaps.sum()/ol.offense_snaps.sum() if ol.offense_snaps.sum()>0 else np.nan,
                          'qb_max_snap_share':shares.max() if len(shares) else np.nan})
        charted['team']=charted.team.replace(ALIASES)
        blocks.append(charted.merge(pd.DataFrame(usage),on=['game_id','team'],how='outer',validate='one_to_one'))
        print(f'Aggregated {year}: {len(blocks[-1])} team games',flush=True)
    stats=pd.concat(blocks,ignore_index=True)
    write_csv(root/'data/research_team_games.csv',stats)
    features=lagged(schedule.loc[schedule.season.ge(2022)],stats)
    history=pd.read_csv(root/'data/nfl_matchups_1999_2025_advanced.csv')
    current=schedule.loc[schedule.season.eq(2026)&schedule.home_score.notna()&schedule.away_score.notna()]
    cols=get_feature_columns(history)
    context=pd.concat([history,current],ignore_index=True)
    snapshots=pd.read_csv(root/'data/current_features_2026.csv').set_index('game_id')
    rows=[]
    for _, game in current.iterrows():
        args={key:game.get(key) for key in ['home_team','away_team','season','week','game_type','roof','surface','temp','wind','div_game','spread_line','total_line']}
        row=make_prediction_row(context,cols,**args)
        row=fill_latest_team_state(row,context,cols,str(game.home_team),str(game.away_team),int(game.season),int(game.week))
        row=apply_current(row,game,snapshots)
        for key in ['game_id','gameday','home_score','away_score']: row[key]=game[key]
        row['home_win']=int(game.home_score>game.away_score) if game.home_score!=game.away_score else np.nan
        rows.append(row)
    data=pd.concat([history,pd.concat(rows,ignore_index=True)],ignore_index=True)
    data=data.merge(features,on='game_id',how='left',validate='one_to_one')
    write_csv(root/'data/research_matchups.csv',data)
    print(f'Research dataset: {len(data)} games, {len(current)} current-season results (ties retained)',flush=True)
    return data


if __name__=='__main__':
    build(Path(__file__).resolve().parent)
