"""Current-season pregame features; no same-game performance enters a forecast."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nflreadpy as nfl
from advanced_features import make_rest_features, make_injury_features, make_roster_features, merge_team_feature_block


def roll_forward(per_game, schedule, value_cols, prefix):
    rows = []
    for game in schedule.itertuples():
        for team in [game.home_team, game.away_team]:
            prior = per_game.loc[per_game.team.eq(team) & per_game.week.lt(game.week)].sort_values('week')
            row = dict(game_id=game.game_id, season=game.season, week=game.week, team=team)
            for col in value_cols:
                values = pd.to_numeric(prior[col], errors='coerce').dropna()
                row[f'{prefix}_{col}_avg'] = values.mean()
                for window in [3, 5]:
                    row[f'{prefix}_{col}_last_{window}'] = values.tail(window).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def build_current(season=2026):
    root = Path(__file__).resolve().parent
    schedule = pd.DataFrame(nfl.load_schedules([season]).to_dicts())
    schedule = schedule.loc[schedule.game_type.eq('REG')].copy()
    finished = schedule.loc[schedule.home_score.notna() & schedule.away_score.notna()]
    output = schedule[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
    output = merge_team_feature_block(output, make_rest_features(schedule))
    coverage = {'season': season, 'completed_games': len(finished), 'sources': {}}
    try:
        pbp = pd.DataFrame(nfl.load_pbp([season]).to_dicts())
        pbp = pbp.loc[pbp.game_id.isin(finished.game_id) & pbp.posteam.notna() & pbp.epa.notna() & ~pbp.play_type.isin(['no_play', 'qb_kneel', 'qb_spike'])].copy()
        pbp['success'] = pbp.epa.gt(0).astype(int)
        pbp['explosive'] = ((pbp.pass_attempt.eq(1) & pbp.yards_gained.ge(20)) | (pbp.rush_attempt.eq(1) & pbp.yards_gained.ge(10))).astype(int)
        pbp['turnover'] = (pbp.interception.eq(1) | pbp.fumble_lost.eq(1)).astype(int)
        pbp['pass_epa'] = pbp.epa.where(pbp.pass_attempt.eq(1))
        pbp['rush_epa'] = pbp.epa.where(pbp.rush_attempt.eq(1))
        offense = pbp.groupby(['game_id', 'week', 'posteam']).agg(
            off_epa_per_play=('epa', 'mean'), off_success_rate=('success', 'mean'),
            off_explosive_rate=('explosive', 'mean'), off_turnover_rate=('turnover', 'mean'),
            off_pass_rate=('pass_attempt', 'mean'), off_pass_epa=('pass_epa', 'mean'), off_rush_epa=('rush_epa', 'mean'),
        ).reset_index().rename(columns={'posteam': 'team'})
        defense = pbp.groupby(['game_id', 'week', 'defteam']).agg(
            def_epa_allowed_per_play=('epa', 'mean'), def_success_allowed_rate=('success', 'mean'),
            def_explosive_allowed_rate=('explosive', 'mean'), def_takeaway_rate=('turnover', 'mean'),
        ).reset_index().rename(columns={'defteam': 'team'})
        stats = offense.merge(defense, on=['game_id', 'week', 'team'], how='outer')
        values = [c for c in stats if c not in ['game_id', 'week', 'team']]
        output = merge_team_feature_block(output, roll_forward(stats, schedule, values, 'team_pbp'))
        coverage['sources']['pbp_games'] = int(pbp.game_id.nunique())
    except Exception as exc:
        coverage['sources']['pbp_error'] = str(exc)
    try:
        stats = pd.DataFrame(nfl.load_player_stats([season]).to_dicts())
        stats = stats.loc[stats.season_type.eq('REG') & stats.position.eq('QB') & stats.attempts.gt(0)]
        completed_teams = pd.concat([finished[['season', 'week', 'home_team']].rename(columns={'home_team': 'team'}), finished[['season', 'week', 'away_team']].rename(columns={'away_team': 'team'})])
        stats = stats.merge(completed_teams, on=['season', 'week', 'team'], validate='many_to_one')
        stats = stats.sort_values('attempts', ascending=False).drop_duplicates(['week', 'team'])
        cols = [c for c in ['attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered', 'passing_epa', 'passing_cpoe'] if c in stats]
        stats = stats.rename(columns={c: 'qb_' + c for c in cols})
        output = merge_team_feature_block(output, roll_forward(stats, schedule, ['qb_' + c for c in cols], 'team_qb'))
        coverage['sources']['qb_team_games'] = len(stats)
    except Exception as exc:
        coverage['sources']['qb_error'] = str(exc)
    for name, loader in [('injuries', make_injury_features), ('rosters', make_roster_features)]:
        try:
            block = loader(schedule, [season])
            output = merge_team_feature_block(output, block)
            coverage['sources'][name + '_team_weeks'] = len(block)
        except Exception as exc:
            coverage['sources'][name + '_error'] = str(exc)
    output.to_csv(root / f'data/current_features_{season}.csv', index=False)
    (root / f'data/current_features_{season}.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    print(json.dumps(coverage, indent=2), flush=True)


def apply_current(row, game, snapshots):
    if snapshots is None or game.get('game_id') not in snapshots.index:
        return row
    fresh = snapshots.loc[game['game_id']]
    changed = set()
    for col in row.columns:
        if col.startswith(('home_team_', 'away_team_')) and col in fresh and pd.notna(fresh[col]):
            row.at[0, col] = fresh[col]
            changed.add(col)
    for col in row.columns:
        if col.startswith('diff_'):
            suffix = col.removeprefix('diff_')
            home, away = 'home_' + suffix, 'away_' + suffix
            if (home in changed or away in changed) and home in row and away in row:
                row.at[0, col] = row.at[0, home] - row.at[0, away]
    return row


if __name__ == '__main__':
    build_current()
