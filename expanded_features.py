"""Public tracking/PBP metrics with week-frozen, prior-game-only features."""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ALIASES = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'WSH': 'WAS'}
NGS = {
    'passing': ('attempts', ['avg_time_to_throw', 'avg_intended_air_yards', 'aggressiveness', 'completion_percentage_above_expectation']),
    'rushing': ('rush_attempts', ['rush_yards_over_expected_per_att', 'rush_pct_over_expected']),
    'receiving': ('targets', ['avg_separation', 'avg_yac_above_expectation']),
}


def aggregate_pbp(pbp):
    p = pbp.loc[pbp.season_type.eq('REG') & pbp.posteam.notna() & pbp.play_type.isin(['pass', 'run'])].copy()
    p['early_epa'] = p.epa.where(p.down.le(2))
    p['sack_rate'] = p.sack.where(p.qb_dropback.eq(1))
    p['hit_rate'] = p.qb_hit.where(p.qb_dropback.eq(1))
    p['third_conversion'] = p.third_down_converted.where(p.down.eq(3))
    p['stuff_rate'] = p.yards_gained.le(0).astype(float).where(p.rush_attempt.eq(1))
    cols = ['early_epa', 'sack_rate', 'hit_rate', 'third_conversion', 'stuff_rate']
    blocks = []
    for side, prefix in [('posteam', 'off'), ('defteam', 'def')]:
        block = p.groupby(['game_id', side])[cols].mean().reset_index().rename(columns={side: 'team', **{c: f'{prefix}_{c}' for c in cols}})
        blocks.append(block)
    result = blocks[0].merge(blocks[1], on=['game_id', 'team'], how='outer', validate='one_to_one')
    drives = p.groupby(['game_id', 'posteam', 'defteam', 'drive']).agg(
        red_zone=('yardline_100', lambda x: x.le(20).any()), td=('touchdown', 'max')).reset_index()
    # A defensive score must not count as an offensive red-zone touchdown.
    offensive_td = p.assign(off_td=p.pass_touchdown.eq(1) | p.rush_touchdown.eq(1)).groupby(['game_id', 'posteam', 'defteam', 'drive']).off_td.max().reset_index()
    drives = drives.drop(columns='td').merge(offensive_td, on=['game_id', 'posteam', 'defteam', 'drive'])
    for side, prefix in [('posteam', 'off'), ('defteam', 'def')]:
        rates = drives.loc[drives.red_zone].groupby(['game_id', side]).off_td.mean().rename(f'{prefix}_red_zone_td_rate').reset_index().rename(columns={side: 'team'})
        result = result.merge(rates, on=['game_id', 'team'], how='left', validate='one_to_one')
    result['team'] = result.team.replace(ALIASES)
    return result


def aggregate_ngs(frame, kind):
    weight, cols = NGS[kind]
    frame = frame.loc[frame.season_type.eq('REG') & frame.week.gt(0)].copy()
    frame['team'] = frame.team_abbr.replace(ALIASES)
    rows = []
    for (season, week, team), group in frame.groupby(['season', 'week', 'team']):
        row = dict(season=season, week=week, team=team)
        for col in cols:
            valid = group[col].notna() & group[weight].gt(0)
            row[f'ngs_{col}'] = float(np.average(group.loc[valid, col], weights=group.loc[valid, weight])) if valid.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pregame_features(schedule, stats):
    """Compute all games in a week before recording any result from that week."""
    schedule = schedule.sort_values(['season', 'week', 'game_id'])
    if schedule.game_id.duplicated().any() or stats.duplicated(['game_id', 'team']).any():
        raise ValueError('Duplicate source keys')
    metrics = [c for c in stats if c not in ['game_id', 'team']]
    lookup = stats.set_index(['game_id', 'team'])
    past = defaultdict(lambda: deque(maxlen=5))
    elo = defaultdict(lambda: 1500.0)
    opponents = defaultdict(list)
    rows = []
    previous_season = None
    for (season, week), games in schedule.groupby(['season', 'week'], sort=True):
        if previous_season is not None and season != previous_season:
            elo = defaultdict(lambda: 1500.0, {team: 1500 + .75 * (rating - 1500) for team, rating in elo.items()})
            opponents.clear()
        previous_season = season
        for game in games.itertuples():
            row = {'game_id': game.game_id}
            for side in ['home', 'away']:
                team = ALIASES.get(getattr(game, side + '_team'), getattr(game, side + '_team'))
                records = [r for r in past[team] if r['_season'] >= season - 1]
                for metric in metrics:
                    values = [r[metric] for r in records if pd.notna(r[metric])]
                    row[f'{side}_team_ext_{metric}_last5'] = float(np.mean(values)) if values else np.nan
                row[f'{side}_team_ext_elo'] = elo[team]
                row[f'{side}_team_ext_opponent_elo'] = float(np.mean(opponents[team])) if opponents[team] else 1500.0
                row[f'{side}_team_ext_observed_games'] = len(records)
            for col in list(row):
                if col.startswith('home_'):
                    row['diff_' + col[5:]] = row[col] - row['away_' + col[5:]]
            rows.append(row)
        for game in games.itertuples():
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            h = ALIASES.get(game.home_team, game.home_team)
            a = ALIASES.get(game.away_team, game.away_team)
            # Ratings for strength of schedule are those available before this week.
            pre = rows[-len(games) + list(games.game_id).index(game.game_id)]
            eh, ea = pre['home_team_ext_elo'], pre['away_team_ext_elo']
            expected = 1 / (1 + 10 ** (-(eh + 55 - ea) / 400))
            actual = 1.0 if game.home_score > game.away_score else 0.0 if game.home_score < game.away_score else .5
            elo[h] += 20 * (actual - expected)
            elo[a] -= 20 * (actual - expected)
            opponents[h].append(ea)
            opponents[a].append(eh)
            for team in [h, a]:
                entry = lookup.loc[(game.game_id, team)].to_dict() if (game.game_id, team) in lookup.index else dict.fromkeys(metrics, np.nan)
                entry['_season'] = season
                past[team].append(entry)
    return pd.DataFrame(rows)


def build(season=2026, fetch=True):
    cache = ROOT / 'data/expanded_cache'
    cache.mkdir(parents=True, exist_ok=True)
    schedule = pd.DataFrame(nfl.load_schedules(list(range(1999, season + 1))).to_dicts())
    schedule = schedule.loc[schedule.game_type.eq('REG')].copy()
    coverage = {'updated_at': datetime.now(timezone.utc).isoformat(), 'through_season': season, 'pbp': {}, 'ngs': {}, 'errors': []}
    blocks = []
    for year in range(2016, season + 1):
        path = cache / f'pbp_{year}.csv'
        try:
            if not path.exists() or (year == season and fetch):
                raw = nfl.load_pbp([year])
                cols = ['game_id', 'season_type', 'posteam', 'defteam', 'play_type', 'epa', 'down', 'sack', 'qb_hit', 'qb_dropback', 'third_down_converted', 'yards_gained', 'rush_attempt', 'drive', 'yardline_100', 'touchdown', 'pass_touchdown', 'rush_touchdown']
                aggregate_pbp(pd.DataFrame(raw.select(cols).to_dicts())).to_csv(path, index=False)
            block = pd.read_csv(path)
            blocks.append(block)
            coverage['pbp'][str(year)] = int(block.game_id.nunique())
            print(f'PBP {year}: {block.game_id.nunique()} games', flush=True)
        except Exception as exc:
            coverage['errors'].append(f'PBP {year}: {exc}')
    if not blocks:
        raise ValueError('No play-by-play aggregates available')
    stats = pd.concat(blocks, ignore_index=True)
    teams = pd.concat([schedule[['game_id', 'season', 'week', side + '_team']].rename(columns={side + '_team': 'team'}) for side in ['home', 'away']], ignore_index=True)
    teams['team'] = teams.team.replace(ALIASES)
    for kind in NGS:
        path = cache / f'ngs_{kind}.csv'
        try:
            if fetch or not path.exists():
                frame = pd.DataFrame(nfl.load_nextgen_stats(list(range(2016, season + 1)), stat_type=kind).to_dicts())
                aggregate_ngs(frame, kind).to_csv(path, index=False)
            block = pd.read_csv(path)
            coverage['ngs'][kind] = {'current_team_weeks': int(block.season.eq(season).sum()), 'latest_week': int(block.loc[block.season.eq(season), 'week'].max())}
            block = teams.merge(block, on=['season', 'week', 'team'], how='left', validate='one_to_one').drop(columns=['season', 'week'])
            stats = stats.merge(block, on=['game_id', 'team'], how='outer', validate='one_to_one')
        except Exception as exc:
            coverage['errors'].append(f'NGS {kind}: {exc}')
    features = pregame_features(schedule, stats)
    base = pd.read_csv(ROOT / f'data/nfl_matchups_1999_{season}_advanced.csv')
    dataset = base.merge(features, on='game_id', how='left', validate='one_to_one')
    dataset.to_csv(ROOT / f'data/nfl_matchups_1999_{season}_expanded.csv', index=False)
    features.loc[features.game_id.isin(schedule.loc[schedule.season.eq(season), 'game_id'])].to_csv(ROOT / f'data/expanded_features_{season}.csv', index=False)
    coverage['dataset_rows'] = len(dataset)
    coverage['new_columns'] = len(features.columns) - 1
    (ROOT / f'data/expanded_coverage_{season}.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    print(json.dumps(coverage, indent=2), flush=True)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--cached', action='store_true')
    args = parser.parse_args()
    build(args.season, fetch=not args.cached)
