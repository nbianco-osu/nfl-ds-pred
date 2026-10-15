"""Reproducible opponent-adjusted EPA; not FTN DVOA or a serving model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from audit_io import write_csv, write_json
from matchup_context import source

SHRINKAGE_PLAYS = 200.0
RATING_FIELDS = ['offense_epa', 'defense_epa_prevented', 'net_epa',
                 'sack_differential_per_game', 'sack_rate_differential',
                 'big_play_differential_per_game', 'big_play_rate_differential']


def prepare(pbp, schedule):
    """Retain scrimmage plays from completed regular-season games only."""
    games = schedule.loc[schedule.game_type.eq('REG') & schedule.home_score.notna()
                         & schedule.away_score.notna()].copy()
    if schedule.game_id.duplicated().any():
        raise ValueError('Duplicate schedule game IDs')
    p = pbp.loc[pbp.game_id.isin(games.game_id) & pbp.play_type.isin(['pass', 'run'])
                & pbp.posteam.notna() & pbp.defteam.notna()].copy()
    if 'play_id' in p and p.duplicated(['game_id', 'play_id']).any():
        raise ValueError('Duplicate play IDs')
    p = p.drop(columns=['season', 'week', 'gameday'], errors='ignore').merge(
        games[['game_id', 'season', 'week', 'gameday']], on='game_id', validate='many_to_one')
    p['gameday'] = pd.to_datetime(p.gameday)
    p['big_play'] = ((p.pass_attempt.eq(1) & p.yards_gained.ge(20)) |
                     (p.rush_attempt.eq(1) & p.yards_gained.ge(10))).astype(float)
    p.loc[p.yards_gained.isna(), 'big_play'] = np.nan
    return p


def fit_ratings(plays, teams, alpha=SHRINKAGE_PLAYS):
    """EPA = intercept + offense effect + defense allowance effect."""
    teams = sorted(set(teams))
    lookup = {team: i for i, team in enumerate(teams)}
    valid = plays.loc[np.isfinite(pd.to_numeric(plays.epa, errors='coerce'))].copy()
    effects = {}
    intercept = None
    if len(valid):
        grouped = valid.groupby(['posteam', 'defteam']).epa.agg(['mean', 'size']).reset_index()
        design = np.zeros((len(grouped), len(teams)*2))
        for i, row in enumerate(grouped.itertuples()):
            design[i, lookup[row.posteam]] = 1
            design[i, len(teams)+lookup[row.defteam]] = 1
        with threadpool_limits(limits=2):
            model = Ridge(alpha=alpha).fit(design, grouped['mean'], sample_weight=grouped['size'])
        intercept = float(model.intercept_)
        effects = {team: (float(model.coef_[i]), float(-model.coef_[len(teams)+i]))
                   for team, i in lookup.items()}
    rows = []
    for team in teams:
        off, defense = plays.loc[plays.posteam.eq(team)], plays.loc[plays.defteam.eq(team)]
        n_games = plays.loc[plays.posteam.eq(team) | plays.defteam.eq(team), 'game_id'].nunique()
        rated = len(valid.loc[valid.posteam.eq(team)]) > 0 and len(valid.loc[valid.defteam.eq(team)]) > 0
        offense_epa, defense_epa = effects.get(team, (None, None)) if rated else (None, None)
        def differential(column, rate=False):
            if off.empty or defense.empty or off[column].isna().any() or defense[column].isna().any():
                return None
            if column == 'sack':
                if off.qb_dropback.isna().any() or defense.qb_dropback.isna().any():
                    return None
                a, b = defense, off
                den_a, den_b = a.qb_dropback.eq(1).sum(), b.qb_dropback.eq(1).sum()
            else:
                a, b = off, defense
                den_a, den_b = len(a), len(b)
            if rate:
                return float(a[column].sum()/den_a-b[column].sum()/den_b) if den_a and den_b else None
            return float((a[column].sum()-b[column].sum())/n_games) if n_games else None
        rows.append({'team': team, 'games': int(n_games), 'offense_plays': len(off),
                     'defense_plays': len(defense), 'epa_plays': int((valid.posteam.eq(team) | valid.defteam.eq(team)).sum()),
                     'offense_epa': offense_epa, 'defense_epa_prevented': defense_epa,
                     'net_epa': offense_epa+defense_epa if rated else None,
                     'sack_differential_per_game': differential('sack'),
                     'sack_rate_differential': differential('sack', True),
                     'big_play_differential_per_game': differential('big_play'),
                     'big_play_rate_differential': differential('big_play', True)})
    return rows, intercept


def build(pbp, schedule, season):
    schedule = schedule.loc[schedule.season.eq(season) & schedule.game_type.eq('REG')].copy()
    teams = sorted(set(schedule.home_team) | set(schedule.away_team))
    plays = prepare(pbp, schedule)
    features, weekly = [], []
    for week, games in schedule.groupby('week', sort=True):
        cutoff = pd.to_datetime(games.gameday).min()
        earlier = plays.loc[plays.week.lt(week) & plays.gameday.lt(cutoff)]
        ratings, mean = fit_ratings(earlier, teams)
        lookup = {r['team']: r for r in ratings}
        weekly.append({'week': int(week), 'cutoff': cutoff.date().isoformat(),
                       'source_games': int(earlier.game_id.nunique()), 'league_epa': mean, 'teams': ratings})
        for game in games.itertuples():
            row = {'game_id': game.game_id, 'season': season, 'week': int(week),
                   'feature_cutoff': cutoff.date().isoformat()}
            for field in RATING_FIELDS:
                h, a = lookup[game.home_team][field], lookup[game.away_team][field]
                row['home_'+field], row['away_'+field] = h, a
                row['diff_'+field] = h-a if h is not None and a is not None else None
            features.append(row)
    current, mean = fit_ratings(plays, teams)
    current.sort(key=lambda r: r['net_epa'] if r['net_epa'] is not None else -np.inf, reverse=True)
    completed_ids = set(schedule.loc[schedule.home_score.notna() & schedule.away_score.notna(), 'game_id'])
    return pd.DataFrame(features), {'season': season, 'status': 'research_only',
        'missing_completed_game_ids': sorted(completed_ids-set(plays.game_id)),
        'shrinkage_plays': SHRINKAGE_PLAYS, 'source_games': int(plays.game_id.nunique()),
        'source_plays': len(plays), 'league_epa': mean, 'latest_ratings': current, 'weekly': weekly,
        'note': 'Opponent-Adjusted EPA, not DVOA. Joint ridge offense/defense effects on nflverse EPA, which already accounts for situation. Positive defense means EPA prevented; positive net is better. Fixed 200-play ridge penalty is an assumption, not tuned evidence. Matchup features freeze before each week; no same-week results. Week 1 and absent teams remain missing. Revised historical EPA is not proof of original availability. Latest rankings include all completed source games and must not be used for historical predictions. Big plays: pass >=20 yards or rush >=10 yards. Sack differential: made minus allowed. Rates use dropbacks for sacks, scrimmage plays for big plays. No production-model change or proven predictive improvement.'}


def export(root, season=2026):
    schedule_path = root/f'data/schedule_{season}.csv'
    pbp = source(root, 'load_pbp')
    if pbp.empty:
        raise ValueError('No archived play-by-play; refusing invented ratings')
    features, report = build(pbp, pd.read_csv(schedule_path), season)
    receipt_path = root/'audit/sources/load_pbp/latest.json'
    report['source_receipt'] = json.loads(receipt_path.read_text())
    report['schedule_sha256'] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    report['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_csv(root/f'data/opponent_features_{season}.csv', features)
    write_json(root/f'data/opponent_ratings_{season}.json', report)
    write_json(root/'public_site/data/opponent_ratings.json', report)
    write_csv(root/f'data/opponent_rankings_{season}.csv', pd.DataFrame(report['latest_ratings']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2026)
    args = parser.parse_args()
    report = export(Path(__file__).resolve().parent, args.season)
    print(f"Calculated {len(report['latest_ratings'])} teams from {report['source_games']} games")
    print(pd.DataFrame(report['latest_ratings'])[['team', 'games', 'net_epa', 'sack_differential_per_game', 'big_play_differential_per_game']].head(10).to_string(index=False))
