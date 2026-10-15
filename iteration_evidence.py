"""R50-R55 read-only evidence assembly and isolated historical research refits.

Only public_site/data/iteration_evidence.json is written. No captures, model
serialization, production fitting, source downloads or promotion occur here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_io import write_json
from historical_metrics import VIEWS, interval, loss
from pregame_inputs import kickoff, timestamp
from research_operations import read
from rolling_backtest import FEATURES, prepare
from publication_changes import canonical, TIMESTAMPS
from matchup_context import source

RESEARCH_PROTOCOL = 2
RESEARCH_SOURCES = ['data/historical_opponent_features.csv', 'data/historical_metrics/schedule.csv',
                    'data/nfl_matchups_1999_2025_advanced.csv']


def fingerprints(root, paths):
    records = []
    for p in paths:
        path = root/p
        is_json = path.suffix == '.json'
        content = None
        if path.exists():
            content = path.read_bytes()
            if is_json:
                content = json.dumps(semantic_json(json.loads(content)), sort_keys=True,
                                     separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')
        records.append({'path': p, 'sha256': hashlib.sha256(content).hexdigest() if content is not None else None,
                        'hash_basis': 'canonical_json_without_operational_timestamps_v1' if is_json else 'file_bytes',
                        **({'excluded_keys': sorted(TIMESTAMPS | {'generated_at'})} if is_json else {})})
    return records


def semantic_json(value):
    value = canonical(value)
    if isinstance(value, dict):
        return {k: semantic_json(v) for k, v in value.items() if k != 'generated_at'}
    if isinstance(value, list):
        return [semantic_json(v) for v in value]
    return value


def special_teams_summary(pbp, schedule):
    note = ('Descriptive raw special-teams EPA only; NOT opponent-adjusted, DVOA, a fitted rating, or predictive evaluation. '
            'Selection is special_teams_play==1 OR kickoff/punt/field-goal/extra-point attempt flags, excluding no_play. '
            'EPA is credited as recorded to posteam and negated for defteam. On kickoffs posteam may be the receiving side; '
            'these are possession-side credits, not kicker/offense attribution. No player or causal claim. '
            'Only completed regular-season games with finite EPA and both teams matching the schedule are used. '
            'Minimum display coverage is 20 valid plays across 2 games per team, a reporting rule not statistical validation. '
            'Latest revised snapshot, not pregame availability evidence; missing plays are not zero.')
    required = {'game_id', 'posteam', 'defteam', 'epa'}
    flags = [c for c in ['special_teams_play', 'kickoff_attempt', 'punt_attempt',
                         'field_goal_attempt', 'extra_point_attempt'] if c in pbp]
    if not required <= set(pbp) or not flags:
        return {'status': 'Partial: source unavailable', 'opponent_adjusted': False, 'teams': [], 'note': note}
    finals = schedule.loc[schedule.home_score.notna() & schedule.away_score.notna() & schedule.game_type.eq('REG')]
    candidate = pbp.loc[pbp.game_id.isin(finals.game_id) & pbp[flags].eq(1).any(axis=1)].copy()
    if 'play_type' in candidate:
        candidate = candidate.loc[candidate.play_type.ne('no_play')]
    if 'play_id' in candidate and candidate.duplicated(['game_id', 'play_id']).any():
        return {'status': 'Partial: duplicate source plays rejected', 'opponent_adjusted': False, 'teams': [], 'note': note}
    checked = candidate.drop(columns=['home_team', 'away_team'], errors='ignore').merge(
        finals[['game_id', 'home_team', 'away_team']], on='game_id', validate='many_to_one')
    ownership = ((checked.posteam.eq(checked.home_team) & checked.defteam.eq(checked.away_team)) |
                 (checked.posteam.eq(checked.away_team) & checked.defteam.eq(checked.home_team)))
    checked['epa'] = pd.to_numeric(checked.epa, errors='coerce')
    valid = checked.loc[ownership & np.isfinite(checked.epa)].copy()
    rows = []
    for team in sorted(set(finals.home_team) | set(finals.away_team)):
        own = valid.loc[valid.posteam.eq(team)]
        other = valid.loc[valid.defteam.eq(team)]
        games = len(set(own.game_id) | set(other.game_id))
        n = len(own)+len(other)
        enough = n >= 20 and games >= 2
        total = float(own.epa.sum()-other.epa.sum())
        rows.append({'team': team, 'games': games, 'plays': n, 'posteam_plays': len(own),
                     'defteam_plays': len(other), 'status': 'Descriptive' if enough else 'Insufficient coverage',
                     'epa_total': total if enough else None, 'epa_per_play': total/n if enough else None})
    return {'status': 'Partial: descriptive only; opponent adjustment not fitted', 'opponent_adjusted': False,
            'teams': rows, 'completed_games': len(finals), 'source_games': int(valid.game_id.nunique()),
            'missing_completed_game_ids': sorted(set(finals.game_id)-set(valid.game_id)),
            'candidate_plays': len(candidate), 'valid_plays': len(valid), 'excluded_plays': len(candidate)-len(valid),
            'note': note}


def select_fields(value, fields):
    return {k: value[k] for k in fields if k in value} if isinstance(value, dict) else None


def matched_baselines(root):
    """Compare saved out-of-sample forecasts, never independently averaged cohorts."""
    paths = ['backtests/feature_value_predictions.csv', 'backtests/advanced_weekly_predictions.csv',
             'backtests/weekly_predictions.csv']
    if not all((root/p).exists() for p in paths):
        return {'status': 'Unavailable', 'models': [], 'reason': 'Matched prediction files missing'}
    feature, advanced, simple = [pd.read_csv(root/p) for p in paths]
    labels = ['season', 'week', 'home_score', 'away_score']
    frames = {'raw_epa_plus_form': feature.loc[feature.view.eq('baseline')].rename(columns={'home_probability': 'p'}),
              'adjusted_epa_plus_form': feature.loc[feature.view.eq('adjusted')].rename(columns={'home_probability': 'p'}),
              'elo': advanced.rename(columns={'elo_probability': 'p'}),
              'simple_form': simple.rename(columns={'form_probability': 'p'}),
              'home_frequency': simple.rename(columns={'home_baseline_probability': 'p'})}
    for name, frame in frames.items():
        if frame.game_id.duplicated().any():
            raise ValueError('Duplicate matched baseline games: '+name)
    ids = sorted(set.intersection(*(set(f.game_id) for f in frames.values())))
    if not ids:
        return {'status': 'Unavailable', 'models': [], 'reason': 'No shared holdout games'}
    reference = frames['simple_form'].set_index('game_id').loc[ids]
    valid = reference.home_score.ne(reference.away_score).to_numpy(copy=True)
    aligned = {}
    for name, frame in frames.items():
        aligned[name] = frame.set_index('game_id').loc[ids]
        if not np.array_equal(aligned[name][labels].to_numpy(), reference[labels].to_numpy()):
            raise ValueError('Matched baseline outcome/cohort mismatch: '+name)
        p = aligned[name].p.to_numpy()
        valid &= np.isfinite(p) & (p >= 0) & (p <= 1)
    held = reference.loc[valid]
    y = held.home_score.gt(held.away_score).to_numpy()
    base = loss(aligned['simple_form'].loc[valid].p.to_numpy(), y)
    clusters = (held.season.astype(str)+'-'+held.week.astype(str)).to_numpy()
    rows = []
    for name, frame in aligned.items():
        p = frame.loc[valid].p.to_numpy()
        delta = loss(p, y)-base
        rows.append({'model': name, 'games': len(held), 'log_loss': float(loss(p, y).mean()) if len(y) else None,
                     'brier': float(np.mean((p-y)**2)) if len(y) else None,
                     'accuracy': float(np.mean((p >= .5) == y)) if len(y) else None,
                     'delta_log_loss_vs_simple': float(delta.mean()) if len(y) else None,
                     'ci95': interval(delta, clusters) if len(set(clusters)) >= 2 else None})
    return {'status': 'Matched saved holdouts', 'models': rows, 'game_ids': held.index.tolist(),
            'shared_games_before_exclusions': len(ids), 'excluded_ties_or_invalid': len(ids)-len(held),
            'sources': fingerprints(root, paths),
            'note': 'Exact shared game IDs and verified identical outcomes across all five models. EPA models use historical_metrics annual chronological fits; simple form/home frequency use rolling_backtest weekly fits; Elo uses advanced_backtest weekly-frozen ratings (K20, home55, offseason75%). Matched holdout cohort, NOT identical training windows or a causal estimator comparison. Raw EPA includes simple score form; not EPA-only. Revised historical data, post-hoc intersection, season-week paired bootstrap, no multiplicity correction or production promotion.'}


def lagged_personnel(data, advanced):
    """Use prior team snapshots only, never the target game's injury/roster row."""
    groups = {}
    for group, prefix in [('qb', 'team_qb_'), ('injury', 'team_injury_'), ('roster', 'team_roster_')]:
        fields = [c[5:] for c in advanced if c.startswith('home_'+prefix)
                  and 'away_'+c[5:] in advanced and (group != 'qb' or c.endswith('_last_5'))]
        groups[group] = ['diff_lagged_'+c for c in fields]
    fields = [c.removeprefix('diff_lagged_') for cols in groups.values() for c in cols]
    if not fields:
        return data, groups, []
    snapshots = []
    for side in ['home', 'away']:
        block = advanced[['season', 'week', 'gameday', side+'_team']+[side+'_'+c for c in fields]].copy()
        block.columns = ['season', 'source_week', 'source_date', 'team']+fields
        snapshots.append(block)
    snapshots = pd.concat(snapshots, ignore_index=True)
    snapshots['source_date'] = pd.to_datetime(snapshots.source_date)
    if snapshots.duplicated(['season', 'source_week', 'team']).any():
        raise ValueError('Duplicate personnel snapshots')
    result = data.copy()
    values = {}
    for side in ['home', 'away']:
        targets = data[['game_id', 'season', 'week', 'feature_cutoff', side+'_team']].rename(columns={side+'_team': 'team'})
        joined = pd.merge_asof(targets.sort_values('feature_cutoff'), snapshots.sort_values('source_date'),
                               left_on='feature_cutoff', right_on='source_date', by=['season', 'team'],
                               direction='backward', allow_exact_matches=False)
        if (joined.source_week.dropna() >= joined.loc[joined.source_week.notna(), 'week']).any():
            raise ValueError('Personnel snapshot overlaps target week')
        values[side] = joined.set_index('game_id').reindex(data.game_id)[fields]
    for field in fields:
        result['diff_lagged_'+field] = (values['home'][field]-values['away'][field]).to_numpy()
    coverage = [{'group': group, 'season': int(year), 'games': len(part),
                 'games_with_any_observed_field': int(part[cols].notna().any(axis=1).sum()),
                 'games_with_all_fields': int(part[cols].notna().all(axis=1).sum()) if cols else 0}
                for group, cols in groups.items() for year, part in result.groupby('season')]
    return result, groups, coverage


def finite(value):
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def pregame_evidence(record, game, time_key='forecast_at', require_training=False):
    """A timestamp alone is insufficient when a known training cutoff overlaps."""
    try:
        start = kickoff(game)
        observed = timestamp(record.get(time_key))
        if observed >= start:
            return False
        saved_start = record.get('kickoff_at_observation')
        if saved_start and observed >= timestamp(saved_start):
            return False
        through = record.get('training_through')
        if require_training and not through:
            return False
        if through and pd.Timestamp(through).date() >= start.tz_convert('America/New_York').date():
            return False
        original = record.get('original_forecast_at')
        return not original or timestamp(original) <= observed
    except (ValueError, TypeError, KeyError):
        return False


def disagreement(records):
    groups = defaultdict(dict)
    conflicts = set()
    for r in records:
        if not all(r.get(k) for k in ['game_id', 'generation', 'horizon', 'task', 'model']):
            continue
        fields = ['home_probability'] if r['task'] == 'winner' else ['predicted_home_score', 'predicted_away_score'] if r['task'] == 'score' else []
        if not fields or not all(finite(r.get(k)) for k in fields):
            continue
        values = tuple(float(r[k]) for k in fields)
        if r['task'] == 'winner' and not 0 <= values[0] <= 1:
            continue
        key = tuple(r[k] for k in ['game_id', 'generation', 'horizon', 'task'])
        previous = groups[key].get(r['model'])
        if previous and previous != values:
            conflicts.add((key, r['model']))
        groups[key][r['model']] = values
    rows = []
    for key, members in sorted(groups.items()):
        members = {m: v for m, v in members.items() if (key, m) not in conflicts}
        values = np.asarray(list(members.values()))
        row = dict(zip(['game_id', 'generation', 'horizon', 'task'], key))
        row.update(models=sorted(members), model_count=len(members), values={m: list(v) for m, v in sorted(members.items())},
                   status='Available' if len(members) >= 2 else 'Insufficient comparable models')
        if len(members) >= 2:
            row['range'] = np.ptp(values, axis=0).tolist()
            row['standard_deviation'] = values.std(axis=0).tolist()
            if key[-1] == 'winner':
                row['pick_split'] = bool((values < .5).any() and (values >= .5).any())
            else:
                row['margin_range'] = float(np.ptp(values[:, 0]-values[:, 1]))
                row['total_range'] = float(np.ptp(values.sum(axis=1)))
        rows.append(row)
    return {'groups': rows, 'conflicting_model_groups_excluded': len(conflicts),
            'note': 'Task, horizon and generation never mix. Ranges use probability units for winner and points for score (home, away). Models share inputs, training data and ensemble members; they are correlated, not independent votes. Dispersion is not calibrated uncertainty or evidence of accuracy. Initial-horizon observations may have different observation times.'}


def grouped_ablation(root, season):
    paths = ['data/historical_opponent_features.csv', 'data/historical_metrics/schedule.csv']
    if not all((root/p).exists() for p in paths):
        return {'status': 'Unavailable', 'reason': 'Historical feature/schedule files missing', 'results': []}
    schedule = pd.read_csv(root/paths[1])
    current = pd.read_csv(root/f'data/schedule_{season}.csv')
    schedule = pd.concat([schedule.loc[schedule.season.ne(season)], current], ignore_index=True)
    features = pd.read_csv(root/paths[0])
    data = prepare(schedule).merge(features.drop(columns=['season', 'week', 'feature_cutoff']),
                                   on='game_id', validate='one_to_one')
    extra = list(VIEWS['all'])
    groups = {'epa': [c for c in extra if 'epa' in c],
              'sacks': [c for c in extra if 'sack' in c],
              'big_plays': [c for c in extra if 'big_play' in c]}
    coverage = []
    advanced_path = root/'data/nfl_matchups_1999_2025_advanced.csv'
    if advanced_path.exists():
        advanced = pd.read_csv(advanced_path)
        # Do not test a wholly unavailable future season as a personnel experiment.
        data = data.loc[data.season.le(advanced.season.max())].copy()
        data, personnel, coverage = lagged_personnel(data, advanced)
        groups.update({k: v for k, v in personnel.items() if v})
        extra += [c for cols in personnel.values() for c in cols]
    groups['all_new'] = list(extra)
    predictions = defaultdict(list)
    folds = []
    for year in range(2024, season+1):
        train = data.loc[data.season.between(2017, year-1)]
        test = data.loc[data.season.eq(year)]
        if train.empty or test.empty:
            continue
        if train.gameday.max() >= test.feature_cutoff.min():
            raise ValueError('Ablation training overlaps test cutoff')
        decisive = train.loc[train.home_score.ne(train.away_score)]
        y = decisive.home_score.gt(decisive.away_score).astype(int)
        if y.nunique() != 2:
            continue
        folds.append({'season': year, 'training_through': str(train.gameday.max().date()),
                      'test_start': str(test.gameday.min().date()), 'games': len(test)})
        for name, dropped in {'none': [], **groups}.items():
            cols = FEATURES+[c for c in extra if c not in dropped]
            def pipeline(model):
                return make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                                    keep_empty_features=True), StandardScaler(), model)
            winner = pipeline(LogisticRegression(C=1, max_iter=1500)).fit(decisive[cols], y)
            score = pipeline(Ridge(alpha=10)).fit(train[cols], train[['home_score', 'away_score']])
            p = winner.predict_proba(test[cols])[:, 1]
            s = np.maximum(0, score.predict(test[cols]))
            out = test[['game_id', 'season', 'week', 'home_score', 'away_score']].copy()
            out['p'], out['h'], out['a'] = p, s[:, 0], s[:, 1]
            predictions[name].append(out)
    results = []
    if predictions:
        full = pd.concat(predictions['none']).set_index('game_id')
        for name, blocks in predictions.items():
            rows = pd.concat(blocks)
            d = rows.loc[rows.home_score.ne(rows.away_score)]
            y = d.home_score.gt(d.away_score).to_numpy()
            delta = loss(d.p.to_numpy(), y)-loss(full.loc[d.game_id, 'p'].to_numpy(), y)
            clusters = (d.season.astype(str)+'-'+d.week.astype(str)).to_numpy()
            results.append({'removed_group': name, 'removed_features': groups.get(name, []),
                            'games': len(rows), 'decisive_games': len(d),
                            'log_loss': float(loss(d.p.to_numpy(), y).mean()),
                            'delta_log_loss_vs_full': float(delta.mean()),
                            'ci95': interval(delta, clusters) if len(set(clusters)) >= 2 else None,
                            'team_score_mae': float((abs(rows.h-rows.home_score)+abs(rows.a-rows.away_score)).mean()/2)})
    return {'status': 'Completed research refits' if results else 'Unavailable', 'results': results, 'folds': folds,
            'personnel_coverage': coverage, 'unavailable_groups': ['special_teams', 'weather']+
            [g for g in ['qb', 'injury', 'roster'] if g not in groups],
            'note': 'True drop-group refits, not masking. Fixed logistic C=1/Ridge alpha=10; identical games/annual chronological folds; training-only imputation/scaling. Positive loss delta means removal hurt. Personnel snapshots are same-season, strictly earlier than target-week cutoff, joined by team before home-away differencing. QB last-five rollups are additionally lagged; injury/roster are PRIOR-game proxies, not current availability or confirmed starters. Advanced source ends 2025, so personnel experiment excludes 2026. Coverage counts precede imputation; original upstream QB zero fills are inherited. Revised historical snapshots do not prove publication-time availability. Season-week bootstrap; no multiplicity correction; post-hoc design. Form inherits 21-point cold-start assumption. No causal or production improvement claim.'}


def joint_scores(root, games):
    archived = read(root, 'public_site/data/score_distributions.json', {})
    metrics = read(root, 'public_site/data/distribution_metrics.json', {})
    eligible, excluded = [], []
    for key, r in sorted(archived.items()):
        game = games.get(r.get('game_id', key))
        valid = game and pregame_evidence(r, game, require_training=True)
        probs = [r.get(k) for k in ['home_win', 'away_win', 'tie']]
        valid = valid and all(finite(p) and 0 <= p <= 1 for p in probs)
        valid = valid and abs(sum(probs)-1) < 1e-8
        if valid:
            eligible.append(r)
        else:
            excluded.append({'game_id': r.get('game_id', key), 'reason': 'Missing/invalid pregame cutoff or probability evidence'})
    return {'games': eligible, 'excluded': excluded, 'historical_evaluation': metrics,
            'method': 'Existing research Ridge paired-residual joint score distribution',
            'note': 'Reuses saved score_distribution exports without generating or capturing forecasts. The existing model resamples paired home/away out-of-sample residuals (last 512; minimum 128), 10,000 draws, seed 42, rounded/clipped at zero. Winner/tie, margin, total and market probabilities derive from the same joint draws, not independent score marginals. Saved training date must precede kickoff date. Weekly historical evaluation used only earlier residuals; revised lines are not verified pregame lines. These are research-model summaries, not a distribution fitted to shadow_next or production point forecasts; no full joint PMF was saved. Historical coverage does not establish calibration for each saved generation.'}


def weekly_recap(predictions, schedule):
    by_id = {r['game_id']: r for r in predictions}
    weeks = []
    for week, frame in schedule.groupby('week', sort=True):
        games = []
        aggregates = defaultdict(list)
        for g in frame.to_dict('records'):
            r = by_id.get(g['game_id'], {})
            completed = finite(g.get('home_score')) and finite(g.get('away_score'))
            tasks = {}
            for task, time_key, version_key in [('winner', 'winner_forecast_at', 'model_version'),
                                                 ('score', 'score_forecast_at', 'score_model_version')]:
                fields = ['home_win_probability'] if task == 'winner' else ['predicted_home_score', 'predicted_away_score']
                available = all(finite(r.get(f)) for f in fields)
                if task == 'winner' and available:
                    available = 0 <= r[fields[0]] <= 1
                genuine = available and pregame_evidence(r, g, time_key)
                cohort = 'genuine_pregame' if genuine else 'legacy_or_unverified' if available else 'missing'
                detail = {'cohort': cohort, 'forecast_at': r.get(time_key),
                          'generation': r.get(version_key) or 'unknown',
                          **{f: r.get(f) for f in fields}, 'grade': 'Pending' if not completed else 'Missing prediction'}
                if completed and available:
                    if task == 'winner':
                        p = float(r[fields[0]])
                        tie = g['home_score'] == g['away_score']
                        correct = (p >= .5) == (g['home_score'] > g['away_score'])
                        detail['grade'] = 'Tie excluded' if tie else 'Correct' if correct else 'Wrong'
                        if not tie:
                            detail['log_loss'] = float(loss(np.array([p]), np.array([g['home_score'] > g['away_score']]))[0])
                    else:
                        h, a = (float(r[f]) for f in fields)
                        detail.update(grade='Graded', team_mae=(abs(h-g['home_score'])+abs(a-g['away_score']))/2,
                                      margin_error=(h-a)-(g['home_score']-g['away_score']),
                                      total_error=(h+a)-(g['home_score']+g['away_score']))
                    aggregates[(task, detail['generation'], cohort)].append(detail)
                tasks[task] = detail
            games.append({'game_id': g['game_id'], 'home_team': g['home_team'], 'away_team': g['away_team'],
                          'status': 'Final' if completed else 'Pending',
                          'home_score': float(g['home_score']) if completed else None,
                          'away_score': float(g['away_score']) if completed else None, 'tasks': tasks})
        summaries = []
        for (task, generation, cohort), rows in sorted(aggregates.items()):
            summary = {'task': task, 'generation': generation, 'cohort': cohort, 'games': len(rows)}
            if task == 'winner':
                graded = [r for r in rows if 'log_loss' in r]
                summary.update(graded=len(graded), ties_excluded=len(rows)-len(graded),
                               correct=sum(r['grade'] == 'Correct' for r in graded),
                               log_loss=float(np.mean([r['log_loss'] for r in graded])) if graded else None)
            else:
                summary['team_mae'] = float(np.mean([r['team_mae'] for r in rows]))
            summaries.append(summary)
        weeks.append({'week': int(week), 'scheduled_games': len(games),
                      'completed_games': sum(g['status'] == 'Final' for g in games), 'games': games, 'summaries': summaries})
    return {'weeks': weeks, 'note': 'All scheduled games, including missing predictions and pending results. Saved production winner and score tasks graded independently; no backfill. Genuine pregame requires an explicit timezone-aware timestamp strictly before actual kickoff, not merely a timestamp present. Legacy/missing/late evidence is separate and never contributes to genuine metrics. Timestamp provenance is not a full input or training audit. Model generations remain separate; ties excluded only for winner grading. No inference from legacy prediction_correct fields.'}


def export(root, season=2026, run_research=False):
    root = Path(root)
    schedule = pd.read_csv(root/f'data/schedule_{season}.csv')
    schedule = schedule.loc[schedule.season.eq(season)].copy()
    if schedule.game_id.duplicated().any():
        raise ValueError('Duplicate schedule game IDs')
    games = schedule.set_index('game_id').to_dict('index')
    shadow = read(root, 'public_site/data/shadow_next.json', {})
    records = list(shadow.get('forecasts', []))
    for path in sorted((root/'audit/shadow_next').glob('*/forecasts.json')):
        records.extend(read(root, path, {}).values())
    eligible = [r for r in records if r.get('game_id') in games and
                pregame_evidence(r, games[r['game_id']], require_training=True)]
    freshness = read(root, 'public_site/data/feature_freshness.json', {})
    fresh = {r['game_id']: r for r in freshness.get('games', [])}
    previous = read(root, 'public_site/data/iteration_evidence.json', {})
    research_sources = fingerprints(root, RESEARCH_SOURCES+[f'data/schedule_{season}.csv'])
    if run_research:
        with threadpool_limits(limits=2):
            ablation = grouped_ablation(root, season)
        ablation.update(protocol_version=RESEARCH_PROTOCOL, season=season, sources=research_sources)
    else:
        ablation = previous.get('grouped_ablation', {})
        if ablation.get('protocol_version') != RESEARCH_PROTOCOL or ablation.get('season') != season:
            ablation = {'status': 'Unavailable', 'results': [], 'folds': [],
                        'reason': 'Run explicitly with --research to persist this research protocol'}
    ablation['source_changed_since_research'] = bool(ablation.get('sources') and
        [(r['path'], r['sha256']) for r in ablation['sources']] != [(r['path'], r['sha256']) for r in research_sources])
    ratings = select_fields(read(root, 'public_site/data/opponent_ratings.json', {}),
                            ['season', 'status', 'latest_ratings', 'source_games', 'source_plays', 'note'])
    feature_evaluation = select_fields(read(root, 'public_site/data/feature_value.json', {}),
                                       ['models', 'folds', 'note'])
    recipe_evaluation = select_fields(read(root, 'public_site/data/next_evidence.json', {}),
                                      ['generation', 'comparisons', 'comparison_note'])
    sources = ['public_site/data/shadow_next.json', 'public_site/data/opponent_ratings.json',
               'public_site/data/feature_value.json', 'public_site/data/next_evidence.json',
               'public_site/data/score_distributions.json', 'public_site/data/distribution_metrics.json',
               'public_site/data/feature_freshness.json', 'public_site/data/predictions.json',
               f'data/schedule_{season}.csv', 'data/historical_opponent_features.csv',
               'data/historical_metrics/schedule.csv', 'audit/sources/load_pbp/latest.json']
    sources += [p.relative_to(root).as_posix() for p in sorted((root/'audit/shadow_next').glob('*/forecasts.json'))]
    baselines = matched_baselines(root)
    st = special_teams_summary(source(root, 'load_pbp'), schedule)
    report = {'schema_version': 2, 'season': season,
              'status': 'research_only', 'promotion': {'status': 'Deferred', 'requirement': 'R58', 'production_changed': False},
              'model_disagreement': disagreement(eligible),
              'opponent_adjusted': {'ratings': ratings,
                                    'matched_feature_evaluation': feature_evaluation,
                                    'matched_recipe_evaluation': recipe_evaluation,
                                    'matched_baselines': baselines,
                                    'simple_baselines': {'status': baselines['status'], 'note': baselines.get('note', baselines.get('reason', '')),
                                                         'results': baselines['models']},
                                    'special_teams': st,
                                    'note': 'Existing Opponent-Adjusted EPA, not DVOA. Latest ratings are descriptive, not historical inputs. Feature-value comparisons and matched estimator-recipe comparisons answer different questions; neither is a newly measured production improvement.'},
              'joint_scores': joint_scores(root, games),
              'grouped_ablation': ablation,
              'input_availability': {'games': [fresh.get(g, {'game_id': g, 'inputs': [], 'status': 'No saved freshness evidence'}) for g in games],
                                     'note': freshness.get('note', '')+' Copied receipt/publication timestamps are not proof of game-specific coverage. Missing games stay unknown; current freshness is not historical pregame evidence.'},
              'weekly_recap': weekly_recap(read(root, 'public_site/data/predictions.json', []), schedule),
              'sources': fingerprints(root, sources)}
    if report != previous:
        write_json(root/'public_site/data/iteration_evidence.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--research', action='store_true', help='Explicitly refit historical ablations and persist results')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = export(Path(__file__).resolve().parent, args.season, run_research=args.research)
    print('Exported iteration_evidence.json:', len(result['weekly_recap']['weeks']), 'weeks;',
          len(result['grouped_ablation']['results']), 'ablation variants; production unchanged')
