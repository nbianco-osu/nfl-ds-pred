"""Focused provenance, grouping and write-boundary tests for R50-R55."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from iteration_evidence import (disagreement, export, joint_scores, pregame_evidence,
                                weekly_recap, lagged_personnel, matched_baselines, fingerprints,
                                special_teams_summary)


def game(game_id='g', week=1):
    return {'game_id': game_id, 'season': 2026, 'week': week, 'gameday': '2026-10-11',
            'gametime': '13:00', 'home_team': 'BUF', 'away_team': 'NYJ',
            'home_score': 21., 'away_score': 17., 'game_type': 'REG'}


class IterationEvidenceTests(unittest.TestCase):
    def test_strict_cutoff_and_training(self):
        r = {'forecast_at': '2026-10-11T16:59:59Z', 'training_through': '2026-10-10'}
        self.assertTrue(pregame_evidence(r, game(), require_training=True))
        for value in [None, 'invalid', '2026-10-11T16:00:00', '2026-10-11T17:00:00Z', '2026-10-12T00:00:00Z']:
            self.assertFalse(pregame_evidence(dict(r, forecast_at=value), game()))
        self.assertFalse(pregame_evidence(dict(r, training_through='2026-10-11'), game()))
        self.assertFalse(pregame_evidence({'forecast_at': r['forecast_at']}, game(), require_training=True))
        self.assertFalse(pregame_evidence(dict(r, kickoff_at_observation='2026-10-11T16:00:00Z'), game()))

    def test_disagreement_never_pools_cohorts(self):
        r = {'game_id': 'g', 'model': 'a', 'task': 'winner', 'generation': 'v1',
             'horizon': 'initial', 'home_probability': .4}
        report = disagreement([r, r, dict(r, model='b', home_probability=.8),
                               dict(r, generation='v2'), dict(r, horizon='24h'),
                               dict(r, task='score', predicted_home_score=21, predicted_away_score=17)])
        self.assertEqual(len(report['groups']), 4)
        group = next(g for g in report['groups'] if g['model_count'] == 2)
        self.assertAlmostEqual(group['range'][0], .4)
        self.assertTrue(group['pick_split'])
        self.assertIn('correlated', report['note'])
        conflict = disagreement([r, dict(r, home_probability=.9)])
        self.assertEqual(conflict['conflicting_model_groups_excluded'], 1)
        self.assertEqual(conflict['groups'][0]['model_count'], 0)

    def test_recap_complete_and_legacy_separate(self):
        predictions = [{'game_id': 'g', 'home_win_probability': .8, 'model_version': 'v1',
                        'winner_forecast_at': '2026-10-11T16:00:00Z',
                        'predicted_home_score': 20., 'predicted_away_score': 18.,
                        'score_forecast_at': '2026-10-11T18:00:00Z'},
                       {'game_id': 'legacy', 'home_win_probability': .8, 'model_version': 'v1'},
                       {'game_id': 'tie', 'home_win_probability': .8,
                        'winner_forecast_at': '2026-10-11T16:00:00Z'}]
        schedule = pd.DataFrame([game(), game('legacy'), game('missing'),
                                 dict(game('tie'), away_score=21),
                                 dict(game('pending', 2), home_score=None, away_score=None)])
        report = weekly_recap(predictions, schedule)
        self.assertEqual(sum(w['scheduled_games'] for w in report['weeks']), 5)
        first = report['weeks'][0]
        self.assertEqual(first['games'][0]['tasks']['winner']['cohort'], 'genuine_pregame')
        self.assertEqual(first['games'][0]['tasks']['score']['cohort'], 'legacy_or_unverified')
        self.assertEqual(first['games'][2]['tasks']['winner']['cohort'], 'missing')
        self.assertEqual(first['games'][3]['tasks']['winner']['grade'], 'Tie excluded')
        cohorts = {r['cohort'] for r in first['summaries'] if r['task'] == 'winner'}
        self.assertEqual(cohorts, {'genuine_pregame', 'legacy_or_unverified'})
        self.assertEqual(report['weeks'][1]['completed_games'], 0)

    def test_joint_rejects_late_and_incoherent(self):
        r = {'game_id': 'g', 'forecast_at': '2026-10-10T00:00:00Z',
             'training_through': '2026-10-09', 'home_win': .6, 'away_win': .38, 'tie': .02}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root/'public_site/data').mkdir(parents=True)
            path = root/'public_site/data/score_distributions.json'
            path.write_text(json.dumps({'g': r}))
            self.assertEqual(len(joint_scores(root, {'g': game()})['games']), 1)
            for bad in [dict(r, training_through='2026-10-11'), dict(r, tie=.5), dict(r, forecast_at=None)]:
                path.write_text(json.dumps({'g': bad}))
                self.assertEqual(len(joint_scores(root, {'g': game()})['games']), 0)

    def test_export_only_writes_one_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root/'data').mkdir()
            pd.DataFrame([game()]).to_csv(root/'data/schedule_2026.csv', index=False)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            report = export(root, 2026)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(set(after)-set(before), {Path('public_site/data/iteration_evidence.json')})
            self.assertTrue(all(after[p] == content for p, content in before.items()))
            self.assertEqual(report['grouped_ablation']['status'], 'Unavailable')
            self.assertFalse(report['promotion']['production_changed'])
            self.assertEqual(report['input_availability']['games'][0]['inputs'], [])
            self.assertEqual(report['opponent_adjusted']['simple_baselines']['results'], [])
            json.loads((root/'public_site/data/iteration_evidence.json').read_text())

    def test_ablation_uses_chronological_refits(self):
        from iteration_evidence import grouped_ablation
        from historical_metrics import VIEWS
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root/'data/historical_metrics').mkdir(parents=True)
            rows = []
            for year in [2023, 2024]:
                for week in range(1, 5):
                    rows.append(dict(game(f'{year}_{week}', week), season=year,
                                     gameday=f'{year}-10-{week:02d}', home_score=24 if week % 2 else 10))
            pd.DataFrame(rows).to_csv(root/'data/historical_metrics/schedule.csv', index=False)
            pd.DataFrame([game()]).to_csv(root/'data/schedule_2026.csv', index=False)
            features = [{'game_id': r['game_id'], 'season': r['season'], 'week': r['week'],
                         'feature_cutoff': r['gameday'], **{c: float(r['week']) for c in VIEWS['all']}} for r in rows]
            pd.DataFrame(features).to_csv(root/'data/historical_opponent_features.csv', index=False)
            with patch('iteration_evidence.interval', return_value=[0., 0.]):
                result = grouped_ablation(root, 2026)
            self.assertEqual(len(result['results']), 5)
            self.assertTrue(all(r['games'] == 4 for r in result['results']))
            self.assertLess(result['folds'][0]['training_through'], result['folds'][0]['test_start'])
            self.assertEqual(next(r for r in result['results'] if r['removed_group'] == 'none')['delta_log_loss_vs_full'], 0)

    def test_export_never_refits_by_default_and_is_stable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root/'data').mkdir()
            pd.DataFrame([game()]).to_csv(root/'data/schedule_2026.csv', index=False)
            with patch('iteration_evidence.grouped_ablation', side_effect=AssertionError('Unexpected fit')):
                self.assertEqual(export(root)['grouped_ablation']['status'], 'Unavailable')
            with patch('iteration_evidence.grouped_ablation', return_value={'status': 'Completed', 'results': [], 'folds': []}) as fit:
                export(root, run_research=True)
                fit.assert_called_once()
            path = root/'public_site/data/iteration_evidence.json'
            before, modified = path.read_bytes(), path.stat().st_mtime_ns
            with patch('iteration_evidence.grouped_ablation', side_effect=AssertionError('Unexpected fit')):
                report = export(root)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, modified)
            self.assertNotIn('generated_at', report)
            changed = pd.DataFrame([dict(game(), home_score=30.)])
            changed.to_csv(root/'data/schedule_2026.csv', index=False)
            with patch('iteration_evidence.grouped_ablation', side_effect=AssertionError('Unexpected fit')):
                self.assertTrue(export(root)['grouped_ablation']['source_changed_since_research'])

    def test_personnel_lags_by_team_before_week(self):
        advanced = pd.DataFrame([
            dict(game('old'), week=1, gameday='2026-10-01', home_team_injury_out_count=2., away_team_injury_out_count=5.),
            dict(game('future'), week=2, gameday='2026-10-08', home_team_injury_out_count=100., away_team_injury_out_count=0.)])
        targets = pd.DataFrame([dict(game('target'), week=2, feature_cutoff=pd.Timestamp('2026-10-08')),
                                dict(game('first'), week=1, feature_cutoff=pd.Timestamp('2026-10-01'))])
        result, groups, coverage = lagged_personnel(targets, advanced)
        self.assertEqual(result.iloc[0].diff_lagged_team_injury_out_count, -3.)
        self.assertTrue(pd.isna(result.iloc[1].diff_lagged_team_injury_out_count))
        advanced.loc[1, 'home_team_injury_out_count'] = -1000
        again, _, _ = lagged_personnel(targets, advanced)
        pd.testing.assert_frame_equal(result, again)
        self.assertTrue(groups['injury'])
        self.assertEqual(next(c for c in coverage if c['group'] == 'injury')['games_with_any_observed_field'], 1)

    def test_baselines_intersect_games_and_verify_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root/'backtests').mkdir()
            shared = dict(game('shared'))
            feature = pd.DataFrame([dict(shared, view=v, home_probability=.6) for v in ['baseline', 'adjusted']]+
                                   [dict(game('only_epa'), view='baseline', home_probability=.9)])
            feature.to_csv(root/'backtests/feature_value_predictions.csv', index=False)
            pd.DataFrame([dict(shared, elo_probability=.7)]).to_csv(root/'backtests/advanced_weekly_predictions.csv', index=False)
            pd.DataFrame([dict(shared, form_probability=.55, home_baseline_probability=.51)]).to_csv(root/'backtests/weekly_predictions.csv', index=False)
            result = matched_baselines(root)
            self.assertEqual(result['game_ids'], ['shared'])
            self.assertEqual(len(result['models']), 5)
            self.assertTrue(all(r['games'] == 1 for r in result['models']))
            pd.DataFrame([dict(shared, home_score=42., elo_probability=.7)]).to_csv(root/'backtests/advanced_weekly_predictions.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                matched_baselines(root)

    def test_semantic_json_provenance_ignores_only_operational_times(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            path = root/'source.json'
            record = {'generated_at': 'old', 'nested': {'retrieved_at': 'old', 'updated_at': 'old',
                      'forecast_at': '2026-10-01T12:00:00Z', 'observed_at': '2026-10-01T11:00:00Z'}, 'probability': .6}
            path.write_text(json.dumps(record))
            before = fingerprints(root, ['source.json'])[0]
            record['generated_at'] = 'new'
            record['nested'].update(retrieved_at='new', updated_at='new')
            path.write_text(json.dumps(record, sort_keys=True, indent=4))
            self.assertEqual(fingerprints(root, ['source.json'])[0], before)
            self.assertEqual(before['hash_basis'], 'canonical_json_without_operational_timestamps_v1')
            record['nested']['forecast_at'] = '2026-10-02T12:00:00Z'
            path.write_text(json.dumps(record))
            self.assertNotEqual(fingerprints(root, ['source.json'])[0]['sha256'], before['sha256'])

    def test_special_teams_ownership_coverage_and_missingness(self):
        schedule = pd.DataFrame([game('a'), game('b'), game('missing')])
        rows = [{'game_id': g, 'play_id': i, 'special_teams_play': 1, 'posteam': 'BUF',
                 'defteam': 'NYJ', 'home_team': 'BUF', 'away_team': 'NYJ',
                 'epa': .5, 'play_type': 'kickoff'} for g in ['a', 'b'] for i in range(10)]
        rows += [dict(rows[0], play_id=99, posteam='INVALID'), dict(rows[0], play_id=100, epa=None)]
        report = special_teams_summary(pd.DataFrame(rows), schedule)
        self.assertFalse(report['opponent_adjusted'])
        self.assertEqual(report['missing_completed_game_ids'], ['missing'])
        self.assertEqual(report['excluded_plays'], 2)
        teams = {r['team']: r for r in report['teams']}
        self.assertEqual(teams['BUF']['epa_total'], 10.)
        self.assertEqual(teams['NYJ']['epa_total'], -10.)
        small = special_teams_summary(pd.DataFrame(rows[:2]), schedule)
        self.assertTrue(all(r['epa_total'] is None for r in small['teams']))
        duplicate = special_teams_summary(pd.DataFrame(rows+[rows[0]]), schedule)
        self.assertEqual(duplicate['teams'], [])


if __name__ == '__main__':
    unittest.main()
