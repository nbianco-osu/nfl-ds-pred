"""Focused mutations of the publication contracts, without fitting models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from publication_gate import validate_transition
from validate_dashboard import validate


class PublicationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'stage'
        self.previous = Path(self.temp.name)/'previous'
        self.now = datetime(2026, 10, 15, 20, tzinfo=timezone.utc)
        self.row = dict(game_id='game', gameday='2026-10-15', status='Scheduled',
                        home_team='H', away_team='A', home_win_probability=.6,
                        away_win_probability=.4, predicted_winner='H', model_version='v1',
                        home_score=None, away_score=None, prediction_result='Pending')
        for root in (self.root, self.previous):
            self.write(root, 'public_site/data/predictions.json', [self.row])
            self.write(root, 'public_site/data/kickoffs.json', {'game': '2026-10-15T19:00:00Z'})
            self.write(root, 'data/current_features_2026.json',
                       {'season': 2026, 'sources': {'pbp_games': 79},
                        'receipts': {'load_pbp': {'retrieved_at': '2099-01-01T00:00:00Z'}}})

    def write(self, root, name, value):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def transition(self):
        return validate_transition(self.root, self.previous, now=self.now)

    def shadow(self):
        features = {'epa': .25, 'injuries': None}
        digest = hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()
        self.input_path = f'audit/shadow_next/test/inputs/{digest}.json'
        self.snapshot = {'features': features, 'observed_at': '2026-10-14T12:00:00Z'}
        self.record = dict(generation='test', model='candidate', game_id='game', horizon='initial',
                           task='winner', home_probability=.6, training_through='2026-10-12',
                           forecast_at='2026-10-14T13:00:00Z',
                           kickoff_at_observation='2026-10-15T19:00:00Z', input_sha256=digest)
        self.write(self.root, self.input_path, self.snapshot)
        self.write(self.root, 'models/retrained/test/report.json', {'training_through': '2026-10-12'})
        self.save_shadow()

    def save_shadow(self):
        self.write(self.root, 'public_site/data/shadow_next.json',
                   {'generation': 'test', 'forecasts': [self.record]})

    def test_completed_actual_scores(self):
        for value in (None, True, '14', -1, 1.5, float('nan'), float('inf')):
            with self.subTest(value=value):
                row = dict(self.row, status='Final', home_score=value, away_score=7, prediction_result='Yes')
                self.write(self.root, 'public_site/data/predictions.json', [row])
                with self.assertRaises(ValueError):
                    validate(self.root)
        for value in (14, 14.0, 0):
            row = dict(self.row, status='Final', home_score=value, away_score=value,
                       prediction_result='Tie (no winner)')
            self.write(self.root, 'public_site/data/predictions.json', [row])
            validate(self.root)

    def test_nonfinal_kickoff_locked_fields(self):
        for field, value in [('model_version', 'v2'), ('winner_forecast_at', '2026-10-15T20:00:00Z'),
                             ('predicted_home_score', 22), ('score_model_version', 'v2')]:
            with self.subTest(field=field):
                self.write(self.root, 'public_site/data/predictions.json', [dict(self.row, **{field: value})])
                with self.assertRaisesRegex(ValueError, 'Locked forecast changed'):
                    self.transition()

    def test_changed_probability_and_moved_kickoff_cannot_unlock(self):
        row = dict(self.row, home_win_probability=.7, away_win_probability=.3)
        self.write(self.root, 'public_site/data/predictions.json', [row])
        self.write(self.root, 'public_site/data/kickoffs.json', {'game': '2027-01-01T00:00:00Z'})
        with self.assertRaisesRegex(ValueError, 'Locked forecast changed'):
            self.transition()

    def test_future_forecast_can_change(self):
        self.now = datetime(2026, 10, 14, tzinfo=timezone.utc)
        self.write(self.root, 'public_site/data/predictions.json', [dict(self.row, model_version='v2')])
        self.assertEqual(self.transition()['locked_forecasts'], 0)

    def test_locked_deletion(self):
        self.write(self.root, 'public_site/data/predictions.json', [dict(self.row, game_id='other')])
        with self.assertRaisesRegex(ValueError, 'Locked forecast removed'):
            self.transition()

    def test_grading_updates_allowed(self):
        self.write(self.root, 'public_site/data/predictions.json',
                   [dict(self.row, status='Final', home_score=20, away_score=7, prediction_result='Yes')])
        self.assertEqual(self.transition()['locked_forecasts'], 1)

    def test_exact_snapshot_mutations(self):
        self.shadow()
        validate(self.root)
        for patch in ({'observed_at': '2026-10-14T14:00:00Z'},
                      {'observed_at': '2026-10-14T12:00:00'},
                      {'features': {'epa': .5}}, {'performance_through': '2026-10-15'}):
            with self.subTest(patch=patch):
                self.write(self.root, self.input_path, dict(self.snapshot, **patch))
                with self.assertRaises(ValueError):
                    validate(self.root)
        (self.root/self.input_path).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing exact-input'):
            validate(self.root)

    def test_training_mutations(self):
        self.shadow()
        for cutoff in ('2026-10-15', '2026-10-14', '2026-10-13', None):
            with self.subTest(cutoff=cutoff):
                self.record['training_through'] = cutoff
                self.save_shadow()
                with self.assertRaises(ValueError):
                    validate(self.root)

    def test_revised_retrieval_not_performance_leakage(self):
        self.shadow()
        self.snapshot.update(retrieved_at='2099-01-01T00:00:00Z', performance_through='2026-10-12')
        self.write(self.root, self.input_path, self.snapshot)
        self.transition()

    def test_coverage_drop_requires_exact_explanation(self):
        self.write(self.root, 'data/current_features_2026.json', {'season': 2026, 'sources': {'pbp_games': 78}})
        with self.assertRaisesRegex(ValueError, 'Unexplained source coverage drop'):
            self.transition()
        explanation = {'key': 'current/2026/pbp_games', 'previous': 79, 'current': 78,
                       'reason': 'Provider removed a duplicate'}
        self.write(self.root, 'public_site/data/coverage_explanations.json', {'drops': [explanation]})
        self.assertEqual(self.transition()['explained_coverage_drops'], 1)
        explanation['current'] = 77
        self.write(self.root, 'public_site/data/coverage_explanations.json', {'drops': [explanation]})
        with self.assertRaises(ValueError):
            self.transition()

    def test_deleted_coverage_source(self):
        (self.root/'data/current_features_2026.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Unexplained source coverage drop'):
            self.transition()

    def test_historical_and_feature_coverage_drops(self):
        cases = [
            ('historical_coverage.json',
             {'seasons': [{'season': 2025, 'scheduled_games': 272, 'pbp_games': 272, 'plays': 32000}]},
             {'seasons': [{'season': 2025, 'scheduled_games': 272, 'pbp_games': 271, 'plays': 32000}]}),
            ('data_quality.json',
             {'checks': [], 'missingness': [{'feature': 'epa', 'rows': 272, 'missing': 16}]},
             {'checks': [], 'missingness': [{'feature': 'epa', 'rows': 272, 'missing': 17}]}),
        ]
        for name, old, new in cases:
            with self.subTest(name=name):
                self.write(self.previous, 'public_site/data/'+name, old)
                self.write(self.root, 'public_site/data/'+name, new)
                with self.assertRaisesRegex(ValueError, 'Unexplained source coverage drop'):
                    self.transition()
                self.write(self.root, 'public_site/data/'+name, old)

    def test_finals_stay_locked_even_with_future_kickoff(self):
        self.write(self.previous, 'public_site/data/predictions.json', [dict(self.row, status='Final')])
        self.write(self.previous, 'public_site/data/kickoffs.json', {'game': '2027-01-01T00:00:00Z'})
        self.write(self.root, 'public_site/data/predictions.json', [dict(self.row, model_version='v2')])
        with self.assertRaisesRegex(ValueError, 'Locked forecast changed'):
            self.transition()

    def test_archived_market_line_is_locked(self):
        old = dict(self.row, line_observed_at='2026-10-14T12:00:00Z', market_total=45)
        self.write(self.previous, 'public_site/data/predictions.json', [old])
        self.write(self.root, 'public_site/data/predictions.json', [dict(old, market_total=46)])
        with self.assertRaisesRegex(ValueError, 'Locked forecast changed'):
            self.transition()

    def test_copied_production_provenance(self):
        self.shadow()
        self.record.update(model='production_v1', original_forecast_at='2026-10-14T12:00:00Z')
        self.record.pop('input_sha256')
        self.save_shadow()
        validate(self.root)
        self.record['original_forecast_at'] = '2026-10-14T14:00:00Z'
        self.save_shadow()
        with self.assertRaisesRegex(ValueError, 'copied production'):
            validate(self.root)

    def test_current_exports(self):
        root = Path(__file__).resolve().parent
        self.assertGreater(validate_transition(root, root)['games'], 0)


if __name__ == '__main__':
    unittest.main()
