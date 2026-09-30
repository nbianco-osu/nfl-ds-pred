import unittest
from datetime import datetime, timezone
from score_markets import comparisons, pregame, grade
from score_markets import refresh, SNAPSHOT_FIELDS
from unittest.mock import patch, Mock
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np


class ScoreMarketsTests(unittest.TestCase):
    def setUp(self):
        self.game = dict(game_id='test', home_team='BUF', away_team='NE', spread_line=3,
                         total_line=47, home_score=27, away_score=20,
                         gameday='2026-10-04', gametime='13:00')
        self.saved = dict(predicted_home_score=28, predicted_away_score=20,
                          market_home_spread=-3, market_total=47)

    def test_home_favorite_and_push(self):
        result = comparisons(self.game, self.saved)
        self.assertEqual(result['market_home_spread'], -3)
        self.assertEqual(result['model_spread_side'], 'BUF')
        self.assertEqual(result['spread_grade'], 'Correct')
        self.assertEqual(result['model_total_side'], 'Over')
        self.assertEqual(result['total_grade'], 'Push')

    def test_archive_beats_changed_source(self):
        self.game['spread_line'] = 10
        self.game['total_line'] = 60
        result = comparisons(self.game, self.saved)
        self.assertEqual(result['actual_spread_side'], 'BUF')
        self.assertEqual(result['actual_total_side'], 'Push')

    def test_no_retrospective_picks(self):
        result = comparisons(self.game)
        self.assertIsNone(result['predicted_home_score'])
        self.assertIsNone(result['model_spread_side'])
        self.assertEqual(result['spread_grade'], 'No archived pick')
        self.assertEqual(result['actual_spread_side'], 'BUF')

    def test_missing_archived_line_never_backfills_pick(self):
        self.saved['market_total'] = None
        result = comparisons(self.game, self.saved)
        self.assertIsNone(result['model_total_side'])
        self.assertEqual(result['total_grade'], 'No archived pick')

    def test_underdog_and_wrong(self):
        self.saved['market_home_spread'] = 8
        self.game['home_score'] = 10
        self.assertEqual(comparisons(self.game, self.saved)['spread_grade'], 'Wrong')
        self.assertEqual(grade('Over', 'Under', True), 'Wrong')
        self.assertEqual(grade('Push', 'Under', True), 'No edge')

    def test_kickoff_freeze_and_missing_time(self):
        self.game.update(home_score=None, away_score=None)
        self.assertTrue(pregame(self.game, datetime(2026,10,4,16,59,tzinfo=timezone.utc)))
        self.assertFalse(pregame(self.game, datetime(2026,10,4,17,tzinfo=timezone.utc)))
        self.game['gametime'] = None
        self.assertFalse(pregame(self.game, datetime(2026,10,4,12,tzinfo=timezone.utc)))

    def test_refresh_preserves_started_forecasts_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'predictions').mkdir()
            (root / 'public_site/data').mkdir(parents=True)
            schedule = pd.DataFrame([dict(self.game, season=2026), dict(self.game, game_id='future', season=2026, home_score=None, away_score=None)])
            archive = root / 'predictions/nfl_2026_score_predictions.csv'
            original = dict(self.saved, game_id='test', score_model_version='old', score_forecast_at='old', line_observed_at='old')
            pd.DataFrame([original], columns=SNAPSHOT_FIELDS).to_csv(archive, index=False)
            model = Mock()
            model.predict.return_value = np.array([[24,20]])
            artifact = dict(models=[model], metrics={}, model_version='new')
            now = datetime(2026,10,1,tzinfo=timezone.utc)
            with patch('score_markets.joblib.load', return_value=artifact), patch('score_markets.build_schedule_features', return_value=(None,pd.DataFrame([{'x':1}]))):
                first = refresh(schedule[['game_id']], schedule, None, None, root, now)
                second = refresh(first, schedule, None, None, root, now)
            pd.testing.assert_frame_equal(first,second)
            saved = pd.read_csv(archive).set_index('game_id')
            self.assertEqual(saved.loc['test','score_model_version'], 'old')
            self.assertEqual(saved.loc['test','predicted_home_score'], 28)
            self.assertEqual(saved.loc['future','predicted_home_score'], 24)
            self.assertFalse(saved.index.duplicated().any())


if __name__ == '__main__':
    unittest.main()
