import unittest

import pandas as pd

from performance import summarize
from rolling_backtest import FEATURES, prepare, run


class PerformanceTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            dict(status='Final', home_team='A', away_team='B', home_score=20, away_score=10,
                 home_win_probability=.8, predicted_winner='A', predicted_home_score=22,
                 predicted_away_score=9, score_forecast_at='2026-09-01', spread_grade='Correct', total_grade='Push'),
            dict(status='Final', home_team='A', away_team='B', home_score=10, away_score=10,
                 home_win_probability=.6, predicted_winner='A', predicted_home_score=None,
                 predicted_away_score=None, score_forecast_at=None, spread_grade='Push', total_grade='No archived pick'),
            dict(status='Scheduled', home_team='A', away_team='B', home_score=None, away_score=None,
                 home_win_probability=.9, predicted_winner='A', predicted_home_score=25,
                 predicted_away_score=5, score_forecast_at='2026-09-01', spread_grade='Pending', total_grade='Pending'),
        ])

    def test_ties_pushes_and_pending_excluded(self):
        result = summarize(self.fixture())
        self.assertEqual((result['completed'], result['winner_n'], result['ties']), (2, 1, 1))
        self.assertEqual(result['correct'], 1)
        self.assertAlmostEqual(result['brier'], .04)
        self.assertEqual(result['spread']['n'], 1)
        self.assertEqual(result['total']['n'], 0)
        self.assertEqual(result['score_n'], 1)
        self.assertEqual(result['team_score_mae'], 1.5)
        self.assertEqual(sum(b['n'] for b in result['calibration']), 1)

    def test_no_results(self):
        result = summarize(self.fixture().iloc[2:])
        self.assertIsNone(result['accuracy'])
        self.assertIsNone(result['margin_mae'])

    def test_missing_score_timestamp_excluded(self):
        rows = self.fixture()
        rows['score_forecast_at'] = None
        self.assertEqual(summarize(rows)['score_n'], 0)

    def test_probability_validation(self):
        rows = self.fixture()
        rows.loc[0, 'home_win_probability'] = 1.2
        with self.assertRaises(ValueError):
            summarize(rows)


class BacktestTests(unittest.TestCase):
    def games(self):
        return pd.DataFrame([dict(game_id=str(i), season=2000+i//16, week=i%16+1,
                                  gameday=(pd.Timestamp('2000-01-01')+pd.Timedelta(days=i*7)).isoformat(),
                                  home_team='A', away_team='B', home_score=14+i%13,
                                  away_score=20+(i%5)) for i in range(300)])

    def test_future_labels_cannot_change_earlier_features(self):
        games = self.games()
        before = prepare(games)
        games.loc[280:, 'home_score'] = 99
        after = prepare(games)
        pd.testing.assert_frame_equal(before.loc[:280, FEATURES], after.loc[:280, FEATURES])

    def test_week_frozen_and_duplicate_rejected(self):
        games = self.games().iloc[:3].copy()
        games.loc[1, 'week'] = 1
        prepared = prepare(games)
        self.assertTrue((prepared.loc[1, FEATURES] == 21).all())
        with self.assertRaises(ValueError):
            prepare(pd.concat([games, games]))

    def test_rolling_predictions_strict_cutoffs_and_future_invariance(self):
        games = self.games()
        predictions, folds = run(games, 2018, 2018)
        self.assertTrue(all(f['training_through'] < f['cutoff'] for f in folds))
        self.assertFalse(predictions.game_id.duplicated().any())
        games.loc[299, 'home_score'] = 99
        altered, _ = run(games, 2018, 2018)
        pd.testing.assert_series_equal(predictions.form_probability, altered.form_probability)


if __name__ == '__main__':
    unittest.main()
