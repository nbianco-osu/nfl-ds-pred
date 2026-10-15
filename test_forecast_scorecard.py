import unittest
import pandas as pd
from forecast_scorecard import before, cohort
from performance import summarize


class StrictScorecardTests(unittest.TestCase):
    def test_timestamp_gate(self):
        k = '2026-10-15T20:00:00Z'
        self.assertTrue(before('2026-10-15T19:00:00Z', k))
        for v in [None, '2026-10-15', k, '2026-10-16T20:00:00Z', 'bad']:
            self.assertFalse(before(v, k))

    def test_independent_score_and_line_provenance(self):
        row = dict(game_id='g', status='Final', home_team='A', away_team='B',
                   home_score=20, away_score=10, predicted_winner='A', home_win_probability=.7,
                   winner_forecast_at=None, score_forecast_at='2026-10-15T19:00:00Z',
                   predicted_home_score=21, predicted_away_score=12,
                   line_observed_at=None, spread_grade='Correct', total_grade='Wrong')
        rows = pd.DataFrame([row]); times = {'g': '2026-10-15T20:00:00Z'}
        self.assertEqual(len(cohort(rows, times, 'winner', 'pregame')), 0)
        result = summarize(cohort(rows, times, 'score', 'pregame'))
        self.assertEqual(result['score_n'], 1)
        self.assertEqual(result['winner_n'], 0)
        self.assertEqual(result['spread']['n'], 0)


if __name__ == '__main__': unittest.main()
