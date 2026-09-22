import unittest
import numpy as np
import pandas as pd
from simulate_season import simulate
from current_features import roll_forward


class SimulationTests(unittest.TestCase):
    def test_conservation_reproducibility_and_fixed_results(self):
        rows = pd.DataFrame([
            dict(game_id='a', home_team='A', away_team='B', home_win_probability=.4, status='Final', home_score=20, away_score=10),
            dict(game_id='b', home_team='B', away_team='A', home_win_probability=.5, status='Scheduled', home_score=None, away_score=None),
        ])
        result = simulate(rows, simulations=5000)
        pd.testing.assert_frame_equal(result, simulate(rows, simulations=5000))
        self.assertAlmostEqual(result.expected_wins.sum(), 2)
        self.assertEqual(result.set_index('team').loc['A', 'actual_wins'], 1)
        self.assertGreaterEqual(result.set_index('team').loc['A', 'low_wins'], 1)
        self.assertAlmostEqual(result.set_index('team').loc['B', 'expected_wins'], .5, delta=.04)

    def test_no_same_game_stats(self):
        stats = pd.DataFrame({'team': ['A', 'A'], 'week': [1, 2], 'epa': [1., 999.]})
        games = pd.DataFrame({'game_id': ['a', 'b', 'c'], 'season': [2026]*3, 'week': [1, 2, 3], 'home_team': ['A']*3, 'away_team': ['B']*3})
        result = roll_forward(stats, games, ['epa'], 'team_pbp').set_index(['game_id', 'team'])
        self.assertTrue(np.isnan(result.loc[('a', 'A'), 'team_pbp_epa_avg']))
        self.assertEqual(result.loc[('b', 'A'), 'team_pbp_epa_avg'], 1)
        self.assertEqual(result.loc[('c', 'A'), 'team_pbp_epa_avg'], 500)


if __name__ == '__main__':
    unittest.main()
