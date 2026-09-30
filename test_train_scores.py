import unittest
import numpy as np
import pandas as pd
from train_scores import score_metrics, market_evaluation, candidates


class TrainingTests(unittest.TestCase):
    def test_metrics_use_home_away_order(self):
        result = score_metrics(np.array([[24, 17]]), np.array([[21, 20]]))
        self.assertEqual(result['total_mae'], 0)
        self.assertEqual(result['margin_mae'], 6)
        self.assertEqual(result['home_mae'], 3)

    def test_market_denominators_exclude_pushes_missing_and_no_edge(self):
        base = dict(game_id='g', home_team='A', away_team='B', home_score=24, away_score=17,
                    spread_line=3, total_line=40)
        games = pd.DataFrame([base, dict(base, spread_line=7, total_line=41),
                              dict(base, spread_line=None, total_line=None),
                              dict(base, spread_line=7, total_line=45)])
        summary, rows = market_evaluation(games, np.array([[28,17], [28,17], [28,17], [26,19]]))
        self.assertEqual(summary['spread']['graded'], 1)
        self.assertEqual(summary['spread']['correct'], 1)
        self.assertEqual(summary['spread']['pushes'], 1)
        self.assertEqual(summary['spread']['no_edge'], 1)
        self.assertEqual(summary['spread']['missing'], 1)
        self.assertEqual(summary['total']['graded'], 1)
        self.assertIsNone(rows[2]['model_total_side'])

    def test_four_families_with_two_configurations(self):
        configs = list(candidates())
        self.assertEqual(len(configs), 4)
        self.assertTrue(all(len(options) == 2 for _, options in configs))


if __name__ == '__main__':
    unittest.main()
