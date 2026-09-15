import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from active_learning import grade_predictions, query_games, update_learning


def fixtures():
    return pd.DataFrame([
        dict(game_id=f'g{i}', season=2099, week=1 if i < 4 else 2,
             gameday='2099-09-20', home_team='SEA', away_team='NE',
             home_win_probability=0.45 + i * .03, predicted_winner='SEA',
             status='Final' if i < 4 else 'Scheduled',
             home_score=(20 if i % 2 else 10) if i < 4 else None,
             away_score=15 if i < 4 else None) for i in range(8)
    ])


class LearningTests(unittest.TestCase):
    def test_grading(self):
        rows = fixtures()
        rows.loc[2, 'home_score'] = 15
        result = grade_predictions(rows)
        self.assertEqual(result.prediction_result[:4].tolist(), ['No', 'Yes', 'Tie (no winner)', 'Yes'])
        self.assertTrue(result.prediction_correct[4:].isna().all())

    def test_query_does_not_use_labels(self):
        rows = fixtures()
        selected = query_games(rows, 3)
        rows['home_score'] = 99
        self.assertEqual(selected, query_games(rows, 3))
        self.assertEqual(len(set(selected)), 3)

    def test_prospective_evaluation_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = fixtures()
            first = update_learning(rows, root)
            self.assertEqual(first['training_games'], 4)
            self.assertEqual(first['evaluated_games'], 0)
            self.assertTrue((root / 'learning/candidate.joblib').exists())
            self.assertEqual(first, update_learning(rows, root))
            saved = json.loads((root / 'learning/state.json').read_text())['forecasts']['g4']
            rows.loc[4, ['status', 'home_score', 'away_score']] = ['Final', 27, 10]
            later = update_learning(rows, root)
            self.assertEqual(later['evaluated_games'], 1)
            evaluation = pd.read_csv(root / 'learning/evaluation.csv')
            self.assertAlmostEqual(evaluation.iloc[0].candidate, saved['candidate'])
            self.assertEqual(later, update_learning(rows, root))


if __name__ == '__main__':
    unittest.main()
