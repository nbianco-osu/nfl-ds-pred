import unittest

import numpy as np
import pandas as pd

from retrain_updated import partitions, pipeline, probability_metrics
from sklearn.linear_model import Ridge


class UpdatedRetrainingTests(unittest.TestCase):
    def test_temporal_partitions(self):
        rows = pd.DataFrame({'season':[2023, 2024, 2025, 2026],
                             'gameday':pd.to_datetime(['2023-10-01', '2024-10-01', '2025-10-01', '2026-10-01'])})
        self.assertEqual([len(x) for x in partitions(rows)], [1, 1, 1, 1])
        rows.loc[0, 'gameday'] = pd.Timestamp('2025-11-01')
        with self.assertRaises(ValueError):
            partitions(rows)

    def test_missing_partition_rejected(self):
        with self.assertRaises(ValueError):
            partitions(pd.DataFrame({'season':[2026], 'gameday':[pd.Timestamp('2026-10-01')]}))

    def test_probability_validation(self):
        for p in [[np.nan, .5], [-.1, .5], [.5, 1.1]]:
            with self.assertRaises(ValueError):
                probability_metrics(np.array([0, 1]), p)
        self.assertEqual(probability_metrics(np.array([0, 1]), [.2, .8])['accuracy'], 1)

    def test_imputation_fits_training_only(self):
        train = pd.DataFrame({'a':[1., 3., np.nan], 'b':[np.nan]*3})
        model = pipeline(Ridge()).fit(train, np.array([[1, 2], [3, 4], [2, 3]]))
        self.assertEqual(model.named_steps['simpleimputer'].statistics_[0], 2.)
        self.assertTrue(np.isfinite(model.predict(pd.DataFrame({'a':[1000.], 'b':[np.nan]}))).all())
        self.assertEqual(model.named_steps['simpleimputer'].statistics_[0], 2.)


if __name__ == '__main__':
    unittest.main()
