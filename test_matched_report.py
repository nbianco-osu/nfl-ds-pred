"""Validate the published R18 artifact independently of the training run."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


class MatchedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        report = root / 'public_site/data/matched_comparison.json'
        if not report.exists():
            raise unittest.SkipTest('Full R18 evaluation not exported yet')
        cls.report = json.loads(report.read_text())
        cls.rows = pd.read_csv(root / 'backtests/matched_predictions.csv')

    def test_identical_cohort_and_valid_probabilities(self):
        rows = self.rows
        self.assertFalse(rows.duplicated(['model_id', 'game_id']).any())
        self.assertEqual(rows.model_id.nunique(), 64)
        cohorts = [set(group.game_id) for _, group in rows.groupby('model_id')]
        self.assertTrue(all(cohort == cohorts[0] for cohort in cohorts))
        self.assertEqual(len(cohorts[0]), self.report['games'])
        self.assertTrue(rows.home_win.isin([0, 1]).all())
        self.assertTrue(rows[['probability', 'baseline_probability']].apply(lambda s: s.between(0, 1)).all().all())
        self.assertTrue(np.isfinite(rows.probability).all())

    def test_metrics_reproduce_and_folds_are_disjoint(self):
        for model in self.report['models']:
            rows = self.rows.loc[self.rows.model_id.eq(model['id'])]
            y, p = rows.home_win.to_numpy(), rows.probability.to_numpy()
            self.assertEqual(model['overall']['n'], len(rows))
            self.assertAlmostEqual(model['overall']['brier'], np.mean((p-y)**2))
            self.assertAlmostEqual(model['overall']['log_loss'], np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
        for fold in self.report['folds']:
            self.assertLess(fold['base_through'], fold['calibration_start'])
            self.assertLess(fold['final_fit_through'], fold['selection_start'])
            self.assertLess(fold['final_calibration_through'], fold['test_start'])
            self.assertEqual(len(fold['recipes']), 64)
            for recipe in fold['recipes']:
                self.assertEqual(recipe['selected'], int(np.argmin(recipe['selection_losses'])))
                self.assertTrue(0 <= recipe['missing_test_fraction'] <= 1)
