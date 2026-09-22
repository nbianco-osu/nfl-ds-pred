import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from gpr_models import CalibratedGPRClassifier, FEATURES, KERNELS


class GPRTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.normal(size=(80, len(FEATURES))), columns=FEATURES)
        self.X["season"] = 2025
        self.X["week"] = np.repeat(np.arange(1, 21), 4)
        self.y = np.tile([0, 1], 40)

    def test_kernels_probabilities_and_serialization(self):
        for kernel in KERNELS:
            with self.subTest(kernel=kernel):
                model = clone(CalibratedGPRClassifier(kernel, calibration_rows=16, max_train_rows=40))
                model.fit(self.X, self.y)
                p = model.predict_proba(self.X)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p >= 0) & (p <= 1)).all())
                np.testing.assert_allclose(p.sum(axis=1), 1)
                self.assertLess(model.training_end_, model.calibration_start_)
                self.assertEqual(model.training_rows_, 40)
                self.assertIn("WhiteKernel", str(model.regressor_.kernel_))
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.joblib"
                    joblib.dump(model, path)
                    np.testing.assert_allclose(p, joblib.load(path).predict_proba(self.X))

    def test_calibration_labels_cannot_affect_regression_fit(self):
        first = CalibratedGPRClassifier(calibration_rows=16).fit(self.X, self.y)
        changed = self.y.copy()
        changed[-16:] = 1 - changed[-16:]
        second = CalibratedGPRClassifier(calibration_rows=16).fit(self.X, changed)
        np.testing.assert_allclose(first.regressor_.alpha_, second.regressor_.alpha_)

    def test_reject_single_class(self):
        with self.assertRaises(ValueError):
            CalibratedGPRClassifier(calibration_rows=16).fit(self.X, np.zeros(80))


if __name__ == "__main__":
    unittest.main()
