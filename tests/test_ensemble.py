"""Checks for the B2 + FT-aug combination rule in ensemble_experiment.py (TensorFlow is not needed)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ensemble_experiment import b2_model, b2_probability, combine, plot


class EnsembleTests(unittest.TestCase):
    def test_equal_weight_mean_and_threshold(self):
        p, labels = combine([0.2, 0.9, 0.4, 0.5], [0.9, 0.0, 0.6, 0.5])
        self.assertTrue(np.allclose(p, [0.55, 0.45, 0.5, 0.5]))
        self.assertEqual(list(labels), ["Diseased", "Healthy", "Diseased", "Diseased"])

    def test_failed_b2_mask_uses_ft_only(self):
        p, labels = combine([np.nan, np.nan], [0.3, 0.8])
        self.assertTrue(np.allclose(p, [0.3, 0.8]))
        self.assertEqual(list(labels), ["Healthy", "Diseased"])

    def test_b2_probability_is_for_diseased(self):
        generator = np.random.default_rng(0)
        healthy, diseased = generator.normal(-2, 1, (40, 3)), generator.normal(2, 1, (40, 3))
        features = np.vstack([healthy, diseased])
        labels = np.array(["Healthy"] * 40 + ["Diseased"] * 40, dtype=object)
        model = b2_model(10.0).fit(features, labels)
        query = np.array([[-3.0, -3.0, -3.0], [3.0, 3.0, 3.0], [np.nan, 0.0, 0.0]])
        p = b2_probability(model, query)
        self.assertLess(p[0], 0.2)
        self.assertGreater(p[1], 0.8)
        self.assertTrue(np.isnan(p[2]))

    def test_plot_writes_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "figure.png"
            plot(np.array([0.1, 0.8, np.nan]), np.array([0.9, 0.7, 0.2]),
                 np.array(["Healthy", "Diseased", "Healthy"], dtype=object), path)
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
