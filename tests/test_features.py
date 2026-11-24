"""Checks for the extra feature groups in feature_experiment.py."""

import unittest

import cv2
import numpy as np

from feature_experiment import GROUPS, extra_features, glcm_features, lesion_features, shape_features


def leaf(spots=0):
    rgb = np.full((256, 256, 3), 170, np.uint8)
    cv2.ellipse(rgb, (128, 128), (60, 90), 0, 0, 360, (40, 140, 50), -1)
    for k in range(spots):
        cv2.circle(rgb, (100 + 15 * (k % 4), 90 + 20 * (k // 4)), 4, (120, 70, 20), -1)  # brown lesions
    mask = np.zeros((256, 256), np.uint8)
    cv2.ellipse(mask, (128, 128), (60, 90), 0, 0, 360, 1, -1)
    return rgb, mask.astype(bool)


class FeatureTests(unittest.TestCase):
    def test_lengths_and_finite(self):
        rgb, mask = leaf(spots=8)
        values = extra_features(rgb, mask)
        self.assertEqual(values.shape, (sum(GROUPS.values()),))
        self.assertTrue(np.isfinite(values).all())

    def test_lesions_are_counted(self):
        healthy, mask = leaf()
        spotted, _ = leaf(spots=8)
        self.assertEqual(lesion_features(healthy, mask)[0], 0)
        share, per_thousand = lesion_features(spotted, mask)[:2]
        self.assertGreater(share, 0.01)
        self.assertAlmostEqual(per_thousand * mask.sum() / 1000, 8)

    def test_glcm_ignores_background(self):
        rgb, mask = leaf()
        changed = rgb.copy()
        changed[~mask] = 0
        self.assertTrue(np.allclose(glcm_features(rgb, mask), glcm_features(changed, mask)))

    def test_shape_of_ellipse(self):
        _, mask = leaf()
        solidity, extent, eccentricity, roughness = shape_features(mask)
        self.assertGreater(solidity, 0.98)
        self.assertAlmostEqual(extent, np.pi / 4, delta=0.03)
        self.assertAlmostEqual(eccentricity, np.sqrt(1 - (60 / 90) ** 2), delta=0.02)
        self.assertLess(roughness, 1.2)


if __name__ == "__main__":
    unittest.main()
