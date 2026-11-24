"""Checks for the augmentation used in finetune_experiment.py (TensorFlow is not needed)."""

import unittest

import numpy as np

from finetune_experiment import SETTINGS, augment


class AugmentTests(unittest.TestCase):
    def test_shape_range_and_background_swap(self):
        leaf = np.full((256, 256, 3), 100, np.uint8)
        mask = np.zeros((256, 256), bool)
        mask[96:160, 96:160] = True
        pool = [np.full((256, 256, 3), 250, np.uint8)]
        size = SETTINGS["input"]
        plain = augment(leaf, mask, pool, np.random.default_rng(0), backgrounds=False)
        self.assertEqual(plain.shape, (size, size, 3))
        self.assertTrue((plain >= 0).all() and (plain <= 255).all())
        self.assertLess(plain.max(), 130)  # no background swap: only the leaf colour, brightness x<=1.2
        swapped = [augment(leaf, mask, pool, np.random.default_rng(seed), backgrounds=True).max() for seed in range(20)]
        self.assertTrue(any(value > 180 for value in swapped))  # the bright background appears in some draws


if __name__ == "__main__":
    unittest.main()
