"""Checks for the CNN-input construction in cnn_experiment.py (TensorFlow is not needed)."""

import unittest

import cv2
import numpy as np

from cnn_experiment import GREY, masked_inputs
from leaf_pipeline import DEFAULT_SEGMENTATION, segment_leaf


class CnnInputTests(unittest.TestCase):
    def test_background_is_painted_grey(self):
        rgb = np.full((256, 256, 3), 170, np.uint8)
        cv2.ellipse(rgb, (128, 128), (65, 95), 0, 0, 360, (30, 130, 45), -1)
        inputs = masked_inputs(rgb, DEFAULT_SEGMENTATION)
        self.assertIs(inputs["A"], rgb)
        mask = segment_leaf(rgb, DEFAULT_SEGMENTATION)
        for method in ("B", "B2"):
            self.assertTrue((inputs[method][~mask if method == "B" else (0, 0)] == GREY).all())
            self.assertTrue(np.array_equal(inputs[method][128, 128], rgb[128, 128]))

    def test_failed_mask_gives_no_input(self):
        flat = np.full((256, 256, 3), 120, np.uint8)  # no leaf: the Lab mask is empty
        inputs = masked_inputs(flat, DEFAULT_SEGMENTATION)
        self.assertIsNone(inputs["B"])
        self.assertIsNotNone(inputs["A"])


if __name__ == "__main__":
    unittest.main()
