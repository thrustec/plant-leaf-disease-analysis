"""Checks that the background-only control never measures leaf pixels near the predicted leaf."""

import unittest

import cv2
import numpy as np

from background_control import background_region
from leaf_pipeline import DEFAULT_SEGMENTATION, extract_features, segment_leaf


class BackgroundControlTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((256, 256, 3), 170, np.uint8)
        cv2.ellipse(self.rgb, (128, 128), (65, 95), 0, 0, 360, (30, 130, 45), -1)

    def test_region_keeps_margin_from_leaf(self):
        leaf = segment_leaf(self.rgb, DEFAULT_SEGMENTATION)
        for margin in (7, 21):
            region = background_region(self.rgb, DEFAULT_SEGMENTATION, margin)
            self.assertFalse((region & leaf).any())
            distance = cv2.distanceTransform((~leaf).astype(np.uint8), cv2.DIST_L2, 5)
            self.assertGreaterEqual(float(distance[region].min()), margin - 1)
        self.assertGreater(background_region(self.rgb, DEFAULT_SEGMENTATION, 7).sum(),
                           background_region(self.rgb, DEFAULT_SEGMENTATION, 21).sum())

    def test_background_features_ignore_leaf_changes(self):
        region = background_region(self.rgb, DEFAULT_SEGMENTATION, 7)
        changed = self.rgb.copy()
        changed[segment_leaf(self.rgb, DEFAULT_SEGMENTATION)] = (200, 40, 40)
        np.testing.assert_allclose(extract_features(self.rgb, region), extract_features(changed, region))


if __name__ == "__main__":
    unittest.main()
