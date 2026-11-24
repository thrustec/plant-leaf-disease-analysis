"""Checks for the post-freeze GrabCut segmentation (segmentation_grabcut.py)."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from extended_analysis import require_case_sensitive
from grabcut_experiment import SKIP_FLAG, reference_mask
from leaf_pipeline import ROOT, DEFAULT_SEGMENTATION, mask_scores, segment_leaf
from segmentation_grabcut import central_component, grabcut_leaf


def leaf_scene():
    """A central leaf joined by a thin stem to a larger neighbouring leaf in the corner,
    the typical field-photo case where the frozen mask keeps both."""
    rgb = np.full((256, 256, 3), (170, 165, 150), np.uint8)
    cv2.ellipse(rgb, (128, 128), (50, 70), 0, 0, 360, (40, 140, 50), -1)
    cv2.ellipse(rgb, (225, 225), (60, 60), 0, 0, 360, (90, 160, 40), -1)
    cv2.line(rgb, (160, 170), (195, 195), (60, 150, 45), 3)
    noise = np.random.default_rng(0).normal(0, 6, rgb.shape)
    rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    leaf = np.zeros((256, 256), np.uint8)
    cv2.ellipse(leaf, (128, 128), (50, 70), 0, 0, 360, 1, -1)
    return rgb, leaf.astype(bool)


class GrabCutTests(unittest.TestCase):
    def test_central_leaf_is_separated_from_neighbour(self):
        rgb, leaf = leaf_scene()
        frozen = mask_scores(segment_leaf(rgb, DEFAULT_SEGMENTATION), leaf)["dice"]
        refined = mask_scores(grabcut_leaf(rgb, DEFAULT_SEGMENTATION), leaf)["dice"]
        self.assertGreater(refined, 0.95)
        self.assertGreater(refined, frozen)

    def test_deterministic(self):
        rgb, _ = leaf_scene()
        first = grabcut_leaf(rgb, DEFAULT_SEGMENTATION)
        self.assertTrue(np.array_equal(first, grabcut_leaf(rgb, DEFAULT_SEGMENTATION)))

    def test_central_component_prefers_centre_over_size(self):
        mask = np.zeros((256, 256), np.uint8)
        cv2.circle(mask, (128, 128), 30, 1, -1)
        cv2.rectangle(mask, (0, 0), (90, 90), 1, -1)  # larger, but in the corner
        chosen = central_component(mask)
        self.assertTrue(chosen[128, 128])
        self.assertFalse(chosen[10, 10])

    def test_plantdoc_annotations_are_complete(self):
        folder = ROOT / "data/annotations/plantdoc"
        drawn = 0
        for path in sorted(folder.glob("pd_*.json")):
            annotation = json.loads(path.read_text())
            self.assertEqual((annotation["imageHeight"], annotation["imageWidth"]), (256, 256))
            if annotation.get("flags", {}).get(SKIP_FLAG):
                continue
            self.assertTrue(all(s["label"] == "leaf" and s["shape_type"] == "polygon" for s in annotation["shapes"]))
            self.assertTrue(reference_mask(annotation).any())
            drawn += 1
        self.assertGreaterEqual(drawn, 30)  # 10 selection + 20 test images

    def test_case_insensitive_plantdoc_disk_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            rust = Path(folder) / "train/Apple rust leaf"
            rust.mkdir(parents=True)
            (rust / "CAR1.jpg").write_bytes(b"first image")
            with self.assertRaises(SystemExit):  # only one of the two files survived
                require_case_sensitive(folder)
            (rust / "car1.jpg").symlink_to(rust / "CAR1.jpg")
            with self.assertRaises(SystemExit):  # both names point to the same file
                require_case_sensitive(folder)
            (rust / "car1.jpg").unlink()
            (rust / "car1.jpg").write_bytes(b"second image")
            require_case_sensitive(folder)


if __name__ == "__main__":
    unittest.main()
