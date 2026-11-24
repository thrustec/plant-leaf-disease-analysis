"""Checks for the background-randomisation experiment (background_experiment.py)."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from background_experiment import background_pool, composite


class BackgroundTests(unittest.TestCase):
    def test_composite_keeps_leaf_and_replaces_background(self):
        leaf = np.full((256, 256, 3), 50, np.uint8)
        background = np.full((256, 256, 3), 200, np.uint8)
        mask = np.zeros((256, 256), bool)
        mask[100:150, 100:150] = True
        result = composite(leaf, mask, background)
        self.assertTrue((result[mask] == 50).all())
        self.assertTrue((result[~mask] == 200).all())

    def test_pool_excludes_evaluated_crops_and_copies(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = {}
            for name in ("train/Corn rust leaf/a.jpg", "train/Corn rust leaf/copy.jpg",
                         "train/Apple leaf/b.jpg", "test/Tomato leaf mosaic virus/c.jpg"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                images[name] = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
                cv2.imwrite(str(path), images[name])
            # "copy.jpg" duplicates the evaluated Apple image, so it must be rejected.
            cv2.imwrite(str(root / "train/Corn rust leaf/copy.jpg"), images["train/Apple leaf/b.jpg"])
            pool, rejected = background_pool(root, [{"path": "train/Apple leaf/b.jpg"}])
            self.assertEqual(pool, ["train/Corn rust leaf/a.jpg"])
            self.assertEqual(rejected, ["train/Corn rust leaf/copy.jpg"])


if __name__ == "__main__":
    unittest.main()
