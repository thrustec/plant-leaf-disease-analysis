"""Checks for scientific invariants, failure accounting, and reviewed-mask gates."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiment import evaluate_final, freeze
from leaf_pipeline import (DEFAULT_SEGMENTATION, LABELS, ROOT, classification_metrics,
                           extract_features, load_rgb, mask_scores, predict_image,
                           segment_leaf, texture_region, write_csv, write_json)
from mask_review import export_reviewed, polygon_mask, reviewed_references
from prepare_data import LABEL_MAPPING, RAW, REVISION, resolve_group, validate_dataset, validate_source_record


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((256, 256, 3), 170, np.uint8)
        cv2.ellipse(self.rgb, (128, 128), (65, 95), 0, 0, 360, (30, 130, 45), -1)
        self.mask = np.zeros((256, 256), np.uint8)
        cv2.ellipse(self.mask, (128, 128), (65, 95), 0, 0, 360, 1, -1)
        self.mask = self.mask.astype(bool)

    def test_feature_shape_and_normalization(self):
        for mask in (None, self.mask):
            features = extract_features(self.rgb, mask)
            self.assertEqual(features.shape, (58,))
            self.assertTrue(np.isfinite(features).all())
            for start, stop in ((0, 16), (16, 32), (32, 48), (48, 58)):
                self.assertAlmostEqual(float(features[start:stop].sum()), 1)

    def test_masked_features_ignore_background(self):
        changed = self.rgb.copy()
        changed[~self.mask] = (255, 0, 255)
        np.testing.assert_allclose(extract_features(self.rgb, self.mask), extract_features(changed, self.mask))
        self.assertFalse(texture_region(self.mask)[~self.mask].any())
        full = texture_region(np.ones((256, 256), bool))
        self.assertFalse(full[0].any())
        self.assertFalse(full[:, -1].any())

    def test_segmentation_preserves_internal_symptoms(self):
        self.rgb[100:150, 110:145] = (170, 140, 100)
        mask = segment_leaf(self.rgb)
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask[125, 125])
        self.assertGreater(mask_scores(mask, self.mask)["iou"], 0.95)

    def test_unusable_masks_fail(self):
        for mask in (np.zeros((256, 256), bool), np.ones((256, 256), bool), np.zeros((10, 10), bool)):
            with self.assertRaises(ValueError):
                extract_features(self.rgb, mask)

    def test_mask_metrics_and_empty_convention(self):
        a = np.array([[1, 1], [0, 0]], bool)
        b = np.array([[1, 0], [1, 0]], bool)
        self.assertAlmostEqual(mask_scores(a, b)["iou"], 1 / 3)
        self.assertAlmostEqual(mask_scores(a, b)["dice"], 0.5)
        self.assertEqual(mask_scores(a, a), {"iou": 1.0, "dice": 1.0})
        self.assertEqual(mask_scores(a * 0, a * 0), {"iou": 0.0, "dice": 0.0})

    def test_failed_predictions_remain_in_metrics(self):
        report = classification_metrics(["Healthy", "Diseased", "Diseased"], ["Healthy", "Diseased", "FAILED"])
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertAlmostEqual(report["macro_f1"], 5 / 6)
        self.assertEqual(report["failures_by_true_class"], {"Healthy": 0, "Diseased": 1})
        self.assertEqual(sum(map(sum, report["covered_confusion_matrix"])) +
                         sum(report["failures_by_true_class"].values()), 3)
        failed = classification_metrics(list(LABELS), ["FAILED", "FAILED"])
        self.assertEqual(failed["macro_f1"], 0)
        self.assertEqual(failed["coverage"], 0)

    def test_label_mapping_and_group_resolution(self):
        self.assertNotIn("Tomato___Spider_mites Two-spotted_spider_mite", LABEL_MAPPING)
        self.assertNotIn("Tomato___Tomato_mosaic_virus", LABEL_MAPPING)
        mapping = {"test 001": ["Apple___healthy:::1", "Other___healthy:::2"]}
        self.assertEqual(resolve_group("uuid___Test 001.JPG", "Apple___healthy", mapping), "Apple___healthy:::1")
        self.assertIsNone(resolve_group("unknown.jpg", "Apple___healthy", mapping))
        mapping["test 001"].append("Apple___healthy:::2")
        self.assertIsNone(resolve_group("uuid___Test 001.JPG", "Apple___healthy", mapping))

    def test_acquisition_rejects_foreign_sources_and_paths(self):
        row = {"original_label": "Apple___healthy", "source_revision": REVISION,
               "source_path": "raw/color/Apple___healthy/leaf.jpg",
               "path": "data/raw/Apple___healthy/leaf.jpg", "source_url": RAW + "raw/color/Apple___healthy/leaf.jpg"}
        validate_source_record(row)
        for changed in ({"path": "../escape.jpg"}, {"source_url": "https://example.com/leaf.jpg"},
                        {"source_revision": "unfixed"}):
            with self.assertRaises(ValueError):
                validate_source_record(dict(row, **changed))

    def test_reload_consistency_and_invalid_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            image = path / "leaf.png"
            cv2.imwrite(str(image), cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
            features = extract_features(self.rgb)
            model = make_pipeline(StandardScaler(), SVC()).fit(
                np.array([features, features * 0.5, features * 0.25, features * 0.1]),
                ["Healthy", "Healthy", "Diseased", "Diseased"])
            artifact = {"method": "A", "pipeline": model, "segmentation": DEFAULT_SEGMENTATION}
            joblib.dump(artifact, path / "model.joblib")
            self.assertEqual(predict_image(image, artifact)[0], predict_image(image, path / "model.joblib")[0])
            (path / "bad.txt").write_text("not an image")
            with self.assertRaises(ValueError):
                load_rgb(path / "bad.txt")
            with self.assertRaises(ValueError):
                load_rgb(path / "missing.png")

    def test_reference_export_requires_both_actual_review_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"image_id": "leaf_0001", "annotation_partition": "development", "plant": "Apple",
                   "health": "Healthy", "path": "data/raw/leaf.png"}
            write_csv(root / "data/manifests/images.csv", [row])
            path = root / "data/annotations/labelme/development/leaf_0001.json"
            annotation = {"imageHeight": 256, "imageWidth": 256, "imagePath": "leaf_0001.png",
                          "flags": {"Emre_reviewed": False, "Merve_reviewed": False},
                          "shapes": [{"label": "leaf", "shape_type": "polygon", "points": [[10, 10], [50, 10], [50, 50]]}]}
            write_json(path, annotation)
            self.assertEqual(export_reviewed(root), [])
            annotation["flags"]["Emre_reviewed"] = True
            write_json(path, annotation)
            self.assertEqual(export_reviewed(root), [])
            annotation["flags"]["Merve_reviewed"] = True
            write_json(path, annotation)
            (root / "data/raw").mkdir(parents=True)
            cv2.imwrite(str(root / row["path"]), self.rgb)
            cv2.imwrite(str(path.with_suffix(".png")), self.rgb)
            self.assertEqual(len(export_reviewed(root)), 1)
            self.assertEqual(len(reviewed_references(root)), 1)
            annotation["shapes"][0]["points"][0] = [11, 11]
            write_json(path, annotation)
            with self.assertRaisesRegex(ValueError, "changed"):
                reviewed_references(root)

    def test_reference_polygon_validation(self):
        with self.assertRaises(ValueError):
            polygon_mask({"imageHeight": 256, "imageWidth": 256, "shapes": []})
        with self.assertRaises(ValueError):
            polygon_mask({"imageHeight": 255, "imageWidth": 256, "shapes": []})

    def test_final_evaluation_and_freeze_are_gated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root / "data/manifests/images.csv", [
                {"image_id": "test", "annotation_partition": "", "plant": "Apple", "health": "Healthy"}])
            with self.assertRaisesRegex(ValueError, "all 30"):
                evaluate_final(root)
            with self.assertRaisesRegex(ValueError, "12 outlines"):
                freeze(root)

    @unittest.skipUnless((ROOT / "data/manifests/images.csv").exists(), "Data acquisition not complete")
    def test_real_dataset_group_and_hash_separation(self):
        self.assertEqual(len(validate_dataset()), 900)


if __name__ == "__main__":
    unittest.main()
