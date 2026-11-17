"""Checks that the interactive notebook tools show the real pipeline and never touch the benchmark."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import explore
from leaf_pipeline import DEFAULT_SEGMENTATION, ROOT, read_manifest, segment_leaf, write_json


class ExploreTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((256, 256, 3), 170, np.uint8)
        cv2.ellipse(self.rgb, (128, 128), (65, 95), 0, 0, 360, (30, 130, 45), -1)
        self.rgb[100:150, 110:145] = (170, 140, 100)

    def test_steps_match_shared_segmentation(self):
        for parameters in (None, {"distance": 12.0, "closing": 3}, {"distance": 30.0, "border": 9}):
            steps = explore.segmentation_steps(self.rgb, parameters)
            expected = segment_leaf(self.rgb, dict(DEFAULT_SEGMENTATION, **(parameters or {})))
            np.testing.assert_array_equal(steps["final"], expected)
            self.assertEqual(steps["distance"].shape, (256, 256))

    def test_invalid_parameters_rejected(self):
        for parameters in ({"threshold": 5}, {"distance": -1}, {"closing": 0}, {"border": 0}, {"border": 129}):
            with self.assertRaises(ValueError):
                explore.segmentation_steps(self.rgb, parameters)
        for border in (1, 128):
            explore.segmentation_steps(self.rgb, {"border": border})

    def test_example_selection_excludes_test_images(self):
        rows = [{"image_id": "leaf_dev", "partition": "development", "path": "data/raw/a.jpg"},
                {"image_id": "leaf_test", "partition": "test", "path": "data/raw/b.jpg"}]
        self.assertEqual(explore.development_image_path(rows, "leaf_dev", Path("/p")), Path("/p/data/raw/a.jpg"))
        with self.assertRaisesRegex(ValueError, "locked test"):
            explore.development_image_path(rows, "leaf_test")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            explore.development_image_path(rows, "leaf_none")

    def test_parameter_experiment_applies_pipeline_gates(self):
        rows = [{"image_id": f"leaf_{i}", "health": ("Healthy", "Diseased")[i % 2],
                 "cv_fold": str(i % 3), "partition": "development"} for i in range(30)]
        features = np.random.default_rng(0).random((30, 58))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(explore, "validate_dataset", return_value=rows), \
                mock.patch.object(explore, "reviewed_references", return_value=[]):
            root = Path(directory)
            too_few = ["unusable"] * 7 + [""] * 23     # 23/30 < 80%
            with mock.patch.object(explore, "feature_matrices", return_value=(features, features, too_few)):
                with self.assertRaisesRegex(ValueError, "80%"):
                    explore.run_parameter_experiment(name="few", root=root)
            self.assertFalse((root / "results/experiments/few/summary.json").exists())
            self.assertTrue((root / "results/experiments/few/unusable_masks.csv").exists())
            # Removing every Healthy image from fold 0 (6 of 30, still >= 80%) empties a validation class.
            lost_class = ["unusable" if i % 3 == 0 and i % 2 == 0 else "" for i in range(30)]
            self.assertGreaterEqual(lost_class.count(""), 24)
            with mock.patch.object(explore, "feature_matrices", return_value=(features, features, lost_class)):
                with self.assertRaisesRegex(ValueError, "missing a health class"):
                    explore.run_parameter_experiment(name="fold", root=root)
            write_json(root / "results/development/summary.json", {
                "segmentation": DEFAULT_SEGMENTATION, "shared_training_images": 30,
                "methods": {}, "cv_macro_f1_B_minus_A": 0.0, "segmentation_status": "x"})
            some = ["unusable"] + [""] * 29
            with mock.patch.object(explore, "feature_matrices", return_value=(features, features, some)):
                summary = explore.run_parameter_experiment(name="ok", root=root)
            self.assertEqual(summary["shared_training_images"], 29)
            self.assertFalse(summary["comparable_to_recorded"])

    def test_figures_for_a_saved_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaf.png"
            cv2.imwrite(str(path), cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
            figure, _ = explore.show_segmentation_steps(path, {"distance": 24.0})
            plt.close(figure)
            figure, features = explore.show_features(path)
            plt.close(figure)
            self.assertEqual(features["A"].shape, (58,))
            self.assertEqual(features["B"].shape, (58,))

    def test_experiment_folder_cannot_reach_protected_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../../final", "../development", "a/b", ".hidden", ""):
                if name:
                    with self.assertRaises(ValueError):
                        explore._experiment_folder(root, name)
            folder = explore._experiment_folder(root, "trial-1")
            self.assertEqual(folder, root / "results/experiments/trial-1")

    def test_reproduction_requires_stored_benchmark_and_leaves_it_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "No stored final"):
                explore.reproduce_final(root)
            methods = {m: {"macro_f1": 0.9, "accuracy": 0.9, "coverage": 1.0} for m in ("A", "B")}
            stored = {"methods": methods, "segmentation": {"iou": {"mean": 0.8}, "dice": {"mean": 0.85}}}
            write_json(root / "results/final/metrics.json", stored)
            before = (root / "results/final/metrics.json").read_bytes()

            def fake_evaluate(scratch):
                self.assertFalse((scratch / "results/final").exists())
                output = scratch / "results/final"
                output.mkdir(parents=True)
                for name in ("metrics.json", "predictions.csv", "segmentation.csv", "confusion_matrices.png"):
                    (output / name).write_text("x")
                return stored

            with mock.patch.object(explore, "evaluate_final", fake_evaluate):
                comparison = explore.reproduce_final(root)
            self.assertTrue(all(row["match"] for row in comparison))
            self.assertEqual((root / "results/final/metrics.json").read_bytes(), before)
            self.assertTrue((root / "results/experiments/final_reproduction/comparison.csv").exists())

    @unittest.skipUnless((ROOT / "data/raw").exists(), "Data acquisition not complete")
    def test_parameter_experiment_on_real_development_data(self):
        protected = sorted(p for d in ("models", "results/development") for p in (ROOT / d).rglob("*") if p.is_file())
        before = {p: p.read_bytes() for p in protected}
        summary = explore.run_parameter_experiment(name="unittest_default")
        recorded = summary.get("recorded_development")
        if recorded and recorded["segmentation"] == DEFAULT_SEGMENTATION:
            for method in ("A", "B"):
                self.assertAlmostEqual(summary["methods"][method]["cv_macro_f1"],
                                       recorded["methods"][method]["cv_macro_f1"])
        self.assertEqual({p: p.read_bytes() for p in protected}, before)
        self.assertEqual(len(read_manifest()), 900)


if __name__ == "__main__":
    unittest.main()
