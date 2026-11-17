"""Interactive exploration for the notebook: intermediate processing steps, method comparison
on any image, editable parameter experiments, and a reproduction of the locked evaluation.

Every output of this module is written under results/experiments/. It never replaces the
models, the development summary, the frozen protocol, or the stored final benchmark, and it
does not modify leaf_pipeline.py or experiment.py, whose hashes the frozen protocol records.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiment import evaluate_final, feature_matrices
from leaf_pipeline import (DEFAULT_SEGMENTATION, IMAGE_SIZE, LABELS, ROOT, extract_features, load_rgb,
                           mask_scores, overlay_mask, predict_image, read_manifest, segment_leaf,
                           texture_region, validate_mask, write_csv, write_json)
from mask_review import reviewed_references
from prepare_data import validate_dataset

EXPERIMENTS = Path("results/experiments")
PROTECTED = ("models", "results/development", "results/final")


def _parameters(parameters):
    merged = dict(DEFAULT_SEGMENTATION, **(parameters or {}))
    unknown = set(merged) - set(DEFAULT_SEGMENTATION)
    if unknown:
        raise ValueError(f"Unknown segmentation parameters: {sorted(unknown)}")
    if merged["distance"] <= 0 or min(int(merged[k]) for k in ("border", "opening", "closing")) < 1:
        raise ValueError("Segmentation parameters must be positive")
    if int(merged["border"]) > IMAGE_SIZE // 2:
        # Wider frames overlap in segment_leaf and would count pixels twice.
        raise ValueError(f"border must be between 1 and {IMAGE_SIZE // 2} pixels")
    return merged


def development_image_path(rows, image_id, root=ROOT):
    """Resolve an image_id for interactive inspection; test images are not available here."""
    by_id = {row["image_id"]: row for row in rows}
    if image_id not in by_id:
        raise ValueError(f"Unknown image_id: {image_id}")
    if by_id[image_id]["partition"] != "development":
        raise ValueError(f"{image_id} is a locked test image; choose a development image")
    return Path(root) / by_id[image_id]["path"]


def segmentation_steps(rgb, parameters=None):
    """Return each intermediate image of Method B's segmentation, in processing order.

    The final step is checked against leaf_pipeline.segment_leaf, so the figures always show
    exactly what the classifier uses.
    """
    parameters = _parameters(parameters)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    width = int(parameters["border"])
    border = np.zeros(rgb.shape[:2], bool)
    border[:width], border[-width:], border[:, :width], border[:, -width:] = True, True, True, True
    background = np.median(lab[border], axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    threshold = (distance > parameters["distance"]).astype(np.uint8)
    steps = {"border": border, "background_lab": background, "distance": distance,
             "threshold": threshold.astype(bool)}
    mask = threshold
    for operation, name in ((cv2.MORPH_OPEN, "opening"), (cv2.MORPH_CLOSE, "closing")):
        size = int(parameters[name])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, operation, kernel)
        steps[name] = mask.astype(bool)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.drawContours(final, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    steps["final"] = final.astype(bool)
    steps["components"] = len(contours)
    if not np.array_equal(steps["final"], segment_leaf(rgb, parameters)):
        raise AssertionError("Intermediate steps disagree with the shared segmentation")
    return steps


def show_segmentation_steps(image_path, parameters=None):
    """Plot the Lab distance map and every morphology step for one image."""
    parameters = _parameters(parameters)
    rgb = load_rgb(image_path)
    steps = segmentation_steps(rgb, parameters)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    framed = rgb.copy()
    framed[steps["border"]] = (0.5 * framed[steps["border"]] + [0, 0, 127]).astype(np.uint8)
    axes[0, 0].imshow(framed)
    axes[0, 0].set_title(f"Input, border used for background ({parameters['border']} px)")
    shown = axes[0, 1].imshow(steps["distance"], cmap="magma")
    axes[0, 1].contour(steps["distance"], levels=[parameters["distance"]], colors="cyan", linewidths=0.8)
    axes[0, 1].set_title(f"Lab distance to background (cyan = {parameters['distance']})")
    figure.colorbar(shown, ax=axes[0, 1], fraction=0.046)
    axes[0, 2].imshow(steps["threshold"], cmap="gray")
    axes[0, 2].set_title("Threshold")
    axes[1, 0].imshow(steps["opening"], cmap="gray")
    axes[1, 0].set_title(f"Opening {parameters['opening']}x{parameters['opening']}")
    axes[1, 1].imshow(steps["closing"], cmap="gray")
    axes[1, 1].set_title(f"Closing {parameters['closing']}x{parameters['closing']} "
                         f"({steps['components']} components)")
    fraction = steps["final"].mean()
    try:
        validate_mask(steps["final"])
        status = "usable"
    except ValueError:
        status = "UNUSABLE"
    axes[1, 2].imshow(overlay_mask(rgb, steps["final"]))
    axes[1, 2].set_title(f"Largest region filled: {fraction:.1%} leaf, {status}")
    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle(Path(image_path).name)
    figure.tight_layout()
    return figure, steps


def show_features(image_path, parameters=None):
    """Compare the 58 features measured on the full image (A) and inside the mask (B)."""
    rgb = load_rgb(image_path)
    mask = segment_leaf(rgb, _parameters(parameters))
    full = extract_features(rgb)
    try:
        masked = extract_features(rgb, mask)
    except ValueError:
        masked = None
    lbp = local_binary_pattern(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), P=8, R=1, method="uniform")
    figure, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    names = ("Hue", "Saturation", "Value", "Uniform LBP")
    for index, (axis, name) in enumerate(zip(axes[:4], names)):
        start, stop = (index * 16, index * 16 + 16) if index < 3 else (48, 58)
        bins = np.arange(stop - start)
        axis.bar(bins - 0.2, full[start:stop], 0.4, label="A: full image", color="#637f78")
        if masked is not None:
            axis.bar(bins + 0.2, masked[start:stop], 0.4, label="B: leaf mask", color="#327349")
        axis.set_title(name)
        axis.set_xlabel("bin")
    axes[0].set_ylabel("normalized frequency")
    axes[0].legend(fontsize=8)
    shown = np.where(texture_region(mask), lbp, np.nan) if masked is not None else lbp
    axes[4].imshow(shown, cmap="viridis")
    axes[4].set_title("LBP codes" + (" inside mask" if masked is not None else " (mask unusable)"))
    axes[4].axis("off")
    figure.tight_layout()
    return figure, {"A": full, "B": masked}


def compare_methods(image_path, root=ROOT, models=None):
    """Predict one image with both saved models and show their feature regions side by side."""
    root = Path(root)
    models = models or {m: root / "models" / f"development_{m}.joblib" for m in ("A", "B")}
    results = {}
    for method, model in models.items():
        try:
            label, rgb, mask = predict_image(image_path, model)
            results[method] = {"prediction": label, "mask": mask, "error": ""}
        except ValueError as error:
            rgb = load_rgb(image_path)
            results[method] = {"prediction": "FAILED", "mask": None, "error": str(error)}
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    for axis, method in zip(axes[1:], ("A", "B")):
        result = results[method]
        region = result["mask"] if result["mask"] is not None else np.zeros(rgb.shape[:2], bool)
        axis.imshow(overlay_mask(rgb, region) if method == "B" else rgb)
        caption = result["prediction"] if not result["error"] else "FAILED: " + result["error"][:40]
        axis.set_title(f"Method {method}: {caption}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure, {m: {k: v for k, v in r.items() if k != "mask"} for m, r in results.items()}


def upload_images(root=ROOT):
    """Colab only: ask for image files and save them under uploads/. Returns their paths."""
    try:
        from google.colab import files
    except ImportError as error:
        raise RuntimeError("Uploading uses Colab; locally, set the image path directly") from error
    target = Path(root) / "uploads"
    target.mkdir(exist_ok=True)
    paths = []
    for name, content in files.upload().items():
        path = target / Path(name).name
        path.write_bytes(content)
        paths.append(path)
    return paths


def _experiment_folder(root, name):
    name = name or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name) or name.startswith("."):
        raise ValueError("Use letters, digits, '.', '_' or '-' for the experiment name")
    folder = Path(root) / EXPERIMENTS / name
    for protected in PROTECTED:
        if folder.resolve().is_relative_to((Path(root) / protected).resolve()):
            raise ValueError("Experiments cannot write into protected benchmark folders")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_parameter_experiment(parameters=None, name=None, root=ROOT, c_values=(0.1, 1.0, 10.0)):
    """Rerun the paired A/B development comparison with edited segmentation parameters.

    Uses only development images and the same fixed folds, scaler, SVM and C search as the
    recorded experiment. Writes to results/experiments/<name>/ and saves no models.
    """
    root = Path(root)
    parameters = _parameters(parameters)
    folder = _experiment_folder(root, name)
    rows = [row for row in validate_dataset(root) if row["partition"] == "development"]
    full, masked, errors = feature_matrices(rows, parameters, root)
    eligible = np.array([not error for error in errors])
    write_csv(folder / "unusable_masks.csv",
              [{"image_id": row["image_id"], "reason": error} for row, error in zip(rows, errors) if error],
              ["image_id", "reason"])
    # Same gates as experiment.run_development, so no score is produced where the recorded pipeline stops.
    if eligible.sum() < len(rows) * 0.8:
        raise ValueError(f"Only {int(eligible.sum())} of {len(rows)} development masks are usable with "
                         f"{parameters}; the recorded pipeline stops below 80%, so no score is reported. "
                         f"See {folder / 'unusable_masks.csv'}")
    labels = np.array([row["health"] for row in rows])[eligible]
    folds = np.array([int(row["cv_fold"]) for row in rows])[eligible]
    cv = [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in range(3)]
    for train, valid in cv:
        if set(labels[train]) != set(LABELS) or set(labels[valid]) != set(LABELS):
            raise ValueError("A cross-validation fold is missing a health class with these parameters")
    summary = {"stage": "exploratory_development", "segmentation": parameters,
               "development_images": len(rows), "shared_training_images": int(eligible.sum()),
               "unusable_masks": int((~eligible).sum()), "methods": {}}
    for method, features in (("A", full), ("B", masked)):
        pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
        search = GridSearchCV(pipeline, {"svc__C": list(c_values)}, scoring="f1_macro",
                              cv=cv, n_jobs=1, refit=False)
        search.fit(features[eligible], labels)
        summary["methods"][method] = {"cv_macro_f1": float(search.best_score_),
                                       "C": float(search.best_params_["svc__C"])}
    summary["cv_macro_f1_B_minus_A"] = (summary["methods"]["B"]["cv_macro_f1"]
                                        - summary["methods"]["A"]["cv_macro_f1"])
    references = reviewed_references(root, "development")
    if references:
        by_id = {row["image_id"]: row for row in rows}
        dice = []
        for reference in references:
            expected = cv2.imread(str(root / reference["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
            predicted = segment_leaf(load_rgb(root / by_id[reference["image_id"]]["path"]), parameters)
            dice.append({"image_id": reference["image_id"], **mask_scores(predicted, expected)})
        write_csv(folder / "development_reference_scores.csv", dice)
        summary["reviewed_development_references"] = len(dice)
        summary["mean_development_dice"] = float(np.mean([row["dice"] for row in dice]))
    recorded = root / "results/development/summary.json"
    if recorded.exists():
        summary["recorded_development"] = {
            k: v for k, v in json.loads(recorded.read_text()).items()
            if k in ("segmentation", "methods", "cv_macro_f1_B_minus_A", "segmentation_status",
                     "shared_training_images")}
        # Scores on a different image set are not a like-for-like comparison with the record.
        summary["comparable_to_recorded"] = (summary["shared_training_images"]
                                             == summary["recorded_development"]["shared_training_images"])
    write_json(folder / "summary.json", summary)
    return summary


def reproduce_final(root=ROOT, name="final_reproduction"):
    """Recompute the locked test evaluation in a scratch copy and compare it with the stored one.

    The stored benchmark is only read. The recomputed metrics, predictions and a comparison are
    written to results/experiments/<name>/.
    """
    root = Path(root)
    stored_path = root / "results/final/metrics.json"
    if not stored_path.exists():
        raise ValueError("No stored final benchmark yet: freeze and evaluate first")
    stored = json.loads(stored_path.read_text())
    folder = _experiment_folder(root, name)
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory) / "project"
        shutil.copytree(root, scratch, symlinks=True, ignore=shutil.ignore_patterns(
            ".git", ".venv*", "uploads", "experiments", "final", "cache", "__pycache__"))
        recomputed = evaluate_final(scratch)
        for name_ in ("metrics.json", "predictions.csv", "segmentation.csv", "confusion_matrices.png"):
            shutil.copy2(scratch / "results/final" / name_, folder / name_)
    comparison = []
    for method in ("A", "B"):
        for metric in ("macro_f1", "accuracy", "coverage"):
            comparison.append({"method": method, "metric": metric,
                               "stored": stored["methods"][method][metric],
                               "recomputed": recomputed["methods"][method][metric]})
    for metric in ("iou", "dice"):
        comparison.append({"method": "B", "metric": f"mean_{metric}",
                           "stored": stored["segmentation"][metric]["mean"],
                           "recomputed": recomputed["segmentation"][metric]["mean"]})
    for row in comparison:
        row["match"] = bool(np.isclose(row["stored"], row["recomputed"], atol=1e-12))
    write_csv(folder / "comparison.csv", comparison)
    return comparison
