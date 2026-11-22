"""Background-only control: how well can Healthy/Diseased be predicted from pixels outside the leaf?

Uses the 720 development images only, with the same fixed group-safe folds, 58 features,
StandardScaler, RBF SVM and C search as the paired experiment. The background region is the
complement of Method B's leaf mask after growing that mask by a safety margin, so leaf edges are
excluded. Leaf leakage into the background region is measured against the 12 development
reference outlines. A high score supports a background-bias explanation for PlantVillage; it
does not by itself prove why Method A outperforms Method B.

Outputs go to results/experiments/background_control/. No model, development summary, frozen
protocol or final result is written.

    .venv/bin/python background_control.py
"""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiment import feature_matrices
from leaf_pipeline import (DEFAULT_SEGMENTATION, LABELS, ROOT, extract_features, load_rgb,
                           segment_leaf, write_csv, write_json)
from mask_review import annotation_path, polygon_mask
from prepare_data import validate_dataset

OUTPUT = Path("results/experiments/background_control")
MARGINS = (7, 21)


def background_region(rgb, parameters, margin):
    """Pixels outside Method B's leaf mask grown by `margin` pixels."""
    leaf = segment_leaf(rgb, parameters).astype(np.uint8)
    size = 2 * int(margin) + 1
    grown = cv2.dilate(leaf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return grown == 0


def fixed_folds(rows):
    folds = np.array([int(row["cv_fold"]) for row in rows])
    labels = np.array([row["health"] for row in rows])
    cv = [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in range(3)]
    for train, valid in cv:
        if set(labels[train]) != set(LABELS) or set(labels[valid]) != set(LABELS):
            raise ValueError("A cross-validation fold is missing a health class")
    return cv, labels


def cv_macro_f1(features, rows):
    cv, labels = fixed_folds(rows)
    pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
    search = GridSearchCV(pipeline, {"svc__C": [0.1, 1.0, 10.0]}, scoring="f1_macro",
                          cv=cv, n_jobs=1, refit=False, error_score="raise")
    search.fit(features, labels)
    folds = [float(search.cv_results_[f"split{fold}_test_score"][search.best_index_]) for fold in range(3)]
    return {"cv_macro_f1": float(search.best_score_), "C": float(search.best_params_["svc__C"]),
            "fold_macro_f1": folds}


def leakage(rows, parameters, margin, root):
    """Fraction of background-region pixels covered by the reference leaf outline."""
    records = []
    for row in rows:
        if row.get("annotation_partition") != "development":
            continue
        path = annotation_path(row, root)
        annotation = json.loads(path.read_text())
        reference = polygon_mask(annotation)
        region = background_region(load_rgb(root / row["path"]), parameters, margin)
        records.append({"image_id": row["image_id"], "margin": margin,
                        "dual_reviewed": all(annotation.get("flags", {}).values()),
                        "background_pixels": int(region.sum()),
                        "leaf_pixels_in_background": int((region & reference).sum()),
                        "leaf_share_of_background": float((region & reference).sum() / max(region.sum(), 1))})
    return records


def run(root=ROOT, margins=MARGINS):
    root = Path(root)
    rows = [row for row in validate_dataset(root) if row["partition"] == "development"]
    recorded = root / "results/development/summary.json"
    parameters = (json.loads(recorded.read_text())["segmentation"] if recorded.exists()
                  else dict(DEFAULT_SEGMENTATION))
    full, masked, errors = feature_matrices(rows, parameters, root)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    summary = {"stage": "exploratory_development_control", "segmentation": parameters,
               "development_images": len(rows), "margins": {}}
    leaks = []
    for margin in margins:
        features, usable = [], []
        for row in rows:
            rgb = load_rgb(root / row["path"])
            try:
                features.append(extract_features(rgb, background_region(rgb, parameters, margin)))
                usable.append(True)
            except ValueError:
                features.append(np.full(58, np.nan))
                usable.append(False)
        # Compare all three feature regions on exactly the same images.
        keep = np.array(usable) & np.array([not error for error in errors])
        kept = [row for row, k in zip(rows, keep) if k]
        result = {"images": int(keep.sum()), "excluded_small_background": int((~np.array(usable)).sum()),
                  "background_only": cv_macro_f1(np.asarray(features)[keep], kept),
                  "A_full_image_same_images": cv_macro_f1(full[keep], kept),
                  "B_leaf_mask_same_images": cv_macro_f1(masked[keep], kept)}
        by_plant = {}
        for plant in sorted({row["plant"] for row in kept}):
            subset = np.array([row["plant"] == plant for row in kept])
            by_plant[plant] = cv_macro_f1(np.asarray(features)[keep][subset],
                                          [row for row, s in zip(kept, subset) if s])["cv_macro_f1"]
        result["background_only_by_plant"] = by_plant
        records = leakage(rows, parameters, margin, root)
        leaks.extend(records)
        result["leakage_check"] = {
            "reference_outlines": len(records),
            "dual_reviewed_outlines": sum(r["dual_reviewed"] for r in records),
            "max_leaf_share_of_background": max(r["leaf_share_of_background"] for r in records),
            "mean_leaf_share_of_background": float(np.mean([r["leaf_share_of_background"] for r in records]))}
        summary["margins"][str(margin)] = result
        print(f"margin {margin}px: background-only CV macro F1 {result['background_only']['cv_macro_f1']:.4f} "
              f"on {result['images']} images (A {result['A_full_image_same_images']['cv_macro_f1']:.4f}, "
              f"B {result['B_leaf_mask_same_images']['cv_macro_f1']:.4f}); max leaf share in background "
              f"{result['leakage_check']['max_leaf_share_of_background']:.4f}", flush=True)
    write_csv(output / "leakage.csv", leaks)
    write_json(output / "summary.json", summary)
    render(rows, parameters, max(margins), root, output / "background_regions.png")
    return summary


def render(rows, parameters, margin, root, path):
    """Development reference images with the background region tinted and the outline drawn."""
    selected = [row for row in rows if row.get("annotation_partition") == "development"]
    figure, axes = plt.subplots(2, 6, figsize=(18, 7.2))
    for axis, row in zip(axes.ravel(), selected):
        rgb = load_rgb(root / row["path"])
        region = background_region(rgb, parameters, margin)
        shown = rgb.copy()
        shown[region] = (0.45 * shown[region] + [140, 0, 140]).astype(np.uint8)
        reference = polygon_mask(json.loads(annotation_path(row, root).read_text())).astype(np.uint8)
        contours, _ = cv2.findContours(reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(shown, contours, -1, (255, 255, 0), 1)
        axis.imshow(shown)
        axis.set_title(f"{row['image_id']} {row['plant'][:6]} {row['health']}", fontsize=9)
        axis.axis("off")
    figure.suptitle(f"Background region (magenta) used by the control, margin {margin}px; "
                    "yellow = reference leaf outline")
    figure.tight_layout()
    figure.savefig(path, dpi=80)
    plt.close(figure)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
