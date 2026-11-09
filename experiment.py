"""Paired development experiments and a reviewed, frozen held-out evaluation."""

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from leaf_pipeline import (DEFAULT_SEGMENTATION, LABELS, ROOT, SEED, classification_metrics,
                           extract_features, file_hash, load_rgb, mask_scores, overlay_mask,
                           predict_image, read_manifest, segment_leaf, write_csv, write_json)
from mask_review import reviewed_references
from prepare_data import validate_dataset


def environment():
    packages = ["numpy", "opencv-python-headless", "scikit-image", "scikit-learn",
                "matplotlib", "joblib", "jupyterlab", "ipykernel", "nbclient", "nbformat", "certifi"]
    result = {"python": platform.python_version(), "os": platform.platform(),
              "machine": platform.machine(), "logical_cpus": os.cpu_count(),
              "opencv_threads": cv2.getNumThreads(),
              "packages": {name: importlib.metadata.version(name) for name in packages}}
    if platform.system() == "Darwin":
        for name, query in (("cpu", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
            result[name] = subprocess.check_output(["sysctl", "-n", query], text=True).strip()
    return result


def select_segmentation(root=ROOT):
    references = reviewed_references(root, "development")
    if len(references) != 12:
        return dict(DEFAULT_SEGMENTATION), "provisional_unreviewed_masks"
    by_id = {row["image_id"]: row for row in read_manifest(root)}
    scores = []
    for distance in (12.0, 18.0, 24.0):
        for closing in (3, 5):
            parameters = dict(DEFAULT_SEGMENTATION, distance=distance, closing=closing)
            dice = []
            for row in references:
                rgb = load_rgb(Path(root) / by_id[row["image_id"]]["path"])
                reference = cv2.imread(str(Path(root) / row["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
                dice.append(mask_scores(segment_leaf(rgb, parameters), reference)["dice"])
            scores.append({"distance": distance, "closing": closing, "mean_dice": float(np.mean(dice))})
    write_csv(Path(root) / "results/development/segmentation_validation.csv", scores)
    best = max(scores, key=lambda row: row["mean_dice"])
    return dict(DEFAULT_SEGMENTATION, distance=best["distance"], closing=best["closing"]), "reviewed_development_masks"


def feature_matrices(rows, parameters, root=ROOT):
    root = Path(root)
    import hashlib
    signature = json.dumps({"parameters": parameters, "ids": [row["image_id"] for row in rows],
                            "manifest": file_hash(root / "data/manifests/images.csv"),
                            "implementation": file_hash(root / "leaf_pipeline.py")}, sort_keys=True)
    key = hashlib.sha256(signature.encode()).hexdigest()[:20]
    cache = root / "data/cache" / f"features_{key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return data["full"], data["masked"], data["errors"].tolist()
    full, masked, errors = [], [], []
    for row in rows:
        rgb = load_rgb(root / row["path"])
        full.append(extract_features(rgb))
        try:
            masked.append(extract_features(rgb, segment_leaf(rgb, parameters)))
            errors.append("")
        except ValueError as error:
            masked.append(np.full(58, np.nan))
            errors.append(str(error))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, full=np.asarray(full), masked=np.asarray(masked), errors=np.asarray(errors))
    return np.asarray(full), np.asarray(masked), errors


def run_development(root=ROOT):
    root = Path(root)
    if (root / "models/frozen_protocol.json").exists():
        raise ValueError("Protocol is frozen; preserve its models and use the recorded development results")
    cv2.setNumThreads(1)
    rows = [row for row in validate_dataset(root) if row["partition"] == "development"]
    parameters, status = select_segmentation(root)
    full, masked, errors = feature_matrices(rows, parameters, root)
    eligible = np.array([not error for error in errors])
    kept = [row for row, valid in zip(rows, eligible) if valid]
    if len(kept) < len(rows) * 0.8:
        raise ValueError("Too many unusable development masks; inspect development segmentation before training")
    labels = np.array([row["health"] for row in kept])
    folds = np.array([int(row["cv_fold"]) for row in kept])
    cv = [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in range(3)]
    for train, valid in cv:
        if set(labels[train]) != set(LABELS) or set(labels[valid]) != set(LABELS):
            raise ValueError("A cross-validation fold is missing a health class")
    output = root / "results/development"
    output.mkdir(parents=True, exist_ok=True)
    exclusions = [{"image_id": row["image_id"], "reason": error}
                  for row, error in zip(rows, errors) if error]
    write_csv(output / "shared_training_exclusions.csv", exclusions, ["image_id", "reason"])
    summary = {"stage": "development_only", "segmentation_status": status,
               "segmentation": parameters, "development_images": len(rows),
               "shared_training_images": len(kept), "held_out_test_evaluated": False,
               "manifest_sha256": file_hash(root / "data/manifests/images.csv"),
               "implementation_sha256": file_hash(root / "leaf_pipeline.py"),
               "development_reference_hashes": {r["image_id"]: r["annotation_sha256"]
                                                for r in reviewed_references(root, "development")},
               "methods": {}}
    for method, features in (("A", full), ("B", masked)):
        pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
        search = GridSearchCV(pipeline, {"svc__C": [0.1, 1.0, 10.0]}, scoring="f1_macro",
                              cv=cv, n_jobs=1, refit=True, error_score="raise")
        start = perf_counter()
        search.fit(features[eligible], labels)
        seconds = perf_counter() - start
        table = [{"C": result["svc__C"], "mean_macro_f1": float(mean), "std_macro_f1": float(std),
                  "mean_fit_seconds": float(fit),
                  **{f"fold_{fold}_macro_f1": float(search.cv_results_[f"split{fold}_test_score"][index])
                     for fold in range(3)}}
                 for index, (result, mean, std, fit) in enumerate(zip(
                     search.cv_results_["params"], search.cv_results_["mean_test_score"],
                     search.cv_results_["std_test_score"], search.cv_results_["mean_fit_time"]))]
        write_csv(output / f"cv_{method}.csv", table)
        artifact = {"method": method, "pipeline": search.best_estimator_, "segmentation": parameters,
                    "labels": list(LABELS), "image_size": [256, 256], "feature_count": 58,
                    "status": status, "seed": SEED, "best_C": search.best_params_["svc__C"],
                    "manifest_sha256": summary["manifest_sha256"],
                    "implementation_sha256": summary["implementation_sha256"],
                    "training_image_ids": [row["image_id"] for row in kept]}
        (root / "models").mkdir(exist_ok=True)
        joblib.dump(artifact, root / "models" / f"development_{method}.joblib")
        summary["methods"][method] = {"cv_macro_f1": float(search.best_score_),
                                       "C": artifact["best_C"], "search_and_refit_seconds": seconds,
                                       "final_refit_seconds": float(search.refit_time_)}
        print(f"Method {method}: development CV macro F1 {search.best_score_:.4f}", flush=True)
    summary["cv_macro_f1_B_minus_A"] = summary["methods"]["B"]["cv_macro_f1"] - summary["methods"]["A"]["cv_macro_f1"]
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment())
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(["A: full image", "B: masked"], [summary["methods"][m]["cv_macro_f1"] for m in ("A", "B")],
                color=["#637f78", "#327349"])
    axes[0].set(ylim=(0, 1), ylabel="Mean development CV macro F1")
    sample = next(row for row, valid in zip(rows, eligible) if valid)
    rgb = load_rgb(root / sample["path"])
    axes[1].imshow(overlay_mask(rgb, segment_leaf(rgb, parameters)))
    axes[1].set_title("Development example: " + sample["image_id"])
    axes[1].axis("off")
    figure.suptitle("Development experiment — held-out evaluation pending reference review")
    figure.tight_layout()
    figure.savefig(output / "development_overview.png", dpi=150)
    plt.close(figure)
    return summary


def freeze(root=ROOT):
    root = Path(root)
    references = reviewed_references(root, "development")
    if len(references) != 12:
        raise ValueError("Freeze requires 12 outlines reviewed by both team members")
    summary = json.loads((root / "results/development/summary.json").read_text())
    if summary["segmentation_status"] != "reviewed_development_masks":
        raise ValueError("Rerun the development experiment with reviewed outlines before freezing")
    if summary["implementation_sha256"] != file_hash(root / "leaf_pipeline.py"):
        raise ValueError("Implementation changed; rerun development before freezing")
    if (summary["manifest_sha256"] != file_hash(root / "data/manifests/images.csv")
            or summary["development_reference_hashes"] != {
                r["image_id"]: r["annotation_sha256"] for r in references}):
        raise ValueError("Development inputs changed; rerun development before freezing")
    record = {"segmentation": summary["segmentation"], "seed": SEED,
              "manifest_sha256": file_hash(root / "data/manifests/images.csv"),
              "implementation_sha256": file_hash(root / "leaf_pipeline.py"),
              "experiment_sha256": file_hash(root / "experiment.py"),
              "development_references": {r["image_id"]: r["annotation_sha256"] for r in references},
              "models": {m: file_hash(root / "models" / f"development_{m}.joblib") for m in ("A", "B")},
              "failure_policy": "Full denominator; failures are FN for true class; covered 2x2 matrix plus failures"}
    if (root / "results/final/metrics.json").exists():
        raise ValueError("Final results already exist; do not replace the frozen protocol after observing test results")
    write_json(root / "models/frozen_protocol.json", record)
    return record


def evaluate_final(root=ROOT):
    root = Path(root)
    cv2.setNumThreads(1)
    references = reviewed_references(root)
    if Counter(r["partition"] for r in references) != {"development": 12, "test": 18}:
        raise ValueError("Final evaluation requires all 30 outlines reviewed and exported")
    frozen_path = root / "models/frozen_protocol.json"
    if not frozen_path.exists():
        raise ValueError("Freeze the development protocol before final evaluation")
    frozen = json.loads(frozen_path.read_text())
    if (frozen["manifest_sha256"] != file_hash(root / "data/manifests/images.csv")
            or frozen["implementation_sha256"] != file_hash(root / "leaf_pipeline.py")
            or frozen["experiment_sha256"] != file_hash(root / "experiment.py")
            or frozen["development_references"] != {
                r["image_id"]: r["annotation_sha256"] for r in references if r["partition"] == "development"}
            or any(frozen["models"][m] != file_hash(root / "models" / f"development_{m}.joblib") for m in ("A", "B"))):
        raise ValueError("Frozen inputs changed; do not evaluate a different protocol")
    final_path = root / "results/final/metrics.json"
    signature = {r["image_id"]: r["annotation_sha256"] for r in references if r["partition"] == "test"}
    if final_path.exists():
        previous = json.loads(final_path.read_text())
        if previous["frozen_protocol_sha256"] != file_hash(frozen_path) or previous["test_references"] != signature:
            raise ValueError("Stored final results do not match the current frozen inputs")
        return previous
    rows = validate_dataset(root)
    test = [row for row in rows if row["partition"] == "test"]
    warmup = next(row for row in rows if row["partition"] == "development")
    result = {"stage": "locked_test", "n_images": len(test), "methods": {},
              "frozen_protocol_sha256": file_hash(frozen_path), "test_references": signature,
              "timing_repetitions": 3, "environment": environment()}
    result["development_training"] = json.loads((root / "results/development/summary.json").read_text())["methods"]
    predictions = []
    for method in ("A", "B"):
        artifact = joblib.load(root / "models" / f"development_{method}.joblib")
        try:
            predict_image(root / warmup["path"], artifact)
        except ValueError:
            pass
        labels, times = [], []
        for row in test:
            attempts, durations, failure = [], [], ""
            for _ in range(3):
                start = perf_counter()
                try:
                    prediction, _, _ = predict_image(root / row["path"], artifact)
                except ValueError as error:
                    prediction, failure = "FAILED", str(error)
                durations.append(perf_counter() - start)
                attempts.append(prediction)
            if len(set(attempts)) != 1:
                raise ValueError("Repeated inference produced inconsistent predictions")
            labels.append(attempts[0])
            times.append(float(np.median(durations)))
            predictions.append({"image_id": row["image_id"], "plant": row["plant"],
                                "truth": row["health"], "method": method, "prediction": attempts[0],
                                "median_seconds": times[-1], "failure": failure})
        metrics = classification_metrics([row["health"] for row in test], labels)
        metrics["median_inference_seconds"] = float(np.median(times))
        metrics["by_plant"] = {plant: classification_metrics(
            [r["health"] for r in test if r["plant"] == plant],
            [label for r, label in zip(test, labels) if r["plant"] == plant]) for plant in sorted({r["plant"] for r in test})}
        result["methods"][method] = metrics
    by_id = {row["image_id"]: row for row in test}
    segmentation = []
    for reference in references:
        if reference["partition"] != "test":
            continue
        row = by_id[reference["image_id"]]
        expected = cv2.imread(str(root / reference["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
        scores = mask_scores(segment_leaf(load_rgb(root / row["path"]), frozen["segmentation"]), expected)
        segmentation.append(dict(image_id=row["image_id"], plant=row["plant"], health=row["health"], **scores))
    result["segmentation"] = {name: {"mean": float(np.mean([r[name] for r in segmentation])),
                                              "minimum": min(r[name] for r in segmentation),
                                              "maximum": max(r[name] for r in segmentation)} for name in ("iou", "dice")}
    result["macro_f1_B_minus_A"] = result["methods"]["B"]["macro_f1"] - result["methods"]["A"]["macro_f1"]
    output = root / "results/final"
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "segmentation.csv", segmentation)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, method in zip(axes, ("A", "B")):
        matrix = np.array(result["methods"][method]["covered_confusion_matrix"])
        axis.imshow(matrix, cmap="Greens")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=LABELS, yticklabels=LABELS,
                 xlabel="Predicted", ylabel="True", title=f"Method {method}: covered predictions")
        for (i, j), count in np.ndenumerate(matrix):
            axis.text(j, i, str(count), ha="center", va="center", color="black", bbox={"facecolor": "white", "alpha": .75})
    figure.tight_layout()
    figure.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(figure)
    write_json(final_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("development", "freeze", "evaluate"))
    args = parser.parse_args()
    if args.action == "development":
        run_development()
    elif args.action == "freeze":
        print(json.dumps(freeze(), indent=2))
    else:
        print(json.dumps(evaluate_final(), indent=2))
