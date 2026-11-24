"""Post-freeze comparison: hand-crafted 58 HSV+LBP features vs pretrained CNN features.

The classifier stays the same (StandardScaler + RBF SVM, C chosen on development grouped
3-fold CV with macro F1). Only the feature extractor changes: MobileNetV2 pretrained on
ImageNet, used as a fixed extractor (no fine-tuning). The 1280-value global-average-pooled
output of its last convolutional block replaces the 58 features.

Three inputs per image, matching Methods A, B and B2:
    A-CNN   the full 256x256 image
    B-CNN   pixels outside the frozen Lab mask painted mid-grey (0 after MobileNetV2 scaling)
    B2-CNN  pixels outside the GrabCut mask (segmentation_grabcut.py) painted mid-grey
A masked variant fails exactly when the matching 58-feature method fails, so the failure
accounting (FAILED counts as wrong) is identical. PlantDoc is never used for training.
Frozen files are only read; outputs go to results/extended/cnn/.

Usage (the features step needs TensorFlow; evaluation does not):
    pip install tensorflow-cpu
    python cnn_experiment.py features --plantdoc PATH   # needs extended_analysis.py external first
    python cnn_experiment.py evaluate
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from extended_analysis import OUTPUT as EXTENDED, classifiers, fit_search, load_data, macro_f1
from grabcut_experiment import CACHE as GRABCUT_CACHE, paired, predict
from leaf_pipeline import ROOT, extract_features, load_rgb, read_manifest, segment_leaf, write_csv, write_json
from segmentation_grabcut import GRABCUT, grabcut_leaf

OUTPUT = EXTENDED / "cnn"
CACHE = ROOT / "data/cache/cnn_features.npz"
NETWORK = "MobileNetV2, ImageNet weights, include_top=False, global average pooling, 224x224 input"
GREY = 128


def masked_inputs(rgb, segmentation):
    """The three network inputs for one image; a failed mask gives None."""
    inputs = {"A": rgb, "B": None, "B2": None}
    for method, make_mask in (("B", lambda: segment_leaf(rgb, segmentation)),
                              ("B2", lambda: grabcut_leaf(rgb, segmentation))):
        try:
            mask = make_mask()
            extract_features(rgb, mask)  # same validity rule as the 58-feature method
        except ValueError:
            continue
        painted = rgb.copy()
        painted[~mask] = GREY
        inputs[method] = painted
    return inputs


def run_features(plantdoc, root=ROOT, batch=32):
    import tensorflow as tf
    root = Path(root)
    frozen = json.loads((root / "models/frozen_protocol.json").read_text())
    with open(EXTENDED / "plantdoc/inventory.csv", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    paths = {"plantvillage": [root / r["path"] for r in read_manifest(root)],
             "plantdoc": [Path(plantdoc) / r["path"] for r in inventory]}
    model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg",
                                              input_shape=(224, 224, 3))
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    arrays = {}
    for dataset, items in paths.items():
        features = {m: np.full((len(items), 1280), np.nan, np.float32) for m in ("A", "B", "B2")}
        pending = {m: [] for m in features}
        def flush(method):
            index, images = zip(*pending[method])
            x = preprocess(np.stack(images).astype(np.float32))
            features[method][list(index)] = model(x, training=False).numpy()
            pending[method].clear()
        for i, path in enumerate(items):
            for method, image in masked_inputs(load_rgb(path), frozen["segmentation"]).items():
                if image is not None:
                    pending[method].append((i, cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)))
                    if len(pending[method]) == batch:
                        flush(method)
            if (i + 1) % 200 == 0:
                print(f"{dataset}: {i + 1}/{len(items)}", flush=True)
        for method in features:
            if pending[method]:
                flush(method)
        for method, values in features.items():
            arrays[f"{dataset}_{method}"] = values
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, network=NETWORK, tensorflow=tf.__version__, grabcut=json.dumps(GRABCUT),
             plantdoc_paths=np.array([r["path"] for r in inventory]), **arrays)
    print({k: int(np.isnan(v).any(axis=1).sum()) for k, v in arrays.items()}, "failed masks")


def run_evaluate(root=ROOT):
    frozen, data = load_data(root)
    cnn, handcrafted = np.load(CACHE), np.load(GRABCUT_CACHE)
    partition = np.array([row["partition"] for row in read_manifest(root)])
    dev, test = data["development"], data["test"]
    features = {"A": (dev["A"], test["A"]), "B": (dev["B"], test["B"]),
                "B2": tuple(handcrafted["plantvillage"][partition == p] for p in ("development", "test"))}
    for method in ("A", "B", "B2"):
        values = cnn[f"plantvillage_{method}"]
        features[f"{method}-CNN"] = (values[partition == "development"], values[partition == "test"])
    # Shared training rule: an image is used only if every compared method has a valid mask for it.
    eligible = dev["eligible"].copy()
    for train, _ in features.values():
        eligible &= ~np.isnan(train).any(axis=1)
    folds = np.array([int(row["cv_fold"]) for row in dev["rows"]])[eligible]
    cv = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in range(3)]
    labels = dev["labels"][eligible]
    truth = test["labels"].astype(object)
    model, grid = classifiers()["svm_rbf"]
    estimators, test_predictions = {}, {}
    result = {"note": "Post-hoc. Same RBF SVM and folds; only the features differ.", "network": str(cnn["network"]),
              "tensorflow": str(cnn["tensorflow"]), "n_train": int(eligible.sum()), "methods": {}}
    for method, (train, held_out) in features.items():
        search, record = fit_search(model, grid, train[eligible], labels, cv)
        estimators[method] = search.best_estimator_
        test_predictions[method] = predict(search.best_estimator_, held_out)
        result["methods"][method] = {**record, "test_macro_f1_posthoc": macro_f1(truth, test_predictions[method])}
        print(f"{method:7s}: dev CV {record['cv_macro_f1']:.4f}  test {result['methods'][method]['test_macro_f1_posthoc']:.4f}",
              flush=True)
    result["test_paired"] = {f"{m}-CNN_vs_{m}": paired(truth, test_predictions[m], test_predictions[f"{m}-CNN"])
                             for m in ("A", "B", "B2")}
    # PlantDoc: hand-crafted predictions from the earlier runs, CNN predictions from the models above.
    with open(EXTENDED / "plantdoc/predictions.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with open(EXTENDED / "grabcut/plantdoc_b2_predictions.csv", newline="") as handle:
        b2 = list(csv.DictReader(handle))
    assert [r["path"] for r in rows] == [r["path"] for r in b2] == list(cnn["plantdoc_paths"])
    field_truth = np.array([r["health"] for r in rows], dtype=object)
    balanced = np.array([r["balanced_subset"] == "True" for r in rows])
    field = {"A": np.array([r["A_svm"] for r in rows], dtype=object),
             "B": np.array([r["B_svm"] for r in rows], dtype=object),
             "B2": np.array([r["B2_svm"] for r in b2], dtype=object)}
    for method in ("A", "B", "B2"):
        field[f"{method}-CNN"] = predict(estimators[f"{method}-CNN"], cnn[f"plantdoc_{method}"])
    t = field_truth[balanced]
    result["plantdoc"] = {
        "n_balanced": int(balanced.sum()),
        "macro_f1_balanced": {m: macro_f1(t, p[balanced]) for m, p in field.items()},
        "macro_f1_all": {m: macro_f1(field_truth, p) for m, p in field.items()},
        "accuracy_by_class_balanced": {m: {h: float(np.mean(p[balanced][t == h] == h)) for h in ("Healthy", "Diseased")}
                                       for m, p in field.items()},
        "paired": {f"{m}-CNN_vs_{m}": paired(t, field[m][balanced], field[f"{m}-CNN"][balanced])
                   for m in ("A", "B", "B2")}}
    result["plantdoc"]["paired"]["B2-CNN_vs_A-CNN"] = paired(t, field["A-CNN"][balanced], field["B2-CNN"][balanced])
    write_csv(OUTPUT / "plantdoc_cnn_predictions.csv",
              [{"path": r["path"], "health": r["health"], "balanced_subset": r["balanced_subset"],
                "A_CNN_svm": field["A-CNN"][i], "B_CNN_svm": field["B-CNN"][i], "B2_CNN_svm": field["B2-CNN"][i]}
               for i, r in enumerate(rows)])
    write_json(OUTPUT / "results.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("features", "evaluate"))
    parser.add_argument("--plantdoc", help="Path to a PlantDoc-Dataset checkout at the pinned commit")
    args = parser.parse_args()
    if args.action == "features":
        if not args.plantdoc:
            parser.error("features needs --plantdoc")
        from extended_analysis import require_case_sensitive
        require_case_sensitive(args.plantdoc)
        run_features(args.plantdoc)
    else:
        run_evaluate()
