"""Post-freeze experiment: training with randomised backgrounds.

PlantVillage leaves are photographed on a fixed background, which models learn as a shortcut
(background-only control: 0.829 macro F1) and which does not exist in field photos. Here each
development leaf is cut out with the GrabCut mask (segmentation_grabcut.py) and pasted, in place,
onto COPIES field photos of OTHER crops. The background is drawn independently of the label,
so it carries no class information. Models are trained on originals plus composites and tested,
without any tuning on PlantDoc, on the PlantVillage test set and the PlantDoc balanced subset.

Pre-registered before any result (24 November 2026):
  background pool  PlantDoc folders outside our label mapping and not tomato, train + test,
                   minus exact or near copies (dHash <= 4) of the evaluation inventory
  composites       COPIES = 3 per eligible development image, seed 42, leaf kept in place
  methods          A-aug (58 features, full image), B2-aug (GrabCut mask recomputed on the
                   composite), A-CNN-aug (MobileNetV2 embedding); same RBF SVM and C grid
  cross-validation composites inherit their source image's fold; validation folds hold
                   originals only, so no leaf appears on both sides
Frozen files are only read; outputs go to results/extended/background/.

Usage (build needs TensorFlow for the CNN features; run the PlantDoc, GrabCut and CNN
experiments first):
    python background_experiment.py build --plantdoc PATH
    python background_experiment.py evaluate
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from extended_analysis import (OUTPUT as EXTENDED, PLANTDOC_MAPPING, classifiers, dhash, fit_search, hamming,
                               load_data, macro_f1, require_case_sensitive)
from grabcut_experiment import CACHE as GRABCUT_CACHE, paired, predict
from cnn_experiment import CACHE as CNN_CACHE
from leaf_pipeline import ROOT, SEED, extract_features, load_rgb, read_manifest, write_csv, write_json
from segmentation_grabcut import grabcut_leaf

OUTPUT = EXTENDED / "background"
CACHE = ROOT / "data/cache/background_features.npz"
COPIES = 3


def background_pool(plantdoc, inventory):
    """Field photos of crops we never evaluate on, minus copies of evaluation images."""
    plantdoc = Path(plantdoc)
    evaluation = [dhash(load_rgb(plantdoc / r["path"])) for r in inventory]
    pool, rejected = [], []
    for split in ("train", "test"):
        for folder in sorted(p.name for p in (plantdoc / split).iterdir() if p.is_dir()):
            if folder in PLANTDOC_MAPPING or folder.lower().startswith("tomato"):
                continue
            for path in sorted((plantdoc / split / folder).glob("*")):
                try:
                    rgb = load_rgb(path)
                except ValueError:
                    rejected.append(str(path.relative_to(plantdoc)))
                    continue
                if min(hamming(dhash(rgb), other) for other in evaluation) <= 4:
                    rejected.append(str(path.relative_to(plantdoc)))
                    continue
                pool.append(str(path.relative_to(plantdoc)))
    return pool, rejected


def composite(leaf, mask, background):
    return np.where(mask[..., None], leaf, background)


def run_build(plantdoc, root=ROOT):
    import tensorflow as tf
    from cnn_experiment import NETWORK
    require_case_sensitive(plantdoc)
    root, plantdoc = Path(root), Path(plantdoc)
    frozen = json.loads((root / "models/frozen_protocol.json").read_text())
    segmentation = frozen["segmentation"]
    rows = read_manifest(root)
    with open(EXTENDED / "plantdoc/inventory.csv", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    pool, rejected = background_pool(plantdoc, inventory)
    print(f"background pool {len(pool)}, rejected {len(rejected)}", flush=True)
    generator = np.random.default_rng(SEED)
    development = [i for i, r in enumerate(rows) if r["partition"] == "development"]
    sources, backgrounds, images = [], [], []
    for count, index in enumerate(development):
        rgb = load_rgb(root / rows[index]["path"])
        try:
            mask = grabcut_leaf(rgb, segmentation)
            extract_features(rgb, mask)
        except ValueError:
            continue
        for choice in generator.choice(len(pool), COPIES, replace=False):
            images.append(composite(rgb, mask, load_rgb(plantdoc / pool[choice])))
            sources.append(index)
            backgrounds.append(pool[choice])
        if (count + 1) % 120 == 0:
            print(f"composites for {count + 1}/{len(development)} images", flush=True)
    full, masked = [], []
    for i, image in enumerate(images):
        full.append(extract_features(image))
        try:
            masked.append(extract_features(image, grabcut_leaf(image, segmentation)))
        except ValueError:
            masked.append(np.full(58, np.nan))
        if (i + 1) % 300 == 0:
            print(f"58 features {i + 1}/{len(images)}", flush=True)
    model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg",
                                              input_shape=(224, 224, 3))
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    deep = []
    for start in range(0, len(images), 32):
        batch = np.stack([cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA) for x in images[start:start + 32]])
        deep.append(model(preprocess(batch.astype(np.float32)), training=False).numpy())
    # Full-image 58 features of the PlantDoc inventory (the frozen run kept them only in memory).
    field = np.array([extract_features(load_rgb(plantdoc / r["path"])) for r in inventory])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, network=NETWORK, sources=np.array(sources), backgrounds=np.array(backgrounds),
             A=np.array(full), B2=np.array(masked), A_CNN=np.concatenate(deep), plantdoc_A=field,
             plantdoc_paths=np.array([r["path"] for r in inventory]))
    write_csv(OUTPUT / "background_pool.csv", [{"path": p} for p in pool])
    preview(images, sources, rows, backgrounds)
    print(f"{len(images)} composites, B2 mask failures {int(np.isnan(np.array(masked)).any(axis=1).sum())}")


def preview(images, sources, rows, backgrounds, count=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = np.random.default_rng(SEED).choice(len(images), count, replace=False)
    figure, axes = plt.subplots(3, 4, figsize=(12, 9.5))
    for axis, i in zip(axes.ravel(), picks):
        axis.imshow(images[i])
        axis.set_title(f"{rows[sources[i]]['plant']} {rows[sources[i]]['health']}\n"
                       f"bg: {backgrounds[i].split('/')[1]}", fontsize=8)
        axis.axis("off")
    figure.suptitle("PlantVillage leaves (GrabCut cut-out) on other-crop field backgrounds")
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "composite_examples.png", dpi=130)
    plt.close(figure)


def run_evaluate(root=ROOT):
    frozen, data = load_data(root)
    rows = read_manifest(root)
    extra, grabcut, cnn = np.load(CACHE), np.load(GRABCUT_CACHE), np.load(CNN_CACHE)
    partition = np.array([r["partition"] for r in rows])
    dev_index = np.flatnonzero(partition == "development")
    dev, test = data["development"], data["test"]
    original = {"A": dev["A"], "B2": grabcut["plantvillage"][dev_index],
                "A-CNN": cnn["plantvillage_A"][dev_index]}
    held_out = {"A": test["A"], "B2": grabcut["plantvillage"][partition == "test"],
                "A-CNN": cnn["plantvillage_A"][partition == "test"]}
    with open(EXTENDED / "plantdoc/predictions.csv", newline="") as handle:
        field_rows = list(csv.DictReader(handle))
    assert [r["path"] for r in field_rows] == list(extra["plantdoc_paths"]) == list(cnn["plantdoc_paths"])
    field = {"A": extra["plantdoc_A"], "B2": grabcut["plantdoc"], "A-CNN": cnn["plantdoc_A"]}
    composites = {"A": extra["A"], "B2": extra["B2"], "A-CNN": extra["A_CNN"]}
    # Same eligible originals as the B2 and CNN experiments.
    eligible = dev["eligible"].copy()
    for values in original.values():
        eligible &= ~np.isnan(values).any(axis=1)
    position = {index: i for i, index in enumerate(dev_index)}
    source = np.array([position[s] for s in extra["sources"]])
    keep = eligible[source]
    folds = np.array([int(r["cv_fold"]) for r in dev["rows"]])
    labels = dev["labels"]
    truth = test["labels"].astype(object)
    field_truth = np.array([r["health"] for r in field_rows], dtype=object)
    balanced = np.array([r["balanced_subset"] == "True" for r in field_rows])
    model, grid = classifiers()["svm_rbf"]
    result = {"note": "Post-hoc. Pre-registered design in the module docstring; nothing tuned on PlantDoc.",
              "copies": COPIES, "n_composites": int(len(source)), "methods": {}}
    predictions = {}
    for method in ("A", "B2", "A-CNN"):
        for augmented in (False, True):
            name = f"{method}-aug" if augmented else method
            x, y, f = original[method][eligible], labels[eligible], folds[eligible]
            n_original = len(y)
            if augmented:
                valid = keep & ~np.isnan(composites[method]).any(axis=1)
                x = np.vstack([x, composites[method][valid]])
                y = np.concatenate([y, labels[source[valid]]])
                f = np.concatenate([f, folds[source[valid]]])
            is_original = np.arange(len(y)) < n_original
            cv = [(np.flatnonzero(f != k), np.flatnonzero((f == k) & is_original)) for k in range(3)]
            search, record = fit_search(model, grid, x, y, cv)
            test_prediction = predict(search.best_estimator_, held_out[method])
            field_prediction = predict(search.best_estimator_, field[method])
            predictions[name] = (test_prediction, field_prediction)
            t, p = field_truth[balanced], field_prediction[balanced]
            result["methods"][name] = {**record, "n_train": int(len(y)),
                                       "test_macro_f1_posthoc": macro_f1(truth, test_prediction),
                                       "plantdoc_macro_f1_balanced": macro_f1(t, p),
                                       "plantdoc_recall_balanced": {h: float(np.mean(p[t == h] == h))
                                                                    for h in ("Healthy", "Diseased")}}
            print(f"{name:10s}: dev CV {record['cv_macro_f1']:.4f}  test {result['methods'][name]['test_macro_f1_posthoc']:.4f}"
                  f"  PlantDoc {result['methods'][name]['plantdoc_macro_f1_balanced']:.4f}", flush=True)
        before, after = predictions[method], predictions[f"{method}-aug"]
        result[f"{method}-aug_vs_{method}"] = {
            "plantvillage_test": paired(truth, before[0], after[0]),
            "plantdoc_balanced": paired(field_truth[balanced], before[1][balanced], after[1][balanced])}
    write_csv(OUTPUT / "plantdoc_predictions.csv",
              [{"path": r["path"], "health": r["health"], "balanced_subset": r["balanced_subset"],
                **{name: values[1][i] for name, values in predictions.items()}} for i, r in enumerate(field_rows)])
    write_json(OUTPUT / "results.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("build", "evaluate"))
    parser.add_argument("--plantdoc", help="Path to a PlantDoc-Dataset checkout at the pinned commit")
    args = parser.parse_args()
    if args.action == "build":
        if not args.plantdoc:
            parser.error("build needs --plantdoc")
        run_build(args.plantdoc)
    else:
        run_evaluate()
