"""Post-freeze experiment: fine-tuning a pretrained CNN end to end on PlantVillage.

cnn_experiment.py used MobileNetV2 only as a fixed feature extractor for the SVM. Here the
network itself is trained to output Healthy/Diseased, on the full image (like Method A).
Two variants:
    FT      standard augmentation (flips, rotation within 15 degrees, brightness x[0.8, 1.2])
    FT-aug  the same, plus, with probability 0.75 per image and epoch, the background outside
            the GrabCut mask is replaced by a random field photo from the pool used in
            background_experiment.py (other crops only)

Pre-registered before any result (24 November 2026), nothing tuned afterwards:
    MobileNetV2 (ImageNet), 224x224, global average pooling, dropout 0.2, 2-way softmax,
    batch-norm layers kept in inference mode. Stage 1: base frozen, Adam 1e-3, 5 epochs.
    Stage 2: block_13 onwards trainable, Adam 1e-5, 5 epochs. Batch 32, balanced class
    weights, seed 42. Development score: the same grouped 3 folds; final model: all eligible
    development images. Test on PlantVillage and the PlantDoc balanced subset, full image.
Frozen files are only read; outputs go to results/extended/finetune/.

Usage (needs TensorFlow; run the PlantDoc, GrabCut, CNN and background experiments first):
    python finetune_experiment.py --plantdoc PATH
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from extended_analysis import OUTPUT as EXTENDED, load_data, macro_f1, require_case_sensitive
from grabcut_experiment import CACHE as GRABCUT_CACHE, paired
from leaf_pipeline import LABELS, ROOT, SEED, extract_features, load_rgb, read_manifest, write_csv, write_json
from segmentation_grabcut import grabcut_leaf

OUTPUT = EXTENDED / "finetune"
SETTINGS = {"input": 224, "dropout": 0.2, "batch": 32, "head_epochs": 5, "head_learning_rate": 1e-3,
            "tune_epochs": 5, "tune_learning_rate": 1e-5, "first_trainable": "block_13_expand",
            "rotation_degrees": 15, "brightness": (0.8, 1.2), "background_probability": 0.75}
CLASSES = list(LABELS)  # index 0/1 of the softmax output


def augment(rgb, mask, pool, generator, backgrounds):
    if backgrounds and mask is not None and generator.random() < SETTINGS["background_probability"]:
        rgb = np.where(mask[..., None], rgb, pool[generator.integers(len(pool))])
    if generator.random() < 0.5:
        rgb = rgb[:, ::-1]
    if generator.random() < 0.5:
        rgb = rgb[::-1]
    angle = generator.uniform(-SETTINGS["rotation_degrees"], SETTINGS["rotation_degrees"])
    matrix = cv2.getRotationMatrix2D((128, 128), angle, 1.0)
    rgb = cv2.warpAffine(np.ascontiguousarray(rgb), matrix, (256, 256), borderMode=cv2.BORDER_REFLECT)
    rgb = np.clip(rgb.astype(np.float32) * generator.uniform(*SETTINGS["brightness"]), 0, 255)
    return cv2.resize(rgb, (SETTINGS["input"], SETTINGS["input"]), interpolation=cv2.INTER_AREA)


def resized(images):
    size = (SETTINGS["input"], SETTINGS["input"])
    return np.stack([cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in images]).astype(np.float32)


def train(tf, images, masks, labels, pool, backgrounds, seed):
    tf.keras.utils.set_random_seed(seed)
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    size = SETTINGS["input"]
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg",
                                             input_shape=(size, size, 3))
    inputs = tf.keras.Input((size, size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(SETTINGS["dropout"])(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2, activation="softmax")(x))
    y = np.array([CLASSES.index(label) for label in labels])
    counts = np.bincount(y, minlength=2)
    class_weight = {k: len(y) / (2 * counts[k]) for k in range(2)}
    generator = np.random.default_rng(seed)

    def epochs(count, learning_rate):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss="sparse_categorical_crossentropy")
        for _ in range(count):
            order = generator.permutation(len(images))
            batch = np.stack([augment(images[i], masks[i], pool, generator, backgrounds) for i in order])
            model.fit(preprocess(batch), y[order], batch_size=SETTINGS["batch"], epochs=1, shuffle=False,
                      class_weight=class_weight, verbose=0)

    base.trainable = False
    epochs(SETTINGS["head_epochs"], SETTINGS["head_learning_rate"])
    base.trainable = True
    first = [layer.name for layer in base.layers].index(SETTINGS["first_trainable"])
    for layer in base.layers[:first]:
        layer.trainable = False
    epochs(SETTINGS["tune_epochs"], SETTINGS["tune_learning_rate"])
    return model


def predict(tf, model, images):
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    probabilities = model.predict(preprocess(resized(images)), batch_size=64, verbose=0)
    return np.array([CLASSES[i] for i in probabilities.argmax(axis=1)], dtype=object)


def run(plantdoc, root=ROOT):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    require_case_sensitive(plantdoc)
    root, plantdoc = Path(root), Path(plantdoc)
    frozen, data = load_data(root)
    rows = read_manifest(root)
    partition = np.array([r["partition"] for r in rows])
    dev, test = data["development"], data["test"]
    grabcut = np.load(GRABCUT_CACHE)
    # Same eligible development images as the B2 / CNN / background experiments.
    eligible = dev["eligible"] & ~np.isnan(grabcut["plantvillage"][partition == "development"]).any(axis=1)
    dev_rows = [r for r, keep in zip(dev["rows"], eligible) if keep]
    images = [load_rgb(root / r["path"]) for r in dev_rows]
    masks = []
    for rgb in images:
        try:
            mask = grabcut_leaf(rgb, frozen["segmentation"])
            extract_features(rgb, mask)
            masks.append(mask)
        except ValueError:
            masks.append(None)
    labels = dev["labels"][eligible]
    folds = np.array([int(r["cv_fold"]) for r in dev_rows])
    with open(EXTENDED / "background/background_pool.csv", newline="") as handle:
        pool = [load_rgb(plantdoc / r["path"]) for r in csv.DictReader(handle)]
    test_images = [load_rgb(root / r["path"]) for r in test["rows"]]
    with open(EXTENDED / "plantdoc/predictions.csv", newline="") as handle:
        field_rows = list(csv.DictReader(handle))
    field_images = [load_rgb(plantdoc / r["path"]) for r in field_rows]
    truth = test["labels"].astype(object)
    field_truth = np.array([r["health"] for r in field_rows], dtype=object)
    balanced = np.array([r["balanced_subset"] == "True" for r in field_rows])
    result = {"note": "Post-hoc. Pre-registered settings in the module docstring; nothing tuned on PlantDoc.",
              "settings": SETTINGS, "tensorflow": tf.__version__, "n_train": len(images), "methods": {}}
    predictions = {}
    for name, backgrounds in (("FT", False), ("FT-aug", True)):
        fold_scores = []
        for fold in range(3):
            train_index = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            model = train(tf, [images[i] for i in train_index], [masks[i] for i in train_index],
                          labels[train_index], pool, backgrounds, SEED + fold)
            fold_scores.append(macro_f1(labels[held].astype(object), predict(tf, model, [images[i] for i in held])))
            print(f"{name} fold {fold}: {fold_scores[-1]:.4f}", flush=True)
        model = train(tf, images, masks, labels, pool, backgrounds, SEED)
        test_prediction, field_prediction = predict(tf, model, test_images), predict(tf, model, field_images)
        predictions[name] = (test_prediction, field_prediction)
        t, p = field_truth[balanced], field_prediction[balanced]
        result["methods"][name] = {
            "cv_macro_f1": float(np.mean(fold_scores)), "cv_std": float(np.std(fold_scores)),
            "cv_folds": fold_scores, "test_macro_f1_posthoc": macro_f1(truth, test_prediction),
            "plantdoc_macro_f1_balanced": macro_f1(t, p), "plantdoc_macro_f1_all": macro_f1(field_truth, field_prediction),
            "plantdoc_recall_balanced": {h: float(np.mean(p[t == h] == h)) for h in ("Healthy", "Diseased")}}
        print(name, json.dumps(result["methods"][name]), flush=True)
    # Comparisons with earlier PlantDoc predictions (same rows, same order).
    earlier = {}
    for path, columns in ((EXTENDED / "cnn/plantdoc_cnn_predictions.csv", {"A-CNN": "A_CNN_svm"}),
                          (EXTENDED / "grabcut/plantdoc_b2_predictions.csv", {"B2": "B2_svm"}),
                          (EXTENDED / "background/plantdoc_predictions.csv", {"A-CNN-aug": "A-CNN-aug"})):
        with open(path, newline="") as handle:
            other = list(csv.DictReader(handle))
        assert [r["path"] for r in other] == [r["path"] for r in field_rows]
        for method, column in columns.items():
            earlier[method] = np.array([r[column] for r in other], dtype=object)
    t = field_truth[balanced]
    result["plantdoc_paired"] = {
        "FT_vs_A-CNN": paired(t, earlier["A-CNN"][balanced], predictions["FT"][1][balanced]),
        "FT-aug_vs_FT": paired(t, predictions["FT"][1][balanced], predictions["FT-aug"][1][balanced]),
        "FT-aug_vs_A-CNN-aug": paired(t, earlier["A-CNN-aug"][balanced], predictions["FT-aug"][1][balanced]),
        "FT-aug_vs_B2": paired(t, earlier["B2"][balanced], predictions["FT-aug"][1][balanced])}
    result["test_paired"] = {"FT-aug_vs_FT": paired(truth, predictions["FT"][0], predictions["FT-aug"][0])}
    write_csv(OUTPUT / "plantdoc_predictions.csv",
              [{"path": r["path"], "health": r["health"], "balanced_subset": r["balanced_subset"],
                "FT": predictions["FT"][1][i], "FT-aug": predictions["FT-aug"][1][i]} for i, r in enumerate(field_rows)])
    write_json(OUTPUT / "results.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plantdoc", required=True, help="Path to a PlantDoc-Dataset checkout at the pinned commit")
    run(parser.parse_args().plantdoc)
