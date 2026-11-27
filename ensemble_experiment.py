"""Post-freeze experiment: combining Method B2 with the fine-tuned CNN (FT-aug).

The error analysis showed opposite biases on PlantDoc: B2 (58 hand-crafted features inside the
GrabCut mask) misses disease, while FT-aug (MobileNetV2 fine-tuned on the full image with random
backgrounds) calls most healthy leaves Diseased. This idea came from looking at PlantDoc errors,
so the result is reported as exploratory and exactly one rule is tried.

Pre-registered before any result (24 November 2026), nothing tuned afterwards:
    B2 probability    the B2 RBF SVM (C chosen on development CV, as before) with Platt scaling
                      (CalibratedClassifierCV, sigmoid, 5 folds, ensemble=False); P(Diseased)
    FT-aug probability retrained with the settings and seeds of finetune_experiment.py;
                      softmax P(Diseased); agreement with the earlier FT-aug predictions is reported
    Rule              P = (P_B2 + P_FT-aug) / 2, Diseased if P >= 0.5; where the B2 mask failed,
                      P = P_FT-aug. No weights or thresholds are tuned.
    Development       the same 720 images and grouped 3 folds, both models trained on the other
                      folds (B2's C fixed); reported only, nothing is selected
    Test (post-hoc)   PlantVillage test and the PlantDoc balanced subset; ensemble vs B2 and vs
                      FT-aug with McNemar and bootstrap; per-class accuracy; accuracy where the two
                      models agree and disagree
The final FT-aug model is saved to data/cache/ for the notebook demo. Outputs go to
results/extended/ensemble/.

Usage (needs TensorFlow; run the PlantDoc, GrabCut, background and fine-tuning experiments first):
    python ensemble_experiment.py --plantdoc PATH
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from extended_analysis import OUTPUT as EXTENDED, classifiers, fit_search, load_data, macro_f1, require_case_sensitive
from finetune_experiment import CLASSES, resized, train
from grabcut_experiment import CACHE as GRABCUT_CACHE, paired, predict
from leaf_pipeline import ROOT, SEED, extract_features, load_rgb, read_manifest, write_csv, write_json
from segmentation_grabcut import grabcut_leaf

OUTPUT = EXTENDED / "ensemble"
MODEL = ROOT / "data/cache/ft_aug_final.keras"
THRESHOLD = 0.5


def combine(p_b2, p_ft):
    """Equal-weight mean of P(Diseased); where the B2 mask failed (NaN) only FT-aug is used."""
    p_b2, p_ft = np.asarray(p_b2, float), np.asarray(p_ft, float)
    p = np.where(np.isnan(p_b2), p_ft, (p_b2 + p_ft) / 2)
    return p, labels_from(p)


def labels_from(p):
    return np.where(p >= THRESHOLD, "Diseased", "Healthy").astype(object)


def b2_model(C):
    """The B2 SVM with Platt scaling: a sigmoid fitted on 5-fold out-of-fold decision values."""
    svm = clone(classifiers()["svm_rbf"][0]).set_params(svc__C=C)
    return CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False)


def b2_probability(model, features):
    """P(Diseased) from the Platt-scaled SVM; NaN where the GrabCut mask failed."""
    p = np.full(len(features), np.nan)
    valid = ~np.isnan(features).any(axis=1)
    if valid.any():
        p[valid] = model.predict_proba(features[valid])[:, list(model.classes_).index("Diseased")]
    return p


def ft_probability(tf, model, images):
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    return model.predict(preprocess(resized(images)), batch_size=64, verbose=0)[:, CLASSES.index("Diseased")]


def scores(truth, predictions):
    return {name: macro_f1(truth, value) for name, value in predictions.items()}


def class_accuracy(truth, predictions):
    return {name: {h: float(np.mean(value[truth == h] == h)) for h in ("Healthy", "Diseased")}
            for name, value in predictions.items()}


def agreement(truth, first, second, combined):
    same = first == second
    return {"n_agree": int(same.sum()), "n_disagree": int((~same).sum()),
            "accuracy_when_agree": float(np.mean(combined[same] == truth[same])) if same.any() else None,
            "ensemble_accuracy_when_disagree": float(np.mean(combined[~same] == truth[~same])) if (~same).any() else None,
            "B2_right_when_disagree": int(np.sum((first == truth)[~same])),
            "FT-aug_right_when_disagree": int(np.sum((second == truth)[~same]))}


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
    village = {"development": grabcut["plantvillage"][partition == "development"],
               "test": grabcut["plantvillage"][partition == "test"]}
    # Same eligible development images, folds and inputs as finetune_experiment.py.
    eligible = dev["eligible"] & ~np.isnan(village["development"]).any(axis=1)
    dev_rows = [r for r, keep in zip(dev["rows"], eligible) if keep]
    features = village["development"][eligible]
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
    cv = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in range(3)]
    with open(EXTENDED / "background/background_pool.csv", newline="") as handle:
        pool = [load_rgb(plantdoc / r["path"]) for r in csv.DictReader(handle)]
    test_images = [load_rgb(root / r["path"]) for r in test["rows"]]
    with open(EXTENDED / "plantdoc/predictions.csv", newline="") as handle:
        field_rows = list(csv.DictReader(handle))
    field_images = [load_rgb(plantdoc / r["path"]) for r in field_rows]
    field_features = grabcut["plantdoc"]
    truth = test["labels"].astype(object)
    field_truth = np.array([r["health"] for r in field_rows], dtype=object)
    balanced = np.array([r["balanced_subset"] == "True" for r in field_rows])
    earlier = {}
    for path, column in ((EXTENDED / "grabcut/plantdoc_b2_predictions.csv", "B2_svm"),
                         (EXTENDED / "finetune/plantdoc_predictions.csv", "FT-aug")):
        with open(path, newline="") as handle:
            other = list(csv.DictReader(handle))
        assert [r["path"] for r in other] == [r["path"] for r in field_rows]
        earlier[column] = np.array([r[column] for r in other], dtype=object)

    # B2: the same C search as grabcut_experiment.py, then the Platt-scaled SVM.
    search, record = fit_search(*classifiers()["svm_rbf"], features, labels, cv)
    C = search.best_params_["svc__C"]
    b2_field_labels = predict(search.best_estimator_, field_features)
    assert (b2_field_labels == earlier["B2_svm"]).all(), "B2 does not reproduce grabcut_experiment.py"

    # Development: out-of-fold probabilities from both models.
    oof = {"B2": np.full(len(labels), np.nan), "FT-aug": np.full(len(labels), np.nan)}
    oof_b2_labels = np.empty(len(labels), dtype=object)
    for fold, (train_index, held) in enumerate(cv):
        model = b2_model(C).fit(features[train_index], labels[train_index])
        oof["B2"][held] = b2_probability(model, features[held])
        oof_b2_labels[held] = clone(search.best_estimator_).fit(features[train_index], labels[train_index]).predict(features[held])
        network = train(tf, [images[i] for i in train_index], [masks[i] for i in train_index],
                        labels[train_index], pool, True, SEED + fold)
        oof["FT-aug"][held] = ft_probability(tf, network, [images[i] for i in held])
        print(f"fold {fold} done", flush=True)

    # Final models on all development images.
    b2_final = b2_model(C).fit(features, labels)
    network = train(tf, images, masks, labels, pool, True, SEED)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    network.save(MODEL)
    probabilities = {
        "development": oof,
        "test": {"B2": b2_probability(b2_final, village["test"]), "FT-aug": ft_probability(tf, network, test_images)},
        "plantdoc": {"B2": b2_probability(b2_final, field_features), "FT-aug": ft_probability(tf, network, field_images)}}
    reference = {"development": {"B2": oof_b2_labels},
                 "test": {"B2": predict(search.best_estimator_, village["test"])},
                 "plantdoc": {"B2": b2_field_labels}}
    truths = {"development": labels.astype(object), "test": truth, "plantdoc": field_truth}

    result = {"note": "Exploratory and post-hoc; one pre-registered rule (module docstring), nothing tuned.",
              "b2_C": C, "b2_development_search": record, "threshold": THRESHOLD, "n_train": len(labels),
              "tensorflow": tf.__version__, "splits": {}}
    outputs = {}
    for split, p in probabilities.items():
        combined, ensemble = combine(p["B2"], p["FT-aug"])
        b2 = reference[split]["B2"]
        ft = labels_from(p["FT-aug"])
        index = balanced if split == "plantdoc" else np.ones(len(ensemble), bool)
        t = truths[split][index]
        predictions = {"B2": b2[index], "FT-aug": ft[index], "Ensemble": ensemble[index]}
        entry = {"n": int(index.sum()), "macro_f1": scores(t, predictions),
                 "class_accuracy": class_accuracy(t, predictions),
                 "ensemble_vs_B2": paired(t, predictions["B2"], predictions["Ensemble"]),
                 "ensemble_vs_FT-aug": paired(t, predictions["FT-aug"], predictions["Ensemble"]),
                 "agreement": agreement(t, predictions["B2"], predictions["FT-aug"], predictions["Ensemble"])}
        valid = ~np.isnan(p["B2"][index])
        entry["b2_platt_label_agreement"] = float(np.mean(labels_from(p["B2"][index][valid]) == predictions["B2"][valid]))
        if split == "development":  # the mean of per-fold scores, as reported for earlier methods
            entry["macro_f1_fold_mean"] = {name: float(np.mean([macro_f1(t[folds == f], value[folds == f])
                                                                for f in range(3)]))
                                           for name, value in predictions.items()}
        if split == "plantdoc":
            entry["ft_aug_agreement_with_earlier_run"] = float(np.mean(ft[balanced] == earlier["FT-aug"][balanced]))
            entry["ft_aug_earlier_run_macro_f1"] = macro_f1(t, earlier["FT-aug"][balanced])
            entry["macro_f1_all_1089"] = scores(field_truth, {"B2": b2, "FT-aug": ft, "Ensemble": ensemble})
            entry["b2_mask_failed"] = int(np.isnan(p["B2"][balanced]).sum())
        result["splits"][split] = entry
        outputs[split] = (p, combined, ensemble, ft)
        print(split, json.dumps(entry["macro_f1"]), flush=True)

    p, combined, ensemble, ft = outputs["plantdoc"]
    write_csv(OUTPUT / "plantdoc_predictions.csv",
              [{"path": r["path"], "health": r["health"], "balanced_subset": r["balanced_subset"],
                "p_B2": "" if np.isnan(p["B2"][i]) else round(float(p["B2"][i]), 4),
                "p_FT-aug": round(float(p["FT-aug"][i]), 4), "p_ensemble": round(float(combined[i]), 4),
                "B2": b2_field_labels[i], "FT-aug": ft[i], "Ensemble": ensemble[i]}
               for i, r in enumerate(field_rows)])
    write_json(OUTPUT / "results.json", result)
    plot(p["B2"][balanced], p["FT-aug"][balanced], field_truth[balanced], OUTPUT / "plantdoc_probabilities.png")
    print(json.dumps(result, indent=2))
    return result


def plot(p_b2, p_ft, truth, path):
    """PlantDoc P(Diseased) from both models; points above the dashed line are called Diseased."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    figure, axis = plt.subplots(figsize=(6.4, 5.6), facecolor="#fcfcfb")
    axis.set_facecolor("#fcfcfb")
    valid = ~np.isnan(p_b2)
    for label, colour, marker in (("Healthy", "#2a78d6", "o"), ("Diseased", "#eb6834", "^")):
        index = valid & (truth == label)
        axis.scatter(p_b2[index], p_ft[index], s=34, c=colour, marker=marker, alpha=0.75,
                     edgecolors="#fcfcfb", linewidths=0.8, label=f"truly {label} (n={index.sum()})")
    axis.plot([0, 1], [1, 0], color=ink, linestyle="--", linewidth=1.5)
    axis.text(0.97, 0.97, "ensemble: Diseased", ha="right", va="top", color=ink, fontsize=9)
    axis.text(0.03, 0.03, "ensemble: Healthy", ha="left", va="bottom", color=ink, fontsize=9)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("B2: P(Diseased)", color=ink)
    axis.set_ylabel("FT-aug: P(Diseased)", color=ink)
    axis.set_title(f"PlantDoc balanced subset ({valid.sum()} photos with a B2 mask)", color=ink, fontsize=11)
    axis.grid(color=grid, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_color(grid)
    axis.tick_params(colors=muted)
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plantdoc", required=True, help="Path to a PlantDoc-Dataset checkout at the pinned commit")
    run(parser.parse_args().plantdoc)
