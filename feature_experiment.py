"""Post-freeze experiment: extra hand-crafted features inside the GrabCut leaf mask (Method B2).

The error analysis showed that B2 mostly misses disease on field photos (small lesions, virus
leaf-curl). Four interpretable feature groups are added to the 58 HSV+LBP features, all computed
inside the GrabCut mask with thresholds fixed in advance:

    lesion   (4)  share of non-green leaf pixels (OpenCV HSV: H < 30, S >= 40, V >= 30), lesions per
                  1000 leaf pixels (components >= 5 px), mean and largest lesion area / leaf area
    moments (18)  mean, standard deviation and skewness of each Lab and HSV channel
    glcm    (12)  grey-level co-occurrence (32 levels, distances 1 and 3, 4 angles averaged):
                  contrast, dissimilarity, homogeneity, energy, correlation, entropy
    shape    (4)  solidity, extent, eccentricity, roughness (perimeter^2 / 4 pi area)

Pre-registered before any result (24 November 2026): candidates are 58, 58+each group and 58+all;
same RBF SVM, C grid, 720 development images and grouped folds; the candidate with the highest
development CV macro F1 is chosen (within 0.001, the one with fewer features). PlantVillage test and
the PlantDoc balanced subset are post-hoc. Group permutation importance on development folds
(10 repeats) and, post-hoc, on PlantDoc. Outputs go to results/extended/features/.

Usage (run the PlantDoc and GrabCut experiments first):
    python feature_experiment.py --plantdoc PATH
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix
from sklearn.base import clone

from extended_analysis import (OUTPUT as EXTENDED, classifiers, fit_search, load_data, macro_f1,
                               require_case_sensitive)
from grabcut_experiment import CACHE as GRABCUT_CACHE, paired, predict
from leaf_pipeline import ROOT, SEED, extract_features, load_rgb, read_manifest, write_csv, write_json
from segmentation_grabcut import grabcut_leaf

OUTPUT = EXTENDED / "features"
CACHE = ROOT / "data/cache/extra_features.npz"
GROUPS = {"lesion": 4, "moments": 18, "glcm": 12, "shape": 4}
GLCM_LEVELS, GLCM_DISTANCES, GLCM_ANGLES = 32, (1, 3), (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def lesion_features(rgb, mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lesion = mask & (hsv[..., 0] < 30) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 30)
    area = mask.sum()
    count, _, stats, _ = cv2.connectedComponentsWithStats(lesion.astype(np.uint8), connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    sizes = sizes[sizes >= 5]
    return np.array([lesion.sum() / area, 1000 * len(sizes) / area,
                     sizes.mean() / area if len(sizes) else 0.0, sizes.max() / area if len(sizes) else 0.0])


def colour_moments(rgb, mask):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[mask]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[mask]
    values = []
    for channels in (lab, hsv):
        for c in range(3):
            x = channels[:, c]
            values += [x.mean(), x.std(), skew(x) if x.std() > 0 else 0.0]
    return np.array(values)


def glcm_features(rgb, mask):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    levels = (grey.astype(np.int32) * GLCM_LEVELS // 256).astype(np.uint8)
    levels[~mask] = GLCM_LEVELS  # an extra level for the background, dropped below
    counts = graycomatrix(levels, GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS + 1, symmetric=True)
    counts = counts[:GLCM_LEVELS, :GLCM_LEVELS].astype(np.float64).sum(axis=3)  # leaf-leaf pairs, angles pooled
    i, j = np.mgrid[0:GLCM_LEVELS, 0:GLCM_LEVELS]
    values = []
    for d in range(len(GLCM_DISTANCES)):
        p = counts[:, :, d] / max(counts[:, :, d].sum(), 1)
        mean_i, mean_j = (i * p).sum(), (j * p).sum()
        std_i, std_j = np.sqrt((((i - mean_i) ** 2) * p).sum()), np.sqrt((((j - mean_j) ** 2) * p).sum())
        correlation = (((i - mean_i) * (j - mean_j) * p).sum() / (std_i * std_j)) if std_i * std_j > 0 else 0.0
        values += [((i - j) ** 2 * p).sum(), (np.abs(i - j) * p).sum(), (p / (1 + (i - j) ** 2)).sum(),
                   np.sqrt((p ** 2).sum()), correlation, -(p[p > 0] * np.log2(p[p > 0])).sum()]
    return np.array(values)


def shape_features(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = max(cv2.contourArea(contour), 1.0)
    hull = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
    spread = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    major, minor = mu20 + mu02 + spread, max(mu20 + mu02 - spread, 0.0)
    eccentricity = np.sqrt(1 - minor / major) if major > 0 else 0.0
    perimeter = cv2.arcLength(contour, True)
    return np.array([area / hull, area / (w * h), eccentricity, perimeter ** 2 / (4 * np.pi * area)])


def extra_features(rgb, mask):
    return np.concatenate([lesion_features(rgb, mask), colour_moments(rgb, mask),
                           glcm_features(rgb, mask), shape_features(mask)])


def build(plantdoc, root, segmentation, inventory):
    """Extra features for every PlantVillage and PlantDoc image; NaN where the B2 mask fails."""
    out = {}
    for name, paths in (("plantvillage", [Path(root) / r["path"] for r in read_manifest(root)]),
                        ("plantdoc", [Path(plantdoc) / r["path"] for r in inventory])):
        values = np.full((len(paths), sum(GROUPS.values())), np.nan)
        for i, path in enumerate(paths):
            rgb = load_rgb(path)
            mask = grabcut_leaf(rgb, segmentation)
            try:
                extract_features(rgb, mask)  # same validity rule as B2
            except ValueError:
                continue
            values[i] = extra_features(rgb, mask)
            if (i + 1) % 300 == 0:
                print(f"{name}: {i + 1}/{len(paths)}", flush=True)
        out[name] = values
    return out


def group_columns():
    columns, start = {"hsv": np.arange(0, 48), "lbp": np.arange(48, 58)}, 58
    for name, size in GROUPS.items():
        columns[name] = np.arange(start, start + size)
        start += size
    return columns


def group_importance(estimator, x, y, generator, repeats=10):
    base = macro_f1(y, estimator.predict(x))
    drops = {}
    for name, columns in group_columns().items():
        scores = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, columns] = x[generator.permutation(len(x))][:, columns]
            scores.append(base - macro_f1(y, estimator.predict(shuffled)))
        drops[name] = float(np.mean(scores))
    return drops


def run(plantdoc, root=ROOT):
    require_case_sensitive(plantdoc)
    root = Path(root)
    frozen, data = load_data(root)
    with open(EXTENDED / "plantdoc/predictions.csv", newline="") as handle:
        field_rows = list(csv.DictReader(handle))
    if CACHE.exists():
        cache = np.load(CACHE)
        extra = {"plantvillage": cache["plantvillage"], "plantdoc": cache["plantdoc"]}
    else:
        extra = build(plantdoc, root, frozen["segmentation"], field_rows)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **extra)
    grabcut = np.load(GRABCUT_CACHE)
    partition = np.array([r["partition"] for r in read_manifest(root)])
    full = {"plantvillage": np.hstack([grabcut["plantvillage"], extra["plantvillage"]]),
            "plantdoc": np.hstack([grabcut["plantdoc"], extra["plantdoc"]])}
    dev_x, test_x = full["plantvillage"][partition == "development"], full["plantvillage"][partition == "test"]
    dev, test = data["development"], data["test"]
    eligible = dev["eligible"] & ~np.isnan(dev_x).any(axis=1)
    folds = np.array([int(r["cv_fold"]) for r in dev["rows"]])[eligible]
    cv = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in range(3)]
    labels = dev["labels"][eligible]
    truth = test["labels"].astype(object)
    field_truth = np.array([r["health"] for r in field_rows], dtype=object)
    balanced = np.array([r["balanced_subset"] == "True" for r in field_rows])
    columns = group_columns()
    base = np.concatenate([columns["hsv"], columns["lbp"]])
    candidates = {"58": base}
    for name in GROUPS:
        candidates[f"58+{name}"] = np.concatenate([base, columns[name]])
    candidates["58+all"] = np.arange(full["plantvillage"].shape[1])
    model, grid = classifiers()["svm_rbf"]
    result = {"note": "Post-hoc test numbers; selection on development CV only (rule in the module docstring).",
              "n_train": int(eligible.sum()), "candidates": {}}
    estimators, test_predictions, field_predictions = {}, {}, {}
    for name, cols in candidates.items():
        search, record = fit_search(model, grid, dev_x[eligible][:, cols], labels, cv)
        estimators[name] = search.best_estimator_
        test_predictions[name] = predict(search.best_estimator_, test_x[:, cols])
        field_predictions[name] = predict(search.best_estimator_, full["plantdoc"][:, cols])
        t, p = field_truth[balanced], field_predictions[name][balanced]
        result["candidates"][name] = {**record, "n_features": int(len(cols)),
                                      "test_macro_f1_posthoc": macro_f1(truth, test_predictions[name]),
                                      "plantdoc_macro_f1_balanced": macro_f1(t, p),
                                      "plantdoc_recall_balanced": {h: float(np.mean(p[t == h] == h)) for h in ("Healthy", "Diseased")}}
        c = result["candidates"][name]
        print(f"{name:12s} ({len(cols):3d}): dev CV {c['cv_macro_f1']:.4f}  test {c['test_macro_f1_posthoc']:.4f}"
              f"  PlantDoc {c['plantdoc_macro_f1_balanced']:.4f}", flush=True)
    best = max(c["cv_macro_f1"] for c in result["candidates"].values())
    chosen = min((n for n, c in result["candidates"].items() if c["cv_macro_f1"] >= best - 0.001),
                 key=lambda n: result["candidates"][n]["n_features"])
    result["chosen_on_development"] = chosen
    t = field_truth[balanced]
    result["paired_vs_58"] = {
        name: {"plantvillage_test": paired(truth, test_predictions["58"], test_predictions[name]),
               "plantdoc_balanced": paired(t, field_predictions["58"][balanced], field_predictions[name][balanced])}
        for name in candidates if name != "58"}  # every candidate reported; only `chosen` is the pre-registered pick
    # Which feature groups does the all-features model rely on?
    generator = np.random.default_rng(SEED)
    x_all, development_drops = dev_x[eligible], []
    parameters = result["candidates"]["58+all"]["best_params"]
    for train_index, held in cv:
        estimator = clone(model).set_params(**parameters).fit(x_all[train_index], labels[train_index])
        development_drops.append(group_importance(estimator, x_all[held], labels[held].astype(object), generator))
    result["group_importance_development"] = {g: float(np.mean([d[g] for d in development_drops]))
                                              for g in columns}
    valid = balanced & ~np.isnan(full["plantdoc"]).any(axis=1)
    result["group_importance_plantdoc_posthoc"] = group_importance(
        estimators["58+all"], full["plantdoc"][valid], field_truth[valid], generator)
    write_csv(OUTPUT / "plantdoc_predictions.csv",
              [{"path": r["path"], "health": r["health"], "balanced_subset": r["balanced_subset"],
                **{n: field_predictions[n][i] for n in candidates}} for i, r in enumerate(field_rows)])
    write_csv(OUTPUT / "candidates.csv", [{"candidate": n, **{k: v for k, v in c.items() if k != "plantdoc_recall_balanced"}}
                                          for n, c in result["candidates"].items()])
    write_json(OUTPUT / "results.json", result)
    plot_importance(result)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
    return result


def plot_importance(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(result["group_importance_development"])
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - 0.2, [result["group_importance_development"][n] for n in names], 0.4, color="#327349",
             label="PlantVillage development folds")
    axis.bar(x + 0.2, [result["group_importance_plantdoc_posthoc"][n] for n in names], 0.4, color="#c07a2c",
             label="PlantDoc field photos (post-hoc)")
    axis.set(xticks=x, ylabel="Macro F1 drop when shuffled", title="Feature group importance (58+all, B2 mask)")
    axis.set_xticklabels(names)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.legend()
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "group_importance.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plantdoc", required=True, help="Path to a PlantDoc-Dataset checkout at the pinned commit")
    run(parser.parse_args().plantdoc)
