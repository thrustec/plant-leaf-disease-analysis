"""Post-freeze extended analyses for the final presentation.

Nothing here changes the frozen benchmark: leaf_pipeline.py, experiment.py,
models/frozen_protocol.json and the frozen models are only read. All outputs go to
results/extended/. Every selection (classifier, hyperparameters, feature subset) is
made on the 720 development images with the same grouped 3-fold CV and macro F1 as
the frozen protocol. Test-set numbers produced here are post-hoc secondary analyses.

Usage:
    python extended_analysis.py classifiers   # classifier comparison + feature ablation
    python extended_analysis.py significance  # McNemar + bootstrap for frozen A vs B
    python extended_analysis.py external --plantdoc PATH  # frozen models on PlantDoc field photos
    python extended_analysis.py all
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiment import feature_matrices
from leaf_pipeline import (LABELS, ROOT, SEED, extract_features, file_hash, load_rgb, overlay_mask,
                           read_manifest, segment_leaf, write_csv, write_json)

OUTPUT = ROOT / "results/extended"
COLOR = slice(0, 48)
TEXTURE = slice(48, 58)
FEATURE_SETS = {"color_hsv_48": COLOR, "texture_lbp_10": TEXTURE, "both_58": slice(0, 58)}


def classifiers():
    """Candidate models and their development grids. svm_rbf is the frozen configuration."""
    scaled = lambda model: make_pipeline(StandardScaler(), model)
    return {
        "svm_rbf": (scaled(SVC(kernel="rbf", gamma="scale", class_weight="balanced")),
                    {"svc__C": [0.1, 1.0, 10.0]}),
        "svm_linear": (scaled(SVC(kernel="linear", class_weight="balanced")),
                       {"svc__C": [0.01, 0.1, 1.0]}),
        "logistic_regression": (scaled(LogisticRegression(class_weight="balanced", max_iter=5000)),
                                {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}),
        "knn": (scaled(KNeighborsClassifier()),
                {"kneighborsclassifier__n_neighbors": [3, 5, 9, 15],
                 "kneighborsclassifier__weights": ["uniform", "distance"]}),
        "random_forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=SEED, n_jobs=1),
                          {"max_depth": [None, 10], "min_samples_leaf": [1, 3]}),
        "gradient_boosting": (HistGradientBoostingClassifier(random_state=SEED),
                              {"learning_rate": [0.05, 0.1], "max_depth": [3, None]}),
    }


def load_data(root=ROOT):
    """Features with the frozen segmentation; same shared-training rule as experiment.py."""
    root = Path(root)
    frozen = json.loads((root / "models/frozen_protocol.json").read_text())
    rows = read_manifest(root)
    full, masked, errors = feature_matrices(rows, frozen["segmentation"], root)
    data = {}
    for partition in ("development", "test"):
        index = np.array([row["partition"] == partition for row in rows])
        part_rows = [row for row, keep in zip(rows, index) if keep]
        data[partition] = {"rows": part_rows, "A": full[index], "B": masked[index],
                           "errors": [e for e, keep in zip(errors, index) if keep],
                           "labels": np.array([row["health"] for row in part_rows])}
    dev = data["development"]
    eligible = np.array([not error for error in dev["errors"]])
    dev["eligible"] = eligible
    folds = np.array([int(row["cv_fold"]) for row in dev["rows"]])[eligible]
    dev["cv"] = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in range(3)]
    return frozen, data


def fit_search(model, grid, features, labels, cv):
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True,
                          error_score="raise")
    search.fit(features, labels)
    best = search.best_index_
    return search, {"cv_macro_f1": float(search.best_score_),
                    "cv_std": float(search.cv_results_["std_test_score"][best]),
                    "best_params": {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                                    for k, v in search.best_params_.items()}}


def predict_test(estimator, data, method, columns=slice(0, 58)):
    """Test predictions; a failed Method B mask is FAILED (a false negative), as in the frozen policy."""
    test = data["test"]
    features = test[method][:, columns]
    predictions = np.array(["FAILED"] * len(features), dtype=object)
    valid = ~np.isnan(features).any(axis=1)
    if valid.any():
        predictions[valid] = estimator.predict(features[valid])
    return predictions


def macro_f1(truth, predictions):
    return float(f1_score(truth, predictions, labels=list(LABELS), average="macro", zero_division=0))


def run_classifiers(root=ROOT):
    frozen, data = load_data(root)
    dev, test = data["development"], data["test"]
    labels = dev["labels"][dev["eligible"]]
    comparison, ablation = [], []
    for name, (model, grid) in classifiers().items():
        for method in ("A", "B"):
            search, record = fit_search(model, grid, dev[method][dev["eligible"]], labels, dev["cv"])
            predictions = predict_test(search.best_estimator_, data, method)
            comparison.append({"classifier": name, "method": method, **record,
                               "test_macro_f1_posthoc": macro_f1(test["labels"], predictions),
                               "best_params": json.dumps(record["best_params"])})
            print(f"{name:20s} {method}: dev CV {record['cv_macro_f1']:.4f} ± {record['cv_std']:.4f}"
                  f"  test {comparison[-1]['test_macro_f1_posthoc']:.4f}", flush=True)
    model, grid = classifiers()["svm_rbf"]
    for feature_set, columns in FEATURE_SETS.items():
        for method in ("A", "B"):
            features = dev[method][dev["eligible"]][:, columns]
            search, record = fit_search(model, grid, features, labels, dev["cv"])
            predictions = predict_test(search.best_estimator_, data, method, columns)
            ablation.append({"features": feature_set, "method": method, **record,
                             "test_macro_f1_posthoc": macro_f1(test["labels"], predictions),
                             "best_params": json.dumps(record["best_params"])})
            print(f"ablation {feature_set:15s} {method}: dev CV {record['cv_macro_f1']:.4f}", flush=True)
    write_csv(OUTPUT / "classifier_comparison.csv", comparison)
    write_csv(OUTPUT / "feature_ablation.csv", ablation)
    plot_comparison(comparison, ablation)
    return comparison, ablation


def frozen_or_reproduced(method, data, root=ROOT):
    """Use the frozen model file if present; otherwise refit the identical frozen configuration."""
    path = Path(root) / "models" / f"development_{method}.joblib"
    if path.exists():
        return joblib.load(path)["pipeline"], "frozen_model_file"
    dev = data["development"]
    model, grid = classifiers()["svm_rbf"]
    search, _ = fit_search(model, grid, dev[method][dev["eligible"]], dev["labels"][dev["eligible"]], dev["cv"])
    return search.best_estimator_, "refit_identical_configuration"


def mcnemar(truth, first, second):
    first_right, second_right = first == truth, second == truth
    b = int(np.sum(first_right & ~second_right))
    c = int(np.sum(~first_right & second_right))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"only_first_correct": b, "only_second_correct": c, "exact_p_value": float(p)}


def bootstrap_difference(truth, first, second, repetitions=5000):
    generator = np.random.default_rng(SEED)
    differences = []
    for _ in range(repetitions):
        index = generator.integers(0, len(truth), len(truth))
        differences.append(macro_f1(truth[index], second[index]) - macro_f1(truth[index], first[index]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"B_minus_A": macro_f1(truth, second) - macro_f1(truth, first),
            "ci95_low": float(low), "ci95_high": float(high), "repetitions": repetitions}


def run_significance(root=ROOT):
    frozen, data = load_data(root)
    dev, test = data["development"], data["test"]
    result = {"note": "A vs B, frozen RBF SVM configuration. FAILED masks count as wrong."}
    predictions, sources = {}, {}
    for method in ("A", "B"):
        estimator, sources[method] = frozen_or_reproduced(method, data, root)
        predictions[method] = predict_test(estimator, data, method)
    truth = test["labels"].astype(object)
    result["test"] = {"n": len(truth), "model_source": sources,
                      "macro_f1": {m: macro_f1(truth, predictions[m]) for m in ("A", "B")},
                      "mcnemar_A_first": mcnemar(truth, predictions["A"], predictions["B"]),
                      "bootstrap": bootstrap_difference(truth, predictions["A"], predictions["B"])}
    # Development: out-of-fold predictions on the same grouped folds give a larger paired sample.
    labels = dev["labels"][dev["eligible"]].astype(object)
    out_of_fold = {}
    for method in ("A", "B"):
        model, grid = classifiers()["svm_rbf"]
        features = dev[method][dev["eligible"]]
        _, record = fit_search(model, grid, features, labels, dev["cv"])
        model.set_params(**record["best_params"])
        out_of_fold[method] = cross_val_predict(model, features, labels, cv=dev["cv"]).astype(object)
    result["development_out_of_fold"] = {
        "n": len(labels), "macro_f1": {m: macro_f1(labels, out_of_fold[m]) for m in ("A", "B")},
        "mcnemar_A_first": mcnemar(labels, out_of_fold["A"], out_of_fold["B"]),
        "bootstrap": bootstrap_difference(labels, out_of_fold["A"], out_of_fold["B"])}
    write_json(OUTPUT / "significance.json", result)
    print(json.dumps(result, indent=2))
    return result


# Pre-registered in /mnt/project-files/merve/EXTERNAL_TEST_PLAN.md (9 November 2026), before any run.
PLANTDOC_COMMIT = "5467f6012d78d1c446145d5f582da6096f852ae8"
PLANTDOC_MAPPING = {
    "Apple leaf": ("Apple", "Healthy"), "Apple Scab Leaf": ("Apple", "Diseased"),
    "Apple rust leaf": ("Apple", "Diseased"),
    "Bell_pepper leaf": ("Pepper,_bell", "Healthy"), "Bell_pepper leaf spot": ("Pepper,_bell", "Diseased"),
    "Tomato leaf": ("Tomato", "Healthy"), "Tomato Early blight leaf": ("Tomato", "Diseased"),
    "Tomato Septoria leaf spot": ("Tomato", "Diseased"), "Tomato leaf bacterial spot": ("Tomato", "Diseased"),
    "Tomato leaf late blight": ("Tomato", "Diseased"), "Tomato mold leaf": ("Tomato", "Diseased"),
    "Tomato leaf yellow virus": ("Tomato", "Diseased"),
}


def dhash(rgb):
    gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((gray[:, 1:] > gray[:, :-1]).ravel())


def hamming(first, second):
    return int(np.unpackbits(np.bitwise_xor(first, second)).sum())


def require_case_sensitive(plantdoc):
    """PlantDoc has two different images whose names differ only in case (Apple rust leaf/CAR1.jpg and
    car1.jpg). A case-insensitive disk (default macOS, Windows) keeps only one of them, which silently
    changes the inventory, so stop with a clear message instead."""
    folder = Path(plantdoc) / "train/Apple rust leaf"
    upper, lower = folder / "CAR1.jpg", folder / "car1.jpg"
    if not (upper.is_file() and lower.is_file()) or upper.samefile(lower) or file_hash(upper) == file_hash(lower):
        raise SystemExit(f"{folder}: CAR1.jpg and car1.jpg must be two different files. This disk is not "
                         "case-sensitive; run the PlantDoc experiments on Colab, Linux or a case-sensitive volume.")


def plantdoc_inventory(plantdoc, root=ROOT):
    """Readable, de-duplicated PlantDoc images in our three plants, minus PlantVillage near-copies."""
    plantdoc = Path(plantdoc)
    require_case_sensitive(plantdoc)
    village = [dhash(load_rgb(Path(root) / row["path"])) for row in read_manifest(root)]
    records, seen, rejected = [], {}, []
    for split in ("train", "test"):
        for folder, (plant, health) in PLANTDOC_MAPPING.items():
            for path in sorted((plantdoc / split / folder).glob("*")):
                relative = str(path.relative_to(plantdoc))
                try:
                    rgb = load_rgb(path)
                except ValueError as error:
                    rejected.append({"path": relative, "reason": f"unreadable: {error}"})
                    continue
                pixels = hashlib.sha256(rgb.tobytes()).hexdigest()
                if pixels in seen:
                    rejected.append({"path": relative, "reason": f"exact duplicate of {seen[pixels]}"})
                    continue
                code = dhash(rgb)
                if min(hamming(code, other) for other in village) <= 4:
                    rejected.append({"path": relative, "reason": "near-duplicate of a PlantVillage image"})
                    continue
                seen[pixels] = relative
                records.append({"path": relative, "folder": folder, "plant": plant, "health": health,
                                "sha256": file_hash(path)})
    return records, rejected


def balanced_subset(records):
    """Per plant: every healthy image and the same number of diseased, round-robin over disease folders."""
    generator = np.random.default_rng(SEED)
    chosen = []
    for plant in sorted({r["plant"] for r in records}):
        healthy = [r for r in records if r["plant"] == plant and r["health"] == "Healthy"]
        by_folder = {}
        for r in records:
            if r["plant"] == plant and r["health"] == "Diseased":
                by_folder.setdefault(r["folder"], []).append(r)
        queues = {f: list(generator.permutation(len(items))) for f, items in sorted(by_folder.items())}
        diseased = []
        while len(diseased) < len(healthy) and any(queues.values()):
            for folder in sorted(queues):
                if queues[folder] and len(diseased) < len(healthy):
                    diseased.append(by_folder[folder][queues[folder].pop(0)])
        chosen += healthy + diseased
    return chosen


def run_external(plantdoc, root=ROOT):
    frozen, data = load_data(root)
    records, rejected = plantdoc_inventory(plantdoc, root)
    output = OUTPUT / "plantdoc"
    write_csv(output / "inventory.csv", records)
    if rejected:
        write_csv(output / "rejected.csv", rejected)
    features = {"A": [], "B": []}
    failures = []
    for record in records:
        rgb = load_rgb(Path(plantdoc) / record["path"])
        features["A"].append(extract_features(rgb))
        try:
            mask = segment_leaf(rgb, frozen["segmentation"])
            features["B"].append(extract_features(rgb, mask))
            failures.append("")
        except ValueError as error:
            features["B"].append(np.full(58, np.nan))
            failures.append(str(error))
    features = {m: np.asarray(v) for m, v in features.items()}
    truth = np.array([r["health"] for r in records], dtype=object)
    balanced_ids = {r["path"] for r in balanced_subset(records)}
    balanced = np.array([r["path"] in balanced_ids for r in records])
    # Models: the frozen RBF SVM (primary) plus every comparison classifier selected on development CV.
    dev = data["development"]
    labels = dev["labels"][dev["eligible"]]
    models = {}
    for method in ("A", "B"):
        models[("svm_rbf_frozen", method)] = frozen_or_reproduced(method, data, root)[0]
        for name, (model, grid) in classifiers().items():
            if name != "svm_rbf":
                models[(name, method)] = fit_search(model, grid, dev[method][dev["eligible"]], labels, dev["cv"])[0].best_estimator_
    predictions, summary = {}, {"plantdoc_commit": PLANTDOC_COMMIT, "segmentation": frozen["segmentation"],
                                "n_images": len(records), "n_rejected": len(rejected),
                                "n_balanced": int(balanced.sum()),
                                "B_mask_failures": int(sum(bool(f) for f in failures)), "results": []}
    for (name, method), estimator in models.items():
        x = features[method]
        result = np.array(["FAILED"] * len(x), dtype=object)
        valid = ~np.isnan(x).any(axis=1)
        result[valid] = estimator.predict(x[valid])
        predictions[(name, method)] = result
        row = {"classifier": name, "method": method,
               "macro_f1_balanced": macro_f1(truth[balanced], result[balanced]),
               "macro_f1_all": macro_f1(truth, result),
               "accuracy_balanced": float(np.mean(truth[balanced] == result[balanced]))}
        for plant in sorted({r["plant"] for r in records}):
            index = balanced & np.array([r["plant"] == plant for r in records])
            row[f"macro_f1_balanced_{plant}"] = macro_f1(truth[index], result[index])
        summary["results"].append(row)
        print(f"{name:20s} {method}: balanced F1 {row['macro_f1_balanced']:.4f}  all {row['macro_f1_all']:.4f}", flush=True)
    a, b = predictions[("svm_rbf_frozen", "A")], predictions[("svm_rbf_frozen", "B")]
    summary["frozen_svm_balanced"] = {"mcnemar_A_first": mcnemar(truth[balanced], a[balanced], b[balanced]),
                                      "bootstrap": bootstrap_difference(truth[balanced], a[balanced], b[balanced])}
    write_csv(output / "metrics.csv", summary["results"])
    write_csv(output / "predictions.csv", [
        {**r, "balanced_subset": bool(s), "B_failure": f, "A_svm": pa, "B_svm": pb}
        for r, s, f, pa, pb in zip(records, balanced, failures, a, b)])
    write_json(output / "summary.json", summary)
    plot_external_examples(plantdoc, records, frozen["segmentation"], output)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return summary


def plot_external_examples(plantdoc, records, parameters, output, count=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    generator = np.random.default_rng(SEED)
    picks = [records[i] for i in generator.choice(len(records), count, replace=False)]
    figure, axes = plt.subplots(3, 4, figsize=(12, 9.5))
    for axis, record in zip(axes.ravel(), picks):
        rgb = load_rgb(Path(plantdoc) / record["path"])
        axis.imshow(overlay_mask(rgb, segment_leaf(rgb, parameters)))
        axis.set_title(f"{record['folder']}", fontsize=8)
        axis.axis("off")
    figure.suptitle("PlantDoc field photos: frozen Lab segmentation (Method B)")
    figure.tight_layout()
    figure.savefig(output / "mask_examples.png", dpi=130)
    plt.close(figure)


def plot_comparison(comparison, ablation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    for axis, rows, key, title in ((axes[0], comparison, "classifier", "Classifier comparison"),
                                   (axes[1], ablation, "features", "Feature ablation (RBF SVM)")):
        names = list(dict.fromkeys(row[key] for row in rows))
        x = np.arange(len(names))
        for offset, method, color in ((-0.2, "A", "#637f78"), (0.2, "B", "#327349")):
            values = [next(r for r in rows if r[key] == n and r["method"] == method) for n in names]
            axis.bar(x + offset, [v["cv_macro_f1"] for v in values], 0.38,
                     yerr=[v["cv_std"] for v in values], capsize=3, color=color,
                     label="A: full image" if method == "A" else "B: leaf mask")
        axis.set(xticks=x, ylim=(0.5, 1.0), ylabel="Development CV macro F1", title=title)
        axis.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower left")
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "classifier_comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("classifiers", "significance", "external", "all"))
    parser.add_argument("--plantdoc", help="Path to a PlantDoc-Dataset checkout at the pinned commit")
    args = parser.parse_args()
    if args.action in ("classifiers", "all"):
        run_classifiers()
    if args.action in ("significance", "all"):
        run_significance()
    if args.action == "external" or (args.action == "all" and args.plantdoc):
        if not args.plantdoc:
            parser.error("external needs --plantdoc")
        run_external(args.plantdoc)
