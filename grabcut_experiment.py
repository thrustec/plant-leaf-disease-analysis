"""Post-freeze experiment: GrabCut leaf segmentation ("Fikir 2") for field photos.

Nothing here changes the frozen benchmark. leaf_pipeline.py, experiment.py and
models/frozen_protocol.json are only read; outputs go to results/extended/grabcut/.

Two actions:

masks     Mask quality (Dice/IoU) against hand-drawn PlantDoc reference masks
          (data/annotations/plantdoc/, drawn in Labelme on 256x256 images). The drawn
          photos are split in their pre-registered order: the first 10 are the selection
          set (the only images used to pick parameters), the next 20 are the test set
          (scored once), the rest are extra. The 12 PlantVillage development references
          are a regression check.

classify  Method B2: the same 58 HSV+LBP features and RBF SVM as Method B, but inside
          the GrabCut mask. C is chosen on development grouped 3-fold CV, as for A and B.
          The PlantVillage test and the PlantDoc balanced subset are post-hoc. PlantDoc is
          never used for training. Needs the outputs of
          `python extended_analysis.py external --plantdoc PATH` (same inventory and
          A/B predictions). The fitted B2 model is saved to models/development_B2.joblib
          (same layout as development_A/B, plus the GrabCut parameters) for the notebook demo.

Usage:
    python grabcut_experiment.py masks --plantdoc PATH
    python grabcut_experiment.py classify --plantdoc PATH
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.model_selection import cross_val_predict

from extended_analysis import (OUTPUT as EXTENDED, bootstrap_difference, classifiers, fit_search, load_data,
                               macro_f1, mcnemar, require_case_sensitive)
from leaf_pipeline import (LABELS, ROOT, SEED, extract_features, file_hash, load_rgb, mask_scores, read_manifest,
                           segment_leaf, write_csv, write_json)
from segmentation_grabcut import GRABCUT, central_lab_mask, grabcut_leaf

OUTPUT = EXTENDED / "grabcut"
ANNOTATIONS = ROOT / "data/annotations/plantdoc"
CACHE = ROOT / "data/cache/grabcut_features.npz"
SELECTION, TEST = 10, 20
SKIP_FLAG = "atla_net_yaprak_yok"  # annotator flag: no single clear leaf, image skipped


def reference_mask(annotation):
    """Filled leaf polygons; points are clipped to the image as Labelme can place them on the edge."""
    mask = np.zeros((256, 256), np.uint8)
    for shape in annotation["shapes"]:
        points = np.clip(np.asarray(shape["points"], dtype=float), 0, 255)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def plantdoc_references(plantdoc, root=ROOT):
    """(name, rgb, reference) for every drawn, non-skipped photo, in pre-registered order."""
    annotations = Path(root) / "data/annotations/plantdoc"
    references = []
    with open(annotations / "candidates.csv", newline="") as handle:
        candidates = sorted(csv.DictReader(handle), key=lambda row: int(row["order"]))
    for row in candidates:
        path = annotations / row["image"].replace(".png", ".json")
        if not path.exists():
            continue
        annotation = json.loads(path.read_text())
        if annotation.get("flags", {}).get(SKIP_FLAG) or not annotation.get("shapes"):
            continue
        source = Path(plantdoc) / row["plantdoc_path"]
        if file_hash(source) != row["sha256"]:
            raise ValueError(f"{source} differs from the annotated PlantDoc file")
        references.append((path.stem, load_rgb(source), reference_mask(annotation)))
    return references


def plantvillage_references(root=ROOT):
    root = Path(root)
    by_id = {row["image_id"]: row for row in read_manifest(root)}
    references = []
    for path in sorted((root / "data/annotations/labelme/development").glob("*.json")):
        references.append((path.stem, load_rgb(root / by_id[path.stem]["path"]),
                           reference_mask(json.loads(path.read_text()))))
    return references


def score(method, references):
    scores = [mask_scores(method(rgb), reference) for _, rgb, reference in references]
    return scores, float(np.mean([s["dice"] for s in scores])), float(np.mean([s["iou"] for s in scores]))


def candidate_methods(segmentation):
    """Small grids, compared on the selection set only."""
    methods = [("frozen_lab", {}, lambda rgb: segment_leaf(rgb, segmentation))]
    for opening in (5, 9, 13):
        methods.append(("central_lab", {"opening": opening},
                        lambda rgb, o=opening: central_lab_mask(rgb, segmentation, o)))
    for opening in (5, 9, 13):
        for central in (0.30, 0.35, 0.40):
            methods.append(("grabcut", {"opening": opening, "central": central},
                            lambda rgb, o=opening, c=central: grabcut_leaf(rgb, segmentation,
                                                                           {"opening": o, "central": c})))
    return methods


def run_masks(plantdoc, root=ROOT):
    frozen = json.loads((Path(root) / "models/frozen_protocol.json").read_text())
    references = plantdoc_references(plantdoc, root)
    selection, test, extra = references[:SELECTION], references[SELECTION:SELECTION + TEST], references[SELECTION + TEST:]
    grid, chosen = [], {}
    for name, parameters, method in candidate_methods(frozen["segmentation"]):
        _, dice, iou = score(method, selection)
        grid.append({"method": name, "parameters": json.dumps(parameters),
                     "selection_dice": dice, "selection_iou": iou})
        if name not in chosen or dice > chosen[name][0]:
            chosen[name] = (dice, parameters, method)
        print(f"{name:12s} {parameters}: selection Dice {dice:.4f}", flush=True)
    sets = {"plantdoc_selection_10": selection, "plantdoc_test_20": test,
            "plantdoc_all": references, "plantvillage_development_12": plantvillage_references(root)}
    names = {n: s for s, items in (("selection", selection), ("test", test), ("extra", extra)) for n, _, _ in items}
    result = {"selection_images": [n for n, _, _ in selection], "test_images": [n for n, _, _ in test],
              "extra_images": [n for n, _, _ in extra], "grabcut_defaults": GRABCUT,
              "chosen_on_selection": {k: v[1] for k, v in chosen.items()}, "methods": {}}
    per_image = []
    for name, (_, parameters, method) in chosen.items():
        result["methods"][name] = {}
        for label, items in sets.items():
            scores, dice, iou = score(method, items)
            result["methods"][name][label] = {"n": len(items), "dice": dice, "iou": iou,
                                              "dice_min": float(min(s["dice"] for s in scores)),
                                              "n_dice_below_0.5": int(sum(s["dice"] < 0.5 for s in scores))}
            if label == "plantdoc_all":
                per_image += [{"image": n, "set": names[n], "method": name, **s}
                              for (n, _, _), s in zip(items, scores)]
    write_csv(OUTPUT / "mask_selection_grid.csv", grid)
    write_csv(OUTPUT / "mask_per_image.csv", per_image)
    write_json(OUTPUT / "mask_results.json", result)
    print(json.dumps(result["methods"], indent=2))
    return result


def grabcut_features(paths, segmentation):
    """B2 features; a mask the feature extractor rejects is NaN (counted as FAILED)."""
    features, errors = [], []
    for path in paths:
        rgb = load_rgb(path)
        try:
            features.append(extract_features(rgb, grabcut_leaf(rgb, segmentation)))
            errors.append("")
        except ValueError as error:
            features.append(np.full(58, np.nan))
            errors.append(str(error))
    return np.asarray(features), errors


def cached_features(plantdoc, inventory, segmentation, root=ROOT):
    root = Path(root)
    rows = read_manifest(root)
    key = json.dumps({"grabcut": GRABCUT, "segmentation": segmentation,
                      "plantvillage": [r["sha256"] for r in rows] if "sha256" in rows[0] else [r["path"] for r in rows],
                      "plantdoc": [r["sha256"] for r in inventory]}, sort_keys=True)
    if CACHE.exists():
        cache = np.load(CACHE)
        if str(cache["key"]) == key:
            return cache["plantvillage"], cache["plantdoc"]
    cv2.setNumThreads(1)
    village, _ = grabcut_features([root / r["path"] for r in rows], segmentation)
    field, _ = grabcut_features([Path(plantdoc) / r["path"] for r in inventory], segmentation)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, key=key, plantvillage=village, plantdoc=field)
    return village, field


def predict(estimator, features):
    predictions = np.array(["FAILED"] * len(features), dtype=object)
    valid = ~np.isnan(features).any(axis=1)
    if valid.any():
        predictions[valid] = estimator.predict(features[valid])
    return predictions


def save_b2_model(estimator, segmentation, training_image_ids, root=ROOT):
    """Store the fitted B2 classifier like experiment.py stores A and B, plus the GrabCut parameters."""
    artifact = {"method": "B2", "pipeline": estimator, "segmentation": dict(segmentation),
                "grabcut_parameters": dict(GRABCUT), "labels": list(LABELS), "image_size": [256, 256],
                "feature_count": 58, "seed": SEED, "best_C": float(estimator.get_params()["svc__C"]),
                "training_image_ids": list(training_image_ids)}
    path = Path(root) / "models" / "development_B2.joblib"
    path.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, path)
    return path


def paired(truth, first, second):
    return {"mcnemar": mcnemar(truth, first, second), "bootstrap": bootstrap_difference(truth, first, second)}


def run_classify(plantdoc, root=ROOT):
    require_case_sensitive(plantdoc)
    frozen, data = load_data(root)
    external = EXTENDED / "plantdoc"
    if not (external / "predictions.csv").exists():
        raise SystemExit("Run `python extended_analysis.py external --plantdoc PATH` first")
    with open(external / "inventory.csv", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    with open(external / "predictions.csv", newline="") as handle:
        field_rows = list(csv.DictReader(handle))
    village, field = cached_features(plantdoc, inventory, frozen["segmentation"], root)
    partition = np.array([row["partition"] for row in read_manifest(root)])
    dev, test = data["development"], data["test"]
    b2 = {"development": village[partition == "development"], "test": village[partition == "test"]}
    # Shared training rule: train only on images whose masks are valid for every compared method.
    eligible = dev["eligible"] & ~np.isnan(b2["development"]).any(axis=1)
    folds = np.array([int(row["cv_fold"]) for row in dev["rows"]])[eligible]
    cv = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in range(3)]
    labels = dev["labels"][eligible]
    model, grid = classifiers()["svm_rbf"]
    train = {"A": dev["A"], "B": dev["B"], "B2": b2["development"]}
    held_out = {"A": test["A"], "B": test["B"], "B2": b2["test"]}
    truth = test["labels"].astype(object)
    estimators, out_of_fold, result = {}, {}, {
        "note": "Post-hoc. B2 = Method B features inside the GrabCut mask (segmentation_grabcut.py).",
        "n_train": int(eligible.sum()),
        "b2_mask_failures": {"development": int(np.isnan(b2["development"]).any(axis=1).sum()),
                             "test": int(np.isnan(b2["test"]).any(axis=1).sum()),
                             "plantdoc": int(np.isnan(field).any(axis=1).sum())},
        "methods": {}}
    test_predictions = {}
    for method in ("A", "B", "B2"):
        search, record = fit_search(model, grid, train[method][eligible], labels, cv)
        estimators[method] = search.best_estimator_
        out_of_fold[method] = cross_val_predict(search.best_estimator_, train[method][eligible], labels,
                                                cv=cv).astype(object)
        test_predictions[method] = predict(search.best_estimator_, held_out[method])
        if method == "B2":
            save_b2_model(search.best_estimator_, frozen["segmentation"],
                          [row["image_id"] for row, keep in zip(dev["rows"], eligible) if keep], root)
        result["methods"][method] = {**record, "test_macro_f1_posthoc": macro_f1(truth, test_predictions[method])}
        print(f"{method:2s}: dev CV {record['cv_macro_f1']:.4f}  test {result['methods'][method]['test_macro_f1_posthoc']:.4f}",
              flush=True)
    labels = labels.astype(object)
    for first in ("A", "B"):
        result[f"test_B2_vs_{first}"] = paired(truth, test_predictions[first], test_predictions["B2"])
        result[f"development_out_of_fold_B2_vs_{first}"] = paired(labels, out_of_fold[first], out_of_fold["B2"])
    # PlantDoc: the frozen A and B predictions from extended_analysis, B2 from the model above.
    assert [r["path"] for r in field_rows] == [r["path"] for r in inventory]
    field_truth = np.array([r["health"] for r in field_rows], dtype=object)
    balanced = np.array([r["balanced_subset"] == "True" for r in field_rows])
    field_predictions = {"A": np.array([r["A_svm"] for r in field_rows], dtype=object),
                         "B": np.array([r["B_svm"] for r in field_rows], dtype=object),
                         "B2": predict(estimators["B2"], field)}
    t, p = field_truth[balanced], {m: v[balanced] for m, v in field_predictions.items()}
    result["plantdoc"] = {
        "n_balanced": int(balanced.sum()), "n_all": len(field_truth),
        "macro_f1_balanced": {m: macro_f1(t, v) for m, v in p.items()},
        "macro_f1_all": {m: macro_f1(field_truth, v) for m, v in field_predictions.items()},
        "B2_vs_A": paired(t, p["A"], p["B2"]), "B2_vs_B": paired(t, p["B"], p["B2"]),
        "B2_confusion_balanced": {h: dict(collections.Counter(p["B2"][t == h])) for h in ("Healthy", "Diseased")}}
    write_csv(OUTPUT / "plantdoc_b2_predictions.csv",
              [{"path": r["path"], "health": r["health"], "balanced_subset": r["balanced_subset"], "B2_svm": b}
               for r, b in zip(field_rows, field_predictions["B2"])])
    write_json(OUTPUT / "classification_results.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("masks", "classify", "all"))
    parser.add_argument("--plantdoc", required=True, help="Path to a PlantDoc-Dataset checkout at the pinned commit")
    args = parser.parse_args()
    if args.action in ("masks", "all"):
        run_masks(args.plantdoc)
    if args.action in ("classify", "all"):
        run_classify(args.plantdoc)
