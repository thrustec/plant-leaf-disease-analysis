"""Post-freeze error analysis of Method B2 on the PlantDoc balanced subset.

Groups B2's field-photo errors by disease folder, plant, GrabCut mask size and original image
resolution, and draws a gallery of random errors (seed 42). Other methods are shown alongside for
comparison. Nothing is retrained or tuned; outputs go to results/extended/errors/.

Usage (run the PlantDoc, GrabCut, CNN, background and fine-tuning experiments first):
    python error_analysis.py --plantdoc PATH
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from extended_analysis import OUTPUT as EXTENDED, require_case_sensitive
from leaf_pipeline import ROOT, SEED, load_rgb, overlay_mask, write_csv, write_json
from segmentation_grabcut import grabcut_leaf

OUTPUT = EXTENDED / "errors"
MASK_BINS = ((0.0, 0.10, "<10%"), (0.10, 0.30, "10-30%"), (0.30, 0.60, "30-60%"), (0.60, 1.01, ">60%"))
SIZE_BINS = ((0, 300, "<300 px"), (300, 600, "300-600 px"), (600, 10 ** 6, ">=600 px"))


def read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_predictions():
    rows = read(EXTENDED / "plantdoc/predictions.csv")
    columns = {"A": [r["A_svm"] for r in rows], "B": [r["B_svm"] for r in rows]}
    for path, mapping in ((EXTENDED / "grabcut/plantdoc_b2_predictions.csv", {"B2": "B2_svm"}),
                          (EXTENDED / "cnn/plantdoc_cnn_predictions.csv", {"A-CNN": "A_CNN_svm"}),
                          (EXTENDED / "finetune/plantdoc_predictions.csv", {"FT-aug": "FT-aug"})):
        other = read(path)
        assert [r["path"] for r in other] == [r["path"] for r in rows]
        for method, column in mapping.items():
            columns[method] = [r[column] for r in other]
    keep = [i for i, r in enumerate(rows) if r["balanced_subset"] == "True"]
    return [rows[i] for i in keep], {m: np.array([v[i] for i in keep], dtype=object) for m, v in columns.items()}


def group_table(rows, predictions, key, truth):
    table = []
    for value in sorted({key(r) for r in rows}):
        index = np.array([key(r) == value for r in rows])
        entry = {"group": value, "n": int(index.sum()),
                 "truth": "/".join(sorted(set(truth[index])))}
        for method, values in predictions.items():
            entry[f"{method}_accuracy"] = round(float(np.mean(values[index] == truth[index])), 3)
        table.append(entry)
    return table


def run(plantdoc, root=ROOT):
    require_case_sensitive(plantdoc)
    plantdoc = Path(plantdoc)
    frozen = json.loads((Path(root) / "models/frozen_protocol.json").read_text())
    rows, predictions = load_predictions()
    truth = np.array([r["health"] for r in rows], dtype=object)
    images, masks, fractions, sizes = [], [], [], []
    for r in rows:
        source = cv2.imread(str(plantdoc / r["path"]))
        sizes.append(min(source.shape[:2]))
        rgb = load_rgb(plantdoc / r["path"])
        mask = grabcut_leaf(rgb, frozen["segmentation"])
        images.append(rgb)
        masks.append(mask)
        fractions.append(float(mask.mean()))
    fractions, sizes = np.array(fractions), np.array(sizes)
    b2 = predictions["B2"]
    wrong = b2 != truth
    result = {"n": len(rows), "B2_errors": int(wrong.sum()),
              "B2_error_types": {"missed_disease (Diseased called Healthy)": int(np.sum((truth == "Diseased") & (b2 == "Healthy"))),
                                 "false_alarm (Healthy called Diseased)": int(np.sum((truth == "Healthy") & (b2 == "Diseased"))),
                                 "mask_failed": int(np.sum(b2 == "FAILED"))},
              "all_five_methods_wrong": int(np.all([v != truth for v in predictions.values()], axis=0).sum())}
    tables = {
        "by_folder": group_table(rows, predictions, lambda r: r["folder"], truth),
        "by_plant_and_health": group_table(rows, predictions, lambda r: f"{r['plant']} {r['health']}", truth)}
    for name, values, bins in (("by_mask_size", fractions, MASK_BINS), ("by_image_size", sizes, SIZE_BINS)):
        table = []
        for low, high, label in bins:
            index = (values >= low) & (values < high)
            table.append({"group": label, "n": int(index.sum()),
                          **{f"{m}_accuracy": round(float(np.mean(v[index] == truth[index])), 3) if index.any() else None
                             for m, v in predictions.items()}})
        tables[name] = table
    for name, table in tables.items():
        write_csv(OUTPUT / f"{name}.csv", table)
    result["tables"] = tables
    gallery(images, masks, rows, truth, b2, fractions)
    write_json(OUTPUT / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def gallery(images, masks, rows, truth, b2, fractions, per_type=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    generator = np.random.default_rng(SEED)
    missed = np.flatnonzero((truth == "Diseased") & (b2 == "Healthy"))
    alarms = np.flatnonzero((truth == "Healthy") & (b2 == "Diseased"))
    picks = list(generator.choice(missed, per_type, replace=False)) + list(generator.choice(alarms, per_type, replace=False))
    figure, axes = plt.subplots(2, per_type, figsize=(3 * per_type, 7))
    for axis, i in zip(axes.ravel(), picks):
        axis.imshow(overlay_mask(images[i], masks[i]))
        kind = "missed disease" if truth[i] == "Diseased" else "false alarm"
        axis.set_title(f"{rows[i]['folder']}\n{kind} · mask {fractions[i]:.0%}", fontsize=8)
        axis.axis("off")
    figure.suptitle("Random B2 errors on PlantDoc (top row: diseased called healthy; bottom row: healthy called diseased)")
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "error_gallery.png", dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plantdoc", required=True, help="Path to a PlantDoc-Dataset checkout at the pinned commit")
    run(parser.parse_args().plantdoc)
