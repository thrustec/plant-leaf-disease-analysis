"""Prepare independent annotation drafts and export only explicitly reviewed references."""

import argparse
import html
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from leaf_pipeline import ROOT, SEED, file_hash, load_rgb, overlay_mask, read_manifest, write_csv, write_json

REVIEWERS = ("Emre_reviewed", "Merve_reviewed")


def annotation_path(row, root=ROOT):
    return Path(root) / "data/annotations/labelme" / row["annotation_partition"] / (row["image_id"] + ".json")


def polygon_mask(annotation):
    if (annotation.get("imageHeight"), annotation.get("imageWidth")) != (256, 256):
        raise ValueError("Reference annotations must use the supplied 256x256 images")
    mask = np.zeros((256, 256), np.uint8)
    shapes = annotation.get("shapes", [])
    if not shapes:
        raise ValueError("Annotation contains no leaf outline")
    for shape in shapes:
        points = np.asarray(shape.get("points", []), dtype=float)
        if (shape.get("label") != "leaf" or shape.get("shape_type") != "polygon"
                or points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or (points < 0).any() or (points > 255).any()):
            raise ValueError("Use leaf polygons with at least three points inside the image")
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    if not mask.any():
        raise ValueError("Reference mask is empty")
    return mask.astype(bool)


def draft_annotation(row, root):
    rgb = load_rgb(root / row["path"])
    output = annotation_path(row, root)
    image = output.with_suffix(".png")
    image.parent.mkdir(parents=True, exist_ok=True)
    if not image.exists():
        cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if output.exists():
        return
    cv2.setRNGSeed(SEED)
    # shortcut: initial GrabCut drafts can include shadows; correct them before reference export.
    labels = np.zeros((256, 256), np.uint8)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels, (1, 1, 254, 254),
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                5, cv2.GC_INIT_WITH_RECT)
    mask = np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    if contours:
        outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.7, True)
        if len(outline) >= 3:
            shapes.append({"label": "leaf", "points": outline[:, 0, :].astype(float).tolist(),
                           "group_id": None, "description": "Unreviewed GrabCut draft; correct before approval.",
                           "shape_type": "polygon", "flags": {}})
    write_json(output, {"version": "6.3.0", "flags": {key: False for key in REVIEWERS},
                        "shapes": shapes, "imagePath": image.name,
                        "imageData": None, "imageHeight": 256, "imageWidth": 256})


def review_status(root=ROOT):
    status = []
    for row in read_manifest(root):
        if not row["annotation_partition"]:
            continue
        path = annotation_path(row, root)
        annotation = json.loads(path.read_text()) if path.exists() else {}
        flags = annotation.get("flags", {})
        status.append(dict(image_id=row["image_id"], partition=row["annotation_partition"],
                           plant=row["plant"], health=row["health"],
                           Emre_reviewed=flags.get(REVIEWERS[0]) is True,
                           Merve_reviewed=flags.get(REVIEWERS[1]) is True,
                           annotation_sha256=file_hash(path) if path.exists() else ""))
    return status


def export_reviewed(root=ROOT):
    root = Path(root)
    records = []
    images = {row["image_id"]: row for row in read_manifest(root)}
    for row in review_status(root):
        if not (row[REVIEWERS[0]] and row[REVIEWERS[1]]):
            continue
        path = root / "data/annotations/labelme" / row["partition"] / (row["image_id"] + ".json")
        annotation = json.loads(path.read_text())
        if annotation.get("imagePath") != row["image_id"] + ".png":
            raise ValueError(f"Annotation image association changed: {row['image_id']}")
        image = path.with_suffix(".png")
        if not np.array_equal(load_rgb(image), load_rgb(root / images[row["image_id"]]["path"])):
            raise ValueError(f"Annotation image differs from the selected source: {row['image_id']}")
        mask = polygon_mask(annotation)
        target = root / "data/annotations/reference_masks" / (row["image_id"] + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), mask.astype(np.uint8) * 255)
        records.append(dict(row, mask_path=str(target.relative_to(root)), mask_sha256=file_hash(target),
                            annotation_image_sha256=file_hash(image)))
    # Rebuild the index: revoked approvals and edited JSON must never leave active references.
    write_json(root / "data/annotations/reviewed_index.json", records)
    return records


def reviewed_references(root=ROOT, partition=None):
    root = Path(root)
    path = root / "data/annotations/reviewed_index.json"
    if not path.exists():
        return []
    current = {row["image_id"]: row for row in review_status(root)}
    valid = []
    for row in json.loads(path.read_text()):
        now = current.get(row["image_id"], {})
        if (not now.get(REVIEWERS[0]) or not now.get(REVIEWERS[1])
                or now.get("annotation_sha256") != row["annotation_sha256"]
                or file_hash(root / "data/annotations/labelme" / row["partition"] /
                             (row["image_id"] + ".png")) != row["annotation_image_sha256"]
                or file_hash(root / row["mask_path"]) != row["mask_sha256"]):
            raise ValueError("Reference files changed; complete review and export again")
        if partition is None or row["partition"] == partition:
            valid.append(row)
    return valid


def render_review(root=ROOT):
    root = Path(root)
    output = root / "results/mask_review"
    output.mkdir(parents=True, exist_ok=True)
    status = {row["image_id"]: row for row in review_status(root)}
    pages = []
    for partition in ("development", "test"):
        rows = [row for row in read_manifest(root) if row["annotation_partition"] == partition]
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            figure, axes = plt.subplots(len(batch), 3, figsize=(9, 3 * len(batch)), squeeze=False)
            for number, row in enumerate(batch):
                rgb = load_rgb(root / row["path"])
                annotation = json.loads(annotation_path(row, root).read_text())
                try:
                    mask = polygon_mask(annotation)
                except ValueError:
                    mask = np.zeros((256, 256), bool)
                for axis, image, name in zip(axes[number], [rgb, overlay_mask(rgb, mask), mask],
                                             ["Original", "Outline overlay", "Draft silhouette"]):
                    axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1 if image.ndim == 2 else 255)
                    axis.axis("off")
                    axis.set_title(name, fontsize=9)
                state = status[row["image_id"]]
                approved = state[REVIEWERS[0]] and state[REVIEWERS[1]]
                axes[number, 0].set_title(
                    f"{row['image_id']} | {row['plant']} | {row['health']}\n"
                    + ("REVIEWED" if approved else "UNREVIEWED"), fontsize=9)
            figure.suptitle(f"{partition.title()} reference outlines — human review required", fontsize=12)
            figure.tight_layout()
            name = f"{partition}_{start // 4 + 1:02d}.png"
            figure.savefig(output / name, dpi=140)
            plt.close(figure)
            pages.append((partition, name))
    write_csv(root / "data/annotations/review_status.csv", list(status.values()))
    links = "".join(f'<h2>{html.escape(partition.title())} — page {name[-6:-4]}</h2>'
                    f'<a href="{name}"><img src="{name}" alt="{partition} mask review"></a>'
                    for partition, name in pages)
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Leaf mask review</title>'
        '<style>body{font:17px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;'
        'background:#f4f5f1;color:#18271f}img{width:100%;background:white}h1{font-size:30px}'
        'code{background:#e2e7df;padding:3px}a{color:#18573a}</style>'
        '<h1>Leaf reference mask review</h1>'
        '<p>These are assisted outline drafts, not validated ground truth. Initial GrabCut '
        'suggestions may be corrected with traced polygons and explicit shadow exclusion. '
        'Check leaf edges, tips, stalks, and all yellow/brown symptomatic tissue. '
        'Correct the polygon in Labelme before either reviewer approves it.</p>'
        '<p>Open <code>Review Development Masks.command</code> first. After development review, '
        'use <code>Review Test Masks.command</code> for the held-out annotation subset. '
        'Each person checks their own <code>Emre_reviewed</code> or <code>Merve_reviewed</code> '
        'image flag and saves the file. Do not judge outlines against Method B predictions.</p>'
        '<p>Development outlines may guide segmentation settings. Test outlines remain excluded '
        'from tuning. Run <code>.venv/bin/python mask_review.py export</code> after saving reviews.</p>'
        + links + '</html>', encoding="utf-8")
    return output / "index.html"


def prepare_reviews(root=ROOT):
    root = Path(root)
    for row in read_manifest(root):
        if row["annotation_partition"]:
            draft_annotation(row, root)
    return render_review(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "render", "export", "status"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare_reviews())
    elif args.action == "render":
        print(render_review())
    elif args.action == "export":
        print(f"Exported {len(export_reviewed())}/30 reviewed reference masks")
        render_review()
    else:
        status = review_status()
        print(json.dumps({"total": len(status), "both_reviewed": sum(
            row[REVIEWERS[0]] and row[REVIEWERS[1]] for row in status)}, indent=2))
