"""Post-freeze segmentation refinement for field photos ("Fikir 2").

The frozen Lab segmentation (leaf_pipeline.segment_leaf) estimates the background colour
from the image border and keeps the largest component. On field photos the border is
other foliage, so the mask often covers the whole green scene. This refinement:

1. computes the same Lab-distance mask with the frozen parameters,
2. opens it with a larger ellipse so leaves touching through thin bridges separate,
3. keeps the component nearest the image centre (the photographed leaf is usually central),
4. seeds OpenCV GrabCut with that component (centre core = sure leaf, 4 px frame = sure
   background, rest = probable) and keeps the central component of the GrabCut result.

The two free parameters (opening size, central box) were chosen on the first 10 drawn
PlantDoc reference masks only; see grabcut_experiment.py. Known failure modes: several
separate leaves of equal importance, an off-centre subject, and a leaf that fills the frame.
"""

import cv2
import numpy as np

from leaf_pipeline import IMAGE_SIZE

GRABCUT = {"opening": 9, "central": 0.35, "iterations": 5, "frame": 4, "seed": 42}
_Y, _X = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
_CENTRE_DISTANCE = np.hypot(_Y - IMAGE_SIZE / 2, _X - IMAGE_SIZE / 2)


def lab_mask(rgb, segmentation):
    """Raw Lab-distance mask before component selection, identical to the frozen rule."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    width = int(segmentation["border"])
    border = np.concatenate((lab[:width].reshape(-1, 3), lab[-width:].reshape(-1, 3),
                             lab[width:-width, :width].reshape(-1, 3),
                             lab[width:-width, -width:].reshape(-1, 3)))
    mask = (np.linalg.norm(lab - np.median(border, axis=0), axis=2) > segmentation["distance"]).astype(np.uint8)
    for operation, name in ((cv2.MORPH_OPEN, "opening"), (cv2.MORPH_CLOSE, "closing")):
        size = int(segmentation[name])
        mask = cv2.morphologyEx(mask, operation, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return mask


def central_component(mask):
    """Filled component nearest the image centre, ignoring specks under 1% of the image."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, bool)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 0.01 * mask.size]
    if candidates:
        best = min(candidates, key=lambda i: _CENTRE_DISTANCE[labels == i].min())
    else:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def central_lab_mask(rgb, segmentation, opening=GRABCUT["opening"]):
    """'Fikir 1': stronger opening, then the central component."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening, opening))
    return central_component(cv2.morphologyEx(lab_mask(rgb, segmentation), cv2.MORPH_OPEN, kernel))


def grabcut_leaf(rgb, segmentation, parameters=None):
    """'Fikir 2': GrabCut seeded by the central Lab component."""
    p = dict(GRABCUT, **(parameters or {}))
    size = IMAGE_SIZE
    seed = central_lab_mask(rgb, segmentation, p["opening"])
    labels = np.full((size, size), cv2.GC_PR_BGD, np.uint8)
    box = (np.abs(_Y - size / 2) < p["central"] * size) & (np.abs(_X - size / 2) < p["central"] * size)
    labels[seed & box] = cv2.GC_PR_FGD
    if seed.mean() < 0.02 or seed.mean() > 0.85:  # unusable seed: fall back to a central prior
        labels[box] = cv2.GC_PR_FGD
    core = cv2.erode(seed.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    labels[core & (_CENTRE_DISTANCE < 0.25 * size)] = cv2.GC_FGD
    if not seed.any() or seed[size // 2, size // 2]:
        labels[_CENTRE_DISTANCE < 0.06 * size] = cv2.GC_FGD
    frame = p["frame"]
    labels[:frame] = labels[-frame:] = cv2.GC_BGD
    labels[:, :frame] = labels[:, -frame:] = cv2.GC_BGD
    cv2.setRNGSeed(p["seed"])
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                p["iterations"], cv2.GC_INIT_WITH_MASK)
    mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return central_component(mask)
