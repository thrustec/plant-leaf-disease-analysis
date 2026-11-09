"""Shared image processing and binary prediction for the notebook and CLI."""

import csv
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
LABELS = ("Healthy", "Diseased")
SEED = 42
IMAGE_SIZE = 256
DEFAULT_SEGMENTATION = {"distance": 18.0, "border": 5, "opening": 3, "closing": 5}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(root=ROOT):
    with (Path(root) / "data/manifests/images.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rgb(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 8:
        raise ValueError(f"Unreadable or too small image: {path.name}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def segment_leaf(rgb, parameters=None):
    parameters = DEFAULT_SEGMENTATION if parameters is None else parameters
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    width = int(parameters["border"])
    border = np.concatenate((lab[:width].reshape(-1, 3), lab[-width:].reshape(-1, 3),
                             lab[width:-width, :width].reshape(-1, 3),
                             lab[width:-width, -width:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    mask = (distance > parameters["distance"]).astype(np.uint8)
    for operation, name in ((cv2.MORPH_OPEN, "opening"), (cv2.MORPH_CLOSE, "closing")):
        size = int(parameters[name])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, operation, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(mask.shape, np.uint8)
    if contours:
        # Filling the external contour preserves discolored tissue inside the leaf.
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return result.astype(bool)


def texture_region(mask):
    # Radius-one LBP uses bilinear neighbors; the complete 3x3 footprint must be valid.
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def validate_mask(mask):
    fraction = float(mask.mean())
    if not 0.01 <= fraction <= 0.97 or not texture_region(mask).any():
        raise ValueError(f"Unusable leaf mask (foreground fraction {fraction:.4f})")


def extract_features(rgb, mask=None):
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError("Features require a 256x256 uint8 RGB image")
    if mask is None:
        mask = np.ones(rgb.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != rgb.shape[:2]:
            raise ValueError("Mask dimensions do not match the image")
        validate_mask(mask)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histograms = []
    for channel, upper in enumerate((180, 256, 256)):
        histogram = np.histogram(hsv[:, :, channel][mask], bins=16, range=(0, upper))[0]
        histograms.append(histogram / histogram.sum())
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(grayscale, P=8, R=1, method="uniform")
    texture = np.histogram(lbp[texture_region(mask)], bins=10, range=(0, 10))[0]
    histograms.append(texture / texture.sum())
    features = np.concatenate(histograms).astype(np.float64)
    if features.shape != (58,) or not np.isfinite(features).all():
        raise ValueError("Invalid feature vector")
    return features


def mask_scores(predicted, reference):
    predicted, reference = np.asarray(predicted, bool), np.asarray(reference, bool)
    if predicted.shape != reference.shape:
        raise ValueError("Mask dimensions differ")
    intersection = int(np.logical_and(predicted, reference).sum())
    union = int(np.logical_or(predicted, reference).sum())
    total = int(predicted.sum() + reference.sum())
    if not predicted.any() or not reference.any():
        return {"iou": 0.0, "dice": 0.0}
    return {"iou": intersection / union, "dice": 2 * intersection / total}


def classification_metrics(truth, predictions):
    truth, predictions = list(truth), list(predictions)
    if not truth or len(truth) != len(predictions) or not set(truth) <= set(LABELS):
        raise ValueError("Invalid evaluation labels")
    if not set(predictions) <= set(LABELS) | {"FAILED"}:
        raise ValueError("Unknown prediction label")
    # FAILED is an abstention status, contributing a false negative to its true class.
    report = classification_report(truth, predictions, labels=list(LABELS),
                                   output_dict=True, zero_division=0)
    covered = np.array([label in LABELS for label in predictions])
    report["macro_f1"] = report["macro avg"]["f1-score"]
    report["accuracy"] = float(np.mean(np.array(truth) == np.array(predictions)))
    report["coverage"] = float(covered.mean())
    report["n_images"] = len(truth)
    report["failures_by_true_class"] = {
        label: sum(t == label and p == "FAILED" for t, p in zip(truth, predictions))
        for label in LABELS}
    report["covered_confusion_matrix"] = (confusion_matrix(
        np.array(truth)[covered], np.array(predictions)[covered], labels=list(LABELS)).tolist()
        if covered.any() else [[0, 0], [0, 0]])
    return report


def overlay_mask(rgb, mask):
    overlay = rgb.copy()
    overlay[mask] = (0.65 * rgb[mask] + 0.35 * np.array([0, 230, 100])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 40, 40), 1)
    return overlay


def predict_image(path, artifact):
    if isinstance(artifact, (str, Path)):
        artifact = joblib.load(artifact)
    rgb = load_rgb(path)
    mask = (segment_leaf(rgb, artifact["segmentation"]) if artifact["method"] == "B"
            else np.ones(rgb.shape[:2], dtype=bool))
    features = extract_features(rgb, mask if artifact["method"] == "B" else None)
    prediction = str(artifact["pipeline"].predict(features.reshape(1, -1))[0])
    if prediction not in LABELS:
        raise ValueError("Model returned an unsupported label")
    return prediction, rgb, mask
