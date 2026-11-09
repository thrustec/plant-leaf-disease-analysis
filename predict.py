"""Predict one leaf image using a saved A or B model."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaf_pipeline import ROOT, overlay_mask, predict_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--method", choices=("A", "B"), default="B")
    parser.add_argument("--model", type=Path, help="Override the saved model path")
    parser.add_argument("--output", type=Path, default=ROOT / "results/prediction")
    args = parser.parse_args()
    model = args.model or ROOT / "models" / f"development_{args.method}.joblib"
    try:
        cv2.setNumThreads(1)
        prediction, rgb, mask = predict_image(args.image, model)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "mask.png"), mask.astype("uint8") * 255)
    cv2.imwrite(str(args.output / "overlay.png"), cv2.cvtColor(overlay_mask(rgb, mask), cv2.COLOR_RGB2BGR))
    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    for axis, image, title in zip(axes, [rgb, mask, overlay_mask(rgb, mask)],
                                 ["Original", "Feature region", prediction]):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "prediction.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"status": "ok", "prediction": prediction,
                      "model": str(model), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
