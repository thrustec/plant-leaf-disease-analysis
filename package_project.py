"""Build the Colab companion archive, excluding raw data, environments, and server secrets."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent


def package():
    patterns = ["*.py", "*.md", "requirements*.txt", "*.command", "labelme_config.yaml",
                "leaf_disease_analysis.ipynb", "tests/*.py", "data/manifests/*",
                "data/annotations/labelme/*/*", "data/annotations/reference_masks/*.png",
                "data/annotations/reviewed_index.json", "data/annotations/review_status.csv",
                "data/annotations/shadow_correction_log.json", "models/*.joblib",
                "models/frozen_protocol.json", "results/development/*", "results/final/*",
                "results/mask_review/development_*.png", "results/mask_review/test_*.png",
                "results/mask_review/index.html"]
    files = sorted({path for pattern in patterns for path in ROOT.glob(pattern) if path.is_file()})
    output = ROOT / "dist/leaf_project_colab.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError("Refusing to archive a link outside the project")
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(output) as archive:
        names = set(archive.namelist())
        assert archive.testzip() is None, "Archive integrity check failed"
        assert {"leaf_pipeline.py", "leaf_disease_analysis.ipynb", "data/manifests/images.csv", "requirements.txt"} <= names
        assert not any(name.startswith((".venv", ".cache/", ".jupyter/", ".git/", "data/raw/")) for name in names)
    return output


ARTIFACT_PATTERNS = ["models/*.joblib", "models/frozen_protocol.json",
                     "results/development/*", "results/final/*", "results/extended/**/*",
                     "data/cache/ft_aug_final.keras"]


def package_artifacts():
    """Build the models/results archive that the Colab notebook downloads from the shared Drive folder.

    Code, manifests and annotations come from the Git repository; raw images come from the
    pinned PlantVillage source. Only files Git does not track are packaged here.
    """
    files = sorted({path for pattern in ARTIFACT_PATTERNS for path in ROOT.glob(pattern) if path.is_file()})
    if not any(path.suffix == ".joblib" for path in files):
        raise ValueError("No saved models to package; run the development experiment first")
    output = ROOT / "dist/shared_artifacts.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError("Refusing to archive a link outside the project")
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(output) as archive:
        assert archive.testzip() is None, "Archive integrity check failed"
        assert all(name.startswith(("models/", "results/")) or name == "data/cache/ft_aug_final.keras"
                   for name in archive.namelist())
    return output


if __name__ == "__main__":
    import sys
    print(package_artifacts() if sys.argv[1:] == ["artifacts"] else package())
