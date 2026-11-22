# Plant Leaf Disease Analysis Using Segmentation and Machine Learning


COM0418 Image Processing, 2026–2027 Fall Semester  
Emre Cantürk and Merve Gündoğdu

This project compares Healthy/Diseased classification using full-image features (A) and predicted leaf-mask features (B). Both methods use the same 58 HSV/LBP features, StandardScaler, RBF SVM, development images, and cross-validation folds. The research outcome is the held-out macro F1 difference, B minus A.

The notebook and report language is English. The course delivery target is 6 December 2026. The detailed experimental protocol is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

The instructor requires successful execution of the core algorithms and evaluation in a fresh Google Colab runtime. Share the Colab notebook with `ngkasapoglu@gmail.com` as Editor, share the required Google Drive project folder, verify loading through another account, and submit both links with the other required artifacts through CATS. The notebook now loads its resources automatically in Colab (see [Run in Colab](#run-in-colab)); a real hosted run from another account is still required. See [the current project overview](PROJECT_OVERVIEW.md) and [the two-person work plan](TEAM_WORK_PLAN.md). No interim report is required; confirm the official final deadline and format through CATS.

## Current stage

The implementation supports data acquisition, paired development experiments, independent reference-mask drafting, prediction, and locked final evaluation. Reference masks require both team members' review. Until that review is complete, any displayed CV scores are provisional development results. Final test metrics and the final report are deliberately pending.

## Run locally

Double-click **Start Jupyter Lab.command**, open `leaf_disease_analysis.ipynb`, and select the **Leaf Disease (.venv)** kernel. Run the cells in order. The notebook restores missing originals from the fixed manifest; an existing split is never resampled.

The local environments are `.venv` for the experiment and `.venv-labelme` for the annotation GUI. The annotation environment is separate so hosted notebook execution does not need Qt or annotation-assistance packages.

This checkout contains a frozen protocol. Before running the notebook or commands below, download [shared_artifacts.zip](https://drive.google.com/file/d/1QHfBnQg6yLIElWTAyRJzjDad9ofgyBgi/view) and extract its `models/` and `results/` folders into the project root.

For a new installation with Python 3.11:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name leaf-disease --display-name 'Leaf Disease (.venv)'
python3 -m venv .venv-labelme
.venv-labelme/bin/python -m pip install -r requirements-annotation.txt
.venv/bin/python prepare_data.py
.venv/bin/python mask_review.py prepare
.venv/bin/python experiment.py evaluate
```

`requirements.txt` contains tested direct package pins after setup. `requirements-local-lock.txt` and `requirements-annotation-lock.txt` record the complete local Python 3.11 environments; those full locks may contain platform-specific packages. Use the direct core requirements in Colab.

## Run in Colab

Open `leaf_disease_analysis.ipynb` in Colab and choose **Runtime > Run all**. The setup cell needs no upload or manual step:

1. It clones this repository at `REPO_REF` into `/content/plant-leaf-disease-analysis`. The repository must be public (or otherwise readable without credentials) for the instructor's runtime.
2. It installs `requirements-colab.txt`: the same processing and model pins as `requirements.txt`, without replacing Colab's own Jupyter/kernel packages. If Colab had already loaded another NumPy, it restarts the runtime once; choose **Run all** again.
3. If `SHARED_ARTIFACTS_ID` is set, it downloads `shared_artifacts.zip` (saved models, frozen protocol, development and final results) from the shared Drive folder with `gdown`. Share that file as "anyone with the link can view".
4. Section 2 restores the original images from their pinned URLs with Git-blob and SHA-256 verification.

Prepare the shared archive locally after training, freezing or evaluating:

```bash
.venv/bin/python package_project.py artifacts   # writes dist/shared_artifacts.zip
```

Upload it to the shared project folder, copy its file ID (the part after `/d/` in the sharing link) into `SHARED_ARTIFACTS_ID`, and commit. Without it, the notebook still runs end to end and refits the development models.

Labelme review stays local. Commit the reviewed JSON files and `reviewed_index.json`; the notebook re-exports reference masks from them. `package_project.py` without arguments still builds the older all-in-one companion ZIP. Local execution and actual hosted Colab execution are recorded separately in `PROJECT_STATUS.md`.

## Review the 30 reference masks

Open **results/mask_review/index.html** for the complete numbered gallery. Each row contains the original image, an outline overlay, and a silhouette. Initial drafts come from GrabCut, independently of the Lab-distance segmentation under evaluation. Nine development drafts were corrected after the team identified cast-shadow inclusion. Those revisions use traced priors, explicit background seeds, and visible-tissue restorations; they still require both reviewers. The correction record is `data/annotations/shadow_correction_log.json`.

1. Double-click **Review Development Masks.command** for the 12 development outlines.
2. Check the actual leaf boundary, tips, and any visible stalk. Preserve the entire visible leaf, including yellow/brown diseased tissue. Include a visible connected petiole; exclude the background, detached objects, and cast shadows. Correct the polygon before marking review.
3. Each person checks their own **Emre_reviewed** or **Merve_reviewed** image flag and saves the JSON file. A flag means that person actually inspected the outline. Leave uncertain outlines unapproved and record the image ID for discussion.
4. Double-click **Review Test Masks.command** and review the 18 test outlines against the originals. Do not use Method B predictions or final metrics to decide the reference boundary.
5. If an already approved outline needs editing, clear both flags first. Both people review the edited outline again.
6. Export the references after saving:

```bash
.venv/bin/python mask_review.py export
.venv/bin/python mask_review.py status
```

The export accepts only polygons with both review flags. It checks their image association and writes raster reference masks with hashes. Edited annotations invalidate the exported reference index. Automated drafts are never counted as reviewed references.

Annotation images use 256 by 256 coordinates and sit next to their Labelme JSON files. Keep the paired filenames and `leaf` polygon label. `labelme_config.yaml` defines the permitted polygon label and the two review flags.

## Development and final evaluation

During development:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python prepare_data.py --verify
.venv/bin/python experiment.py development
```

Both methods use the same fixed three folds, `C={0.1,1,10}`, `gamma='scale'`, and balanced class weights. Segmentation selects distance `{12,18,24}` and closing size `{3,5}` by Dice on the 12 reviewed development outlines. Until those references exist, the experiment uses a declared default without pretending that mask tuning is complete.

After all 30 outlines are reviewed and exported, rerun development, then freeze and evaluate:

```bash
.venv/bin/python experiment.py development
.venv/bin/python experiment.py freeze
.venv/bin/python experiment.py evaluate
```

The final operation checks the reviewed references, model hashes, implementation, and fixed manifest. It stores final results once and returns the stored result on unchanged reruns. It does not tune from test results. A frozen protocol prevents accidental model replacement during subsequent notebook runs.

Failed Method B masks remain unsuccessful end-to-end cases. Accuracy includes every test image. For the two-class macro F1, a failure contributes a false negative for its true class. The covered 2 by 2 confusion matrix is accompanied by failure counts by true class. Timing includes input decoding, resizing, feature extraction, and prediction; Method B also includes segmentation. Model loading is outside the timing boundary. The same CPU/thread setting and three repetitions are used for each method.

## Predict one image

```bash
.venv/bin/python predict.py '/absolute/path/to/leaf.jpg' --method B
```

The default model is `models/development_B.joblib`. Use `--method A` for full-image features or `--model` for an explicit saved artifact. Outputs go to `results/prediction/`: original/mask/overlay figure and separate mask/overlay PNGs. The script returns a nonzero exit status and a clear error for invalid inputs or unusable Method B masks. Method A's displayed feature region is the whole image, not a predicted leaf mask.

## Interactive exploration

Notebook section 5b uses `explore.py` to show every intermediate step of Method B's segmentation (border, Lab distance map, threshold, opening, closing, filled region), the A and B feature histograms, and both saved models' predictions for any image. In Colab, set `UPLOAD_NEW_IMAGE = True` to test your own photo. Edit `SEGMENTATION_PARAMETERS` to rerun the paired development comparison with other settings; it uses development images only and writes to `results/experiments/<name>/`.

After the final evaluation exists, the notebook also recomputes it from the frozen models in a temporary copy and compares the headline metrics with the stored benchmark. Nothing in `explore.py` writes to `models/`, `results/development/` or `results/final/`, and it does not change `leaf_pipeline.py` or `experiment.py`, whose hashes the frozen protocol records.

## Data and experimental records

- `data/manifests/images.csv`: fixed source, plant, original/binary labels, hashes, leaf and split groups, partitions, folds, and annotation selection.
- `data/manifests/source_inventory.csv`: available verified source groups by original label.
- `data/manifests/exclusions.csv` and `label_mapping.json`: unresolved mappings and explicitly excluded categories.
- `data/manifests/duplicate_pairs.csv`: exact/near-duplicate candidates conservatively grouped before splitting.
- `data/annotations/labelme/`: editable reference polygons and paired images.
- `data/annotations/reviewed_index.json`: only currently reviewed and exported references.
- `results/development/`: CV tables, common training exclusions, environment, and development figures.
- `results/final/`: locked classification, mask, timing, and coverage evidence after review.
- `models/`: fitted scaler/SVM pipelines and, after freeze, the protocol record.

The subset uses one selected image per verified supplied leaf group. Duplicate screening uses exact decoded-pixel hashes or pHash distance at most 6 plus 64 by 64 RGB mean absolute error at most 8. This heuristic supports the supplied grouping records; it cannot establish that all visually related leaves have been identified. Approximate balance is preserved within the group constraints.

## Sources and interpretation

Primary data: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), revision `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`. Cite Mohanty, Hughes, and Salathé (2016), [Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419). The [author dataset card](https://huggingface.co/datasets/mohanty/PlantVillage) reports CC BY-SA 3.0; preserve attribution and check the source terms before distributing image derivatives. The companion archive retrieves originals from the source rather than bundling them.

Tools: [OpenCV](https://docs.opencv.org/4.x/), [scikit-image LBP](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html), [scikit-learn](https://scikit-learn.org/stable/), and [Labelme](https://github.com/wkentaro/labelme).

Controlled-dataset classification does not establish field-ready diagnosis. The 18 segmentation test references form a small diagnostic sample. Report improvements and regressions with the same evidence, including false Healthy predictions and processing failures. The final English report and presentation follow the locked experiment and will not invent missing reference scores.
