# Project Status

Historical snapshot from 28 October 2026. For the 9 November evidence inspection and mandatory Colab delivery requirements, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). The revised two-person schedule is [TEAM_WORK_PLAN.md](TEAM_WORK_PLAN.md). Application-running statements below refer to the original setup session, not current server state.

Updated 28 October 2026. The local prototype is running. Reference annotation review is the current dependency for segmentation tuning and final evaluation. The final report and presentation follow the locked experiment.

## Installed and running

- Local core environment: `.venv`, Python 3.11.0. Direct versions and full local package locks are saved.
- Local annotation environment: `.venv-labelme`, Labelme 6.3.1. Dependency checks passed and the Qt application opened the supplied annotations successfully.
- JupyterLab is running on `http://127.0.0.1:8888`; the project kernel is `Leaf Disease (.venv)`.
- Notebook: `leaf_disease_analysis.ipynb`. A clean project kernel executed all seven code cells successfully after the implementation and annotation provenance updates.
- The notebook shares processing and prediction code with `predict.py`.

## Data and experiment readiness

- 900 images from the pinned PlantVillage source revision, with 900 distinct supplied leaf groups.
- Apple, Bell pepper, and Tomato; 150 Healthy and 150 Diseased images per plant.
- 720 development and 180 test images; each plant/health stratum has 120 development and 30 test samples.
- All selected originals passed source-blob, SHA-256, and decode checks. Original files are read-only.
- No exact decoded-image or configured near-duplicate pair was found in the sample. The grouping and hash checks found no partition or CV-fold overlap.
- Grape was replaced because healthy grouping metadata was absent. Apple black rot and Tomato mosaic were excluded because grouping correspondence was unresolved. Target Spot had zero resolved source images and is absent from the sampled subset; this is recorded in the inventory/exclusions.
- Both saved development SVM/scaler pipelines exist. All 720 development images were eligible for the initial paired experiment.

The initial, unreviewed-mask development CV macro F1 was 0.9444 for A and 0.8971 for B. These are provisional development results. They are not final test results and do not establish the outcome of the hypothesis. Segmentation tuning awaits all 12 reviewed development outlines.

## Reference review

There are 30 editable Labelme drafts: 12 development and 18 test. The gallery is `results/mask_review/index.html`. A saved review requires each person to inspect the outline and mark their own image flag.

The team reported cast-shadow inclusion in development outlines `leaf_0121`, `leaf_0156`, `leaf_0381`, `leaf_0395`, `leaf_0519`, `leaf_0689`, `leaf_0743`, `leaf_0849`, and `leaf_0887`. These nine were corrected using assistant-traced priors, explicit shadow background seeds, GrabCut edge refinement, and restoration of visible tissue/petioles. Original JSON files were backed up. Both review flags were reset for the corrected revisions; other annotations were preserved.

These corrections remain assisted drafts until the team completes review. The 18 test drafts also need review. No final IoU/Dice or held-out classification result has been computed. Reload an image in an already open Labelme window to see an externally updated JSON file.

## Verification evidence

- `results/logs/tests.log`: 13 focused tests passed, including actual dataset group/hash separation, failure accounting, feature invariants, model reload consistency, source/path validation, reference review requirements, and the final-evaluation gate.
- `results/logs/notebook-execution.log`: clean-kernel notebook execution passed, seven code cells.
- `results/logs/prediction_smoke.json`: 12 real development predictions across both labels and all three plants retained the same labels after model reload.
- The CLI passed a valid-image test and returned clear failures for an unreadable file and an unusable blank-image mask.
- Both environments passed `pip check`. Jupyter's authenticated local status endpoint returned HTTP 200.
- All 30 annotation JSON files passed polygon and coordinate-dimension validation. Geometric validity does not establish outline accuracy.

The Colab route is implemented and a companion archive can be built with `package_project.py`. Actual execution in Google's hosted Colab runtime has not been verified. Local JupyterLab is the tested execution route.

## Next actions

1. Review the nine corrected development masks and the remaining references. Save actual reviewer flags; export the reviewed index.
2. Rerun development using all 12 reviewed development outlines to choose segmentation settings.
3. Freeze the protocol, then evaluate the 180 locked test images and the 18 reviewed test masks.
4. Prepare the English final report using Humanizer and saved experiment evidence; prepare the presentation and final delivery check.
