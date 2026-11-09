# Plant Leaf Disease Analysis Project Plan

This project will test whether leaf segmentation improves binary Healthy/Diseased classification compared with features measured over the full image. The deliverables are a reproducible Jupyter notebook, shared Python implementation, command-line prediction tool, experiment evidence, final report, and presentation. The implementation and evaluation must work even if segmentation produces no improvement.

Delivery update, 9 November 2026: the supplied instructor announcement requires executable Google Colab notebooks, fresh-runtime Run all verification, a shared Google Drive project folder, instructor Editor access to the notebook, and links submitted through CATS. There is no interim report. Use `PROJECT_OVERVIEW.md` for the current evidence and `TEAM_WORK_PLAN.md` for the revised two-person schedule. These delivery requirements and revised targets take precedence over optional Colab wording in the earlier plan.

The plan follows the supplied Project Proposal and the decisions confirmed on 28 October 2026. Notebook explanations and the final report will be in English. The final target is 6 December 2026. The dates below are internal milestones rather than independently verified instructor deadlines.

## Working sequence

1. Complete planning and settle the experimental protocol.
2. Set up the project inside this workspace and prepare the data.
3. Build segmentation, features, classification, and prediction using Ponytail.
4. Test the implementation, freeze settings, and run the final experiment.
5. Write the final report using Humanizer and verified experiment outputs.
6. Prepare the presentation and verify the complete delivery package.

Chat responses use Caveman. Persisted code, notebook explanations, and report prose use clear, complete English. Implementation is now under way; current installation and verification evidence is recorded in `PROJECT_STATUS.md`.

## Confirmed scope

- Input: one leaf image. Core outputs: leaf mask, visual overlay, and Healthy or Diseased prediction, or an explicit processing error.
- Method A: full-image features followed by StandardScaler and an RBF SVM.
- Method B: the same features and classifier measured inside the predicted leaf region.
- Each feature vector contains 58 values: three normalized 16-bin HSV histograms and one 10-bin uniform LBP histogram with 8 neighbors and radius 1.
- Images are processed at 256 by 256 pixels. RGB, HSV, Lab, and grayscale conversions follow the same conventions during training and inference.
- The measured effect is Method B minus Method A macro F1, accompanied by segmentation quality, runtime, coverage, and error analysis.
- Species and original disease names are metadata for sampling and analysis. The prediction target has two health classes.

The initial implementation uses existing CPU hardware. Neural networks, species prediction, disease subtype prediction, severity estimation, a web/mobile interface, and hardware deployment are deferred. PlantDoc or PlantWild would be a separate optional external evaluation. The TÜBİTAK application and its budget are separate work and do not block the course project.

## Environment and installation plan

The workspace was empty at inspection. The machine has an Apple M2 CPU, 8 CPU cores, 16 GiB RAM, and macOS 26.5.2. Python 3.11 and an Anaconda Jupyter executable are already available. These observations support using a CPU implementation; runtime and memory performance will still be measured.

- Create a project-local `.venv` and install a compatible, pinned package set. Use the existing Python 3.11 interpreter if the dependency compatibility check passes.
- Core packages: NumPy, OpenCV, scikit-image, scikit-learn, Matplotlib, joblib, JupyterLab, ipykernel, and notebook execution support. Use the standard library for downloading, CSV/JSON records, hashing, and paths where sufficient.
- Keep Labelme GUI dependencies out of the Colab requirements. Install the open-source Python tool in a separate local annotation environment. Current upstream documentation assigns Python 3.11 to the Labelme 6.3 maintenance series; choose and test an exact compatible release during setup.
- Record the Python version, package versions, random seed, hardware, and source revisions. Pin the working versions after installation and smoke tests.
- Keep downloaded data, caches, temporary files, models, and environment folders inside the project workspace. Create an appropriate `.gitignore` and use Git for source and small reproducibility records.
- Keep one shared implementation for the notebook and prediction script. Do not add a service, training framework, or packaging layer without a concrete requirement.

Local JupyterLab is the reference development route. The same `.ipynb` must execute core processing and evaluation in Google Colab with relative working paths, dependency installation, and resource loading that works through the instructor's own account. The existing archive upload/extraction cell is a supplementary route; complete shared-resource loading before submission. Colab loads supplied reference annotations without launching a desktop annotation GUI. A successful fresh-runtime Colab Run all and verified instructor access are required completion evidence. Local execution alone does not establish that result.

## Data preparation protocol

1. Use the original color images from the primary PlantVillage repository. Inspect source metadata before downloading image files. Prefer downloading only the selected images and required metadata.
2. Start with Tomato, Grape, and Bell pepper as candidate plants. Final selection depends on quality and enough verified independent leaf groups in both health states. Record any replacement and its reason before fixing the split.
3. Target about 150 Healthy and 150 Diseased images per plant, about 900 images in total. Distribute diseased samples across eligible original disease categories where available and document the resulting composition. Counts may change to preserve group separation and quality.
4. Define an explicit allowlist from original labels to Healthy/Diseased. Exclude ambiguous labels, pest-only damage, and nutrient/stress categories. In particular, do not silently map Tomato spider-mite images into Diseased.
5. Validate image decoding, inspect quality, identify exact and near duplicates, and resolve leaf/photo groups using upstream grouping metadata. Use hashes and inspection to supplement supplied IDs. Record missing or ambiguous group mappings; exclude unresolved cases rather than treating unverified filenames as independent leaves.
6. Use a fixed seed and group-aware approximately 80/20 development/test split, balancing plant and health state as closely as grouping permits. Aim for about 720 development and 180 test images. Save the actual counts and the fixed manifest.
7. Use the same samples and partition for A and B. Check that groups and duplicate content never cross partitions or cross-validation folds.
8. Preserve the selected original images without modification. Record source URL and revision, relative path, plant, original label, binary label, image hash, leaf/photo group, split, and exclusion reasons.
9. Retain attribution and document applicable reuse terms before distributing images. Upstream derivative color/grayscale/segmented representations must not be counted as independent observations.

The upstream repository contains leaf grouping metadata. Revision observed during planning: `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`. Pin the acquisition revision and record any change from this observation. Verify metadata against the actual selected files; a dataset card's claim of group-safe splits is not a substitute for our checks.

## Reference masks and segmentation

Select annotation images from the fixed manifest before tuning:

| Partition | Per plant | Health balance per plant | Total |
| --- | --- | --- | --- |
| Development | 4 | 2 Healthy and 2 Diseased | 12 |
| Test | 6 | 3 Healthy and 3 Diseased | 18 |

Create polygon outlines in Labelme and save both source JSON and raster masks. Preserve the full visible leaf, including yellow/brown symptomatic tissue, according to a written annotation convention. Record the image coordinate space and use nearest-neighbor resizing for binary masks. Emre and Merve must review and resolve the outlines together. Review status and reviewer decisions must be recorded accurately.

Predicted masks are not independent reference masks. Upstream segmented images are not substitutes for the team's reviewed outlines. Assisted outlines, if used, require correction and recorded review. Do not report final reference-mask IoU/Dice until the required reference outlines have actually been reviewed.

Implement segmentation using Lab background estimates from image borders, color-distance thresholding, morphological opening/closing, hole filling, and largest-component selection. Tune a small, recorded set of parameters using development masks only. Inspect whether diseased tissue is lost. Test outlines must not influence parameter choices.

For Method B, restrict LBP sampling to neighborhoods fully inside the predicted mask. Use development data to define unusable-mask checks before the final experiment. Cache masks/features with enough source and configuration information to prevent stale results after changes.

## Training and final evaluation

- Use the same three development folds for A and B. Use stratified, group-aware cross-validation; inspect plant/health distributions and preserve group separation.
- Fit StandardScaler inside the training portion of every fold through a scikit-learn Pipeline.
- Give both methods the same search budget: `C` in `{0.1, 1, 10}`, `gamma="scale"`, `class_weight="balanced"`, and macro F1 for selection.
- Record cross-validation results, choose parameters using development data, and freeze image processing, mask rules, features, label mapping, and hyperparameters before final test scoring.
- Refit each selected pipeline on the same eligible development samples. Resolve unusable development masks using development data before cross-validation. If a predeclared exclusion is necessary, apply it to both methods' training samples and record the change. Do not silently replace Method B features with Method A features, and do not exclude failed test cases.
- Evaluate the locked test set after implementation checks and settings are complete. Do not retune from test findings. Reproducibility reruns must preserve the frozen protocol and must not become new model-selection cycles.
- Save both fitted scaler/model pipelines, the binary label mapping, segmentation settings, feature conventions, source/split manifests, seed, and environment metadata.

Required results:

- Macro F1 as the primary classification metric; accuracy and class precision, recall, F1, and support. Diseased is the positive class.
- Confusion matrices, plant-specific results, and examples of diseased leaves incorrectly predicted Healthy.
- IoU and Dice for the 18 reviewed test masks, including mean, range, successes, and failures. Empty predicted masks score zero.
- Training time and median preprocessing-to-prediction time on the same reference CPU. Use identical timing boundaries and record repetitions and hardware.
- Mask failure counts and prediction coverage, plus B minus A macro F1 and runtime differences.

An unusable mask produces an explicit failure status, not a fabricated health label. Keep every failed case in the end-to-end accuracy denominator. For the primary two-class F1 computation, a failure contributes a false negative to its true class. Report the 2 by 2 confusion matrix for covered predictions together with failure counts by true class, so all test cases are accounted for. Any metric conditional on successful processing must be identified separately. Fix and document this policy before viewing final test results.

## Minimal implementation and notebook structure

```text
Image Processing/
  PROJECT_PLAN.md
  README.md
  requirements.txt
  requirements-annotation.txt
  leaf_disease_analysis.ipynb
  leaf_pipeline.py
  predict.py
  tests/
  data/
    raw/
    manifests/
    annotations/
    masks/
    cache/
  models/
  results/
    tables/
    figures/
    logs/
  reports/
  .venv/
  .venv-labelme/
```

This is an initial organization, not a requirement to create unused files. Keep shared code in one module until an actual complexity problem justifies a split. The notebook and command-line tool must call the same image processing and feature functions.

Notebook sections:

1. Research question, labels, setup, paths, and reproducibility settings.
2. Data acquisition, quality checks, mapping, and fixed partitions.
3. Development-only exploratory figures and reference annotation instructions.
4. Segmentation examples and development-mask parameter selection.
5. Full-image and masked feature extraction with shape/validity checks.
6. Paired cross-validation and parameter freeze.
7. Final classification, segmentation, runtime, and coverage results.
8. Error analysis, conclusions, saved artifacts, and single-image demonstration.

## Verification and completion gates

Keep tests small and focused on behavior that could invalidate the experiment:

- Data validation: permitted label mapping, unreadable images, fixed counts, no group/hash overlap across partitions, and reproducible manifest generation.
- Image processing: mask dimensions/type, symptomatic tissue checks, finite 58-value features, normalized histograms, and LBP neighborhoods restricted to the mask.
- Metric checks: synthetic known IoU/Dice cases, empty-mask scores, and failure accounting across the full test set.
- Prediction: valid/invalid image inputs, explicit failure handling, two permitted successful labels, and matching predictions after model reload.
- Execution: a small development-only smoke run, then a clean-kernel local notebook run and a command-line demonstration. A successful fresh-runtime Colab Run all, executable evaluation, editable input/parameter demonstrations, and shared-resource access checks are mandatory before final submission.
- Final package: frozen parameters, manifests, 30 reviewed reference masks, saved models, tables/figures, run instructions, and a report whose quantitative claims match saved experiment outputs.

Classification improvement is not a completion gate. A working, reproducible comparison with a negative result satisfies the research objective. Missing independent reference annotations or hidden test failures do not.

## Schedule

| Target | Work | Completion evidence |
| --- | --- | --- |
| 1 November 2026 | Local setup, data selection, labels, fixed split | Working kernel, source records, validated manifests |
| 8 November 2026 | Segmentation prototype and 30 reference outlines | Annotation JSON/masks and both reviewers' records |
| 15 November 2026 | Features and Method A | Passing focused checks and working binary baseline |
| 22 November 2026 | Method B, paired validation, settings freeze | Recorded CV results and frozen configurations |
| 29 November 2026 | Final classification, masks, runtime, errors | Saved final tables, figures, and failure accounting |
| 6 December 2026 | Report, presentation, demo, delivery check | Reproducible final package |

Emre and Merve retain the responsibilities proposed in the source document. Assisted work will be described accurately; execution, annotation review, and authorship will not be attributed to a team member unless they performed it.

## Final report plan

Write the English report after testing and final evaluation. Apply Humanizer while preserving technical precision and the team's actual results. Deliver a Word report by default; confirm any instructor template if one becomes available. Prepare presentation slides after the report's evidence and conclusions are settled.

The report will cover the problem and hypothesis, dataset and exclusions, group-safe protocol, reference annotations, segmentation and 58-value features, paired SVM comparison, measured results, representative errors, runtime, limitations, reproducibility, and references. Describe background bias, the small segmentation test sample, and limited evidence for unseen field conditions. Populate tables and figures from saved experiment outputs. Report improvements, regressions, and failures with the same standard of evidence.

## Sources checked during planning

- Supplied proposal: `/Users/emrecanturk/Downloads/Project Proposal.docx`.
- [Primary PlantVillage repository and grouping metadata](https://github.com/spMohanty/PlantVillage-Dataset).
- [PlantVillage author dataset card](https://huggingface.co/datasets/mohanty/PlantVillage).
- [Upstream PlantVillage loader and label definitions](https://github.com/spMohanty/PlantVillage-Dataset/blob/master/plant_village.py).
- [Labelme installation and Python compatibility](https://github.com/wkentaro/labelme#installation).
- [scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [JupyterLab installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html).
- Instructor execution and sharing announcement supplied on 9 November: `/Users/emrecanturk/.codex/attachments/9d51b3d5-e723-477b-9c23-8d5622068d9f/Yapıştırılan metin.txt`. Share the notebook with `ngkasapoglu@gmail.com` as Editor, share the required Drive project resources, and submit both links through CATS. The announcement contains no submission date.

Current status: local installation, fixed data acquisition, working prototype, development experiment, and local verification are complete. Reference review is in progress. Segmentation tuning, locked final evaluation, the final report, and presentation follow the reviewed references. See `PROJECT_STATUS.md` for the measured results and current limitations.
