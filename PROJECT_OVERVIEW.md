# Plant leaf disease analysis project overview

Status date: 9 November 2026  
Course: COM0418 Image Processing, 2026–2027 Fall Semester  
Team: Emre Cantürk, 2300006608; Merve Gündoğdu, 2200004952  
Project title: Plant Leaf Disease Analysis Using Segmentation and Machine Learning

The local prototype, sampled dataset, and initial development experiments are available. The immediate work is to finish reviewing the reference outlines and make the implementation executable from shared resources in Google Colab. The saved results currently favor full-image classification over classification with the default predicted mask. Final test performance and segmentation quality remain unmeasured.

The agreed final delivery target is 6 December 2026. The instructor's supplied announcement contains no submission date, report template, or page limit; confirm the official deadline and format through CATS. The team confirmed that no interim report is required. The [two-person work plan](TEAM_WORK_PLAN.md) assigns the remaining work through final delivery.

## Instructor requirements that govern delivery

The instructor's announcement makes Google Colab compatibility, successful execution, and instructor access mandatory. Local JupyterLab remains a development route. Local files and a companion ZIP are supplementary until a shared notebook can execute the core algorithms and experimental evaluation in Colab.

| Requirement | Required delivery or proof | Current position |
| --- | --- | --- |
| Notebook format | Organized `.ipynb` files with explanatory Markdown cells | Existing notebook |
| Notebook sections | Environment setup; dataset loading; algorithms; experimental procedures; results/visualization; conclusions | Present in the local narrative; Colab workflow needs completion |
| Fresh Colab execution | Sequential `Runtime > Run all` with installation and loading instructions; no undocumented modifications | Not yet verified in hosted Colab |
| Accessible resources | Images, samples, models, code, annotations, and required files accessible using the instructor's own account | Local resources exist; shared Drive folder not recorded |
| Notebook sharing | Add `ngkasapoglu@gmail.com` with Editor permission | No shared notebook link or permission evidence recorded |
| Project-folder sharing | Share the corresponding Drive folder with the instructor and verify file access | No shared folder link or access test recorded |
| Evaluation interaction | Rerunnable algorithms, parameter experiments, new-image inputs, intermediate outputs, and quantitative comparisons | Local implementation exists; hosted interaction/evaluation needs proof |
| CATS submission | Colab notebook link(s), shared Drive folder link, execution instructions, report, presentation | Final submission pending |

The shared folder must contain the required dataset, sample images, trained models, and companion files. Do not rely on a personal `/content/drive/MyDrive/...` path. Sharing a folder alone does not make that path work in another person's runtime. Resource loading must resolve the shared files using documented identifiers and the signed-in user's authorized access. Verify this with the other team member's account before submission.

The current upload-a-ZIP Colab setup is an incomplete delivery route: the final notebook must include a reproducible resource-loading path from the shared project files, followed by a tested fresh-session Run all. Document any account authorization required to access the shared folder. Both main processing and experimental evaluation must execute; cached results alone are insufficient.

Report the frozen scientific benchmark separately from editable demonstration experiments. The instructor must be able to change a parameter or input image and execute the relevant code without overwriting the recorded final benchmark. The notebook's current final-evaluation switch and stored-result behavior need an explicit hosted rerun path, with isolated experiment outputs.

No external sharing, upload, invitation, or CATS submission has been performed as part of preparing these Markdown documents. The work plan assigns those delivery tasks to the team.

## 1 Research question and scope

The project asks whether measuring color and texture inside a segmented leaf improves Healthy/Diseased classification compared with measuring the same features over the whole image. The hypothesis is that removing the background improves test macro F1. The experiment may support or reject that hypothesis.

The input is a leaf image. Successful predictions have one of two labels, `Healthy` or `Diseased`. The masked method also produces a leaf mask and overlay. Invalid images and unusable masks produce an explicit processing failure.

The comparison uses two methods:

| Component | Method A | Method B |
| --- | --- | --- |
| Feature region | Full image | Predicted leaf region |
| Color features | Three normalized 16-bin HSV histograms | Same histograms inside the mask |
| Texture features | Normalized 10-bin uniform LBP histogram | Same histogram with neighborhoods inside the mask |
| Feature count | 58 | 58 |
| Classifier | StandardScaler and RBF SVM | Same approach |
| Training and validation | Fixed development samples and three folds | Same eligible samples and folds |

Plant species and original disease names support sampling and error analysis. They are not model outputs. Neural networks, disease subtype prediction, lesion severity, a web/mobile interface, and hardware deployment are outside the core experiment. External field-image evaluation and TÜBİTAK preparation are optional separate work.

## 2 Current completion status

| Work item | Status on 9 November | Evidence or next dependency |
| --- | --- | --- |
| Research question and protocol | Defined | Original proposal and `PROJECT_PLAN.md` |
| Local Python and annotation environments | Created; earlier execution checks recorded | `.venv`, `.venv-labelme`, requirements and logs |
| Dataset acquisition | Complete | 900 original color images and fixed manifest |
| Binary mapping and group separation | Implemented and validated in recorded tests | Manifests, source records, test log |
| Full-image and masked feature extraction | Implemented | `leaf_pipeline.py` |
| Paired SVM training and prediction | Implemented; provisional models saved | `experiment.py`, `predict.py`, saved pipelines |
| Notebook | Local clean-kernel execution recorded | Seven executed code cells with no stored error outputs |
| Reference outline drafts | 30 prepared | 12 development, 18 test |
| Both-person outline review | Incomplete | Saved flags currently approve 1 development outline |
| Segmentation parameter selection | Implemented, awaiting reviewed development references | Requires all 12 development masks |
| Frozen protocol and final evaluation | Pending | No frozen protocol or final metrics file |
| Hosted Colab execution | Required; pending verification | Notebook route and companion ZIP exist |
| Instructor access and CATS links | Required; pending | Shared notebook, Drive folder, and access verification needed |
| Final report and presentation | Pending | Follow the locked final experiment |

Existing logs record the full test suite, notebook, and prediction checks performed during prototype preparation on 28 October. On 9 November, read-only dataset validation, both environments' dependency checks, and reference-index validation passed again. The full suite, notebook, training, and final evaluation were not repeated for this documentation update. The old Jupyter log records shutdown; start the server when needed.

## 3 Dataset and selection decisions

The primary source is the original color-image collection in the [PlantVillage repository](https://github.com/spMohanty/PlantVillage-Dataset). Acquisition is pinned to revision `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`; the seed is `42`.

| Plant | Healthy development | Diseased development | Healthy test | Diseased test | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple | 120 | 120 | 30 | 30 | 300 |
| Bell pepper | 120 | 120 | 30 | 30 | 300 |
| Tomato | 120 | 120 | 30 | 30 | 300 |
| Total | 360 | 360 | 90 | 90 | 900 |

Bell pepper is recorded as `Pepper,_bell` in source labels. There are 720 development images, 180 locked test images, and 900 distinct supplied leaf groups. The subset selects one image per supplied group. Each development validation fold contains 240 images: 40 for each plant/health combination. Each training fold therefore contains 480 images before any shared mask-related exclusion.

The selected original categories are:

| Plant | Original category | Selected images |
| --- | --- | ---: |
| Apple | Apple scab | 81 |
| Apple | Cedar apple rust | 69 |
| Apple | Healthy | 150 |
| Bell pepper | Bacterial spot | 150 |
| Bell pepper | Healthy | 150 |
| Tomato | Bacterial spot | 25 |
| Tomato | Early blight | 25 |
| Tomato | Late blight | 25 |
| Tomato | Leaf mold | 25 |
| Tomato | Septoria leaf spot | 25 |
| Tomato | Tomato yellow leaf curl virus | 25 |
| Tomato | Healthy | 150 |

Disease categories map to `Diseased`; verified healthy categories map to `Healthy`. The manifest stores the original categories. The numerical metadata mapping is Healthy = 0 and Diseased = 1; the saved classifiers return string labels.

Changes from the initial candidate selection are documented:

- Grape was replaced by Apple because healthy grape grouping metadata was absent in the selected revision.
- Apple black rot was omitted because its source grouping label differed and was not resolved for this subset.
- Tomato mosaic was omitted because verified grouping metadata was unavailable.
- Tomato Target Spot remains in the allowlist but supplied keys resolved to zero usable source images. It contributes no images to this sample.
- Tomato spider-mite damage was excluded as a pest-only category. Ambiguous health labels and nutrient/stress categories are outside the binary disease experiment.

Original files are preserved read-only. Each image has a pinned URL, Git blob checksum, SHA-256 checksum, decoded-pixel hash, source label, plant, leaf group, split group, partition, and fold record. The acquisition code verifies source paths and checksums and rejects unreadable images.

Duplicate screening uses identical decoded pixels or a pHash distance at most 6 together with a 64×64 RGB mean absolute error at most 8. The sampled manifest records no qualifying duplicate pairs. Supplied leaf groups and hash checks show no overlap between development and test or between validation folds. The near-duplicate rule is a heuristic and cannot prove that every visually related photograph has been found.

The [author dataset card](https://huggingface.co/datasets/mohanty/PlantVillage) records the source attribution and stated license. Preserve attribution and check applicable terms when distributing image derivatives. The Colab archive retrieves originals from pinned source URLs instead of bundling the full image collection.

## 4 Image processing and feature extraction

The shared implementation decodes images with OpenCV, converts BGR to RGB, and resizes them to 256×256 pixels. Feature extraction uses HSV and grayscale. Segmentation uses floating-point Lab.

Method B estimates the background as the median Lab color of the image border. It thresholds Euclidean Lab color distance, applies morphological opening and closing, and fills the largest external contour. Current provisional parameters are:

| Parameter | Current value |
| --- | ---: |
| Lab distance threshold | 18.0 |
| Border width | 5 pixels |
| Opening kernel | 3×3 ellipse |
| Closing kernel | 5×5 ellipse |

Once 12 development references have been reviewed, the code compares distances `{12, 18, 24}` and closing sizes `{3, 5}`. It selects by mean development-reference Dice. Test references must never determine these settings.

The HSV descriptor contains 16 bins per channel, using OpenCV ranges H `[0,180)`, S `[0,256)`, and V `[0,256)`. Each channel histogram is normalized separately. Uniform LBP uses 8 neighbors, radius 1, and 10 normalized bins. A complete 3×3 neighborhood must lie within the feature region, including for full-image texture at the outer border. The concatenated vector contains 48 color values and 10 texture values.

Method B rejects a mask with foreground below 1% or above 97%, or without a valid texture neighborhood. Feature vectors must contain 58 finite values. Shared code in `leaf_pipeline.py` keeps notebook and CLI preprocessing consistent. Feature caches include source/configuration hashes to avoid reusing results after relevant changes.

## 5 Reference masks and the shadow issue

The 30 annotation images were selected from the fixed manifest. For each plant, development has two Healthy and two Diseased examples; test has three of each.

| Reference subset | Required masks | Saved dual review | Current exported references |
| --- | ---: | ---: | ---: |
| Development | 12 | 1 | 1 |
| Test | 18 | 0 | 0 |
| Total | 30 | 1 | 1 |

The approved development outline is `leaf_0113`. The other outlines remain pending according to saved flags. If reviews were performed without saving their flags, the files must be updated by the actual reviewers.

Initial reference drafts used GrabCut, separately from the Lab-distance method under evaluation. The team identified cast-shadow inclusion in these nine development images:

`leaf_0121`, `leaf_0156`, `leaf_0381`, `leaf_0395`, `leaf_0519`, `leaf_0689`, `leaf_0743`, `leaf_0849`, `leaf_0887`.

Those drafts were corrected using traced priors, explicit shadow-background seeds, GrabCut edge refinement, and restorations of visible tissue or petioles. The original JSON files were backed up. Both review flags were cleared for the corrected revisions. These are assisted drafts awaiting renewed review; correction alone does not make them ground truth.

All 12 development annotation IDs are:

`leaf_0113`, `leaf_0121`, `leaf_0156`, `leaf_0198`, `leaf_0381`, `leaf_0395`, `leaf_0465`, `leaf_0519`, `leaf_0689`, `leaf_0743`, `leaf_0849`, `leaf_0887`.

Both team members must inspect all 30 final references. Preserve visible leaf tissue, including yellow/brown lesions and a connected visible petiole. Exclude cast shadows, background, and detached objects. Judge each reference against the original image. Clear both review flags before changing a previously approved outline, then review the revision again.

Labelme stores polygons and paired 256×256 PNGs. Each person marks their own `Emre_reviewed` or `Merve_reviewed` image flag and saves. The exporter accepts only dual-reviewed polygons, checks their image association, and records mask/annotation hashes. Changes invalidate the previously exported reference record. `FAILED` predictions and unreviewed drafts cannot serve as reference labels.

The review gallery is [results/mask_review/index.html](results/mask_review/index.html). Review launchers are `Review Development Masks.command` and `Review Test Masks.command`. The correction history is recorded in `data/annotations/shadow_correction_log.json`.

## 6 Training protocol and preliminary findings

Both classifiers use an RBF SVM with `gamma="scale"`, `class_weight="balanced"`, and `C` in `{0.1, 1, 10}`. The same three fixed development folds and macro F1 selection rule apply to both methods. A scikit-learn Pipeline fits StandardScaler within each training fold.

The initial experiment retained all 720 development images. If a future development image has an unusable mask, the current implementation excludes it from both methods' training and validation and records the common exclusion. Report the resulting coverage and sample counts; the final test denominator must retain every image.

The saved development summary reports:

| Measurement | Method A | Method B |
| --- | ---: | ---: |
| Selected C | 10 | 10 |
| Mean three-fold development macro F1 | 0.9444438657 | 0.8971290264 |
| Shared eligible development images | 720 | 720 |

B minus A is `-0.0473148393`, or about -4.73 macro F1 points on a 0–100 scale. The saved segmentation status is `provisional_unreviewed_masks`, with the declared default mask parameters. The measured difference is an early development observation. It does not establish the final masking effect or statistical significance.

The cast-shadow problem was identified in assisted reference drafts. It motivates inspection of development masks, but it does not by itself prove why Method B has lower CV macro F1. Re-examine the Lab method after the reference review and retain negative findings if they persist.

Use [results/development/summary.json](results/development/summary.json), `cv_A.csv`, and `cv_B.csv` for report values. Search/refit timings in those records describe classifier fitting on features; they are not end-to-end inference benchmarks. After reviewed references are available, rerun the paired development experiment and identify the new settings and results as a new development revision.

## 7 Final evaluation requirements

The final evaluation requires 12 reviewed development references for parameter selection, all 30 references exported for evaluation, and a frozen record of parameters, manifest, implementation, model files, and development-reference hashes. Neither `models/frozen_protocol.json` nor `results/final/metrics.json` exists in the inspected project snapshot.

Required outputs are:

- Classification macro F1, accuracy, per-class precision/recall/F1/support, and B minus A macro F1.
- A covered-prediction 2×2 confusion matrix for each method, with failures by true class and prediction coverage.
- Per-plant classification results and discussion of Diseased leaves predicted Healthy.
- IoU and Dice on the 18 reviewed test outlines, with mean, minimum, maximum, and representative errors.
- Training time and median preprocessing-to-prediction time on the same reference CPU. The code requests one OpenCV thread and three inference repetitions per test image. Model loading is outside the inference timing boundary. Existing saved environment metadata reports eight OpenCV threads, so verify and record the actual setting during the final run before asserting consistent timing conditions.

IoU is intersection divided by union. Dice is twice the intersection divided by the sum of mask areas. The project convention gives empty masks zero. Invalid Method B masks are explicit failures, remain in the full accuracy denominator, and contribute a false negative to their true class in the two-class F1 calculation. Covered confusion matrices and separate failure counts must account for the complete test set.

The locked test results are stored once for the frozen inputs. An unchanged rerun returns stored results. Do not change settings after reading test results to improve the reported comparison. Add and test a separate Colab evaluation rerun for the instructor; distinguish reproduced fixed-protocol metrics from exploratory parameter changes and preserve the submitted benchmark outputs.

## 8 Environment and execution

The recorded reference machine is an Apple M2 with 8 CPU cores and 16 GiB RAM. The saved environment records macOS 26.5.2 and Python 3.11.0. The project uses CPU processing and does not require a dedicated GPU or paid API.

| Package | Pinned version |
| --- | --- |
| NumPy | 2.4.6 |
| OpenCV headless | 4.14.0.94 |
| scikit-image | 0.26.0 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| joblib | 1.6.0 |
| JupyterLab | 4.6.4 |
| ipykernel | 7.4.0 |
| nbclient | 0.11.0 |
| nbformat | 5.11.1 |
| certifi | 2026.10.23 |
| Labelme, separate annotation environment | 6.3.1 |

Direct requirements are in `requirements.txt` and `requirements-annotation.txt`. Complete local locks are `requirements-local-lock.txt` and `requirements-annotation-lock.txt`. The notebook/kernel uses `.venv`; Labelme uses `.venv-labelme`. Qt GUI dependencies are excluded from hosted notebook setup.

Run commands from `/Users/emrecanturk/Desktop/Image Processing`. The launchers set local cache/Jupyter paths.

```bash
# Open JupyterLab and select the Leaf Disease (.venv) kernel.
./"Start Jupyter Lab.command"

# Check the fixed dataset and implementation.
.venv/bin/python prepare_data.py --verify
.venv/bin/python -m unittest discover -s tests -v

# Open, review, and save development outlines in Labelme.
./"Review Development Masks.command"
.venv/bin/python mask_review.py status
.venv/bin/python mask_review.py export
.venv/bin/python mask_review.py render

# Refit the paired development experiment after reference review.
.venv/bin/python experiment.py development

# Demonstrate prediction on a supplied image.
.venv/bin/python predict.py '/absolute/path/to/leaf.jpg' --method B

# Rebuild the Colab companion archive after changes.
.venv/bin/python package_project.py
```

Final commands are for the later evaluation stage after review and parameter checks:

```bash
.venv/bin/python experiment.py freeze
.venv/bin/python experiment.py evaluate
```

The current Colab prototype opens `leaf_disease_analysis.ipynb` and requests `dist/leaf_project_colab.zip` through an upload cell. The archive supplies companion code and manifests, and the notebook restores originals from the pinned source. Hosted execution has not been verified. Retain this as a supplementary route while implementing the mandatory shared-resource loading and evaluation workflow described above. Regenerate the archive after changing annotations, results, or documentation.

## 9 Verification evidence and its limits

Recorded checks include 13 passing focused tests, a clean local notebook run covering seven code cells, 12 real development-image predictions across all plant/health combinations and both methods, and consistent predictions after model reload. The CLI accepted a valid image and rejected an unreadable file and an unusable blank-image mask. Both environments passed dependency checks during setup.

The tests cover feature shape/normalization, exclusion of background from masked features, synthetic symptomatic-tissue preservation, unusable masks, known IoU/Dice cases, failure accounting, label/group resolution, source/path validation, saved-model consistency, polygon validity, actual dataset separation, and review requirements for final evaluation.

These checks establish software behavior for the recorded cases. Synthetic symptom checks do not establish accurate segmentation on every leaf. Valid polygon coordinates do not establish correct leaf boundaries. Controlled PlantVillage images and 18 test references provide limited evidence for field photographs, unseen plants, or general agricultural diagnosis.

## 10 File map and reproducibility records

| File or directory | Purpose |
| --- | --- |
| `leaf_disease_analysis.ipynb` | Experiment narrative, local/Colab setup, development results, demonstration, final-evaluation switch |
| `leaf_pipeline.py` | Shared preprocessing, segmentation, features, metrics, prediction |
| `prepare_data.py` | Pinned acquisition, sampling, grouping, duplicate checks, fixed partitions |
| `mask_review.py` | Draft preparation, gallery, review checks, reference export |
| `experiment.py` | Development CV, segmentation selection, freeze, locked evaluation |
| `predict.py` | Single-image CLI and output figures |
| `package_project.py` | Companion ZIP builder excluding environments, raw dataset, and server files |
| `tests/test_pipeline.py` | Focused behavior and experiment-integrity tests |
| `data/manifests/images.csv` | Per-image source, label, hashes, grouping, partition, folds, annotation selection |
| `data/manifests/source_inventory.csv` | Available source groups by original category |
| `data/manifests/exclusions.csv` | Missing or ambiguous image/group correspondence |
| `data/manifests/label_mapping.json` | Label mapping and excluded/absent categories |
| `data/manifests/dataset_summary.json` | Source revision, sample counts, seed, duplicate rule |
| `data/manifests/duplicate_pairs.csv` | Duplicate-screening candidate record |
| `data/raw/` and `data/source/` | Preserved originals and source metadata |
| `data/cache/` | Cached features identified by input/configuration hashes |
| `data/annotations/labelme/development/` and `test/` | Paired editable JSON/PNG reference drafts |
| `data/annotations/reviewed_index.json` | Exported reference index with reviewer state and hashes |
| `data/annotations/reference_masks/` | Exported raster masks for reviewed outlines |
| `data/annotations/shadow_correction_log.json` | The nine development-outline corrections |
| `models/development_A.joblib` and `development_B.joblib` | Provisional fitted scaler/SVM artifacts |
| `results/development/` | CV tables, summary, environment, exclusions, development figure |
| `results/logs/` | Test, notebook, setup, prediction, and application evidence |
| `results/mask_review/` | Numbered annotation-review gallery |
| `dist/leaf_project_colab.zip` | Generated companion archive |
| `reports/` | Intended report location; report authoring remains pending |

`README.md` contains operating instructions. `PROJECT_PLAN.md` preserves the original protocol and milestones. `PROJECT_STATUS.md` is the earlier 28 October status record. This overview records the 9 November inspection. The Git repository is initialized; the inspected files are still untracked, so versioned snapshots and a shared backup remain team tasks.

## 11 Final report and presentation

Write the English final report after the reviewed-reference and locked-evaluation stages. Include the actual sample counts, feature definitions, comparison protocol, final classification and segmentation results, runtime, coverage, errors, and limitations. Explain the shadow issue and document how references were reviewed. Use Humanizer for prose and retain the technical details and measured outcomes.

A suitable report outline is: objective and hypothesis; dataset and exclusions; group separation and reference annotations; segmentation/features/classifier; development parameter selection; final results and timing; errors and limitations; reproducibility and contributions; references. Adapt length and format to the course submission requirements when confirmed. The presentation should explain the same experiment, demonstrate executable code, and use consistent results.

Use `dataset_summary.json` and `images.csv` for counts; development `summary.json`, `cv_A.csv`, and `cv_B.csv` for parameter selection; the gallery and correction log for annotation work; and execution logs for reproducibility evidence. Add final metrics, confusion matrices, segmentation results, and timing only after the locked experiment. Preserve preliminary labels when discussing earlier CV results. Include fresh-Colab execution evidence and verified resource access in the delivery instructions.

Assign future tasks according to `TEAM_WORK_PLAN.md`. Credit completed contributions only when the person has performed and checked them. Assisted implementation and outline drafting must be described consistently with the course's applicable rules.

## 12 References and project records

- Supplied proposal: `/Users/emrecanturk/Downloads/Project Proposal.docx`.
- Instructor announcement supplied by the user: `/Users/emrecanturk/.codex/attachments/9d51b3d5-e723-477b-9c23-8d5622068d9f/Yapıştırılan metin.txt`. It establishes mandatory Colab execution, sharing, access verification, and submission artifacts; it gives no dates.
- [PlantVillage repository](https://github.com/spMohanty/PlantVillage-Dataset) and [author dataset card](https://huggingface.co/datasets/mohanty/PlantVillage).
- Mohanty, Hughes, and Salathé (2016), [Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419).
- [OpenCV color conversion](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html), [morphology](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html), and [GrabCut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html).
- [scikit-image LBP example](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html).
- [scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [Labelme](https://github.com/wkentaro/labelme), [local project instructions](README.md), [original plan](PROJECT_PLAN.md), and [two-person work plan](TEAM_WORK_PLAN.md).
