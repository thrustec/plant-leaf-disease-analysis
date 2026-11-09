# Two-person work plan and final delivery schedule

Planning date: 9 November 2026  
Team: Emre Cantürk and Merve Gündoğdu  
Agreed final delivery target: 6 December 2026

The project now needs reviewed reference masks, completed experiments, and a Google Colab delivery that the instructor can access and execute. No interim report is required. The supplied announcement does not state a deadline, page limit, or report template; confirm official course dates and formats through CATS. The dates below are internal targets supporting the agreed 6 December delivery.

The [project overview](PROJECT_OVERVIEW.md) records the current implementation, methods, dataset, preliminary results, and evidence. This plan assigns future work. Existing assisted code and outline drafts do not establish either student's completed contribution; record work after it has actually been performed and checked.

## 1 Delivery requirements from the announcement

Final submission must include:

1. Google Colab notebook link(s), shared with `ngkasapoglu@gmail.com` with Editor permission.
2. A shared Google Drive project folder containing the required dataset, sample images, saved models, and companion files, accessible to the instructor.
3. Instructions and executable code for setup, resource loading, algorithms, experiments, results, and conclusions.
4. A successful fresh Google Colab `Runtime > Run all`, without undocumented manual changes.
5. The report and presentation in the course's required formats.
6. Colab link(s) and the shared Drive folder link submitted through the relevant CATS section.

Local development may continue in JupyterLab. A migration to Anaconda is not required by the announcement. The existing project environments remain useful; the required hosted execution must be tested independently.

The instructor may change parameters, use new input images, inspect intermediate steps, and recompute quantitative comparisons. Preserve the fixed benchmark and provide an editable experiment path that writes separate outputs. Screenshots and saved metrics alone do not satisfy execution requirements.

The notebook must access resources through the instructor's own account. A student's personal mounted Drive path will not transfer to that account. Use documented shared resource identifiers and a working access procedure, and verify the route from the other team member's account before submission. Document any normal authorization step.

No upload, sharing invitation, or CATS submission is completed by this planning document. Each external delivery task needs an actual link, permission check, or submission confirmation before it is marked done.

## 2 Current position

| Area | Verified position on 9 November | Next dependency |
| --- | --- | --- |
| Dataset | 900 images; 720 development, 180 test; fresh integrity check passed | Preserve the fixed manifest |
| Local implementation | Notebook, shared processing, both SVMs, CLI, saved models available | Complete reference-based selection and hosted delivery |
| Prior execution | 13 tests, seven notebook code cells, 12 real prediction cases passed on 28 October | Repeat relevant checks after code/Colab changes |
| Environments | Both environments passed dependency checks again on 9 November | Verify hosted package compatibility |
| Reference masks | 1/12 development and 0/18 test masks have dual review and valid export | 11 development and 18 test references still need both reviewers |
| Initial development CV | A macro F1 0.9444; B 0.8971; default, provisional mask settings | Rerun after all 12 development references are available |
| Frozen benchmark | No freeze or final metrics file exists | Finish development choices and checks before test evaluation |
| Colab | Notebook upload route and ZIP exist; hosted execution unverified | Shared loading, executable evaluation, fresh Run all |
| Instructor access | No shared notebook/folder links or access proof recorded | Create/share delivery resources and test access |
| Report and presentation | Pending | Use final experiment outputs |

The original 8 November mask-review target has passed without completed review. Local implementation is already available. Prioritize the annotation dependency and Colab delivery now, while keeping final evaluation isolated from development choices.

## 3 Ownership and cross-review

| Work area | Primary owner | Other member's role | Evidence of completion |
| --- | --- | --- | --- |
| Data sources, binary labels, grouping, sample tables | Emre | Merve checks counts and exclusions | Manifest-based dataset table and source record |
| Reference-mask coordination and segmentation | Emre coordinates; both edit their allocated images | Each independently reviews every final reference | Dual-reviewed JSON files and valid exported index |
| Features, scaler/SVM, CV, experiment records | Merve | Emre checks preprocessing, folds, sample consistency | Paired CV tables and selected settings |
| Colab notebook structure, loading code, installation, parameter controls | Merve | Emre executes from a fresh runtime and another account | Runnable hosted notebook with all six required sections |
| Shared project-folder contents, samples, model files, permissions | Emre | Merve checks account access and completeness | Shared folder link and successful resource download |
| Notebook sharing and link/access record | Merve | Emre verifies notebook opens with the required role | Instructor Editor permission and notebook link |
| Local tests, CLI, execution instructions | Emre | Merve repeats a demonstration and checks instructions | Updated execution logs and readable run instructions |
| Formal final experiment and classification analysis | Merve | Emre checks freeze, failures, denominators | Locked results, per-plant metrics, confusion matrices |
| Segmentation quality and runtime | Emre | Merve checks references, settings, units, failures | IoU/Dice, timing, coverage, success/failure examples |
| Report dataset, image-processing, reproducibility sections | Emre | Merve reviews the text | English sections tied to source/output files |
| Report objective, classifier, results, discussion; report assembly | Merve | Emre checks numerical and technical consistency | Complete report with supported conclusions |
| Presentation and CATS delivery | Both | Each checks the final package | Rehearsed demonstration and submission confirmation |

Both people must explain the research question, grouping and leakage prevention, the 58 features, the A/B comparison, segmentation failure handling, and the distinction between development and final evaluation. Primary ownership includes understanding and validating the work, even where the code was prepared with assistance.

## 4 First work sessions

Work can start in parallel without using the locked test images for parameter selection.

| Internal target | Emre | Merve | Joint checkpoint |
| --- | --- | --- | --- |
| 9–11 November | Check data records; inspect the assigned corrected development outlines; inventory the files needed in Drive. | Read the CV records; inspect the assigned development outlines; map notebook gaps against the announcement. | Confirm official course requirements and settle the annotation convention. |
| 12 November | Finish first-pass corrections and independent review of Merve's subset; export references when agreed. | Finish first-pass corrections and independent review of Emre's subset; check review/index consistency. | All 12 development references are reviewed by both people and exported, or remaining disputed IDs have a clear correction owner. |
| 13–16 November | Verify development examples, tests, and source counts; prepare the shared data/sample/model folder; test the hosted loading route from another account. | Select segmentation on reviewed development masks; rerun paired CV; complete hosted setup/loading and configurable demonstrations. | Working development-stage Colab Run all with accessible companion files and actual parameter/input execution. |

If a boundary remains uncertain, leave its review unapproved, record the image ID and the disputed region, and resolve it together. Do not change the sample partition or substitute test references to finish development tuning.

## 5 Reference-mask allocation

Split the first editing pass, then exchange the subsets. Both reviewers still inspect every final outline. Emre coordinates the list and exports; Merve checks the exported counts and hashes.

| First editor | Development reference IDs |
| --- | --- |
| Emre | `leaf_0121`, `leaf_0156`, `leaf_0381`, `leaf_0395`, `leaf_0519`, `leaf_0198` |
| Merve | `leaf_0689`, `leaf_0743`, `leaf_0849`, `leaf_0887`, `leaf_0465`, `leaf_0113` |

`leaf_0113` already has saved dual approval; preserve it unless someone finds a boundary error. Nine corrected drafts require renewed review. `leaf_0198` and `leaf_0465` also await their saved approvals.

Each person checks their own `Emre_reviewed` or `Merve_reviewed` image flag and saves. Include visible leaf tissue, yellow/brown symptoms, and a connected visible petiole. Exclude cast shadows and background. Clear both flags before editing a previously approved outline, then repeat both reviews. Reload an externally changed file in Labelme before editing it again.

For the test-reference first pass:

| First editor | Test reference IDs |
| --- | --- |
| Emre | `leaf_0098`, `leaf_0109`, `leaf_0132`, `leaf_0167`, `leaf_0199`, `leaf_0211`, `leaf_0337`, `leaf_0346`, `leaf_0396` |
| Merve | `leaf_0454`, `leaf_0539`, `leaf_0557`, `leaf_0606`, `leaf_0622`, `leaf_0732`, `leaf_0811`, `leaf_0816`, `leaf_0880` |

Exchange all nine for independent review after editing. Judge test outlines only against originals; keep them out of segmentation selection. The internal target is 19 November for all 30 valid references.

## 6 Task tracker

`Ready` means work can start now. `Waiting` names a dependency. Mark `Done` only after the listed evidence exists.

| ID | Task | Lead | Dependency | Internal target | Current status | Completion evidence |
| --- | --- | --- | --- | --- | --- | --- |
| P01 | Check official CATS dates, report/presentation format, and required submission section | Merve | Course announcements | 11 November | Ready | Requirements and official dates recorded |
| P02 | Verify dataset records and prepare source/sample tables | Emre | Existing manifests | 11 November | Ready | 900 images, 720/180 split, category exclusions, source citation |
| P03 | Finish all development-mask edits and independent reviews | Emre coordinates; both edit/review | Existing drafts and correction record | 12 November | In progress | 12 actual dual approvals |
| P04 | Export references and rerun segmentation selection plus paired CV | Emre exports; Merve runs experiments | P03 | 15 November | Waiting | Reviewed index; `reviewed_development_masks` summary; selected settings; CV tables |
| P05 | Create the shared delivery folder and inventory all required resources | Emre | Existing code, data, samples, models; update after review | 15 November | Ready | Folder link with dataset, samples, models, annotations, code, requirements |
| P06 | Replace local ZIP dependence with documented shared-resource loading and verify hosted dependencies | Merve | P05 shared resource identifiers | 16 November | Waiting | Fresh-runtime setup and successful downloads using the other member's account |
| P07 | Make parameter changes, new inputs, intermediate results, and evaluation reruns executable in Colab | Merve | P06; preserve fixed benchmark outputs | 19 November | Waiting | Documented editable experiment path with separate outputs |
| P08 | Complete test-reference edits, both reviews, and export | Both, allocated subsets | Section 5; never use test references for tuning | 19 November | Ready | 12 development plus 18 test valid references |
| P09 | Run relevant tests and clean notebook; verify local/hosted settings and finalize development choices | Emre checks; Merve confirms experiment choices | P04, P06, P07 | 21 November | Waiting | Passing checks; sample/fold consistency; recorded packages and actual thread settings |
| P10 | Freeze the final protocol | Merve; Emre reviews | P08 and P09 | 22 November | Waiting | Hashes for manifest, implementation, models, development references, parameters |
| P11 | Execute final classification, segmentation, runtime, and coverage evaluation | Merve: classification; Emre: masks/timing | P10 | 26 November | Waiting | Final tables, matrices, IoU/Dice, per-plant results, failures, timing |
| P12 | Analyze errors and write the English final report | Merve assembles; both author assigned sections | P11 | 30 November–3 December | Waiting | Report values/figures match saved experiment outputs |
| P13 | Prepare presentation and rehearse parameter/input demonstrations | Both | P11 and report conclusions | 3 December | Waiting | Slides, working hosted demonstration, both students can explain the code |
| P14 | Verify final hosted Run all, other-account resource access, and instructor Editor permission | Emre runs; Merve checks sharing | Final notebook/resources/results | 4–5 December | Waiting | Recorded fresh Colab run; accessible folder; shared notebook; no missing files |
| P15 | Submit all required links and artifacts through CATS and preserve a shared backup | Both | P01, P12, P13, P14 | 6 December agreed target, subject to official CATS date | Waiting | Submitted version, links, report, presentation, confirmation |

If official dates differ, adjust the internal targets while preserving the order of reference review, development selection, freeze, and final evaluation. The first hosted loading test should occur in mid-November; it must not wait until delivery week.

## 7 Colab notebook changes and acceptance checks

| Required section | Implementation work | Acceptance check |
| --- | --- | --- |
| Environment setup | Install needed packages in the hosted Python version; state imports and restart instructions only if actually necessary | Fresh runtime completes setup with recorded versions |
| Dataset loading | Load code, the fixed sample, reviewed references, and saved models from accessible shared resources; use relative working paths | Other member's account can obtain every required resource without a private mounted path |
| Image-processing algorithms | Explain Lab distance, morphology, HSV/LBP, and classifier settings; expose the parameters needed for demonstrations | Change a development-image parameter and rerun the processing cell |
| Experimental procedures | Separate development CV, the frozen benchmark, and exploratory reruns; implement executable evaluation beyond merely displaying cached metrics | Recompute comparison metrics under documented settings without overwriting the benchmark |
| Results and visualization | Show original/mask/overlay, metric tables, graphs, intermediate outputs, and failures | Run all produces inspectable outputs; a new input image can be processed |
| Conclusions | Compare A and B using recorded results and explain limits | Text, figures, denominators, and metrics agree |

The current notebook's `RUN_FINAL_EVALUATION=False` default and stored-result behavior are development safeguards. They need a documented evaluation reproduction route for the instructor. Implement that route before freezing the final protocol so evaluation-code changes do not invalidate the freeze. Use development data or small synthetic checks to verify the route before the formal test evaluation.

Keep Labelme as a local annotation tool. Colab must receive the reviewed JSON/PNG/index artifacts and execute segmentation/evaluation without launching a Qt GUI. The saved SVM/scaler files should be available for immediate prediction; full retraining can remain a documented reproducibility option.

Formal CPU timing uses one declared reference environment. Record the actual OpenCV thread setting: old environment metadata says eight threads although code requests one. Treat Colab timings as a separate hardware measurement when the host differs. Compare reproduced classifications and metrics against the fixed protocol without tuning from discrepancies in test results.

## 8 Shared folder, access, and submission checklist

Suggested shared folder contents:

```text
Plant Leaf Disease Analysis/
  leaf_disease_analysis.ipynb
  companion Python modules and requirements
  data/
    selected original images and sample inputs
    fixed manifests and source attribution
    reviewed annotation JSON files and paired images
    exported reference masks and reviewed index
  models/
    saved scaler/SVM pipelines and frozen protocol
  results/
    development and final evidence
  reports/
    final report
  presentation/
    final slides
  execution instructions and project Markdown documents
```

This is a delivery layout to prepare, not a claim that an online folder already exists. Retain the source's reuse terms and attribution. Omit virtual environments, local server state, caches, credentials, and unrelated personal files.

- [ ] Save/open the final notebook in Google Colab and retain its link.
- [ ] Add `ngkasapoglu@gmail.com` with Editor permission to the notebook.
- [ ] Share the required project folder with the instructor and check the permission covers every dependency.
- [ ] Test resource loading from the other student's account using the documented route.
- [ ] Run the final notebook sequentially in a fresh hosted runtime.
- [ ] Confirm the core algorithms and evaluation execute, including changed parameters and a new input image.
- [ ] Record runtime, package versions, seed, and the required setup/loading steps.
- [ ] Confirm report, slides, and notebook use the same final result revision.
- [ ] Submit Colab link(s), Drive folder link, instructions, report, and presentation through CATS as required.
- [ ] Retain the submitted version and confirmation; both members keep access to the backup.

Sharing the folder and passing the other-account loading test are separate checks. Marking sharing tasks complete requires actual online permissions and links. No sharing invitation or submission has been sent while preparing these documents.

## 9 Report and presentation allocation

Emre writes the dataset, source selection, image preprocessing, segmentation, reference annotations, segmentation errors, timing setup, and execution instructions. Merve writes the research objective, feature/classifier protocol, development selection, final classification results, per-plant errors, discussion, and assembles the report. Both write limitations and actual contributions, verify citations, and read the complete report.

Prepare figures and tables from the saved experiment outputs. Retain negative masking results if they persist. The current CV difference of about -4.73 macro F1 points is preliminary and must not be copied into a final-test table. The annotation-draft shadow problem is documented; its connection to classifier errors still needs evidence.

Use Humanizer for English report prose and Ponytail for the smallest complete implementation changes. Both students rehearse explaining code and changing a parameter or input in Colab. Screenshots supplement the live implementation.

## 10 Progress record and scope control

One person edits a shared file at a time; the other reviews a named saved revision. After a session, record the task ID, owner, files changed, evidence path, reviewer, and next dependency. Do not assign completed contributions from planned ownership alone.

| Session date | Task IDs | Person | Evidence or revision | Other member's review | Next action |
| --- | --- | --- | --- | --- | --- |
| To fill after the next work session |  |  |  |  |  |

The Git repository is initialized but inspected project files are still untracked. Emre prepares the first versioned snapshot and shared backup; Merve checks that source, manifests, reviewed references, and reports are included while environments and local server files are excluded. Rebuild the supplementary Colab ZIP with `package_project.py` after documentation, annotation, or implementation changes.

A working comparison can conclude that masking reduces performance. Additional plant types, neural models, a separate web/mobile interface, external datasets, and funding work must not delay required references, Colab execution, final evaluation, report, or presentation. Confirm any scope change together before starting it.

## Source

The instructor's supplied announcement is `/Users/emrecanturk/.codex/attachments/9d51b3d5-e723-477b-9c23-8d5622068d9f/Yapıştırılan metin.txt`. It establishes the Colab, access, and submission requirements. The 6 December target comes from the team's earlier agreed schedule. The user confirmed on 9 November that no interim report is required.
