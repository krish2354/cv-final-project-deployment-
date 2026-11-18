# CellCount AI Studio · Recorded Experiment Results

**MAIB AI 221 · Project 1: Healthcare · BCCD experiment · 18 November 2026**

These are **measured values from the uploaded Colab output archive**, not invented performance targets or estimates from the deployed app. All results describe a small annotated teaching/research dataset, **not clinical validation**.

## Dataset and protocol

The source Colab notebook records the MIT-licensed BCCD dataset with **364** paired annotated image files. Images were divided with seed **42** into **254 training / 55 validation / 55 untouched test** images. Fine-tuning used a YOLO11n model with **45 epochs requested** (the results file does not establish the actual epoch at which training ended). The classical reference is a deterministic OpenCV color-thresholding and watershed approach. Counting MAE is the mean number of **absolute cell-count mistakes per image**, separately for each cell type. Lower is better.

## Object detection on the 55-image held-out test set

| Metric | Value |
|---|---:|
| YOLO11n mAP@0.5 | **0.9014** |
| YOLO11n mAP@0.5:0.95 | **0.6094** |

| Cell type | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|---|---:|---:|---:|---:|
| RBC | 0.7770 | 0.8252 | 0.8665 | 0.5954 |
| WBC | 0.9444 | 1.0000 | 0.9854 | 0.7827 |
| Platelets | 0.7930 | 0.8356 | 0.8523 | 0.4499 |

These scores reflect held-out **detection performance** and must not be called clinical accuracy, precision of a blood count, or proof of patient safety.

## Mean absolute counting error (55 original held-out images)

| Cell type | Classical baseline MAE | Fine-tuned YOLO MAE | Direction |
|---|---:|---:|---|
| RBC | 15.673 | 5.018 | Lower error with YOLO |
| WBC | 26.600 | 0.091 | Lower error with YOLO |
| Platelets | 78.855 | 0.400 | Lower error with YOLO |

The baseline produces extremely high error for WBC and platelets, so the comparison demonstrates improvement over **this specific weak classical method**, not over a clinical gold standard. See `artifacts/classical_baseline_test.csv` and `artifacts/yolo_original_test_counts.csv`.

## Controlled super-resolution experiment

The experiment selected **20 distinct images from the held-out test set** and created a paired 160 × 120 low-resolution input for each, as well as a degraded counterpart (blur, low-resolution, low light, or eyepiece-style appearance). Each low-resolution input was evaluated **without SR** and with a **Real-ESRGAN 4× enhancement before YOLO11n**. For all four conditions, ground-truth cell counts came from the original BCCD annotations.

| Input condition | RBC MAE | WBC MAE | Platelet MAE |
|---|---:|---:|---:|
| Clean low-resolution, raw | **4.60** | **0.10** | **0.35** |
| Clean low-resolution, +SR | **6.55** | **0.10** | **0.40** |
| Degraded low-resolution, raw | **5.20** | **0.10** | **0.55** |
| Degraded low-resolution, +SR | **4.65** | **0.10** | **0.65** |

**Interpretation:** super-resolution is **not uniformly beneficial**. On degraded images, RBC count MAE improved by 0.55 cells per image, while platelet count MAE worsened by 0.10. On clean low-res images, RBC MAE worsened by 1.95 cells per image. The WBC count MAE was unchanged, but under degraded images the IoU-matched WBC localization recall dropped from 1.000 without SR to 0.895 with SR, while RBC detection precision also worsened (0.671 → 0.628). Qualitative inspection is needed before calling any new false-positive box an “SR hallucination.”

**Timing recorded in Colab (not Streamlit Cloud):** degraded low-res raw averaged 0.216 s/image; degraded low-res +SR averaged 1.143 s/image. The deployed target of ~5 s/image must still be checked after launch.

## Error analysis and responsible AI

Known difficult cases: overlapping RBCs, incomplete platelet detection, stain and light variation, blur and eyepiece vignetting. The embedded annotated example figure provides visual examples, but **does not supply ground-truth verification itself**. The automatic “needs manual count” state is heuristic, not calibrated as a validated medical safety feature. **All output counts require human confirmation**, and identifiable/sensitive patient specimens should not be uploaded to public infrastructure.

The dataset is narrow and does not include an independently established external clinical/site/device subgroup study. Therefore, reliability across microscopes, hospitals, patient populations, or pathology classes remains **unestablished**.

## Reproducibility and evidence files

- `artifacts/yolo_test_overall.json`: held-out mAP.
- `artifacts/yolo_test_per_class_metrics.csv`: per-class scores.
- `artifacts/split_manifest.json`: test-image-disjoint split.
- `artifacts/classical_baseline_test.csv`, `artifacts/yolo_original_test_counts.csv`: 55-image counting evidence.
- `artifacts/robustness_count_results.csv`: 20 images × four conditions × three classes = **240 counting rows**.
- `artifacts/robustness_detection_errors_aggregate.csv`: greedy IoU@0.5 detection matches, **not** official mAP.
- `artifacts/image_quality_flags.csv`, `artifacts/robustness_manifest.csv`: degraded-input description and flagged images; the source paths refer to a temporary Colab environment and are not available inside this deployment package.
- `artifacts/example_predictions.png`: qualitative inference figure, not a raw input.

**Scope boundary:** This demonstrates a teaching/research pipeline for microscopic field counting, **not** a diagnostic blood test, complete blood count, or substitute for laboratory review. No patient-level clinical conclusions are justified by these results.
