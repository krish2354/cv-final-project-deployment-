# Model Card · CellCount AI Studio (Research / Educational)

**Model:** YOLO11n fine-tuned for BCCD microscopy (ONNX); optional Real-ESRGAN x4v3 (ONNX). **Classes:** red blood cells (RBC), white blood cells (WBC), platelets. **Status:** student prototype, **not clinically validated**.

**Intended use:** Illustrative cell localization and counting from anonymized microscope-smear images, with review by a trained technician or trainee. **Not intended for:** disease diagnosis, clinical decisions, replacing laboratory analyzers, patient-level CBC interpretation, or autonomous reporting.

**Data:** 364 BCCD annotated images (MIT, as specified by project brief); seed 42; disjoint train/val/test counts **254/55/55**. The model was adapted by student fine-tuning; test data were kept outside training. An additional **20-image paired synthetic degradation experiment** evaluates super-resolution robustness.

**Measured held-out detection performance:** mAP@0.5 **0.9014**; mAP@0.5:0.95 **0.6094**. Class AP@0.5: **RBC 0.8665**, **WBC 0.9854**, **platelets 0.8523**. Mean absolute counting error per original test image: **RBC 5.018**, **WBC 0.091**, **platelets 0.400**; per-class count MAE can conceal localization misses.

**Generative-AI experiment:** On 20 degraded 160×120 held-out samples, RBC MAE **5.20 → 4.65** (improved), platelet MAE **0.55 → 0.65** (worse), and WBC MAE **0.10 → 0.10**; degraded-image WBC detection recall fell **1.000 → 0.895** with SR. On clean low-res, RBC MAE increased **4.60 → 6.55**. SR is optional and not uniformly beneficial.

**Known failures / population & subgroup limits:** Close or clumped cells, varying smear coloration, small platelets, low light, blur, scope/phone optics, and synthetic super-resolution artifacts. BCCD has a limited lab-camera domain; **external clinical, device, stain, geography, and demographic subgroup performance is not established**. It is inappropriate to infer fairness from this dataset.

**Human oversight:** Show boxes and confidence, quality warnings, correction controls, manual confirmation, and an auditable anonymized CSV. Require trained human interpretation; no unsupervised decisions.

**Privacy & deployment:** No external paid AI API. Uploaded images are processed in the app server’s memory and not deliberately stored. Hosting metadata may be logged. Do not upload identifiable or real sensitive patient data to a public demonstration site.

**Licences:** BCCD MIT per brief; Ultralytics AGPL-3.0; Real-ESRGAN BSD-3-Clause; confirm conditions/attributions for redistributed ONNX conversions. Sources and full quantitative evidence are documented in `README.md`, `LICENSE_NOTICES.md`, and `EXPERIMENT_RESULTS.md`.
