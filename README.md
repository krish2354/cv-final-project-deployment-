# 🩸 CellCount AI Studio — Functional Streamlit App

**MAIB AI 221 · Project 1 · Healthcare**  
**Research/education only. Not a diagnostic device.**

This application is the **deployment companion** to `CellCount_AI_Studio_Colab.ipynb`. It does **real ONNX inference** with the fine-tuned YOLO11n and compact Real-ESRGAN model exported by that notebook. It provides an OpenCV color-threshold + watershed baseline, confidence filtering, image quality flags, annotated bounding boxes, RBC/WBC/platelet counts, batch processing, technician corrections, CSV and PNG export, and a measured-evidence dashboard.

> **This integrated release includes the user-supplied, trained ONNX weights and measured Colab evidence.** Both `artifacts/cellcount_yolo11n.onnx` (10.67 MB) and `artifacts/realesr_general_x4v3.onnx` (4.87 MB), along with test-set CSV/JSON results, are already staged. The UI reads actual evidence; it does not invent performance results. Models are not trained inside Streamlit.

## 1 · What the professor expects

| Project item | Implementation |
|---|---|
| Public BCCD dataset | Training notebook; 364 images, MIT |
| Baseline | OpenCV color masking + watershed separation |
| Trained model | Your YOLO11n fine-tuned ONNX (`cellcount_yolo11n.onnx`) |
| Generative AI | Compact Real-ESRGAN 4× (`realesr_general_x4v3.onnx`) |
| Measured experiment | Paired 20-image held-out degraded set, raw vs 4× SR |
| App interface | Streamlit Community Cloud |
| Quantitative evidence | Per-class AP50, count MAE, TP/FP/FN proxy, latency |
| Human-in-the-loop | Editable technician counts + per-image confirmation |
| Privacy | Non-identifiable uploads only; no deliberate image persistence |

## 2 · Get your Colab outputs

1. Open the notebook you ran earlier: `CellCount_AI_Studio_Colab.ipynb`.
2. Confirm the FINAL notebook cell ran and downloaded **`cellcount_colab_outputs.zip`**. It must include these files under its `artifacts/` directory:
   ```text
   artifacts/
       cellcount_yolo11n.onnx
       realesr_general_x4v3.onnx
       split_manifest.json
       yolo_test_overall.json
       yolo_test_per_class_metrics.csv
       classical_baseline_test.csv
       yolo_original_test_counts.csv
       robustness_count_results.csv
       robustness_detection_errors_aggregate.csv
       ... more optional evidence
   ```
3. Keep your ZIP — it is your actual evaluated experimental evidence.

**If you need to regenerate the file**, execute this in a new Colab cell:

```python
from google.colab import files
files.download('/content/cellcount_ai/cellcount_colab_outputs.zip')
```

## 3 · Run locally

Tested for Python 3.10–3.12-style packages; use a clean environment.

### Windows (PowerShell)

```powershell
cd CellCount_AI_Studio_Streamlit
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
cd CellCount_AI_Studio_Streamlit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

**Fastest test:** Because this integrated project already includes your real ONNX weights and metric files in `artifacts/`, start Streamlit and upload a *non-identifiable* blood-smear photo on **Analysis Lab**. The optional sidebar ZIP importer remains available if you later retrain and replace the model for one session.

**Model files are already embedded in this integrated release.** If you retrain, run `python prepare_artifacts.py path/to/cellcount_colab_outputs.zip` to replace the evidence from that new experiment. Do not copy patient images.

## 4 · Deploy publicly on Streamlit Community Cloud (free)

1. Create a PUBLIC GitHub repository, e.g., `cellcount-ai-studio`.
2. Push **the CONTENTS** of this integrated project folder (`app.py`, `cellcount/`, `.streamlit/config.toml`, `requirements.txt`, `artifacts/`, `README.md`, etc.) to the repository root. Both trained models and measured CSV/JSON files are **already in** `artifacts/`. Do not push the parent ZIP as the Streamlit app.
3. Check GitHub's per-file size limits. The original notebook aimed for models under 20MB, but check actual sizes in your Colab ZIP. Files >100MB need Git LFS or other legitimate hosting.
4. Go to https://share.streamlit.io/, select *Create app*, choose your GitHub repository and branch, and set **entrypoint** to `app.py`.
5. Deploy; test the URL from a desktop and mobile phone. Keep the app live until grading is complete.
6. Verify both ONNX models load and a real image can be analyzed. Also test exporting reviewed CSV and viewing Evaluation Evidence.

Streamlit Community Cloud may put an idle app to sleep. Wake the app before the professor's demonstration. Keep within free resource limits (especially memory and CPU). It is possible that super-resolution on a busy free CPU takes longer than the target <5 seconds — measure real deployed timings; reduce input dimensions carefully and document the tradeoff if needed.

## 5 · Functional user journey

1. **Analysis Lab:** Upload up to ten JPG/PNG microscope images, choose YOLO11n or classical baseline, set confidence/NMS, optionally enable Real-ESRGAN 4× A/B comparison and classical comparison, click **Run real image analysis**.
2. **Inspect results:** Live counts, WBC/RBC within-field ratio (not a clinical CBC), per-image diagnostic flags, bounding boxes, confidence, speed and optional comparisons.
3. **Technician Review:** Inspect each image, correct the automated cell counts if needed, check **I have manually reviewed**, download an anonymized CSV.
4. **Evaluation Evidence:** Review **actual imported** Colab held-out mAP@0.5, per-class scores, classic-vs-YOLO MAE, 20-image paired clean/degraded with/without-SR MAE and false-detection approximations.
5. **Model Card & Help:** Document intended use, known failure modes, bias, data/permissions, privacy, generation risks, and model card download.

**Note on super-resolution:** The app uses the same **160×120** source for two competing inference paths (raw low-res YOLO vs Real-ESRGAN 4× + YOLO), and separately shows original-resolution detections as context. This prevents confusing the high-resolution original with the actual SR control arm. A visual improvement does not imply count improvement. Hallucination should be assessed against ground-truth annotations and manually inspected bounding boxes. Evidence is reported from the held-out study, not inferred from arbitrary uploaded images.

## 6 · Test commands

```bash
python -m pip install pytest
python -m pytest -q
python -m compileall -q app.py cellcount
```

The included tests validate letterboxing, raw YOLO11 output decoding, class-wise NMS, classical processing, user CSV corrections, safety-oriented artifact import, and invalid input handling. **Passing tests does not replace inference testing with your actual ONNX models and Colab evidence.**

## 7 · Model files and licensing

- BCCD dataset: [Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset) — MIT (as recorded in project brief).
- Ultralytics training code: [Ultralytics](https://github.com/ultralytics/ultralytics) — AGPL-3.0.
- Real-ESRGAN: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — BSD-3-Clause.
- Real-ESRGAN ONNX conversion used by the Colab notebook: [vlad-grig/realesr-general-x4v3-onnx](https://huggingface.co/vlad-grig/realesr-general-x4v3-onnx); verify redistribution terms and attribution before release.

Retain original license notices. Do not include real patient/identifiable images or unlicensed datasets in the public repository.

## 8 · Important limitations

- **Not diagnostic.** No clinical validity or safety claim. Manual review required even on high-confidence images.
- Baseline thresholds are tuned to BCCD-like stains and image sizes, not general microscopy.
- ONNX inference and NMS are implemented in OpenCV/NumPy to fit small free deployments, not Ultralytics `YOLO()` runtime. Compare count predictions for a small sample against the notebook's `onnx_model.predict()` before final submission; small differences may arise in letterboxing and NMS.
- User-uploaded images and manual corrections are held in Streamlit session memory during usage, not deliberately saved or transmitted to a third-party AI API. The hosting provider may nevertheless retain operational metadata; do not use sensitive patient data.
- The app shows measured Colab evidence only when imported; it does not calculate mAP for arbitrary uploads (ground-truth boxes are unavailable).
- A public, working URL, correct weights and measured latency must be verified after deployment. The project here is the **ready-to-deploy code package**, not a live app by itself.

## 9 · Actual integrated Colab results (November 2026)

**Recorded model evaluation, not a new live deployment benchmark:**

- Disjoint split: **254 train / 55 val / 55 test** BCCD images; seed 42.
- YOLO11n held-out detection **mAP@0.5 = 0.9014**, mAP@0.5:0.95 = 0.6094 (these are mAP scores, not patient-level accuracy).
- Per-class AP@0.5: RBC **0.8665**; WBC **0.9854**; platelets **0.8523**.
- Original 55-image held-out average absolute **counting error per image** (classical → YOLO): RBC **15.673 → 5.018**, WBC **26.600 → 0.091**, platelets **78.855 → 0.400**. Classical baseline performs very poorly on this dataset and is provided for comparison only.
- On **20 held-out degraded low-resolution images**, RBC counting MAE is **5.20 without SR vs 4.65 with SR** (small improvement), platelet MAE is **0.55 vs 0.65** (worse), and WBC counting MAE is **0.10 vs 0.10**. On *clean* low-resolution images, RBC MAE **worsens** from 4.60 to 6.55 with SR.
- Degraded-image WBC detection recall (IoU 0.5 greedy match) drops from **1.00 to 0.895** after SR despite unchanged counting MAE; this illustrates why count-only evaluation is insufficient.
- In the recorded Colab run, mean degraded low-resolution image processing took **0.216 s without SR vs 1.143 s with SR**, calculated from `robustness_count_results.csv`. This does **not** establish inference time on Streamlit Community Cloud.

See `EXPERIMENT_RESULTS.md` and `MODEL_CARD.md` for the full interpretation. The `robustness_manifest.csv` includes *Colab absolute paths* for local reference: the actual original BCCD photos are not bundled, so those paths will not resolve after deployment. The `example_predictions.png` is an evidence figure containing model drawings, **not** a raw smear photo or ground-truth validation.

## 10 · Verification state and remaining submission tasks

- Model files are present, and both have executed via a local **OpenCV DNN ONNX smoke test** with correctly shaped outputs: YOLO `[1, 7, 8400]` and Real-ESRGAN 4×. This checks model parse and forward only.
- Python test suite and compile checks cover inference postprocessing, paired SR protocol, and CSV workflow.
- The deployed **ONNX Runtime + Streamlit UI path remains to be verified** in an environment that has those dependencies installed, then on Streamlit Community Cloud.
- Still required for the grading submission: a live public URL, tested desktop/mobile experience, a measured hosted CPU latency test, a 3–5 minute demo of three real cases including failure, the 3,000–4,000 word technical report, and the 10-minute presentation.
