"""CellCount AI Studio | Streamlit deployment for MAIB AI221 Project 1.

Detection is powered ONLY by the user's trained ONNX model. When artifacts are
not yet supplied, the UI clearly identifies the missing model rather than
fabricating counts, scores or evaluation metrics.
"""
from __future__ import annotations
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from cellcount.artifacts import (
    DETECTOR, SUPERRESOLUTION, get_csv, get_json, import_colab_bundle, status
)
from cellcount.inference import CLASSES, YOLODetector, RealESRGAN
from cellcount.workflow import process_image, make_summary, summarize_review_status

BASE = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = BASE / 'artifacts'

st.set_page_config(page_title='CellCount AI Studio', page_icon='🩸', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family: 'DM Sans', Arial, sans-serif;}
[data-testid="stAppViewContainer"] {background: #f4f7fa;}
[data-testid="stSidebar"] {background:#10243c !important;}
[data-testid="stSidebar"] * {color:#edf8fc !important;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {background:rgba(255,255,255,.06);border-radius:12px;padding:4px;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color:#f4fcff !important;}
[data-testid="stSidebar"] .stRadio label {color:#edf8fc !important;}
.block-container {max-width:1450px;padding-top:1.5rem;padding-bottom:4rem;}
.banner {padding:29px 30px 25px;border-radius:22px;background:linear-gradient(115deg,#122944 0%,#183c52 65%,#117c87 100%);color:#fff;box-shadow:0 12px 32px rgba(16,39,67,.13);margin-bottom:22px;}
.banner .eyebrow {letter-spacing:2px;font-weight:700;font-size:11px;color:#7bede3;}
.banner h1 {color:#fff;font-size:clamp(28px,3.2vw,44px);font-weight:800;letter-spacing:-.5px;margin:8px 0 7px;}
.banner p {color:#d4eef1;font-size:14px;margin-bottom:0;max-width:840px;}
.eyebrow-dark {font-weight:800;letter-spacing:1.9px;font-size:11px;color:#187c87;margin-bottom:6px;}
.panel {border:1px solid #e4ebf1;border-radius:16px;background:#fff;padding:19px 22px;margin:9px 0 16px;}
.panel h3 {font-size:18px;margin-bottom:8px;color:#16334c;}
.panel p {color:#53697a;line-height:1.7;margin:0;}
.pill {display:inline-flex;padding:5px 10px;border-radius:30px;font-weight:750;font-size:11px;letter-spacing:.3px;}
.good {background:#d8faf1;color:#067666;}
.warn {background:#fff1d5;color:#946011;}
.err {background:#ffeded;color:#a6313d;}
.section-title {color:#122e49;font-weight:800;margin:16px 0 3px;}
.caption-muted {font-size:12px;color:#637789;margin-bottom:12px;}
[data-testid="stMetric"] {border:1px solid #e1ebf2;border-radius:14px;padding:15px 17px;background:#fff;min-height:115px;}
[data-testid="stMetricValue"] {font-weight:800;color:#15384e;}
div.stButton > button[kind="primary"] {background:#07a99a;border:none;border-radius:10px;color:#fff;font-weight:700;min-height:46px;}
div.stButton > button[kind="primary"]:hover {background:#067b72;}
[data-testid="stDownloadButton"] button {border-radius:9px;}
</style>''', unsafe_allow_html=True)


def init_state():
    defaults = {'results': [], 'manual_counts': {}, 'approvals': {}, 'active_folder': None}
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


@st.cache_resource(show_spinner='Loading trained YOLO11n ONNX model...')
def load_yolo(path: str, mtime_ns: int):
    return YOLODetector(Path(path))


@st.cache_resource(show_spinner='Loading compact Real-ESRGAN ONNX model...')
def load_sr(path: str, mtime_ns: int):
    return RealESRGAN(Path(path))


folder = Path(st.session_state.active_folder) if st.session_state.active_folder else DEFAULT_ARTIFACTS
available = status(folder)

with st.sidebar:
    st.markdown('### 🩸 CellCount AI Studio')
    st.caption('MAIB AI 221  /  Project 01  /  Healthcare')
    st.divider()
    page = st.radio('WORKSPACE', ('Analysis Lab', 'Technician Review', 'Evaluation Evidence', 'Model Card & Help'),
                    index=0, label_visibility='collapsed')
    st.divider()
    st.markdown('**Model readiness**')
    st.markdown('🟢 Trained YOLO11n' if available['detector'] else '🔴 YOLO11n ONNX missing')
    st.markdown('🟢 Real-ESRGAN' if available['super_resolution'] else '🟡 Real-ESRGAN ONNX missing')
    st.markdown('🟢 Evaluation results loaded' if available['test_results'] else '🟡 Measured evaluation not loaded')
    st.caption('ONNX inference runs on the free server. No external AI API or signup.')
    with st.expander('Import your completed Colab ZIP', expanded=not available['detector']):
        st.caption('Use the **cellcount_colab_outputs.zip** produced by the final notebook cell. Only models and evaluation evidence are extracted. Images are not stored.')
        evidence_zip = st.file_uploader('Colab output archive', type=['zip'], key='evidence_zip')
        if st.button('Activate Colab results', disabled=evidence_zip is None, use_container_width=True):
            target = Path(tempfile.gettempdir()) / f'cellcount_{uuid.uuid4().hex}'
            try:
                filenames = import_colab_bundle(evidence_zip.getvalue(), target)
                st.session_state.active_folder = str(target)
                # Prevent stale inference results from a previous model bundle.
                st.session_state.results = []
                st.session_state.manual_counts = {}
                st.session_state.approvals = {}
                st.success(f'Imported {len(filenames)} artifacts.')
                st.rerun()
            except Exception as ex:
                st.error(f'Could not import bundle: {ex}')
        st.caption('For permanent public deployment, copy the ONNX/CSV files into the repository artifacts/ folder. Temporary ZIP imports are session-only.')
    st.divider()
    st.caption('EDUCATIONAL TOOL · NOT FOR CLINICAL DIAGNOSIS')

st.markdown('''<div class="banner">
<div class="eyebrow">HEALTHCARE  /  COMPUTER VISION  /  GENERATIVE AI</div>
<h1>CellCount <span style="color:#68e5d5">AI Studio</span></h1>
<p>Blood-smear cell detection and technician review. Measure RBCs, WBCs and platelets with a trained YOLO11n model, compare against a classical baseline, and assess super-resolution responsibly.</p>
</div>''', unsafe_allow_html=True)


def badge(ok, yes='READY', no='ACTION REQUIRED'):
    return f'<span class="pill {"good" if ok else "warn"}">{yes if ok else no}</span>'


def need_model_prompt():
    st.warning('**Trained ONNX model not found.** Import your Colab output ZIP in the left sidebar or place `cellcount_yolo11n.onnx` inside `artifacts/`. You can still preview the classical baseline, but results are not the trained detector.')


def evidence_error(e):
    st.error(f'Could not read evaluation evidence: {e}. Check the Colab ZIP and retry.')


if page == 'Analysis Lab':
    st.markdown('#### Image Analysis Lab')
    st.caption('Upload non-identifiable microscopic smear images. Every automated result requires laboratory technician verification.')
    a,b,c,d = st.columns(4)
    run_results = st.session_state.results
    with a: st.metric('Processed images', len(run_results))
    with b: st.metric('RBC detections', sum(r['counts']['RBC'] for r in run_results))
    with c: st.metric('WBC detections', sum(r['counts']['WBC'] for r in run_results))
    with d: st.metric('Platelet detections', sum(r['counts']['Platelets'] for r in run_results))
    left, right = st.columns([1,1.15], gap='large')
    with left:
        st.markdown('**01 / Input images**')
        images = st.file_uploader('Upload JPG, JPEG or PNG images', type=['jpg','jpeg','png'],
                                  accept_multiple_files=True, help='Up to 10 images per batch, 12 megapixels/image. No patient identifiers.')
        if images and len(images) > 10:
            st.error('Please upload a maximum of 10 images per batch.')
        st.markdown('**02 / Inference configuration**')
        methods = ['YOLO11n'] if available['detector'] else []
        methods.append('Classical baseline')
        method = st.selectbox('Detection method', methods,
                              help='YOLO11n uses the fine-tuned Colab ONNX weights; classical baseline uses OpenCV heuristics.')
        if not available['detector']:
            need_model_prompt()
        conf = st.slider('Detection confidence threshold', 0.05, 0.90, 0.25, 0.05,
                         disabled=(method != 'YOLO11n'))
        nms = st.slider('NMS overlap threshold (IoU)', 0.20, 0.85, 0.50, 0.05,
                        disabled=(method != 'YOLO11n'))
        compare_sr = st.checkbox('Compare Real-ESRGAN 4× super-resolution', value=False,
                                 disabled=not (available['super_resolution'] and available['detector'] and method=='YOLO11n'),
                                 help='Uses a 160×120 input before 4× enhancement, as in the Colab robustness experiment. Not a clinical enhancement.')
        compare_classical = st.checkbox('Compare classical baseline against YOLO', value=False,
                                        disabled=(method != 'YOLO11n'))
        run = st.button('▶  Run real image analysis', type='primary', use_container_width=True,
                        disabled=not images or len(images)>10)
    with right:
        st.markdown('**Pipeline status**')
        st.markdown(f'''<div class="panel"><h3>Detection runtime {badge(available['detector'],'ONNX MODEL FOUND','BASELINE ONLY')}</h3>
<p>1 · Upload & quality check &nbsp; → &nbsp; 2 · YOLO / OpenCV &nbsp; → &nbsp; 3 · Cell counting &nbsp; → &nbsp; 4 · Technician verification<br/><br/>
<b>Optional experiment:</b> 160×120 input → Real-ESRGAN 4× → YOLO. Count changes alone do not prove accuracy gains or hallucinations.</p></div>''', unsafe_allow_html=True)
        st.info('**Privacy:** Processing happens on the app server. Do not upload identifiable patient images or sensitive clinical data. No images are deliberately saved by the app, but hosting providers may log operational metadata.')
        if run_results:
            reviewed,total = summarize_review_status(run_results, st.session_state.approvals)
            st.metric('Confirmed by technician', f'{reviewed}/{total}')
            if st.button('Open technician review →', use_container_width=True):
                st.info('Select **Technician Review** in the left sidebar to approve and export results.')
    if run:
        detector = None
        enhancer = None
        if method == 'YOLO11n' or compare_sr:
            model_path = folder / DETECTOR
            try:
                detector = load_yolo(str(model_path), model_path.stat().st_mtime_ns)
            except Exception as ex:
                st.error(f'YOLO model failed to load: {ex}. Make sure you uploaded the trained ONNX model.')
                st.stop()
        if compare_sr:
            sr_path = folder / SUPERRESOLUTION
            try:
                enhancer = load_sr(str(sr_path), sr_path.stat().st_mtime_ns)
            except Exception as ex:
                st.error(f'Real-ESRGAN model failed to load: {ex}')
                st.stop()
        new_results = []
        progress = st.progress(0, text='Beginning real ONNX/OpenCV processing...')
        for index, upload in enumerate(images):
            try:
                value = process_image(upload.getvalue(), method, detector=detector,
                                      enhancer=enhancer, compare_sr=compare_sr,
                                      compare_classical=compare_classical,
                                      confidence=conf, nms_iou=nms)
                value['sample_label'] = f'Sample {index+1:02d}'
                new_results.append(value)
            except Exception as ex:
                st.error(f'Image {index+1} could not be processed: {ex}')
            progress.progress((index+1)/len(images), text=f'Processed {index+1} / {len(images)} images')
        st.session_state.results = new_results
        st.session_state.manual_counts = {}
        st.session_state.approvals = {}
        st.success(f'Finished real processing: {len(new_results)} of {len(images)} image(s). Review results below.') if new_results else st.error('No images were processed successfully.')
    if st.session_state.results:
        st.markdown('### Latest run · Actual detections')
        selected = st.selectbox('Choose an analyzed image',
                                range(len(st.session_state.results)),
                                format_func=lambda k: st.session_state.results[k]['sample_label'])
        r = st.session_state.results[selected]
        st.markdown(f"**{r['sample_label']} · {r['method']}** — {r['quality']['status']}")
        if r['quality']['reasons']:
            st.warning('Manual count recommended: ' + ', '.join(r['quality']['reasons']))
        cols = st.columns(4)
        for i, cls in enumerate(CLASSES):
            cols[i].metric(cls, r['counts'][cls])
        cols[3].metric('WBC / RBC ratio', f"{r['counts']['WBC'] / r['counts']['RBC']:.4f}" if r['counts']['RBC'] else 'N/A')
        st.caption('The WBC/RBC value is a ratio within this photographed field only; it is NOT a complete blood count or a clinical reference measurement.')
        col_img, col_info = st.columns([1.5,1], gap='large')
        with col_img:
            st.image(r['annotated_png'], caption=f"{r['sample_label']} · model detections · colored bounding boxes", use_container_width=True)
            st.download_button('Download annotated image (PNG)', r['annotated_png'],
                               file_name=f"{r['sample_label'].replace(' ','_').lower()}_detections.png", mime='image/png')
        with col_info:
            st.markdown('**Run diagnostics**')
            st.metric('End-to-end processing time', f"{r['end_to_end_seconds']:.3f} s")
            st.caption('Measured in this app session, including model and optional comparison overhead. Deploy and retest on Streamlit Community Cloud.')
            st.metric('Detector / baseline time', f"{r['time_seconds']:.3f} s")
            st.metric('Mean confidence' if r['mean_confidence'] is not None else 'Confidence unavailable',
                      f"{r['mean_confidence']:.1%}" if r['mean_confidence'] is not None else 'Classical heuristic')
            st.caption(f"Brightness: {r['quality']['brightness']} · Sharpness variance: {r['quality']['sharpness_var']}")
            st.write('**RBC** / **WBC** / **Platelets** are highlighted with different colors. Bounding boxes are suggestions to review, not verified labels.')
        if r.get('sr'):
            st.markdown('#### Generative AI A/B comparison · same source at 160 × 120')
            st.caption('Fair comparison: low-resolution image → YOLO11n **without SR** versus the same low-resolution image → Real-ESRGAN 4× → YOLO11n **with SR**. Full-resolution detections above are a separate reference.')
            z1, z2 = st.columns(2)
            with z1:
                st.image(r['sr']['raw_low_res_annotated_png'], caption='160 × 120 input → YOLO11n · no enhancement', use_container_width=True)
                st.write('Without SR: ' + ', '.join(f"{k}: {v}" for k, v in r['sr']['raw_low_res_counts'].items()))
                st.caption(f"Detection: {r['sr']['raw_low_res_seconds']:.3f}s")
            with z2:
                st.image(r['sr']['annotated_png'], caption='Same 160 × 120 input → Real-ESRGAN 4× → YOLO11n', use_container_width=True)
                st.write('With SR: ' + ', '.join(f"{k}: {v}" for k, v in r['sr']['counts'].items()))
                st.caption(f"Enhancement: {r['sr']['enhancement_seconds']:.3f}s + detection: {r['sr']['detector_seconds']:.3f}s")
            st.dataframe(pd.DataFrame([{
                'Class': cls, 'Low-res raw': r['sr']['raw_low_res_counts'][cls],
                'Low-res + SR': r['sr']['counts'][cls],
                'Change (SR − raw)': r['sr']['counts'][cls] - r['sr']['raw_low_res_counts'][cls]
            } for cls in CLASSES]), hide_index=True, use_container_width=True)
            st.warning('No ground-truth labels accompany this uploaded image. A changed count alone is not evidence of improved accuracy or hallucination. Refer to the measured held-out study in Evaluation Evidence.')
        if r.get('classical'):
            st.markdown('#### Baseline comparison')
            z1, z2 = st.columns(2)
            with z1:
                st.image(r['annotated_png'], caption='Trained YOLO11n', use_container_width=True)
            with z2:
                st.image(r['classical']['annotated_png'], caption='OpenCV threshold + watershed baseline', use_container_width=True)
            st.dataframe(pd.DataFrame([{'Cell': cls, 'YOLO': r['counts'][cls],
                                        'Classical': r['classical']['counts'][cls]} for cls in CLASSES]), hide_index=True)
        with st.expander('Individual cell detection coordinates and confidence'):
            st.dataframe(pd.DataFrame(r['details']), use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="panel"><h3>No images analyzed yet</h3><p>Use the uploader above and click Run real image analysis to obtain results. This application never displays fabricated model predictions.</p></div>', unsafe_allow_html=True)

elif page == 'Technician Review':
    st.markdown('### Human-in-the-Loop Review')
    st.caption('Review every result, adjust counts where needed, and explicitly confirm. Exports contain anonymous sample identifiers only.')
    results = st.session_state.results
    if not results:
        st.info('No results in this session. Open Analysis Lab and run the model on at least one image.')
    else:
        approved, total = summarize_review_status(results, st.session_state.approvals)
        a, b, c = st.columns(3)
        a.metric('Images analyzed', total)
        b.metric('Technician confirmations', approved)
        c.metric('Awaiting confirmation', total - approved)
        for i, r in enumerate(results, start=1):
            with st.expander(f"SAMPLE_{i:03d}   •   {r['quality']['status']}   •   {r['method']}", expanded=i==1):
                image_col, review_col = st.columns([1.1,1])
                with image_col:
                    st.image(r['annotated_png'], use_container_width=True)
                    if r['quality']['reasons']:
                        st.warning('Quality flags: ' + ', '.join(r['quality']['reasons']))
                    st.caption('Cell-level detections are indicative; assess the raw smear manually if uncertain.')
                with review_col:
                    st.markdown('**Review observed counts**')
                    past = st.session_state.manual_counts.get(r['id'], r['counts'])
                    data = pd.DataFrame([
                        {'Cell class': cls, 'Model count': r['counts'][cls], 'Technician count': int(past[cls])}
                        for cls in CLASSES
                    ])
                    edited = st.data_editor(data, use_container_width=True, hide_index=True,
                        disabled=['Cell class','Model count'],
                        column_config={'Technician count': st.column_config.NumberColumn(
                            'Technician count', min_value=0, max_value=10000, step=1, required=True)},
                        key=f"edit_{r['id']}")
                    corrected = {str(row['Cell class']): int(row['Technician count']) for _,row in edited.iterrows()}
                    st.session_state.manual_counts[r['id']] = corrected
                    confirm = st.checkbox('I have manually reviewed these counts',
                                key=f"approval_{r['id']}", value=st.session_state.approvals.get(r['id'], False))
                    st.session_state.approvals[r['id']] = bool(confirm)
                    st.caption('Confirmation must be performed by an appropriate lab trainee or technician. It is not a medical diagnosis.')
        summary = make_summary(results, st.session_state.manual_counts, st.session_state.approvals)
        st.markdown('#### Anonymous batch summary')
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button('⬇ Download reviewed CSV report', summary.to_csv(index=False).encode('utf-8'),
                           file_name='cellcount_technician_review.csv', mime='text/csv', use_container_width=True,
                           help='Exports automatic counts, technician-adjusted counts, quality flags and manual approval status.')
        st.caption('If any image is still unconfirmed, its `technician_confirmed` field remains false. No patient identifiers or original image filenames are included.')

elif page == 'Evaluation Evidence':
    st.markdown('### Evidence & Generalization')
    st.caption('All scores below were measured in the supplied Colab run. These are BCCD test-set results, not clinical validation and not guaranteed results on arbitrary images.')
    try:
        overview = get_json(folder, 'yolo_test_overall.json')
        cls_table = get_csv(folder, 'yolo_test_per_class_metrics.csv')
        count_df = get_csv(folder, 'robustness_count_results.csv')
        classical_df = get_csv(folder, 'classical_baseline_test.csv')
        yolo_orig_df = get_csv(folder, 'yolo_original_test_counts.csv')
        err_df = get_csv(folder, 'robustness_detection_errors_aggregate.csv')
        split = get_json(folder, 'split_manifest.json')
    except Exception as ex:
        evidence_error(ex)
        st.stop()
    if overview is None:
        st.warning('Measured evaluation files are missing. Import your completed Colab ZIP to display test-set mAP and robustness metrics.')
    else:
        a,b,c = st.columns(3)
        a.metric('Held-out mAP@0.5', f"{overview['mAP50']:.3f}")
        b.metric('Held-out mAP@0.5:0.95', f"{overview['mAP50_95']:.3f}")
        c.metric('Training epochs requested', str(overview.get('epochs_requested', '—')))
        st.caption('Values are from the notebook model validation step. Epochs requested are not necessarily epochs actually completed (e.g., early stopping).')
        st.success('**Measured research result:** YOLO11n substantially reduces average counting error compared with the classical baseline on the 55 held-out BCCD images. Super-resolution had mixed effects: it helped degraded-image RBC counting slightly but worsened platelet error and RBC counting on clean low-resolution images. Do not claim universal improvement.')
    if split is not None:
        st.markdown('#### Dataset split (image disjoint)')
        splits = split.get('splits', {})
        c1,c2,c3 = st.columns(3)
        c1.metric('Train', len(splits.get('train', [])))
        c2.metric('Validation', len(splits.get('val', [])))
        c3.metric('Held-out test', len(splits.get('test', [])))
    if cls_table is not None:
        st.markdown('#### YOLO11n per-class precision, recall and AP50')
        st.dataframe(cls_table, use_container_width=True, hide_index=True)
        if {'class', 'AP50'}.issubset(cls_table.columns):
            st.bar_chart(cls_table.set_index('class')[['AP50']])
    if classical_df is not None and yolo_orig_df is not None:
        st.markdown('#### Original held-out images: baseline vs YOLO11n count error')
        comparison = pd.concat([classical_df,yolo_orig_df], ignore_index=True)
        clean_metrics = comparison.groupby(['condition','class'])['absolute_error'].mean().reset_index()
        st.dataframe(clean_metrics.rename(columns={'absolute_error':'count_MAE'}), use_container_width=True, hide_index=True)
        st.bar_chart(clean_metrics.pivot(index='class',columns='condition',values='absolute_error'))
    if count_df is not None:
        st.markdown('#### Paired robustness experiment · 20 held-out images × four conditions')
        if {'condition','class','absolute_error'}.issubset(count_df.columns):
            mae = count_df.groupby(['condition','class'])['absolute_error'].mean().reset_index()
            st.dataframe(mae.rename(columns={'absolute_error':'count_MAE'}), use_container_width=True, hide_index=True)
            st.bar_chart(mae.pivot(index='class',columns='condition',values='absolute_error'))
            st.caption('Mean absolute counting error (MAE): lower is better. Clean/degraded images were compared raw vs 4× Real-ESRGAN.')
            if {'stem','detector_seconds','enhancement_seconds'}.issubset(count_df.columns):
                timing = count_df.drop_duplicates(['stem','condition']).copy()
                timing['total_seconds'] = timing['detector_seconds'] + timing['enhancement_seconds']
                runtime = timing.groupby('condition')['total_seconds'].agg(['mean','max']).reset_index()
                st.markdown('**Colab inference time (seconds/image, end-to-end)**')
                st.dataframe(runtime.round(3), hide_index=True, use_container_width=True)
                st.caption('Timing measurements were captured in Colab, not on Streamlit Community Cloud. Re-test after public deployment against the professor’s roughly 5-second target.')
            pivot = mae.pivot(index='class',columns='condition',values='absolute_error')
            if {'degraded_lr_raw','degraded_lr_sr'}.issubset(pivot.columns):
                effect = (pivot['degraded_lr_raw'] - pivot['degraded_lr_sr']).rename('MAE reduction from SR')
                st.markdown('**Per-class effect of enhancement on degraded images**')
                st.dataframe(effect.to_frame(), use_container_width=True)
                st.caption('Positive reduction indicates improvement; negative means enhancement made counting worse.')
    if err_df is not None:
        st.markdown('#### Approximate IoU-matched detection errors (TP / FP / FN)')
        st.dataframe(err_df, use_container_width=True, hide_index=True)
        st.caption('These detection errors use greedy confidence-ordered IoU matching from the Colab experiment, not the official mAP implementation.')
    example_image = folder / 'example_predictions.png'
    if example_image.is_file():
        with st.expander('Visual evidence · predictions on difficult images'):
            st.image(str(example_image), caption='Actual Colab output: raw/degraded vs Real-ESRGAN-enhanced detections. Annotations shown are model predictions, not confirmed ground truth.', use_container_width=True)
            st.caption('Inspect changes in predicted boxes manually and cross-reference IoU-matched false positives. The figure alone cannot prove hallucinations.')
    if any([overview is not None, count_df is not None, cls_table is not None]):
        st.markdown('#### Download supporting research evidence')
        for name in ['yolo_test_per_class_metrics.csv', 'robustness_count_results.csv',
                     'robustness_detection_errors_aggregate.csv', 'yolo_original_test_counts.csv',
                     'classical_baseline_test.csv']:
            file = folder / name
            if file.is_file():
                st.download_button(f'⬇ {name}', file.read_bytes(), file_name=name, mime='text/csv', key=f'dl_{name}')
    st.info('**Interpretation:** A change in cell count after super-resolution is not automatically an improvement. Compare with withheld ground truth and check potential false positives visually. This is research on BCCD, not proof of clinical validity.')

elif page == 'Model Card & Help':
    st.markdown('### Model Card & Operating Guide')
    st.markdown('**01 · Intended use**')
    st.write('Research and educational blood-smear cell counting by technicians or trainees. Classes: RBC, WBC, and platelets. Bounding boxes and counts are estimates requiring review.')
    st.markdown('**02 · Out-of-scope use**')
    st.error('Do NOT use this system to diagnose disease, guide treatment, interpret blood disorders, replace clinical analyzers or perform autonomous medical decisions. WBC/RBC in one image is not a patient CBC result.')
    st.markdown('**03 · Models & data**')
    st.write('BCCD annotated blood-smear dataset (364 images, MIT license); classical thresholding + watershed baseline; YOLO11n adapted on disjoint BCCD training images and exported to ONNX; Real-ESRGAN general x4v3 ONNX, used experimentally at 160×120 input resolution.')
    st.markdown('**04 · Responsible AI safeguards**')
    st.write('Quality flags for blurred/dark images, no-cells fallback, mandatory manual confirmation workflow, anonymous export IDs, no uploaded images intentionally saved, no external API, and per-class robustness reporting where experiment results exist.')
    st.markdown('**05 · Known limitations**')
    st.write('Platelets are small and may be missed; overlapping/clumped cells and stain variation can distort estimates. Class imbalance and the limited BCCD domain reduce generalizability. Thresholds for image rejection are heuristics, not calibrated clinical safety thresholds. Super-resolution may introduce misleading image features.')
    st.markdown('**06 · Quick start**')
    st.markdown('''1. In Colab, run the entire notebook through the final **Export evidence** cell.
2. Download `cellcount_colab_outputs.zip` and import it from the left sidebar (temporary session), or extract its `artifacts/` contents into this repository's `artifacts/` directory (permanent deployment).
3. In **Analysis Lab**, choose trained YOLO11n, upload images and click Run real image analysis.
4. Compare original and generative enhancement where available; use the Evidence page for actual measured test results.
5. In **Technician Review**, adjust the counts, confirm each sample, and download a CSV.
6. Deploy `app.py` via Streamlit Community Cloud, linking a public GitHub repository containing your model artifacts.''')
    st.markdown('**07 · Licences and distribution**')
    st.write('BCCD: MIT (verify original dataset licence); Real-ESRGAN: BSD-3-Clause; Ultralytics training software: AGPL-3.0. Include original notices and review model redistribution terms. Do not publish patient images.')
    try:
        overall = get_json(folder,'yolo_test_overall.json')
        class_results = get_csv(folder, 'yolo_test_per_class_metrics.csv')
        robustness = get_csv(folder, 'robustness_count_results.csv')
    except Exception:
        overall = class_results = robustness = None
    m50 = f"{overall['mAP50']:.4f}" if overall else 'Not provided: run Colab evaluation'
    clines = class_results.to_string(index=False) if class_results is not None else 'Not provided'
    robustlines = (robustness.groupby(['condition','class'])['absolute_error'].mean().round(3).to_string()
                   if robustness is not None else 'Not provided')
    card = f'''# Model Card — CellCount AI Studio

**Intended use:** Educational lab cell detection and technician decision support only. Not diagnostic.

**Detector:** Fine-tuned YOLO11n ONNX; three classes (RBC, WBC, Platelets); confidence default 0.25; IoU 0.5.

**Data:** BCCD, MIT license; image-disjoint 70/15/15 train/val/test split with seed 42.

**Held-out mAP@0.5:** {m50}

**Per-class metrics:**\n```text\n{clines}\n```

**Robustness count MAE (20 paired degraded held-out samples, optional super-resolution):**\n```text\n{robustlines}\n```

**Known failure modes:** Overlapping and clumped cells, small platelets, variable stain, blurred and dim inputs, unrealistic changes from super-resolution.

**Human oversight:** All predictions reviewed; manual adjustments and confirmation recorded in anonymized CSV. No patient identifiers.

**Fairness/domain check:** BCCD is a narrow microscope-image domain; subgroup and device-specific generalization to clinical workflows is not established.

**Privacy:** Do not upload identifiable or sensitive patient data to public hosted apps.

**Licensing:** BCCD MIT; Real-ESRGAN BSD-3-Clause; Ultralytics AGPL-3.0. Confirm ONNX conversion distribution terms.
'''
    st.download_button('⬇ Download evidence-backed model card (.md)',card,
                       file_name='CellCount_Model_Card.md',mime='text/markdown')

st.divider()
st.caption('CellCount AI Studio · MAIB AI 221 Final Project · Research/educational decision support only · No paid APIs · Manual confirmation required')
