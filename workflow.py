"""Real, user-triggered inference and review-report utilities."""
from __future__ import annotations
import io
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .inference import (CLASSES, read_image, quality_check, as_counts,
                        annotate, to_png, prepare_sr_input)
from .classical import detect_classical


def process_image(image_bytes: bytes, method: str, detector=None, enhancer=None,
                  compare_sr: bool = False, compare_classical: bool = False,
                  confidence: float = 0.25, nms_iou: float = 0.5):
    started_at = time.perf_counter()
    bgr = read_image(image_bytes)
    quality = quality_check(bgr)
    if method == 'YOLO11n':
        if detector is None:
            raise ValueError('Trained cellcount_yolo11n.onnx is missing; upload the Colab output ZIP.')
        primary, duration = detector.detect(bgr, confidence, nms_iou)
    elif method == 'Classical baseline':
        primary, duration = detect_classical(bgr)
    else:
        raise ValueError('Unsupported method')
    counts = as_counts(primary)
    if not primary:
        quality['status'] = 'NEEDS MANUAL COUNT'
        quality['reasons'].append('no cells detected')
    result = {
        'id': uuid.uuid4().hex[:12], 'method': method,
        'image_shape': tuple(bgr.shape), 'quality': quality,
        'counts': counts, 'time_seconds': round(duration,4),
        'annotated_png': to_png(annotate(bgr, primary)),
        'mean_confidence': round(float(np.mean([d.score for d in primary if d.score >= 0])),3)
           if any(d.score >= 0 for d in primary) else None,
        'details': [{'class': CLASSES[d.class_id], 'confidence': round(d.score, 4) if d.score >= 0 else None,
                    'x1': round(d.box[0],1), 'y1': round(d.box[1],1),
                    'x2': round(d.box[2],1), 'y2': round(d.box[3],1)} for d in primary],
        'sr': None, 'classical': None,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds')
    }
    if compare_sr:
        if enhancer is None or detector is None:
            raise ValueError('Both YOLO ONNX and Real-ESRGAN ONNX are needed for a super-resolution comparison.')
        small = prepare_sr_input(bgr)
        # Paired comparison: detector sees exactly the same 160x120 source,
        # either directly or after 4x super-resolution. Original-resolution
        # inference remains a separate reference, never the SR control arm.
        low_res_dets, low_res_seconds = detector.detect(small, confidence, nms_iou)
        enhanced, sr_seconds = enhancer.enhance(small)
        sr_dets, sr_detect_seconds = detector.detect(enhanced, confidence, nms_iou)
        result['sr'] = {
            'counts': as_counts(sr_dets), 'enhancement_seconds': round(sr_seconds,4),
            'detector_seconds': round(sr_detect_seconds,4),
            'annotated_png': to_png(annotate(enhanced, sr_dets)),
            'input_size': tuple(small.shape[:2]),
            'mean_confidence': round(float(np.mean([d.score for d in sr_dets])), 3) if sr_dets else None,
            'raw_low_res_counts': as_counts(low_res_dets),
            'raw_low_res_seconds': round(low_res_seconds,4),
            'raw_low_res_annotated_png': to_png(annotate(small, low_res_dets)),
        }
    if compare_classical and method == 'YOLO11n':
        baseline, baseline_seconds = detect_classical(bgr)
        result['classical'] = {
            'counts': as_counts(baseline),
            'seconds': round(baseline_seconds, 4),
            'annotated_png': to_png(annotate(bgr, baseline))
        }
    result['end_to_end_seconds'] = round(time.perf_counter() - started_at, 4)
    return result


def make_summary(results: list[dict], manual_counts: dict, approvals: dict) -> pd.DataFrame:
    records = []
    for index, r in enumerate(results, start=1):
        reviewed = manual_counts.get(r['id'], r['counts'])
        sr = r.get('sr') or {}
        row = {
            'sample_id': f'SAMPLE_{index:03d}', 'run_id': r['id'],
            'method': r['method'], 'image_width': r['image_shape'][1],
            'image_height': r['image_shape'][0], 'quality_status': r['quality']['status'],
            'quality_flags': '|'.join(r['quality']['reasons']),
            'technician_confirmed': bool(approvals.get(r['id'], False)),
            'inference_seconds': r['time_seconds'], 'end_to_end_seconds': r.get('end_to_end_seconds', np.nan), 'mean_detection_confidence': r['mean_confidence'],
            'enhancement_seconds': sr.get('enhancement_seconds', np.nan),
            'sr_detection_seconds': sr.get('detector_seconds', np.nan),
            'timestamp_utc': r['created_at'],
        }
        for cls in CLASSES:
            row[f'{cls}_detected'] = int(r['counts'][cls])
            row[f'{cls}_manual'] = int(reviewed[cls])
            row[f'{cls}_sr'] = sr.get('counts', {}).get(cls, np.nan)
        records.append(row)
    return pd.DataFrame(records)


def summarize_review_status(results: list[dict], approvals: dict) -> tuple[int, int]:
    return sum(bool(approvals.get(r['id'], False)) for r in results), len(results)
