"""Safely load ONLY the named output artifacts from the Colab ZIP."""
from __future__ import annotations
import io
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import pandas as pd

DETECTOR = 'cellcount_yolo11n.onnx'
SUPERRESOLUTION = 'realesr_general_x4v3.onnx'
METRICS_CSVS = (
    'yolo_test_per_class_metrics.csv', 'classical_baseline_test.csv',
    'yolo_original_test_counts.csv', 'baseline_vs_yolo_mae.csv',
    'robustness_count_results.csv', 'robustness_iou_errors.csv',
    'robustness_detection_errors_aggregate.csv', 'paired_sr_effects.csv',
    'image_quality_flags.csv', 'robustness_manifest.csv',
)
METRICS_JSONS = ('yolo_test_overall.json', 'robustness_summary.json', 'split_manifest.json')
ALLOWED = set((DETECTOR, SUPERRESOLUTION) + METRICS_CSVS + METRICS_JSONS + ('README_EVIDENCE.md','example_predictions.png'))


def import_colab_bundle(zip_bytes: bytes, target: Path) -> list[str]:
    """Accept notebook output with 'artifacts/' paths, reject Zip Slip, only extract allowlisted names.

    Extraction is not a general-purpose ZIP extractor; patient images are NOT extracted.
    """
    target.mkdir(parents=True, exist_ok=True)
    imported = []
    try:
        with ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                p = Path(info.filename)
                if len(p.parts) != 2 or p.parts[0] != 'artifacts' or p.parts[1] not in ALLOWED:
                    continue
                limit = 100 * 1024**2 if p.suffix == '.onnx' else 20 * 1024**2
                if info.file_size > limit:
                    raise ValueError(f"Artifact too large: {p.name}")
                # Read a bounded amount, without trusting ZIP declared sizes.
                with zf.open(info) as src:
                    data = src.read(limit + 1)
                    if len(data) > limit:
                        raise ValueError(f"Artifact exceeds size limit: {p.name}")
                (target / p.name).write_bytes(data)
                imported.append(p.name)
    except BadZipFile as ex:
        raise ValueError('The supplied file is not a valid Colab output ZIP.') from ex
    if not imported:
        raise ValueError('No recognized artifacts in ZIP: expected artifacts/cellcount_yolo11n.onnx, etc.')
    return imported


def status(folder: Path):
    return {"detector": (folder / DETECTOR).is_file(), "super_resolution": (folder / SUPERRESOLUTION).is_file(),
            "test_results": (folder / 'yolo_test_overall.json').is_file()}


def get_csv(folder: Path, filename: str) -> pd.DataFrame | None:
    if filename not in METRICS_CSVS:
        raise ValueError('Unrecognized evidence file')
    p = folder / filename
    if not p.is_file():
        return None
    return pd.read_csv(p)


def get_json(folder: Path, filename: str):
    if filename not in METRICS_JSONS:
        raise ValueError('Unrecognized evidence file')
    p = folder / filename
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding='utf-8'))
