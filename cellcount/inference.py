"""Small, CPU-oriented ONNX implementations for the exact models exported in CellCount Colab.

No Ultralytics/PyTorch runtime dependency. YOLO11 postprocessing assumes exported
raw output of shape (1, 4 + nc, N), as confirmed by the project's notebook.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CLASSES = ("RBC", "WBC", "Platelets")
PALETTE_BGR = ((172, 181, 33), (229, 128, 91), (76, 163, 237))


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    box: tuple[float, float, float, float]


def letterbox(bgr: np.ndarray, size: int = 640):
    """Ultralytics-style letterbox with centered gray padding."""
    h, w = bgr.shape[:2]
    if min(h, w) <= 0:
        raise ValueError("Empty image dimensions")
    r = min(size / w, size / h)
    new_w, new_h = round(w * r), round(h * r)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (w, h) != (new_w, new_h) else bgr
    dw = size - new_w
    dh = size - new_h
    left, right = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
    top, bottom = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return blob, r, left, top


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def classwise_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                  iou_threshold: float = 0.5, max_det: int = 300) -> np.ndarray:
    """Return original indices, with independent suppression within each class."""
    picked = []
    for cls in np.unique(labels):
        ids = np.flatnonzero(labels == cls)
        order = ids[np.argsort(scores[ids])[::-1]]
        while order.size:
            current = int(order[0]); picked.append(current)
            if order.size == 1:
                break
            overlaps = box_iou_one_to_many(boxes[current], boxes[order[1:]])
            order = order[1:][overlaps <= iou_threshold]
    picked = sorted(picked, key=lambda i: scores[i], reverse=True)
    return np.asarray(picked[:max_det], dtype=np.int64)


def decode_yolo(raw: np.ndarray, ratio: float, left: int, top: int,
                original_shape: tuple[int, int], confidence: float = 0.25,
                nms_iou: float = 0.5, num_classes: int = 3,
                max_det: int = 300) -> list[Detection]:
    """Decode raw YOLO11 output (xywh + 3 class probabilities, no objectness)."""
    arr = np.asarray(raw)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Unexpected YOLO tensor shape {arr.shape}; expected (1, 7, N).")
    candidates = arr[0]
    nc_plus4 = 4 + num_classes
    if candidates.shape[0] == nc_plus4:
        candidates = candidates.T
    elif candidates.shape[1] == nc_plus4:
        pass
    else:
        raise ValueError(f"Unsupported YOLO output {arr.shape}; expected 3-class (1,7,N) or (1,N,7).")
    if candidates.shape[0] == 0:
        return []
    scores_by_class = candidates[:, 4:4+num_classes]
    labels = np.argmax(scores_by_class, axis=1).astype(np.int64)
    scores = scores_by_class[np.arange(len(labels)), labels]
    selection = np.isfinite(scores) & (scores >= confidence)
    xywh = candidates[selection, :4]
    labels = labels[selection]; scores = scores[selection]
    if not len(scores):
        return []
    boxes = np.stack((xywh[:, 0]-xywh[:, 2]/2, xywh[:, 1]-xywh[:, 3]/2,
                      xywh[:, 0]+xywh[:, 2]/2, xywh[:, 1]+xywh[:, 3]/2), axis=1)
    # Recover original image coordinates after letterbox padding.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
    h, w = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & np.isfinite(boxes).all(axis=1)
    boxes = boxes[valid]; scores = scores[valid]; labels = labels[valid]
    if not len(scores):
        return []
    # Avoid quadratic work with extremely low thresholds.
    if len(scores) > 3000:
        top_ids = np.argsort(scores)[-3000:]
        boxes = boxes[top_ids]; scores = scores[top_ids]; labels = labels[top_ids]
    keep = classwise_nms(boxes, scores, labels, nms_iou, max_det)
    return [Detection(int(labels[i]), float(scores[i]), tuple(float(v) for v in boxes[i])) for i in keep]


def _ort_session(path: Path, cpu_threads: int = 2):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = cpu_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class YOLODetector:
    def __init__(self, model_path: Path):
        self.session = _ort_session(model_path)
        ins = self.session.get_inputs()
        if len(ins) != 1:
            raise ValueError(f"Expected one YOLO input, got {len(ins)}")
        self.input_name = ins[0].name
        shape = ins[0].shape
        self.size = int(shape[-1]) if isinstance(shape[-1], int) else 640
        if isinstance(shape[-2], int) and self.size != int(shape[-2]):
            raise ValueError("This app expects square YOLO11 input size.")

    def detect(self, image_bgr: np.ndarray, confidence: float = 0.25, iou: float = 0.5):
        blob, ratio, left, top = letterbox(image_bgr, self.size)
        start = time.perf_counter()
        raw = self.session.run(None, {self.input_name: blob})[0]
        pred = decode_yolo(raw, ratio, left, top, image_bgr.shape[:2], confidence, iou)
        return pred, time.perf_counter() - start


class RealESRGAN:
    """Compact general-x4v3 exported to ONNX, in RGB [0,1]."""
    def __init__(self, model_path: Path):
        self.session = _ort_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def enhance(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = np.ascontiguousarray(rgb.transpose(2,0,1)[None], dtype=np.float32) / 255.0
        start = time.perf_counter()
        result = self.session.run(None, {self.input_name: x})[0]
        elapsed = time.perf_counter() - start
        if result.ndim != 4 or result.shape[0] != 1 or result.shape[1] != 3:
            raise ValueError(f"Unexpected super-resolution output shape: {result.shape}")
        out = np.uint8(np.clip(result[0].transpose(1,2,0), 0, 1)*255 + 0.5)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR), elapsed


def read_image(contents: bytes, max_pixels: int = 12_000_000) -> np.ndarray:
    arr = np.frombuffer(contents, dtype=np.uint8)
    out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if out is None:
        raise ValueError("Unsupported or corrupted image.")
    if out.shape[0] * out.shape[1] > max_pixels:
        raise ValueError("Image exceeds the 12-megapixel size limit. Resize before uploading.")
    return out


def quality_check(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    reasons = []
    if sharpness < 40: reasons.append("low sharpness")
    if brightness < 55: reasons.append("too dark")
    if brightness > 235: reasons.append("overexposed")
    if min(bgr.shape[:2]) < 120: reasons.append("very low resolution")
    return {"status": "NEEDS MANUAL COUNT" if reasons else "REVIEW REQUIRED",
            "sharpness_var": round(sharpness, 2), "brightness": round(brightness, 2), "reasons": reasons}


def as_counts(detections: list[Detection]):
    return {name: sum(d.class_id == i for d in detections) for i, name in enumerate(CLASSES)}


def annotate(bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    view = bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        col = PALETTE_BGR[d.class_id]
        cv2.rectangle(view, (x1, y1), (x2, y2), col, 2)
        label = f"{CLASSES[d.class_id]} {d.score:.2f}" if d.score >= 0 else CLASSES[d.class_id]
        cv2.putText(view, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.41, col, 1, cv2.LINE_AA)
    return view


def to_png(bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', bgr)
    if not ok:
        raise ValueError("Could not encode annotated PNG.")
    return encoded.tobytes()


def prepare_sr_input(bgr: np.ndarray) -> np.ndarray:
    """Replicate notebook robustness protocol: 160x120 before 4x enhancement.

    Deliberate: does not show 4x enhancements as equivalent to high-res originals.
    """
    return cv2.resize(bgr, (160, 120), interpolation=cv2.INTER_AREA)
