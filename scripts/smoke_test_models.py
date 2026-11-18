"""Local ONNX integrity smoke test with OpenCV DNN (not the deployed ONNX Runtime).

Usage: python scripts/smoke_test_models.py
Does not validate clinical performance or deploy the app.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Y = ROOT / 'artifacts' / 'cellcount_yolo11n.onnx'
SR = ROOT / 'artifacts' / 'realesr_general_x4v3.onnx'

if not (Y.is_file() and SR.is_file()):
    raise SystemExit('Missing staged ONNX model files in artifacts/.')

print('OpenCV version:', cv2.__version__)
print('YOLO bytes:', Y.stat().st_size)
print('Real-ESRGAN bytes:', SR.stat().st_size)

detector = cv2.dnn.readNetFromONNX(str(Y))
detector.setInput(np.zeros((1,3,640,640), dtype=np.float32))
pred = detector.forward()
assert tuple(pred.shape) == (1,7,8400), f'Unexpected detector output {pred.shape}'
assert np.isfinite(pred).all(), 'YOLO forward returned nonfinite values'
print('YOLO forward OK:', pred.shape)

upscaler = cv2.dnn.readNetFromONNX(str(SR))
upscaler.setInput(np.zeros((1,3,32,32), dtype=np.float32))
enhanced = upscaler.forward()
assert tuple(enhanced.shape) == (1,3,128,128), f'Unexpected SR output {enhanced.shape}'
assert np.isfinite(enhanced).all(), 'SR forward returned nonfinite values'
print('Real-ESRGAN forward OK:', enhanced.shape)
print('PASS: Both actual supplied ONNX models executed under OpenCV DNN.')
print('NEXT: Verify Streamlit + onnxruntime and real microscopy photos in the target deployment environment.')
