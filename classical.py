"""Classical color threshold + watershed baseline adapted from user's Colab notebook.

Heuristics are calibrated for BCCD-like 640x480 smear photos, not patient use.
"""
import time
import cv2
import numpy as np
from .inference import Detection


def detect_classical(image_bgr):
    start = time.perf_counter()
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    purple = cv2.inRange(hsv, (100, 40, 20), (178, 255, 225))
    purple = cv2.morphologyEx(purple, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(purple, 8)
    detections = []
    wbc_mask = np.zeros((h,w), np.uint8)
    for i in range(1, n):
        x,y,bw,bh,area = [int(z) for z in stats[i]]
        if 140 <= area <= 30000:
            pad = int(0.35 * max(bw, bh))
            box = (max(0,x-pad), max(0,y-pad), min(w,x+bw+pad), min(h,y+bh+pad))
            cv2.rectangle(wbc_mask, box[:2], box[2:], 255, -1)
            detections.append(Detection(1, -1, box))
        elif 4 <= area < 100 and 2 <= bw <= 20 and 2 <= bh <= 20:
            box = (max(0,x-2), max(0,y-2), min(w,x+bw+2), min(h,y+bh+2))
            detections.append(Detection(2, -1, box))
    achannel = cv2.GaussianBlur(lab[:,:,1], (5,5), 0)
    _, rbc = cv2.threshold(achannel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rbc = cv2.bitwise_and(rbc, cv2.bitwise_not(wbc_mask))
    rbc = cv2.morphologyEx(rbc, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    dist = cv2.distanceTransform(rbc, cv2.DIST_L2, 5)
    if dist.max() > 0:
        peaks = cv2.dilate(dist, np.ones((13,13), np.uint8))
        localmax = np.uint8((dist==peaks) & (dist > max(2,0.24*float(dist.max())))) * 255
        _, markers = cv2.connectedComponents(localmax)
        markers = markers.astype(np.int32) + 1
        markers[rbc == 0] = 1
        unknown = cv2.subtract(cv2.dilate(rbc, np.ones((3,3), np.uint8), iterations=2), localmax)
        markers[unknown>0] = 0
        ws = cv2.watershed(image_bgr.copy(), markers)
        for label in np.unique(ws):
            if label < 2: continue
            pixels = np.uint8(ws == label)
            area = int(pixels.sum())
            if 70 <= area <= 4000:
                yx = np.argwhere(pixels)
                y1,x1 = yx.min(axis=0); y2,x2 = yx.max(axis=0)+1
                ratio = (x2-x1)/max(1,y2-y1)
                if 0.35 <= ratio <= 2.8:
                    detections.append(Detection(0, -1, (int(x1),int(y1),int(x2),int(y2))))
    return detections, time.perf_counter() - start
