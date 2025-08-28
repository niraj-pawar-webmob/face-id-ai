from typing import List
import numpy as np
import cv2

def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

def landmarks_to_bbox(pts: np.ndarray) -> List[int]:
    mn = np.min(pts, axis=0); mx = np.max(pts, axis=0)
    x1, y1 = mn.astype(int).tolist(); x2, y2 = mx.astype(int).tolist()
    return [x1, y1, x2, y2]

def landmarks_convex_hull(pts: np.ndarray) -> list[list[int]]:
    if pts is None or len(pts) == 0:
        return []
    hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2)
    return hull.tolist()
