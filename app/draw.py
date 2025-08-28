import numpy as np
import cv2
from typing import Any, Dict, List
from .utils import encode_jpeg_base64

def draw_box(img, box, color=(0, 200, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_text_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.6, text_color=(255, 255, 255),
                 bg_color=(0, 0, 0), thickness=1, pad=3):
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x = max(0, x); y = max(th + pad, y)
    cv2.rectangle(img, (x, y - th - 2*pad), (x + tw + 2*pad, y + pad), bg_color, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, font_scale, text_color, thickness, cv2.LINE_AA)

def _as_dict(model_or_dict):
    if hasattr(model_or_dict, "model_dump"):
        return model_or_dict.model_dump()
    if hasattr(model_or_dict, "dict"):
        return model_or_dict.dict()
    return model_or_dict

def render_match_debug_image(img_bgr, matches, unmatched):
    vis = img_bgr.copy()
    # matched: green
    for m in matches:
        md = _as_dict(m)
        box = md.get("bbox", [])
        draw_box(vis, box, (0, 200, 0), 2)
        pid = md.get("person_id", "id?")
        score = md.get("score", 0.0)
        meta = md.get("metadata", {}) or {}
        kv = "; ".join(f"{k}:{v}" for k, v in list(meta.items())[:3])
        label = f"{pid} | {score:.2f}" + (f" | {kv}" if kv else "")
        draw_text_bg(vis, label, (int(box[0]), int(box[1]) - 8), bg_color=(0,120,0))
        contour = md.get("contour") or []
        if contour:
            pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0,200,0), 1)
    # unmatched: red
    for u in unmatched:
        ud = _as_dict(u)
        box = ud.get("bbox", [])
        draw_box(vis, box, (0, 0, 220), 2)
        score = ud.get("score", 0.0)
        draw_text_bg(vis, f"unmatched | {score:.2f}", (int(box[0]), int(box[1]) - 8), bg_color=(0,0,120))
        contour = ud.get("contour") or []
        if contour:
            pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0,0,220), 1)
    return encode_jpeg_base64(vis)
