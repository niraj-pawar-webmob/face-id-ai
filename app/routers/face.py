from typing import Optional, List, Dict
import json
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from ..config import MATCH_THRESHOLD, MAX_FACES_PER_FRAME
from ..schemas import EnrollResponse, MatchFace, MatchResponse
from ..utils import decode_image_from_input
from ..engine import engine_faces
from ..geometry import iou, landmarks_to_bbox, landmarks_convex_hull
from ..draw import draw_box, draw_text_bg, render_match_debug_image
from ..crud import upsert_person
from ..matcher import best_match

router = APIRouter()

@router.get("/health")
def health():
    return {"ok": True}

@router.post("/enroll", response_model=EnrollResponse, response_model_exclude_none=True)
async def enroll_person(
    person_id: str = Form(...),
    metadata: Optional[str] = Form(None),
    return_image: bool = Form(False),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    try:
        img = await decode_image_from_input(file, image_base64)
        dets = engine_faces.detect_and_embed(img, max_faces=1)
        if not dets:
            return JSONResponse({"ok": False, "person_id": person_id, "error": "No face found."}, status_code=400)

        emb = dets[0]["embedding"]; bbox = dets[0]["bbox"]
        md = json.loads(metadata) if metadata else {}
        upsert_person(person_id, md, emb)

        resp = {"ok": True, "person_id": person_id}
        if return_image:
            vis = img.copy()
            draw_box(vis, bbox, (0, 200, 0), 2)
            label = person_id
            if md:
                kv = "; ".join(f"{k}:{v}" for k, v in list(md.items())[:2])
                if kv: label += f" | {kv}"
            draw_text_bg(vis, label, (int(bbox[0]), int(bbox[1]) - 8), bg_color=(0,120,0))
            resp["bbox"] = bbox
            resp["debug_image_base64"] = render_match_debug_image(img, [], [{"bbox": bbox, "score": 1.0}])  # quick reuse
        return resp
    except Exception as e:
        return JSONResponse({"ok": False, "person_id": person_id, "error": str(e)}, status_code=400)

@router.post("/match", response_model=MatchResponse, response_model_exclude_none=True)
async def match_faces(
    threshold: Optional[float] = Form(None),
    return_image: bool = Form(False),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    th = float(threshold) if threshold is not None else MATCH_THRESHOLD
    try:
        img = await decode_image_from_input(file, image_base64)
        dets = engine_faces.detect_and_embed(img)
        all_landmarks = engine_faces.landmarks_for_all(img)
        lm_boxes = [landmarks_to_bbox(pts) for pts in all_landmarks]

        matches_out: List[MatchFace] = []
        unmatched_out: List[Dict[str, Any]] = []

        for det in dets:
            bbox = det["bbox"]; emb = det["embedding"]

            best_pts = None; best_i = 0.0
            for pts, box in zip(all_landmarks, lm_boxes):
                i = iou(bbox, box)
                if i > best_i:
                    best_i = i; best_pts = pts

            contour_pts = landmarks_convex_hull(best_pts) if best_pts is not None else []
            lm_count = int(best_pts.shape[0]) if best_pts is not None else 0

            pid, meta, score = best_match(emb, th)
            if pid is not None:
                matches_out.append(MatchFace(
                    person_id=pid, score=score, bbox=bbox,
                    contour=contour_pts, landmarks_count=lm_count, metadata=meta or {}
                ))
            else:
                unmatched_out.append({
                    "score": float(score), "bbox": bbox,
                    "contour": contour_pts, "landmarks_count": lm_count
                })

        response = {"matches": matches_out, "unmatched": unmatched_out}
        if return_image:
            response["debug_image_base64"] = render_match_debug_image(img, matches_out, unmatched_out)
        return response
    except Exception as e:
        return JSONResponse({"matches": [], "unmatched": [], "error": str(e)}, status_code=400)
