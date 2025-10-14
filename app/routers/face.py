import time
from typing import Optional, List, Dict
import json
from fastapi import APIRouter, File, UploadFile, Form
# NOTE: we no longer return JSONResponse with 400; just return dicts (HTTP 200)
from ..config import MATCH_THRESHOLD, MAX_FACES_PER_FRAME
from ..schemas import EnrollResponse, MatchFace, MatchResponse
from ..utils import decode_image_from_input, resize_for_inference
from ..engine import engine_faces
from ..geometry import landmarks_convex_hull
from ..draw import render_match_debug_image
from ..crud import upsert_person, person_exists
from ..matcher import best_match_in_session
from ..db import SessionLocal

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
        # Was this ID already present?
        existed = person_exists(person_id)

        img = await decode_image_from_input(file, image_base64)
        dets = engine_faces.detect_and_embed(img, max_faces=1)
        if not dets:
            # HTTP 200 with failure payload
            return {"ok": False, "person_id": person_id, "exists": existed, "error": "No face found."}

        emb = dets[0]["embedding"]; bbox = dets[0]["bbox"]
        md = json.loads(metadata) if metadata else {}
        upsert_person(person_id, md, emb)

        resp = {"ok": True, "person_id": person_id, "exists": existed}
        if return_image:
            resp["bbox"] = bbox
            resp["debug_image_base64"] = render_match_debug_image(img, [], [{"bbox": bbox, "score": 1.0}])
        return resp

    except Exception as e:
        # HTTP 200 with failure payload (schema-compatible)
        return {"ok": False, "person_id": person_id, "exists": False, "error": str(e)}

@router.post("/match", response_model=MatchResponse, response_model_exclude_none=True)
async def match_faces(
    threshold: Optional[float] = Form(None),
    return_image: bool = Form(False),
    return_contours: bool = Form(False),
    max_faces: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    th = float(threshold) if threshold is not None else MATCH_THRESHOLD
    t0 = time.perf_counter()
    try:
        img = await decode_image_from_input(file, image_base64)
        img_proc, scale = resize_for_inference(img)
        inv = (1.0 / scale) if scale != 1.0 else 1.0

        # 1) Detect & embed
        k = max_faces or MAX_FACES_PER_FRAME
        dets = engine_faces.detect_and_embed(img_proc, max_faces=k)
        t_det = time.perf_counter()
        print(f"[perf] detect+embed: {(t_det - t0)*1000:.1f} ms  (faces={len(dets)})")

        # 2) DB match (single session)
        matches_meta = []  # (i, pid, meta, score)
        unmatched = []
        with SessionLocal() as s:
            s.begin()
            for i, det in enumerate(dets):
                pid, meta, score = best_match_in_session(det["embedding"], th, s)
                if pid:
                    matches_meta.append((i, pid, meta or {}, float(score)))
                else:
                    unmatched.append({"bbox": det["bbox"], "score": float(score)})
        t_match = time.perf_counter()
        print(f"[perf] matching: {(t_match - t_det)*1000:.1f} ms")

        # 3) (Optional) contours for matched faces
        matches_out: List[MatchFace] = []
        contours_needed = return_contours or return_image
        if contours_needed and matches_meta:
            matched_bboxes = [dets[i]["bbox"] for (i, *_ ) in matches_meta]
            pts_list = engine_faces.landmarks_for_bboxes(img_proc, matched_bboxes, expand=0.25)
        else:
            pts_list = [None] * len(matches_meta)
        t_lm = time.perf_counter()
        if contours_needed:
            print(f"[perf] landmarks: {(t_lm - t_match)*1000:.1f} ms")

        # 4) Build response
        for (i, pid, meta, score), pts in zip(matches_meta, pts_list):
            bbox = dets[i]["bbox"]
            if scale != 1.0:
                bbox = [int(b*inv) for b in bbox]
            contour = landmarks_convex_hull(pts) if pts is not None else []
            if scale != 1.0 and contour:
                contour = [[int(x*inv), int(y*inv)] for (x, y) in contour]
            lm_count = int(pts.shape[0]) if pts is not None else 0
            matches_out.append(MatchFace(
                person_id=pid, score=score, bbox=bbox,
                contour=contour if contours_needed else [],
                landmarks_count=lm_count if contours_needed else 0,
                metadata=meta
            ))

        if scale != 1.0:
            for u in unmatched:
                u["bbox"] = [int(b*inv) for b in u["bbox"]]

        response = {"matches": matches_out, "unmatched": unmatched or None}
        if return_image:
            response["debug_image_base64"] = render_match_debug_image(img, matches_out, unmatched)
        return response

    except Exception as e:
        # HTTP 200 with failure payload (schema-compatible)
        return {"matches": [], "unmatched": None, "error": str(e)}
