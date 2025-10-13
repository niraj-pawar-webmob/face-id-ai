# --------------------------------------------------------------------------------------------------------------------
# The below given code has rearranged order of the operations

import time
from typing import Optional, List, Dict
import json
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
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
    """Health check endpoint to verify the API is running."""
    return {"ok": True}

@router.post("/enroll", response_model=EnrollResponse, response_model_exclude_none=True)
async def enroll_person(
    person_id: str = Form(...),
    metadata: Optional[str] = Form(None),
    return_image: bool = Form(False),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
        Enroll a new person into the database.

        - **person_id**: Required unique identifier for the person.
        - **metadata**: JSON string with any extra information.
        - **file**: Image upload (face will be detected & embedded).
        - **image_base64**: Base64 image alternative.
    """
    try:
        # Check existence BEFORE upsert (so you know if it was already there)
        existed = person_exists(person_id)

        img = await decode_image_from_input(file, image_base64)
        dets = engine_faces.detect_and_embed(img, max_faces=1)
        if not dets:
            return JSONResponse({"ok": False, "person_id": person_id, "error": "No face found."}, status_code=400)

        emb = dets[0]["embedding"]; bbox = dets[0]["bbox"]
        md = json.loads(metadata) if metadata else {}

        upsert_person(person_id, md, emb)

        resp = {"ok": True, "person_id": person_id, "exists": existed}   # <-- include flag
        if return_image:
            resp["bbox"] = bbox
            resp["debug_image_base64"] = render_match_debug_image(img, [], [{"bbox": bbox, "score": 1.0}])
        return resp
    except Exception as e:
        return JSONResponse({"ok": False, "person_id": person_id, "error": str(e)}, status_code=400)

@router.post("/match", response_model=MatchResponse, response_model_exclude_none=True)
async def match_faces(
    threshold: Optional[float] = Form(None),
    return_image: bool = Form(False),
    return_contours: bool = Form(False),
    max_faces: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Match one or more faces against the enrolled database.

    - **threshold**: Optional threshold override.
    - **return_image**: Whether to return annotated debug image.
    - **return_contours**: Include facial contours in output.
    - **file**: Image upload.
    - **image_base64**: Base64 image alternative.
    """

    th = float(threshold) if threshold is not None else MATCH_THRESHOLD
    t0 = time.perf_counter()
    try:
        img = await decode_image_from_input(file, image_base64)
        # Optional resize to speed up
        img_proc, scale = resize_for_inference(img)
        inv = (1.0 / scale) if scale != 1.0 else 1.0

        # 1) Detect & embed (limit faces early)
        k = max_faces or MAX_FACES_PER_FRAME
        dets = engine_faces.detect_and_embed(img_proc, max_faces=k)
        t_det = time.perf_counter()
        print(f"[perf] detect+embed: {(t_det - t0)*1000:.1f} ms  (faces={len(dets)})")

        # 2) Match all faces using a single DB session
        matches_meta = []  # (i, pid, meta, score)
        unmatched = []     # {"bbox": ..., "score": ...}
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

        # 3) Landmarks only for matched faces (and only if requested)
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

        # 4) Build response (unscale if resized)
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

        # Also include unmatched (optional)
        if scale != 1.0:
            for u in unmatched:
                u["bbox"] = [int(b*inv) for b in u["bbox"]]

        response = {"matches": matches_out, "unmatched": unmatched if unmatched else None}

        # 5) Optional overlay image (draw on original-size image for consistency)
        if return_image:
            response["debug_image_base64"] = render_match_debug_image(img, matches_out, unmatched)

        return response
    except Exception as e:
        return JSONResponse({"matches": [], "unmatched": [], "error": str(e)}, status_code=400)
