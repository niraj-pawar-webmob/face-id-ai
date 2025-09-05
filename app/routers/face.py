# from typing import Optional, List, Dict
# import logging
# import json
# import time
# import numpy as np
# from fastapi import APIRouter, File, UploadFile, Form
# from fastapi.responses import JSONResponse

# from ..config import MATCH_THRESHOLD, MAX_FACES_PER_FRAME
# from ..schemas import EnrollResponse, MatchFace, MatchResponse
# from ..utils import decode_image_from_input
# from ..engine import engine_faces
# from ..geometry import iou, landmarks_to_bbox, landmarks_convex_hull
# from ..draw import draw_box, draw_text_bg, render_match_debug_image
# from ..crud import upsert_person
# from ..matcher import best_match_in_session     # best_match


# router = APIRouter()

# @router.get("/health")
# def health():
#     return {"ok": True}

# @router.post("/enroll", response_model=EnrollResponse, response_model_exclude_none=True)
# async def enroll_person(
#     person_id: str = Form(...),
#     metadata: Optional[str] = Form(None),
#     return_image: bool = Form(False),
#     file: Optional[UploadFile] = File(None),
#     image_base64: Optional[str] = Form(None)
# ):
#     try:
#         img = await decode_image_from_input(file, image_base64)
#         dets = engine_faces.detect_and_embed(img, max_faces=1)
#         if not dets:
#             return JSONResponse({"ok": False, "person_id": person_id, "error": "No face found."}, status_code=400)

#         emb = dets[0]["embedding"]; bbox = dets[0]["bbox"]
#         md = json.loads(metadata) if metadata else {}
#         upsert_person(person_id, md, emb)

#         resp = {"ok": True, "person_id": person_id}
#         if return_image:
#             vis = img.copy()
#             draw_box(vis, bbox, (0, 200, 0), 2)
#             label = person_id
#             if md:
#                 kv = "; ".join(f"{k}:{v}" for k, v in list(md.items())[:2])
#                 if kv: label += f" | {kv}"
#             draw_text_bg(vis, label, (int(bbox[0]), int(bbox[1]) - 8), bg_color=(0,120,0))
#             resp["bbox"] = bbox
#             resp["debug_image_base64"] = render_match_debug_image(img, [], [{"bbox": bbox, "score": 1.0}])  # quick reuse
#         return resp
#     except Exception as e:
#         return JSONResponse({"ok": False, "person_id": person_id, "error": str(e)}, status_code=400)

# # The below given api doesn't return the contours when multiple matches are found in a single img.
# @router.post("/match_face", response_model=MatchResponse, response_model_exclude_none=True)
# async def match_faces(
#     threshold: Optional[float] = Form(None),
#     return_image: bool = Form(False),
#     file: Optional[UploadFile] = File(None),
#     image_base64: Optional[str] = Form(None)
# ):
#     th = float(threshold) if threshold is not None else MATCH_THRESHOLD
#     try:
#         img = await decode_image_from_input(file, image_base64)
#         dets = engine_faces.detect_and_embed(img)
#         all_landmarks = engine_faces.landmarks_for_all(img)
#         lm_boxes = [landmarks_to_bbox(pts) for pts in all_landmarks]

#         matches_out: List[MatchFace] = []
#         unmatched_out: List[Dict[str, Any]] = []

#         for det in dets:
#             bbox = det["bbox"]; emb = det["embedding"]

#             best_pts = None; best_i = 0.0
#             for pts, box in zip(all_landmarks, lm_boxes):
#                 i = iou(bbox, box)
#                 if i > best_i:
#                     best_i = i; best_pts = pts

#             contour_pts = landmarks_convex_hull(best_pts) if best_pts is not None else []
#             lm_count = int(best_pts.shape[0]) if best_pts is not None else 0

#             pid, meta, score = best_match(emb, th)
#             if pid is not None:
#                 matches_out.append(MatchFace(
#                     person_id=pid, score=score, bbox=bbox,
#                     contour=contour_pts, landmarks_count=lm_count, metadata=meta or {}
#                 ))
#             else:
#                 unmatched_out.append({
#                     "score": float(score), "bbox": bbox,
#                     "contour": contour_pts, "landmarks_count": lm_count
#                 })

#         # response = {"matches": matches_out, "unmatched": unmatched_out}
#         # The below given line is to be uncommented when you require unmatched_out
#         response = {"data": matches_out}

#         # 👇 add annotated image only when requested
#         if return_image:
#             # The below given code to be included when you need to convert the image file to base64
#             dbg_b64 = render_match_debug_image(img, matches_out, unmatched_out)
#             response["debug_image_base64"] = dbg_b64

#             # response["raw_image"] = img

#         return response
    
#     except Exception as e:
#         return JSONResponse({"matches": [], "unmatched": [], "error": str(e)}, status_code=400)

# @router.post("/match", response_model=MatchResponse, response_model_exclude_none=True)
# async def match_faces(
#     threshold: Optional[float] = Form(None),
#     return_image: bool = Form(False),
#     file: Optional[UploadFile] = File(None),
#     image_base64: Optional[str] = Form(None)
# ):
#     t0 = time.perf_counter()
#     th = float(threshold) if threshold is not None else MATCH_THRESHOLD
#     try:
#         img = await decode_image_from_input(file, image_base64)
       
#         dets = engine_faces.detect_and_embed(img)
#         t_det = time.perf_counter();
#         print(f"[perf] detect+embed: {(t_det - t0)*1000: 1f} ms")
#         bboxes = [d["bbox"] for d in dets]

#         # NEW: one landmark set per detection (or None)
#         landmarks_per_det = engine_faces.landmarks_for_bboxes(img, bboxes, expand=0.25)
#         t_lm = time.perf_counter(); print(f"[perf] landmarks: {(t_lm - t_det)*1000:.1f} ms")

#         matches_out: List[MatchFace] = []
#         unmatched_out: List[Dict[str, Any]] = []

#         for det, best_pts in zip(dets, landmarks_per_det):
#             bbox = det["bbox"]
#             emb  = det["embedding"]

#             contour_pts = landmarks_convex_hull(best_pts) if best_pts is not None else []
#             lm_count = int(best_pts.shape[0]) if best_pts is not None else 0

#             pid, meta, score = best_match(emb, th)
#             if pid is not None:
#                 matches_out.append(MatchFace(
#                     person_id=pid,
#                     score=score,
#                     bbox=bbox,
#                     contour=contour_pts,
#                     landmarks_count=lm_count,
#                     metadata=meta or {}
#                 ))
#             else:
#                 unmatched_out.append({
#                     "score": float(score),
#                     "bbox": bbox,
#                     "contour": contour_pts,
#                     "landmarks_count": lm_count
#                 })
#         t_match = time.perf_counter();
#         print(f"[perf] matching: {(t_match - t_lm)*1000: 1f} ms")
#         response = {"data": matches_out}  # add "unmatched": unmatched_out if you want it
#         if return_image:
#             dbg_b64 = render_match_debug_image(img, matches_out, unmatched_out)
#             response["debug_image_base64"] = dbg_b64
#         return response

#     except Exception as e:
#         return JSONResponse({"matches": [], "unmatched": [], "error": str(e)}, status_code=400)



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
from ..crud import upsert_person
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
        img = await decode_image_from_input(file, image_base64)
        dets = engine_faces.detect_and_embed(img, max_faces=1)
        if not dets:
            return JSONResponse({"ok": False, "person_id": person_id, "error": "No face found."}, status_code=400)

        emb = dets[0]["embedding"]; bbox = dets[0]["bbox"]
        md = json.loads(metadata) if metadata else {}
        upsert_person(person_id, md, emb)

        resp = {"ok": True, "person_id": person_id}
        if return_image:
            # quick overlay using match renderer (only bbox)
            resp["bbox"] = bbox
            resp["debug_image_base64"] = render_match_debug_image(img, [], [{"bbox": bbox, "score": 1.0}])
        return resp
    except Exception as e:
        return JSONResponse({"ok": False, "person_id": person_id, "error": str(e)}, status_code=400)

@router.post("/match", response_model=MatchResponse, response_model_exclude_none=True)
async def match_faces(
    threshold: Optional[float] = Form(None),
    return_image: bool = Form(False),
    return_contours: bool = Form(True),
    max_faces: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
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

        # response = {"data": matches_out, "unmatched": unmatched if unmatched else None}
        response = {"data": matches_out}

        # 5) Optional overlay image (draw on original-size image for consistency)
        if return_image:
            response["debug_image_base64"] = render_match_debug_image(img, matches_out, unmatched)

        t_cp = time.perf_counter()
        print(f"[perf] complete process: {(t0 - t_cp)*1000:.1f} ms")
        return response
    except Exception as e:
        return JSONResponse({"data": [], "unmatched": [], "error": str(e)}, status_code=400)
