# import os
# import numpy as np
# from typing import Optional
# import cv2
# from typing import List
# import mediapipe as mp
# from insightface.app import FaceAnalysis
# from .config import DET_SIZE, MAX_FACES_PER_FRAME

# class FaceEngine:
#     def __init__(self):
#         self.app = FaceAnalysis(name="buffalo_l")
#         ctx = 0 if os.getenv("GPU", "0") == "1" else -1
#         try:
#             providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx == 0 else ["CPUExecutionProvider"]
#             self.app.prepare(ctx_id=ctx, det_size=DET_SIZE, providers=providers)
#         except TypeError:
#             self.app.prepare(ctx_id=ctx, det_size=DET_SIZE)
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.mesh = self.mp_face_mesh.FaceMesh(
#                 static_image_mode=False, # faster still okay for single images
#                 max_num_faces = MAX_FACES_PER_FRAME,
#                 refine_landmarks = True,
#                 min_detection_confidence = 0.5,
#                 min_tracking_confidence = 0.5
#             )

#     def detect_and_embed(self, img_bgr: np.ndarray, max_faces: int = MAX_FACES_PER_FRAME):
#         faces = self.app.get(img_bgr)
#         faces = sorted(faces, key=lambda f: getattr(f, "det_score", 0.0), reverse=True)
#         out = []
#         for f in faces[:max_faces]:
#             bbox = f.bbox.astype(int).tolist()
#             emb = f.normed_embedding.astype(np.float32)
#             det_score = float(getattr(f, "det_score", 1.0))
#             out.append({"bbox": bbox, "embedding": emb, "det_score": det_score})
#         return out

#     def landmarks_for_all(self, img_bgr: np.ndarray, max_faces: int = MAX_FACES_PER_FRAME) -> List[np.ndarray]:
#         ih, iw = img_bgr.shape[:2]
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         res = self.mesh.process(img_rgb)
#         if not res.multi_face_landmarks:
#             return []
#         out = []
#         for fl in res.multi_face_landmarks:
#             pts = np.array([[lm.x*iw, lm.y*ih] for lm in fl.landmark], dtype=np.float32)
#             out.append(pts)
#         return out

#     def landmarks_for_bboxes(self, img_bgr: np.ndarray, bboxes: List[List[int]], expand: float = 0.25) -> List[Optional[np.ndarray]]:
#         """
#         For each detector bbox, run FaceMesh on a cropped region and return 468x2 landmarks
#         in full-image coordinates. If FaceMesh fails on a crop, returns None for that face.
#         """
#         ih, iw = img_bgr.shape[:2]
#         out: List[Optional[np.ndarray]] = []
       
#         for (x1, y1, x2, y2) in bboxes:
#             # expand box a bit to include chin/forehead
#             w = x2 - x1; h = y2 - y1
#             cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
#             half_w = w * (1.0 + expand) / 2.0
#             half_h = h * (1.0 + expand) / 2.0
#             rx1 = int(max(0, cx - half_w)); ry1 = int(max(0, cy - half_h))
#             rx2 = int(min(iw, cx + half_w)); ry2 = int(min(ih, cy + half_h))
#             crop = img_bgr[ry1:ry2, rx1:rx2]
#             if crop.size == 0:
#                 out.append(None); continue

#             rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#             res = self.mesh.process(rgb)
#             if not res.multi_face_landmarks:
#                 out.append(None); continue

#             lm = res.multi_face_landmarks[0]
#             cw, ch = (rx2 - rx1), (ry2 - ry1)
#             pts = np.array([[lmpt.x * cw + rx1, lmpt.y * ch + ry1] for lmpt in lm.landmark], dtype=np.float32)
#             out.append(pts)
#         return out

# engine_faces = FaceEngine()  # singleton


# This code is in regards with optimization
import os
from typing import List, Optional
import numpy as np
import cv2
import mediapipe as mp
from insightface.app import FaceAnalysis
from .config import DET_SIZE, MAX_FACES_PER_FRAME, FACE_PACK

class FaceEngine:
    def __init__(self):
        # InsightFace
        self.app = FaceAnalysis(name=FACE_PACK)
        ctx = 0 if os.getenv("GPU", "0") == "1" else -1
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx == 0 else ["CPUExecutionProvider"]
            self.app.prepare(ctx_id=ctx, det_size=DET_SIZE, providers=providers)
        except TypeError:
            self.app.prepare(ctx_id=ctx, det_size=DET_SIZE)

        # Persistent FaceMesh graph (faster than per-call context)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=MAX_FACES_PER_FRAME,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_and_embed(self, img_bgr: np.ndarray, max_faces: int = MAX_FACES_PER_FRAME):
        faces = self.app.get(img_bgr)
        faces = sorted(faces, key=lambda f: getattr(f, "det_score", 0.0), reverse=True)[:max_faces]
        out = []
        for f in faces:
            bbox = f.bbox.astype(int).tolist()
            emb = f.normed_embedding.astype(np.float32)  # L2-normalized
            det_score = float(getattr(f, "det_score", 1.0))
            out.append({"bbox": bbox, "embedding": emb, "det_score": det_score})
        return out

    def landmarks_for_bboxes(self, img_bgr: np.ndarray, bboxes: List[List[int]], expand: float = 0.25) -> List[Optional[np.ndarray]]:
        """
        Run FaceMesh per detection crop (max 1 face per crop) and map landmarks to full-image coords.
        """
        ih, iw = img_bgr.shape[:2]
        out: List[Optional[np.ndarray]] = []
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as mesh:
            for (x1, y1, x2, y2) in bboxes:
                w = x2 - x1; h = y2 - y1
                cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
                half_w = w * (1.0 + expand) / 2.0
                half_h = h * (1.0 + expand) / 2.0
                rx1 = int(max(0, cx - half_w)); ry1 = int(max(0, cy - half_h))
                rx2 = int(min(iw, cx + half_w)); ry2 = int(min(ih, cy + half_h))

                crop = img_bgr[ry1:ry2, rx1:rx2]
                if crop.size == 0:
                    out.append(None); continue

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res = mesh.process(rgb)
                if not res.multi_face_landmarks:
                    out.append(None); continue

                lm = res.multi_face_landmarks[0]
                cw, ch = (rx2 - rx1), (ry2 - ry1)
                pts = np.array([[p.x * cw + rx1, p.y * ch + ry1] for p in lm.landmark], dtype=np.float32)
                out.append(pts)
        return out

engine_faces = FaceEngine()  # singleton

