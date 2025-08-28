import os
import numpy as np
import cv2
from typing import List
import mediapipe as mp
from insightface.app import FaceAnalysis
from .config import DET_SIZE, MAX_FACES_PER_FRAME

class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        ctx = 0 if os.getenv("GPU", "0") == "1" else -1
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx == 0 else ["CPUExecutionProvider"]
            self.app.prepare(ctx_id=ctx, det_size=DET_SIZE, providers=providers)
        except TypeError:
            self.app.prepare(ctx_id=ctx, det_size=DET_SIZE)
        self.mp_face_mesh = mp.solutions.face_mesh

    def detect_and_embed(self, img_bgr: np.ndarray, max_faces: int = MAX_FACES_PER_FRAME):
        faces = self.app.get(img_bgr)
        faces = sorted(faces, key=lambda f: getattr(f, "det_score", 0.0), reverse=True)
        out = []
        for f in faces[:max_faces]:
            bbox = f.bbox.astype(int).tolist()
            emb = f.normed_embedding.astype(np.float32)
            det_score = float(getattr(f, "det_score", 1.0))
            out.append({"bbox": bbox, "embedding": emb, "det_score": det_score})
        return out

    def landmarks_for_all(self, img_bgr: np.ndarray, max_faces: int = MAX_FACES_PER_FRAME) -> List[np.ndarray]:
        ih, iw = img_bgr.shape[:2]
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as mesh:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = mesh.process(img_rgb)
            if not res.multi_face_landmarks:
                return []
            out = []
            for fl in res.multi_face_landmarks:
                pts = np.array([[lm.x*iw, lm.y*ih] for lm in fl.landmark], dtype=np.float32)
                out.append(pts)
            return out

engine_faces = FaceEngine()  # singleton
