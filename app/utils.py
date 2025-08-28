import base64
import numpy as np
import cv2
from typing import Optional
from fastapi import UploadFile

async def decode_image_from_input(file: Optional[UploadFile], b64: Optional[str]) -> np.ndarray:
    if file is not None:
        data = await file.read()  # async read
    elif b64 is not None:
        data = base64.b64decode(b64)
    else:
        raise ValueError("No image provided (file or base64).")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes.")
    return img

def encode_jpeg_base64(img_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
