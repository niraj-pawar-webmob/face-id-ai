from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class EnrollResponse(BaseModel):
    ok: bool
    person_id: str
    bbox: Optional[List[int]] = None
    debug_image_base64: Optional[str] = None

class MatchFace(BaseModel):
    person_id: Optional[str] = None
    score: float
    bbox: List[int] = Field(..., description="[x1,y1,x2,y2]")
    contour: List[List[int]]
    landmarks_count: int
    metadata: Dict[str, Any] = {}

class MatchResponse(BaseModel):
    matches: List[MatchFace]
    unmatched: Optional[List[Dict[str, Any]]] = None
    debug_image_base64: Optional[str] = None
