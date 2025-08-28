from typing import Any, Dict, Optional, Tuple
import numpy as np
from sqlalchemy import text, bindparam
from pgvector.sqlalchemy import Vector
from .db import SessionLocal
from .config import EMBED_DIM

def best_match(vec: np.ndarray, threshold: float) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
    try:
        q = vec.astype(np.float32).tolist()
        stmt = text("""
            SELECT person_id,
                   metadata,
                   1 - (embedding <=> :q) AS score
            FROM persons
            ORDER BY embedding <=> :q
            LIMIT 1
        """).bindparams(bindparam("q", type_=Vector(EMBED_DIM)))
        with SessionLocal() as s:
            rows = s.execute(stmt, {"q": q}).mappings().all()
        if not rows:
            return None, None, 0.0
        r = rows[0]
        score = float(r["score"]) if r["score"] is not None else 0.0
        if score >= threshold:
            return r["person_id"], r["metadata"], score
        return None, None, score
    except Exception as e:
        print("best_match error:", repr(e))
        return None, None, 0.0
