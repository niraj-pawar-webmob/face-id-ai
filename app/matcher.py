from typing import Any, Dict, Optional, Tuple
import numpy as np
from sqlalchemy import text, bindparam
from pgvector.sqlalchemy import Vector
from .db import SessionLocal
from .config import EMBED_DIM

# def best_match(vec: np.ndarray, threshold: float) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
#     try:
#         q = vec.astype(np.float32).tolist()
#         stmt = text("""
#             SELECT person_id,
#                    metadata,
#                    1 - (embedding <=> :q) AS score
#             FROM persons
#             ORDER BY embedding <=> :q
#             LIMIT 1
#         """).bindparams(bindparam("q", type_=Vector(EMBED_DIM)))
#         with SessionLocal() as s:
#             rows = s.execute(stmt, {"q": q}).mappings().all()
#         if not rows:
#             return None, None, 0.0
#         r = rows[0]
#         score = float(r["score"]) if r["score"] is not None else 0.0
#         if score >= threshold:
#             return r["person_id"], r["metadata"], score
#         return None, None, score
#     except Exception as e:
#         print("best_match error:", repr(e))
#         return None, None, 0.0


# This code is in regards with the latency reduction
# Face matching in the single session 
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sqlalchemy import text, bindparam
from pgvector.sqlalchemy import Vector
from .config import EMBED_DIM, PGVECTOR_PROBES

# Prepared statement (bind vector param once)
BEST_STMT = text("""
    SELECT person_id, metadata, 1 - (embedding <=> :q) AS score
    FROM persons
    ORDER BY embedding <=> :q
    LIMIT 1
""").bindparams(bindparam("q", type_=Vector(EMBED_DIM)))

def best_match_in_session(vec: np.ndarray, threshold: float, session) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
    q = vec.astype(np.float32).tolist()
    # Optional: tune probes for IVFFlat
    if PGVECTOR_PROBES > 0:
        session.execute(text("SET LOCAL ivfflat.probes = :p"), {"p": PGVECTOR_PROBES})
    row = session.execute(BEST_STMT, {"q": q}).mappings().first()
    if not row:
        return None, None, 0.0
    score = float(row["score"] or 0.0)
    if score >= threshold:
        return row["person_id"], row["metadata"], score
    return None, None, score
