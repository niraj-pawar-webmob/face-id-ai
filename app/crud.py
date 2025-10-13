import json
import numpy as np
from sqlalchemy.dialects.postgresql import insert
from .db import SessionLocal
from .models import Person

def person_exists(person_id: str) -> bool:
    """Return True if a row with this person_id already exists."""
    with SessionLocal() as s:
        return s.get(Person, person_id) is not None

def upsert_person(person_id: str, metadata: dict, embedding: np.ndarray):
    vec = embedding.astype(np.float32).tolist()
    ins = insert(Person).values(person_id=person_id, meta=(metadata or {}), embedding=vec)
    upsert = ins.on_conflict_do_update(
        index_elements=[Person.person_id],
        set_={"metadata": ins.excluded["metadata"], "embedding": ins.excluded["embedding"]}
    )
    with SessionLocal() as s, s.begin():
        s.execute(upsert)
