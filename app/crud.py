import json
import numpy as np
from sqlalchemy.dialects.postgresql import insert
from .db import SessionLocal
from .models import Person

def upsert_person(person_id: str, metadata: dict, embedding: np.ndarray):
    vec = embedding.astype(np.float32).tolist()
    ins = insert(Person).values(person_id=person_id, meta=(metadata or {}), embedding=vec)
    upsert = ins.on_conflict_do_update(
        index_elements=[Person.person_id],
        set_={"metadata": ins.excluded["metadata"], "embedding": ins.excluded["embedding"]}
    )
    with SessionLocal() as s, s.begin():
        s.execute(upsert)
