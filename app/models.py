from typing import Any, Dict, List
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from .config import EMBED_DIM

class Base(DeclarativeBase):
    pass

class Person(Base):
    __tablename__ = "persons"
    person_id: Mapped[str] = mapped_column(primary_key=True)
    # Python attr "meta" -> DB column "metadata" (since "metadata" is reserved by SQLAlchemy)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(EMBED_DIM), nullable=False)
