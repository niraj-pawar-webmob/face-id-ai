from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .config import DATABASE_URL
from .models import Base  # registers mapped classes

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# Initialize extension + tables
with engine.begin() as conn:
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception:
        pass
    Base.metadata.create_all(conn)
