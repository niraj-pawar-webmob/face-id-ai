import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .config import DATABASE_URL
from .models import Base

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# --- bootstrap (extension, table, index, analyze) ---
BOOTSTRAP_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE INDEX IF NOT EXISTS persons_embedding_ivfflat
  ON persons USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

ANALYZE persons;
"""

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(conn)
    conn.execute(text(BOOTSTRAP_SQL))

# --- readiness check (your function) ---
def verify_db_ready():
    checks = {
        "server_version": "SELECT current_setting('server_version')",
        "pgvector_version": "SELECT extversion FROM pg_extension WHERE extname='vector'",
        "persons_table": """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = current_schema() AND table_name = 'persons'
        """,
        "embedding_column": """
            SELECT data_type, udt_name
            FROM information_schema.columns
            WHERE table_name='persons' AND column_name='embedding'
        """,
        "ivfflat_index": """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'persons'
              AND indexname = 'persons_embedding_ivfflat'
        """,
    }
    with engine.connect() as conn:
        print("\n[DB readiness checks]")
        for name, sql in checks.items():
            try:
                res = list(conn.execute(text(sql)))
                print(f"- {name}: {res[0] if res else 'NOT FOUND'}")
            except Exception as e:
                print(f"- {name}: ERROR -> {e}")
        print()

# call once on import (gate with env if you want to silence in prod)
if os.getenv("DB_VERIFY", "1") == "1":
    verify_db_ready()
