import os
from dotenv import load_dotenv

load_dotenv(".env")

EMBED_DIM = 512
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.35"))
MAX_FACES_PER_FRAME = int(os.getenv("MAX_FACES", "20"))
DET_SIZE = (640, 640)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL, e.g. postgresql+psycopg2://user:pass@host:5432/db")

# CPU by default; silence GPU probing noise on Windows unless GPU=1
if os.getenv("GPU", "0") != "1":
    os.environ["ORT_PROVIDER_EXCLUDE"] = "CUDA;ROCM;DirectML"
