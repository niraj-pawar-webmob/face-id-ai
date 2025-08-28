from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.routers.face import router as face_router

app = FastAPI(title="Face Match API (Postgres/pgvector)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(face_router)

# uvicorn main:app --host 0.0.0.0 --port 8000
