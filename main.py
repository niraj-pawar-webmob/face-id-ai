# # main.py
# from fastapi import FastAPI
# from starlette.middleware.cors import CORSMiddleware
# from app.routers.face import router as face_router
# from app.db import verify_db_ready

# app = FastAPI(title="Face Match API (Postgres/pgvector)", version="1.0.0")
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# app.include_router(face_router)

# @app.on_event("startup")
# async def _db_checks():
#     # will print once when the server starts up
#     verify_db_ready()

# # uvicorn main:app --host 0.0.0.0 --port 8000


# main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.routers.face import router as face_router
from app.db import verify_db_ready

# Create FastAPI app with metadata for Swagger/Redoc
app = FastAPI(
    title="Face Match API (Postgres/pgvector)",
    description="""
API for **face recognition and matching** using embeddings stored in Postgres with pgvector.

### Endpoints
- **Health Check** → `/health`
- **Enroll Person** → `/enroll`
- **Match Faces** → `/match`

### Features
- Detect and embed faces
- Store embeddings linked to person IDs
- Match faces against enrolled persons
- Return debug images and contours (optional)
    """,
    version="1.0.0",
    # contact={
    #     "name": "Your Team",
    #     "url": "https://your-company.com",
    #     "email": "support@your-company.com",
    # },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers with a prefix and tags for Swagger grouping
# app.include_router(face_router, prefix="/api/v1", tags=["Face Recognition"])
app.include_router(face_router, tags=["Face Recognition"])

# Startup hook for DB check
@app.on_event("startup")
async def _db_checks():
    # will print once when the server starts up
    verify_db_ready()

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

