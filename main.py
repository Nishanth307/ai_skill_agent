import os
from contextlib import asynccontextmanager

from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import candidates, assessment, plans
from app.memory.vector_store import resume_vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    """

    """
    # ── STARTUP ──
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"   Backend : {settings.LLM_BACKEND.upper()}")
    print(f"   Env     : {settings.ENVIRONMENT}")

    # Skip heavy startup hooks in unit tests
    if settings.ENVIRONMENT != "test":
        await resume_vector_store.initialize()

    yield
    
    # ── SHUTDOWN ──

    print(f"🛑 Shutting down {settings.APP_NAME}")
    
app = FastAPI(
    title = settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## AI-Powered Skill Assessment & Personalised Learning Plan Agent

    Takes a **Job Description** and **candidate resume**, conversationally assesses
    real proficiency on each required skill, identifies gaps, and generates a
    personalised learning plan with curated resources and time estimates.
    
    ### Flow
    1. `POST /candidates` — register candidate + resume
    2. `POST /assessments` — start assessment against a JD
    3. `POST /assessments/chat` — conversational Q&A loop
    4. `POST /plans/generate/{id}` — generate learning plan
    5. `GET  /plans/{id}` — fetch the final plan
    """,
    docs_url = "/docs",
    redoc_url = "/redoc",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ROUTERS ──
app.include_router(candidates.router, prefix="/api/v1")
app.include_router(assessment.router, prefix="/api/v1")
app.include_router(plans.router, prefix="/api/v1")

@app.get("/health", tags=["system"])
async def health_check():
    return {
        "status": "OK",
        "version": settings.APP_VERSION,
        "llm_backend": settings.LLM_BACKEND,
        "embedding_backend": settings.EMBEDDING_BACKEND,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
    }

@app.get("/", tags=["system"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/docs",
    }
