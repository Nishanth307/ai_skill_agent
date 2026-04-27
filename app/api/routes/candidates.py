"""
app/api/routes/candidates.py
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from app.memory.storage import storage
from app.schemas.schemas import CandidateCreate, CandidateResponse
from app.memory.vector_store import resume_vector_store

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.post("/", response_model=CandidateResponse, status_code=201)
async def create_candidate(
    payload: CandidateCreate,
):
    """Register a candidate with their resume text."""

    # 1. Check if email already exists
    if payload.email:
        existing = await storage.get_candidate_by_email(payload.email)
        if existing:
            raise HTTPException(status_code=409, detail="Candidate with this email already exists")

    # 2. Embed resume into ChromaDB for semantic retrieval
    chroma_doc_id = await resume_vector_store.add_resume(
        candidate_email=payload.email,
        candidate_name=payload.name,
        resume_text=payload.resume_text,
    )

    # 3. Save candidate to in-memory storage
    candidate = await storage.create_candidate(
        name=payload.name,
        email=payload.email,
        resume_text=payload.resume_text,
        chroma_doc_id=chroma_doc_id,
    )

    return candidate


@router.post("/upload", response_model=CandidateResponse, status_code=201)
async def upload_candidate_resume(
    name: str = Form("Anonymous"),
    email: str = Form(None),
    file: UploadFile = File(..., description="Resume PDF or DOCX"),
):
    """Upload a resume file (PDF/DOCX) — extracts text automatically."""
    from app.core.document_parser import extract_text_from_file

    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        raise HTTPException(status_code=422, detail="Only PDF and DOCX files are accepted")

    resume_text = await extract_text_from_file(file)

    if len(resume_text.strip()) < 50:
        raise HTTPException(status_code=422, detail="Could not extract meaningful text from the file")

    payload = CandidateCreate(name=name, email=email, resume_text=resume_text)
    return await create_candidate(payload)


@router.get("/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(
    candidate_id: uuid.UUID,
):
    candidate = await storage.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@router.get("/search/semantic")
async def semantic_search_candidates(
    query: str,
    top_k: int = 5,
):
    """
    Semantic search across all resumes in ChromaDB.
    Example: query="distributed systems Kafka microservices"
    Returns top-k most relevant candidates.
    """
    results = await resume_vector_store.search(query=query, top_k=top_k)
    return {"query": query, "results": results}