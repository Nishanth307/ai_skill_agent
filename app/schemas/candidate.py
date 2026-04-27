import uuid 
from datetime import datetime 
from typing import Optional 
from pydantic import BaseModel, Field, EmailStr

 
# ══════════════════════════════════════════════════════
# CANDIDATE SCHEMAS
# ══════════════════════════════════════════════════════
 

class CandidateCreate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    email: Optional[EmailStr] = None
    resume_text: str = Field(..., min_length=1, description="Raw text from resume")

class CandidateResponse(BaseModel):

    id: uuid.UUID 
    name: Optional[str]
    email: Optional[EmailStr]
    chroma_doc_id: Optional[str]
    created_at: datetime 
    model_config = {"from_attributes":True}