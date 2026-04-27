import uuid 
import enum
from datetime import datetime 
from typing import Optional 
from pydantic import BaseModel, Field

class AssessmentStatus(str, enum.Enum):
    pending = "pending"
    extracting = "extracting"
    assessing = "assessing"
    analyzing = "analyzing"
    completed = "completed"
    failed = "failed"

# ══════════════════════════════════════════════════════
# ASSESSMENT SCHEMAS
# ══════════════════════════════════════════════════════

class AssessmentCreate(BaseModel):
    candidate_id: uuid.UUID 
    job_description: str = Field(..., min_length=10)
    job_title: Optional[str] = None 

class AssessmentResponse(BaseModel):
    id: uuid.UUID 
    candidate_id: uuid.UUID
    job_title: Optional[str]
    status: AssessmentStatus
    thread_id: Optional[str]
    required_skills: Optional[list]
    assessment_results: Optional[list]
    skill_gaps: Optional[list]
    overall_score: Optional[float]
    progress: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[datetime]
    created_at: datetime 
    model_config = {"from_attributes":True}

class ChatMessage(BaseModel):
     """Single message in the conversational assessment."""
     role: str = Field(..., pattern="^(user|assistant)$")
     content: str = Field(..., min_length=1)

class AssessmentChatRequest(BaseModel):
    """User sends this to continue a conversational assessment."""
    thread_id: str 
    message:str = Field(..., min_length=1)

class AssessmentChatResponse(BaseModel):
    """Agent's response during conversational assessment."""
    thread_id: str 
    message:str 
    is_complete: bool = False 
    current_skill: Optional[str] = None 
    progress: Optional[str] = None 


