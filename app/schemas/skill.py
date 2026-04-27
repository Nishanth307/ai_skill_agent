import uuid 
from datetime import datetime 
from typing import Optional 
from pydantic import BaseModel, Field, EmailStr

# ══════════════════════════════════════════════════════
# LLM OUTPUT SCHEMAS (used with .with_structured_output())
# ══════════════════════════════════════════════════════

class SkillInfo(BaseModel):
    """Single skill extracted from JD or resume."""
    name: str
    required_level: str = Field(
        ..., description="beginner | intermediate | advanced | expert"
    )
    claimed_by_candidate: bool = Field(
        ..., description="True if candidate mentioned this in resume"
    )
    context: str = Field(
        ..., description="Brief quote or context from JD/resume"
    )

class ExtractedSkills(BaseModel):
    """Structured output from the Skill Extractor node."""
    required_skills: list[SkillInfo]
    candidate_claimed_skills: list[str]
    preliminary_gaps: list[str] = Field(
        ..., description="Skills in JD not found in resume"
    )
    job_title: Optional[str] = None
    seniority_level: Optional[str] = None

class SkillAssessmentScore(BaseModel):
    """Structured output after assessing one skill."""
    skill: str
    rating: str = Field(..., description="STRONG | AVERAGE | WEAK | MISSING")
    score: int = Field(..., ge=1, le=5, description="Internal numeric score (1-5)")
    confidence: str = Field(..., description="low | medium | high")
    evidence: str = Field(..., description="Detailed justification for the rating")
    assessment_complete: bool = Field(
        ..., description="True if enough signal gathered for this skill"
    )