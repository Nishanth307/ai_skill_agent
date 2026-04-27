import uuid 
from datetime import datetime 
from typing import Optional 
from pydantic import BaseModel, Field, EmailStr

# ══════════════════════════════════════════════════════
# LEARNING PLAN SCHEMAS
# ══════════════════════════════════════════════════════
 

class Resource(BaseModel):
    """Single learning resource."""
    title: str 
    url: Optional[str] = None 
    type: str = Field(..., description="course | book | doc | video | practice")
    platform: Optional[str] = None 
    is_free: bool = True
    estimated_hours: int = Field(..., ge=1)

class SkillLearningItem(BaseModel):
    """Learning plan for a single skill gap."""
    skill: str
    current_score: int = Field(..., ge=1, le=5)
    target_score: int = Field(..., ge=1, le=5)
    priority: str = Field(..., description="high | medium | low")
    rationale: str                    # why this skill matters for the JD
    resources: list[Resource]
    estimated_hours: int = 0
    milestones: list[str]

class WeeklySchedule(BaseModel):
    week: int
    focus_skills: list[str]
    goals: list[str]
    hours_per_day: float

class LearningPlanResponse(BaseModel):
    id: uuid.UUID
    assessment_id: uuid.UUID
    skill_items: list[SkillLearningItem]
    weekly_schedule: list[WeeklySchedule]
    total_hours: int
    duration_weeks: int
    summary: str                      # 2-3 sentence human-readable overview
    assessment_report: Optional[str] = None  # The complete PHASE 7 report
    created_at: datetime
 
    model_config = {"from_attributes": True}
    