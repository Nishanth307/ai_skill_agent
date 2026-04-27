"""
app/graph/state.py

The AgentState is the single source of truth that flows through every node
in the LangGraph. Think of it as a shared whiteboard — each node reads from
it and writes back to it.

Key LangGraph concept:
  - TypedDict fields are replaced on each node update (default)
  - EXCEPT fields annotated with Annotated[list, add_messages] — those APPEND
    This is what gives us automatic conversation history management.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SkillInfo(TypedDict):
    name: str
    required_level: str          # "beginner" | "intermediate" | "advanced" | "expert"
    claimed_by_candidate: bool   # True if candidate mentioned this skill in resume
    context: str                 # snippet from JD or resume that mentions this skill


class AssessmentResult(TypedDict):
    skill: str
    score: int                   # Internal score (1–5)
    rating: str                  # "STRONG" | "AVERAGE" | "WEAK" | "MISSING"
    confidence: str              # "low" | "medium" | "high"
    evidence: str                # detailed justification
    questions_asked: list[str]   # all questions asked for this skill


class SkillGap(TypedDict):
    skill: str
    current_score: int
    required_level: str
    gap_severity: str            # "critical" | "moderate" | "minor"
    is_adjacent: bool            # True = candidate can realistically bridge this gap


class AgentState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    job_description: str
    resume: str

    # ── Skill Extraction (Node 1 output) ──────────────────────────────────────
    required_skills: list[SkillInfo]
    claimed_skills: list[str]
    job_title: Optional[str]
    seniority_level: Optional[str]

    # ── Conversational Assessment (Node 2 — loops) ────────────────────────────
    # add_messages = append-only; LangGraph merges new messages into this list
    # automatically. Never overwrite this field — always append.
    messages: Annotated[list[BaseMessage], add_messages]

    # Pointer: which skill are we currently assessing?
    current_skill_index: int

    # Accumulates as each skill is assessed
    assessment_results: list[AssessmentResult]

    # How many questions asked for the CURRENT skill (reset per skill)
    questions_asked_for_current_skill: int

    # NEW: Tracking for non-repetition and global limits
    asked_questions: list[str]
    total_questions_asked: int

    # Flag: set True when all skills have been assessed
    assessment_complete: bool

    # ── Gap Analysis (Node 3 output) ──────────────────────────────────────────
    skill_gaps: list[SkillGap]
    strengths: list[str]         # skills where candidate scored well
    overall_score: float         # 0–100

    # ── Learning Plan (Node 4 output) ─────────────────────────────────────────
    learning_plan: dict          # full structured plan (matches LearningPlanResponse schema)

    # ── Error handling ────────────────────────────────────────────────────────
    error: Optional[str]