"""
app/graph/nodes/plan_generator.py

Node 4: Learning Plan Generator
─────────────────────────────────
Input  : skill_gaps, strengths, job_title, seniority_level (from state)
Output : learning_plan (full structured plan written to state)

Final node — runs once after gap analysis.
Generates a personalised, week-by-week learning plan with curated resources.
"""

from app.graph.state import AgentState
from app.core.llm import get_planning_llm
from app.prompts.templates import PLAN_GENERATOR_PROMPT
from pydantic import BaseModel, Field
from typing import Optional


# ── Structured output schema ───────────────────────────────────────────────────

class ResourceSchema(BaseModel):
    title: str
    url: Optional[str] = None
    type: str = Field(description="course | book | doc | video | practice")
    platform: Optional[str] = None
    is_free: bool = True
    estimated_hours: int


class SkillLearningItemSchema(BaseModel):
    skill: str
    current_score: int
    target_score: int
    priority: str = Field(description="high | medium | low")
    rationale: str
    resources: list[ResourceSchema]
    estimated_hours: int
    milestones: list[str]


class WeeklyScheduleSchema(BaseModel):
    week: int
    focus_skills: list[str]
    goals: list[str]
    hours_per_day: float


class LearningPlanSchema(BaseModel):
    skill_items: list[SkillLearningItemSchema]
    weekly_schedule: list[WeeklyScheduleSchema]
    total_hours: int
    duration_weeks: int
    summary: str
    assessment_report: str = Field(..., description="The complete PHASE 7: FINAL OUTPUT report as specified by the user.")


async def plan_generator_node(state: AgentState) -> dict:
    """
    Generates the final personalised learning plan.
    """
    print("📚 [plan_generator] Generating personalised learning plan...")

    skill_gaps = state.get("skill_gaps", [])
    strengths = state.get("strengths", [])

    # We ALWAYS run the planner to get the final assessment report (PHASE 7),
    # even if there are no gaps (Perfect Match case).
    if not skill_gaps and not strengths:
         print("   ⚠️ No gaps or strengths found. Returning generic plan.")
         return {
            "learning_plan": {
                "skill_items": [],
                "weekly_schedule": [],
                "total_hours": 0,
                "duration_weeks": 0,
                "summary": "Assessment ended before specific strengths or gaps could be identified.",
                "assessment_report": "Partial assessment. No strengths or gaps identified."
            }
         }

    try:
        # Use with_structured_output for clean plan generation
        llm = get_planning_llm().with_structured_output(LearningPlanSchema)
        chain = PLAN_GENERATOR_PROMPT | llm

        # Format skill gaps for the prompt
        gaps_text = _format_gaps_for_prompt(skill_gaps)

        result: LearningPlanSchema = await chain.ainvoke({
            "job_title": state.get("job_title", "the role"),
            "seniority_level": state.get("seniority_level", "mid-level"),
            "strengths": ", ".join(strengths) if strengths else "general programming knowledge",
            "hours_per_week": 10,    # sensible default; could be user-configurable
            "skill_gaps": gaps_text,
        })

        plan_dict = result.model_dump()

        print(f"   ✅ Plan generated: {result.duration_weeks} weeks | "
              f"{result.total_hours} total hours | "
              f"{len(result.skill_items)} skills to address")

        return {"learning_plan": plan_dict}

    except Exception as e:
        print(f"   ❌ [plan_generator] Error: {e}")
        # Fallback: generate a basic plan without LLM
        return {"learning_plan": _fallback_plan(skill_gaps, state)}


def _format_gaps_for_prompt(skill_gaps: list) -> str:
    """Format skill gaps into a readable string for the LLM."""
    lines = []
    for gap in skill_gaps:
        adjacent = " (adjacent — can leverage existing knowledge)" if gap.get("is_adjacent") else ""
        lines.append(
            f"- {gap['skill']}: current score {gap['current_score']}/5, "
            f"required level: {gap['required_level']}, "
            f"severity: {gap['gap_severity']}{adjacent}"
        )
    return "\n".join(lines)


def _fallback_plan(skill_gaps: list, state: AgentState) -> dict:
    """
    Rule-based fallback plan if LLM call fails.
    Points to generic high-quality resources.
    """
    skill_items = []
    total_hours = 0

    for gap in skill_gaps:
        hours = {"critical": 40, "moderate": 20, "minor": 10}.get(
            gap.get("gap_severity", "moderate"), 20
        )
        total_hours += hours
        skill_items.append({
            "skill": gap["skill"],
            "current_score": gap["current_score"],
            "target_score": min(gap["current_score"] + 2, 5),
            "priority": "high" if gap.get("gap_severity") == "critical" else "medium",
            "rationale": f"Required for the {state.get('job_title', 'role')}",
            "resources": [
                {
                    "title": f"Official {gap['skill']} documentation",
                    "url": "https://www.google.com/search?q=" + gap['skill'].replace(' ', '+') + "+official+documentation",
                    "type": "doc",
                    "platform": "Web",
                    "is_free": True,
                    "estimated_hours": max(1, hours // 2),
                },
                {
                    "title": f"{gap['skill']} hands-on exercise",
                    "url": "https://www.google.com/search?q=" + gap['skill'].replace(' ', '+') + "+tutorial",
                    "type": "practice",
                    "platform": "Web",
                    "is_free": True,
                    "estimated_hours": max(1, hours // 2),
                },
            ],
            "estimated_hours": hours,
            "milestones": [
                f"Complete basic {gap['skill']} tutorial",
                f"Build a small project using {gap['skill']}",
            ],
        })

    duration_weeks = max(1, total_hours // 10)

    return {
        "skill_items": skill_items,
        "weekly_schedule": [
            {
                "week": i + 1,
                "focus_skills": [gap["skill"] for gap in skill_gaps[:2]],
                "goals": ["Complete assigned resources", "Build practice project"],
                "hours_per_day": 1.5,
            }
            for i in range(duration_weeks)
        ],
        "total_hours": total_hours,
        "duration_weeks": duration_weeks,
        "summary": (
            f"Your learning plan covers {len(skill_gaps)} skill gaps "
            f"over {duration_weeks} weeks (~{total_hours} hours total). "
            f"Start with the highest priority items first."
        ),
        "assessment_report": (
            "----------------------------------------\n"
            "📊 ASSESSMENT REPORT\n\n"
            f"✅ Strong Skills:\n- {', '.join(strengths) if strengths else 'None'}\n\n"
            "⚠️ Weak/Missing Skills:\n" + "\n".join([f"- {g['skill']}" for g in skill_gaps])
        )
    }