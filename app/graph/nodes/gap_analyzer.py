"""
app/graph/nodes/gap_analyzer.py

Node 3: Gap Analyzer
─────────────────────
Input  : required_skills + assessment_results (from state)
Output : skill_gaps, strengths, overall_score

Runs ONCE after all skills have been assessed.
Classifies gaps by severity and identifies adjacent skills
the candidate can realistically bridge.
"""

from app.graph.state import AgentState, SkillGap
from app.core.llm import get_structured_llm
from app.prompts.templates import GAP_ANALYZER_PROMPT
from pydantic import BaseModel, Field


# ── Structured output schema for gap analysis ─────────────────────────────────

class GapAnalysisOutput(BaseModel):
    skill_gaps: list[dict] = Field(
        description="List of skill gaps with severity and adjacency"
    )
    strengths: list[str] = Field(
        description="Skills where candidate scored 4 or 5"
    )
    overall_score: float = Field(
        ge=0, le=100,
        description="Weighted overall score 0-100"
    )
    summary: str = Field(
        description="2-3 sentence narrative summary of the candidate's profile"
    )


async def gap_analyzer_node(state: AgentState) -> dict:
    """
    Analyzes assessment results to produce structured gap analysis.
    """
    print("🔬 [gap_analyzer] Analyzing skill gaps...")

    required_skills = state["required_skills"]
    assessment_results = state["assessment_results"]
    claimed_skills = state.get("claimed_skills", [])

    # Build a combined view for the LLM
    skills_and_scores = _build_skills_summary(required_skills, assessment_results, claimed_skills)

    try:
        llm = get_structured_llm(GapAnalysisOutput)
        chain = GAP_ANALYZER_PROMPT | llm

        result: GapAnalysisOutput = await chain.ainvoke({
            "job_title": state.get("job_title", "the role"),
            "seniority_level": state.get("seniority_level", "mid-level"),
            "skills_and_scores": skills_and_scores,
        })

        # Convert to typed SkillGap list
        skill_gaps: list[SkillGap] = []
        for gap in result.skill_gaps:
            skill_gaps.append(SkillGap(
                skill=gap.get("skill", ""),
                current_score=gap.get("current_score", 1),
                required_level=gap.get("required_level", "intermediate"),
                gap_severity=gap.get("gap_severity", "moderate"),
                is_adjacent=gap.get("is_adjacent", False),
            ))

        print(f"   ✅ Found {len(skill_gaps)} gaps | "
              f"Strengths: {result.strengths} | "
              f"Overall score: {result.overall_score:.1f}/100")

        return {
            "skill_gaps": skill_gaps,
            "strengths": result.strengths,
            "overall_score": result.overall_score,
        }

    except Exception as e:
        print(f"   ❌ [gap_analyzer] Error: {e}")
        # Fallback: compute gaps deterministically without LLM
        return _fallback_gap_analysis(required_skills, assessment_results, claimed_skills)


def _build_skills_summary(required_skills: list, assessment_results: list, claimed_skills: list) -> str:
    """
    Build a human-readable summary for the LLM to analyze.
    Maps assessment scores back to required skills and includes resume status.
    """
    score_map = {r["skill"]: r for r in assessment_results}
    claimed_set = {s.lower() for s in claimed_skills}
    lines = []

    for skill in required_skills:
        name = skill["name"]
        in_resume = name.lower() in claimed_set
        assessment = score_map.get(name)
        score = assessment["score"] if assessment else 0
        rating = assessment["rating"] if assessment else "MISSING"
        confidence = assessment["confidence"] if assessment else "N/A"
        
        resume_status = "PRESENT in resume" if in_resume else "NOT in resume"
        
        lines.append(
            f"- {name}: {resume_status}, required={skill['required_level']}, "
            f"score={score}/5, rating={rating}, confidence={confidence}"
        )

    return "\n".join(lines)


def _fallback_gap_analysis(required_skills: list, assessment_results: list, claimed_skills: list) -> dict:
    """
    Simple rule-based fallback if the LLM call fails.
    Updated with Resume vs Assessment logic.
    """
    score_map = {r["skill"]: r["score"] for r in assessment_results}
    claimed_set = {s.lower() for s in claimed_skills}
    level_threshold = {"beginner": 2, "intermediate": 3, "advanced": 4, "expert": 5}

    gaps = []
    strengths = []
    total_score = 0

    for skill in required_skills:
        name = skill["name"]
        score = score_map.get(name, 1)
        required = skill["required_level"]
        threshold = level_threshold.get(required, 3)
        total_score += score

        if score >= 4:
            strengths.append(name)

        if score < threshold:
            severity = "critical" if score <= 2 else "moderate" if score <= 3 else "minor"
            gaps.append(SkillGap(
                skill=name,
                current_score=score,
                required_level=required,
                gap_severity=severity,
                is_adjacent=False,   # conservative fallback
            ))

    overall = (total_score / (len(required_skills) * 5) * 100) if required_skills else 0

    return {
        "skill_gaps": gaps,
        "strengths": strengths,
        "overall_score": round(overall, 1),
    }