"""
app/graph/nodes/skill_extractor.py

Node 1: Skill Extractor
───────────────────────
Input  : job_description, resume (from AgentState)
Output : required_skills, claimed_skills, job_title, seniority_level (written back to state)

This node runs ONCE at the start of every assessment.
It uses .with_structured_output() to force the LLM to return a validated
Pydantic object — no regex parsing, no JSON wrangling.
"""

from langchain_core.messages import AIMessage
import re

from app.graph.state import AgentState, SkillInfo
from app.core.llm import get_structured_llm
from app.core.config import settings
from app.schemas.schemas import ExtractedSkills
from app.prompts.templates import (
    SKILL_EXTRACTOR_PROMPT,
    ASSESSMENT_INTRO_MESSAGE,
)


async def skill_extractor_node(state: AgentState) -> dict:
    """
    Parses JD + resume and extracts structured skill data.

    LangGraph concept: nodes return a PARTIAL state dict.
    LangGraph merges it into the full state — you only return
    what changed, not the entire state.
    """
    print("🔍 [skill_extractor] Extracting skills from JD and resume...")

    try:
        # Build a structured LLM — returns ExtractedSkills Pydantic object
        # temperature=0 for maximum consistency in extraction
        llm = get_structured_llm(ExtractedSkills)

        # Build the chain: prompt | llm
        # The | operator is LangChain's pipe syntax (like Unix pipes)
        chain = SKILL_EXTRACTOR_PROMPT | llm

        # Invoke with template variables
        result: ExtractedSkills = await chain.ainvoke({
            "job_description": state["job_description"],
            "resume": state["resume"],
            "max_skills": settings.MAX_SKILLS_TO_ASSESS,
        })

        # Convert Pydantic objects to TypedDicts for state compatibility
        required_skills: list[SkillInfo] = [
            SkillInfo(
                name=skill.name,
                required_level=skill.required_level,
                claimed_by_candidate=skill.claimed_by_candidate,
                context=skill.context,
            )
            for skill in result.required_skills
        ]

        # Build the intro message to kick off the conversation
        first_skill = required_skills[0]["name"] if required_skills else "your skills"
        job_title = result.job_title or "this role"
        intro = ASSESSMENT_INTRO_MESSAGE.format(
            total_skills=len(required_skills),
            job_title=job_title,
            first_skill=first_skill,
        )

        print(f"   ✅ Extracted {len(required_skills)} skills | Job: {job_title}")
        print(f"   📋 Skills: {[s['name'] for s in required_skills]}")
        print(f"   🔴 Preliminary gaps: {result.preliminary_gaps}")

        # Return only the fields that changed — LangGraph merges these
        return {
            "required_skills": required_skills,
            "claimed_skills": result.candidate_claimed_skills,
            "job_title": result.job_title,
            "seniority_level": result.seniority_level,
            "current_skill_index": 0,
            "assessment_results": [],
            "questions_asked_for_current_skill": 0,
            "assessment_complete": False,
            "skill_gaps": [],
            "strengths": [],
            "overall_score": 0.0,
            "error": None,
            # Add the intro message to kick off the conversation
            "messages": [AIMessage(content=intro)],
        }

    except Exception as e:
        print(f"   ❌ [skill_extractor] Error: {e}")
        # Fallback to deterministic keyword extraction so the interview can still proceed.
        fallback_skills = _fallback_extract_skills(
            job_description=state.get("job_description", ""),
            resume=state.get("resume", ""),
            max_skills=settings.MAX_SKILLS_TO_ASSESS,
        )
        intro = ASSESSMENT_INTRO_MESSAGE.format(
            total_skills=len(fallback_skills),
            job_title="this role",
            first_skill=fallback_skills[0]["name"] if fallback_skills else "your core skills",
        )
        return {
            "error": f"Skill extraction failed: {str(e)}",
            "required_skills": fallback_skills,
            "claimed_skills": [s["name"] for s in fallback_skills if s["claimed_by_candidate"]],
            "current_skill_index": 0,
            "assessment_results": [],
            "questions_asked_for_current_skill": 0,
            "assessment_complete": False,
            "skill_gaps": [],
            "strengths": [],
            "overall_score": 0.0,
            "messages": [AIMessage(content=intro)],
        }


def _fallback_extract_skills(job_description: str, resume: str, max_skills: int) -> list[SkillInfo]:
    """Rule-based extraction fallback when LLM extraction fails."""
    jd = (job_description or "").lower()
    cv = (resume or "").lower()
    canonical_skills = [
        "python", "java", "javascript", "typescript", "fastapi", "django", "flask",
        "react", "node.js", "sql", "postgresql", "mysql", "mongodb", "redis",
        "docker", "kubernetes", "aws", "gcp", "azure", "system design", "microservices",
        "kafka", "rabbitmq", "ci/cd", "terraform", "linux", "testing",
    ]
    found = []
    for skill in canonical_skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, jd):
            found.append(
                SkillInfo(
                    name=skill.title(),
                    required_level="intermediate",
                    claimed_by_candidate=bool(re.search(pattern, cv)),
                    context=f"Detected in JD via fallback extractor: {skill}",
                )
            )
        if len(found) >= max_skills:
            break
    if not found:
        found = [
            SkillInfo(
                name="Core Technical Skills",
                required_level="intermediate",
                claimed_by_candidate=True,
                context="Fallback skill when no concrete keywords were detected.",
            )
        ]
    return found