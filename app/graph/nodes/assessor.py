"""
app/graph/nodes/assessor.py

Node 2: Assessor (Conversational Loop)
───────────────────────────────────────
This is the most complex node — it handles multi-turn conversation.

The loop works like this:
  1. Ask an opening question for the current skill
  2. Human answers  → LangGraph pauses here (human-in-the-loop interrupt)
  3. Node resumes → evaluate answer, decide: follow-up OR score & move on
  4. Repeat until all skills assessed

Key LangGraph concepts used here:
  - interrupt_before=["assessor"] → pauses graph, waits for human input
  - current_skill_index pointer → tracks which skill we're on
  - Conditional edge (defined in graph.py) decides if we loop or move forward
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import AgentState, AssessmentResult
from app.core.llm import get_assessor_llm, get_structured_llm
from app.core.config import settings
from app.schemas.schemas import SkillAssessmentScore
from app.prompts.templates import (
    ASSESSOR_OPENING_PROMPT,
    ASSESSOR_FOLLOWUP_PROMPT,
    ASSESSOR_SCORING_PROMPT,
    ASSESSMENT_TRANSITION_MESSAGE,
    ASSESSMENT_COMPLETE_MESSAGE,
)


async def assessor_node(state: AgentState) -> dict:
    """
    Drives the conversational skill assessment.

    Called after every human message. Decides whether to:
      A) Ask a follow-up question (loop back)
      B) Score the skill and move to the next one
      C) Mark assessment complete (all skills done)
    """
    current_idx = state.get("current_skill_index", 0)
    required_skills = state.get("required_skills", [])
    messages = state.get("messages", [])
    questions_asked = state.get("questions_asked_for_current_skill", 0)
    total_asked = state.get("total_questions_asked", 0)
    MAX_TOTAL = 15

    # ── High Priority: Check for Exit Command ────────────────────────────────
    last_human = _last_human_message(messages)
    last_human_content = (_get_text_content(last_human) or "").strip().lower()
    if last_human and last_human_content in ["end", "exit", "quit", "stop"]:
        print(f"   🛑 User triggered early exit via '{last_human_content}' at entry.")
        return await _end_assessment_immediately(state)

    # Hard Stop: limit reached
    if total_asked >= MAX_TOTAL:
        print(f"   🛑 Hard stop reached ({total_asked} questions). Moving to evaluation.")
        return _mark_complete(state)

    # Guard: all skills assessed
    if current_idx >= len(required_skills):
        return _mark_complete(state)

    current_skill = required_skills[current_idx]
    skill_name = current_skill["name"]

    print(f"💬 [assessor] Global Progress: {total_asked}/{MAX_TOTAL} | "
          f"Skill {current_idx + 1}/{len(required_skills)}: {skill_name} "
          f"| Questions asked (this skill): {questions_asked}")

    # ── CASE 1: No messages yet for this skill → ask opening question ──────────
    # Check if the last message is from AI (means we just transitioned to new skill)
    # or if this is the very first message
    last_message = messages[-1] if messages else None
    is_opening = (
        last_message is None
        or (isinstance(last_message, AIMessage) and questions_asked == 0)
    )

    if is_opening and questions_asked == 0:
        return await _ask_opening_question(state, current_skill)

    # ── CASE 2: Human just answered → evaluate and decide ─────────────────────
    return await _evaluate_and_decide(state, current_skill)


async def _ask_opening_question(state: AgentState, current_skill: dict) -> dict:
    """Generate and return the opening question for a skill."""
    llm = get_assessor_llm()
    chain = ASSESSOR_OPENING_PROMPT | llm

    required_skills = state["required_skills"]
    current_idx = state["current_skill_index"]
    question_style = _question_style_for_index(current_idx)

    response = await chain.ainvoke({
        "skill_name": current_skill["name"],
        "required_level": current_skill["required_level"],
        "claimed_by_candidate": current_skill["claimed_by_candidate"],
        "skill_number": state["current_skill_index"] + 1,
        "total_skills": len(state["required_skills"]),
        "question_style": question_style,
        "difficulty_level": "medium",
        "asked_questions": state.get("asked_questions", []),
    })

    print(f"   ❓ Opening question: {response.content[:80]}...")
    
    q_content = response.content
    asked = state.get("asked_questions", []) + [q_content]
    total_asked = state.get("total_questions_asked", 0) + 1

    return {
        "messages": [AIMessage(content=q_content)],
        "questions_asked_for_current_skill": 1,
        "asked_questions": asked,
        "total_questions_asked": total_asked,
    }


async def _evaluate_and_decide(state: AgentState, current_skill: dict) -> dict:
    """
    After a human answer: decide to follow-up OR score and advance.
    """
    questions_asked = state["questions_asked_for_current_skill"]
    max_questions = settings.MAX_QUESTIONS_PER_SKILL
    last_human = _last_human_message(state.get("messages", []))
    last_human_content = _get_text_content(last_human) if last_human else ""
    
    # ── High Priority: Check for "end" command ────────────────────────────────
    if last_human and last_human_content.strip().lower() in ["end", "exit", "quit", "stop"]:
        print(f"   🛑 User triggered early exit via '{last_human_content}' command.")
        return await _end_assessment_immediately(state)

    if last_human and _is_skip_message(last_human_content):
        print(f"   ⏭️ Candidate skipped {current_skill['name']} question.")
        return await _score_skipped_and_advance(state, current_skill)

    # Build conversation transcript for this skill only
    # (messages from the current skill's conversation)
    conversation_transcript = _extract_skill_conversation(
        state["messages"], questions_asked
    )

    # ── Should we ask a follow-up? ─────────────────────────────────────────────
    total_asked = state.get("total_questions_asked", 0)
    MAX_TOTAL = 15
    
    if questions_asked < max_questions and total_asked < MAX_TOTAL:
        llm = get_assessor_llm()
        chain = ASSESSOR_FOLLOWUP_PROMPT | llm

        response = await chain.ainvoke({
            "skill_name": current_skill["name"],
            "required_level": current_skill["required_level"],
            "questions_asked": questions_asked,
            "max_questions": max_questions,
            "conversation_so_far": conversation_transcript,
            "question_style": _next_question_style(state["current_skill_index"], questions_asked),
            "difficulty_adjustment": _difficulty_adjustment_from_answer(
                last_human_content
            ),
            "asked_questions": state.get("asked_questions", []),
        })

        # Check if the LLM decided it has enough signal
        if "[ASSESSMENT_COMPLETE]" in response.content:
            print(f"   ✅ LLM has enough signal for {current_skill['name']} — scoring now")
            return await _score_and_advance(state, current_skill, conversation_transcript)

        print(f"   ↩️  Follow-up: {response.content[:80]}...")
        
        q_content = response.content
        asked = state.get("asked_questions", []) + [q_content]
        
        return {
            "messages": [AIMessage(content=q_content)],
            "questions_asked_for_current_skill": questions_asked + 1,
            "asked_questions": asked,
            "total_questions_asked": total_asked + 1,
        }

    # ── Max questions reached → score and advance ──────────────────────────────
    print(f"   🏁 Max questions reached for {current_skill['name']} — scoring")
    return await _score_and_advance(state, current_skill, conversation_transcript)


async def _score_and_advance(
    state: AgentState,
    current_skill: dict,
    conversation_transcript: str,
) -> dict:
    """
    Score the current skill using LLM structured output,
    then advance to next skill or mark assessment complete.
    """
    # Score the skill with structured output
    scoring_llm = get_structured_llm(SkillAssessmentScore)
    scoring_chain = ASSESSOR_SCORING_PROMPT | scoring_llm

    score_result: SkillAssessmentScore = await scoring_chain.ainvoke({
        "skill_name": current_skill["name"],
        "required_level": current_skill["required_level"],
        "conversation_transcript": conversation_transcript,
    })

    print(f"   📊 Rating for {current_skill['name']}: {score_result.rating} "
          f"({score_result.confidence} confidence)")

    # Build AssessmentResult
    result = AssessmentResult(
        skill=current_skill["name"],
        score=score_result.score,
        rating=score_result.rating.upper(),  # Ensure canonical STRONG|AVERAGE|WEAK|MISSING
        confidence=score_result.confidence,
        evidence=score_result.evidence,
        questions_asked=_extract_questions(state["messages"]),
    )

    # Accumulate results
    updated_results = list(state.get("assessment_results", [])) + [result]

    # ── Advance to next skill ──────────────────────────────────────────────────
    next_idx = state["current_skill_index"] + 1
    required_skills = state["required_skills"]

    if next_idx >= len(required_skills):
        # All skills done!
        complete_msg = ASSESSMENT_COMPLETE_MESSAGE.format(
            total_skills=len(required_skills)
        )
        return {
            "assessment_results": updated_results,
            "assessment_complete": True,
            "current_skill_index": next_idx,
            "questions_asked_for_current_skill": 0,
            "messages": [AIMessage(content=complete_msg)],
        }

    return {
        "assessment_results": updated_results,
        "current_skill_index": next_idx,
        "questions_asked_for_current_skill": 0,
        "messages": [], # No transition message, let next assessor run ask opening question
    }


def _mark_complete(state: AgentState) -> dict:
    """Mark assessment complete and notify the user."""
    from app.prompts.templates import ASSESSMENT_COMPLETE_MESSAGE
    required_skills = state.get("required_skills", [])
    msg = ASSESSMENT_COMPLETE_MESSAGE.format(total_skills=len(required_skills))
    return {
        "assessment_complete": True,
        "messages": [AIMessage(content=msg)],
    }


def _extract_skill_conversation(messages: list, questions_asked: int) -> str:
    """
    Extract the last N message pairs relevant to the current skill.
    questions_asked tells us how deep into the conversation we are.
    """
    # Each skill has: AI question → Human answer → (optional AI followup → Human answer)
    # So the last questions_asked * 2 messages are for the current skill
    relevant = messages[-(questions_asked * 2):]  if questions_asked > 0 else messages[-2:]
    lines = []
    for msg in relevant:
        role = "Interviewer" if isinstance(msg, AIMessage) else "Candidate"
        content = _get_text_content(msg)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_text_content(msg) -> str:
    """Safely extract string content from a message (handles lists/dicts from newer LLMs)."""
    if not msg:
        return ""
    content = msg.content if hasattr(msg, "content") else str(msg.get("content", ""))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                 text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content)


def _extract_questions(messages: list) -> list[str]:
    """Extract AI questions from message history (fallback to parsing if needed)."""
    questions = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            content = _get_text_content(msg)
            if "?" in content:
                questions.append(content)
    return questions


def _last_human_message(messages: list):
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


async def _end_assessment_immediately(state: AgentState) -> dict:
    """
    Handle the 'end' command by scoring current skill and marking rest as MISSING.
    """
    current_idx = state["current_skill_index"]
    required_skills = state["required_skills"]
    results = list(state.get("assessment_results", []))
    
    # 1. Process current skill (if not already processed)
    if current_idx < len(required_skills):
        current_skill = required_skills[current_idx]
        questions_asked = state["questions_asked_for_current_skill"]
        conversation_transcript = _extract_skill_conversation(state["messages"], questions_asked)
        
        # If we have at least one question and answer, try to score it
        if questions_asked >= 1:
            print(f"   📊 Scoring {current_skill['name']} with partial data...")
            scoring_llm = get_structured_llm(SkillAssessmentScore)
            scoring_chain = ASSESSOR_SCORING_PROMPT | scoring_llm
            try:
                score_result: SkillAssessmentScore = await scoring_chain.ainvoke({
                    "skill_name": current_skill["name"],
                    "required_level": current_skill["required_level"],
                    "conversation_transcript": conversation_transcript,
                })
                results.append(AssessmentResult(
                    skill=current_skill["name"],
                    score=score_result.score,
                    rating=score_result.rating.upper(),
                    confidence="medium",
                    evidence=score_result.evidence + " (Assessment ended early by user)",
                    questions_asked=_extract_questions(state["messages"]),
                ))
            except Exception:
                 # Fallback to MISSING if scoring fails
                 results.append(_create_missing_result(current_skill["name"], state))
        else:
            results.append(_create_missing_result(current_skill["name"], state))

    # 2. Mark all remaining skills as MISSING
    for i in range(current_idx + 1, len(required_skills)):
        results.append(_create_missing_result(required_skills[i]["name"], state))

    return {
        "assessment_results": results,
        "assessment_complete": True,
        "current_skill_index": len(required_skills),
        "questions_asked_for_current_skill": 0,
        "messages": [AIMessage(content=ASSESSMENT_COMPLETE_MESSAGE.format(total_skills=len(required_skills)))],
    }


def _create_missing_result(skill_name: str, state: AgentState) -> AssessmentResult:
    return AssessmentResult(
        skill=skill_name,
        score=1,
        rating="MISSING",
        confidence="high",
        evidence="Assessment ended early by user before this skill was evaluated.",
        questions_asked=[],
    )


def _is_skip_message(content: str) -> bool:
    normalized = (content or "").strip().lower()
    return normalized in {
        "skip", "pass", "idk", "i don't know", "dont know", "no idea", "next", 
        "don't have any idea", "don't know", "skip this", "move on", "next question"
    }


def _score_to_rating(score: int) -> str:
    if score >= 4:
        return "Strong"
    if score == 3:
        return "متوسط (Average)"
    if score == 2:
        return "Weak"
    return "Missing"


def _question_style_for_index(skill_index: int) -> str:
    styles = ["scenario", "mcq", "coding"]
    return styles[skill_index % len(styles)]


def _next_question_style(skill_index: int, questions_asked: int) -> str:
    styles = ["scenario", "mcq", "coding"]
    return styles[(skill_index + questions_asked) % len(styles)]


def _difficulty_adjustment_from_answer(answer: str) -> str:
    text = (answer or "").strip().lower()
    if len(text.split()) < 3 or any(token in text for token in ["idk", "don't know", "not sure", "no idea"]):
        return "decrease"
    if len(text) > 80 and any(
        token in text for token in ["trade-off", "latency", "complexity", "scal", "failure", "edge"]
    ):
        return "increase"
    return "maintain"


async def _score_skipped_and_advance(state: AgentState, current_skill: dict) -> dict:
    """Assign MISSING rating when candidate explicitly skips."""
    result = AssessmentResult(
        skill=current_skill["name"],
        score=1,
        rating="MISSING",
        confidence="high",
        evidence="Candidate skipped the assessment question for this skill.",
        questions_asked=_extract_questions(state["messages"]),
    )
    updated_results = list(state.get("assessment_results", [])) + [result]
    next_idx = state["current_skill_index"] + 1
    required_skills = state["required_skills"]

    if next_idx >= len(required_skills):
        complete_msg = ASSESSMENT_COMPLETE_MESSAGE.format(total_skills=len(required_skills))
        return {
            "assessment_results": updated_results,
            "assessment_complete": True,
            "current_skill_index": next_idx,
            "questions_asked_for_current_skill": 0,
            "messages": [AIMessage(content=complete_msg)],
        }

    return {
        "assessment_results": updated_results,
        "current_skill_index": next_idx,
        "questions_asked_for_current_skill": 0,
        "messages": [],
    }


# ── Conditional edge function ──────────────────────────────────────────────────
def should_continue_assessment(state: AgentState) -> str:
    """
    LangGraph conditional edge — decides what happens after assessor runs.

    Returns:
      "continue"  → loop back to assessor (waiting for human input)
      "analyze"   → move to gap_analyzer node
      "error"     → move to error handler
    """
    if state.get("error"):
        return "error"

    if state.get("assessment_complete"):
        print("🏁 [router] Assessment complete → moving to gap_analyzer")
        return "analyze"

    print("🔄 [router] Assessment ongoing → waiting for human input")
    return "continue"