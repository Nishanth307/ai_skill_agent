import uuid
import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.memory.storage import storage
from app.schemas.assessment import AssessmentStatus
from app.schemas.schemas import (
    AssessmentCreate,
    AssessmentResponse,
    AssessmentChatRequest,
    AssessmentChatResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

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

router = APIRouter(prefix="/assessments", tags=["assessments"])


@router.post("/", response_model=AssessmentResponse, status_code=201)
async def start_assessment(
    payload: AssessmentCreate,
    background_tasks: BackgroundTasks,
):
    """
    Kick off a new skill assessment for a candidate against a job description.
    """
    # 1. Validate candidate
    candidate = await storage.get_candidate(payload.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # 2. Create assessment
    thread_id = str(uuid.uuid4())
    assessment = await storage.create_assessment(
        candidate_id=payload.candidate_id,
        job_description=payload.job_description,
        job_title=payload.job_title,
        thread_id=thread_id,
    )

    # 3. Run skill extraction in background
    background_tasks.add_task(
        _run_skill_extraction,
        assessment_id=assessment["id"],
        resume_text=candidate["resume_text"],
        job_description=payload.job_description,
        thread_id=thread_id,
    )

    return assessment


@router.get("/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment(
    assessment_id: uuid.UUID,
):
    """Poll assessment status and results."""
    assessment = await storage.get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    # Fetch progress from graph state if not completed
    progress = None
    if assessment["status"] != AssessmentStatus.completed and assessment.get("thread_id"):
        try:
            from app.graph.graph import get_graph
            graph = get_graph()
            state = await graph.aget_state({"configurable": {"thread_id": assessment["thread_id"]}})
            if state and state.values:
                current_idx = state.values.get("current_skill_index", 0)
                total_skills = len(state.values.get("required_skills", []))
                if total_skills:
                    progress = f"Skill {current_idx + 1} of {total_skills}"
        except Exception:
            pass

    return AssessmentResponse(
        **assessment,
        progress=None
    )


@router.delete("/{assessment_id}")
async def delete_assessment(assessment_id: uuid.UUID):
    """Permanently delete an assessment and its associated data."""
    assessment = await storage.get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    # 1. Delete from vector store if candidate has chroma_doc_id
    candidate = await storage.get_candidate(assessment["candidate_id"])
    if candidate and candidate.get("chroma_doc_id"):
        from app.memory.vector_store import resume_vector_store
        await resume_vector_store.delete_resume(candidate["chroma_doc_id"])
    
    # 2. Delete candidate and assessment from storage
    await storage.delete_candidate(assessment["candidate_id"])
    await storage.delete_assessment(assessment_id)
    
    return {"status": "success", "message": "Assessment and candidate data deleted"}


@router.post("/chat")
async def chat_assessment(
    payload: AssessmentChatRequest,
):
    """
    Continue the conversational assessment with streaming support.
    """
    from app.graph.graph import get_graph
    from langchain_core.messages import AIMessage, HumanMessage

    graph = get_graph()
    config = {"configurable": {"thread_id": payload.thread_id}}

    if payload.message == "__init__":
        result = await _wait_for_assessment_ready_state(graph=graph, config=config)
        
        # Collect all AIMessages that aren't raw JSON
        messages = result.get("messages", [])
        filtered_contents = []
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            content = _get_text_content(msg)
            
            # HARD FILTER: If content looks like a structured JSON skill list or result, SKIP
            c_strip = content.strip()
            if (c_strip.startswith("{") and c_strip.endswith("}")) or ("required_skills" in c_strip and "name" in c_strip):
                print(f"   🚫 Filtering out JSON-like message: {c_strip[:50]}...")
                continue
            
            if content:
                filtered_contents.append(content)
        
        full_message = "\n\n".join(filtered_contents)
        
        # If this is a resume (not first turn), only return the last meaningful AI message
        # to avoid repeating the intro in the UI.
        questions_asked_total = sum(len(r.get("questions_asked", [])) for r in result.get("assessment_results", []))
        if questions_asked_total > 0 or result.get("questions_asked_for_current_skill", 0) > 1:
             # Just return the last one if we have more than one. 
             # (Actually, better to return everything since the last HumanMessage)
             last_human_idx = -1
             for i, m in enumerate(messages):
                 if isinstance(m, HumanMessage):
                     last_human_idx = i
             
             relevant_messages = messages[last_human_idx+1:]
             full_message = "\n\n".join([m.content for m in relevant_messages if isinstance(m, AIMessage) and m.content])
        is_complete = result.get("assessment_complete", False)
        
        current_idx = result.get("current_skill_index", 0)
        total_skills = len(result.get("required_skills", []))
        progress = f"Skill {current_idx + 1} of {total_skills}" if total_skills and not is_complete else None
        current_skill = result["required_skills"][current_idx].get("name") if result.get("required_skills") and current_idx < total_skills and not is_complete else None

        return AssessmentChatResponse(
            thread_id=payload.thread_id,
            message=full_message,
            is_complete=is_complete,
            current_skill=current_skill,
            progress=None,
        )

    # Use a generator for streaming
    async def chat_generator():
        final_state = None
        async for event in graph.astream(
            {"messages": [HumanMessage(content=payload.message)]},
            config=config,
            stream_mode="messages",
        ):
            # We want to yield the content of the AI message as it's being generated
            message, metadata = event
            if isinstance(message, AIMessage) and message.content:
                content = _get_text_content(message)
                c_strip = content.strip()
                
                # HARD FILTER: skip if it looks like raw JSON, structured list, or has internal keywords
                # We check for telltale signs of the skill extractor or assessor structured outputs
                is_json_like = (c_strip.startswith("{") and c_strip.endswith("}")) or \
                              (c_strip.startswith("[") and c_strip.endswith("]"))
                has_internal_keys = any(key in c_strip for key in ["required_skills", "assessment_results", "skill_gaps", "name", "score"])
                
                if is_json_like or (has_internal_keys and len(c_strip) > 100):
                    print(f"   🚫 Filtering out structured data from stream: {c_strip[:40]}...")
                    continue
                yield content

        # After stream finishes, check final state
        snapshot = await graph.aget_state(config)
        if snapshot and snapshot.values.get("assessment_complete"):
            # Update assessment status in background
            assessment = await storage.get_assessment_by_thread(payload.thread_id)
            if assessment:
                from app.schemas.assessment import AssessmentStatus
                await storage.update_assessment(
                    assessment["id"],
                    status=AssessmentStatus.completed,
                    assessment_results=snapshot.values.get("assessment_results", []),
                    skill_gaps=snapshot.values.get("skill_gaps", []),
                    completed_at=datetime.utcnow(),
                )

    return StreamingResponse(chat_generator(), media_type="text/plain")


async def _wait_for_assessment_ready_state(graph, config: dict, timeout: int = 60) -> dict:
    """
    Internal wait loop for the graph state to be ready.
    Eliminates the need for high-frequency client-side polling.
    """
    from langchain_core.messages import AIMessage
    
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        snapshot = await graph.aget_state(config)
        if snapshot and getattr(snapshot, "values", None):
            values = snapshot.values
            messages = values.get("messages", [])
            
            # Check if we have at least one AI message that is a question
            ai_messages = [m for m in messages if isinstance(m, AIMessage)]
            
            def get_content(m):
                c = m.content if hasattr(m, "content") else str(m.get("content", ""))
                if isinstance(c, list):
                    return "".join([p if isinstance(p, str) else p.get("text", "") for p in c])
                return str(c)

            has_question = any("?" in get_content(m) for m in ai_messages)
            
            if len(ai_messages) >= 1 and has_question:
                return values
        
        # Check if assessment failed in storage
        assessment = await storage.get_assessment_by_thread(config["configurable"]["thread_id"])
        if assessment and assessment.get("status") == "failed":
            print(f"⚠️ Assessment {assessment['id']} failed during initialization.")
            return {}

        await asyncio.sleep(0.5) # Poll every 0.5s internally
            
    return {}


async def _run_skill_extraction(
    assessment_id: uuid.UUID,
    resume_text: str,
    job_description: str,
    thread_id: str,
):
    """
    Background task: runs the skill extractor node and updates storage.
    """
    from app.graph.graph import get_graph

    try:
        # Update status → EXTRACTING
        await storage.update_assessment(assessment_id, status=AssessmentStatus.extracting)

        # Run the graph up to the first human checkpoint
        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "job_description": job_description,
            "resume": resume_text,
            "messages": [],
            "current_skill_index": 0,
            "questions_asked_for_current_skill": 0,
            "assessment_results": [],
            "skill_gaps": [],
            "strengths": [],
            "overall_score": 0.0,
            "asked_questions": [],
            "total_questions_asked": 0,
            "assessment_complete": False,
        }

        final_state = await graph.ainvoke(initial_state, config=config)

        # Update status → ASSESSING
        snapshot = await graph.aget_state(config)
        required_skills = []
        if snapshot and getattr(snapshot, "values", None):
            required_skills = snapshot.values.get("required_skills", [])
        
        status = AssessmentStatus.assessing
        if final_state.get("assessment_complete") and not required_skills:
            # Extraction/graph bootstrap failed; do not present this as a completed interview.
            status = AssessmentStatus.failed
        await storage.update_assessment(
            assessment_id,
            status=status,
            required_skills=required_skills,
        )

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            error_msg = "Gemini API Quota Exhausted (limit: 20/day). Please switch to Ollama in .env for unlimited local use."
        
        await storage.update_assessment(
            assessment_id, 
            status=AssessmentStatus.failed,
            error=error_msg
        )
        print(f"Error in _run_skill_extraction: {e}")