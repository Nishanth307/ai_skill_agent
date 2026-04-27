import uuid
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from app.memory.storage import storage
from app.schemas.assessment import AssessmentStatus
from app.schemas.schemas import LearningPlanResponse

router = APIRouter(prefix="/plans", tags=["learning-plans"])


@router.post("/generate/{assessment_id}", status_code=202)
async def generate_plan(
    assessment_id: uuid.UUID,
    background_tasks: BackgroundTasks,
):
    """
    Trigger learning plan generation for a completed assessment.
    """
    assessment = await storage.get_assessment(assessment_id)

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    if assessment["status"] != AssessmentStatus.completed:
        raise HTTPException(
            status_code=422,
            detail=f"Assessment must be completed before generating a plan. Current status: {assessment['status']}"
        )

    # Check if plan already exists
    existing_plan = await storage.get_learning_plan_by_assessment(assessment_id)
    if existing_plan:
        raise HTTPException(status_code=409, detail="Learning plan already exists for this assessment")

    background_tasks.add_task(_generate_plan_task, assessment_id=assessment_id)

    return {
        "message": "Learning plan generation started",
        "assessment_id": str(assessment_id),
        "poll_url": f"/api/v1/plans/{assessment_id}",
    }


@router.get("/{assessment_id}", response_model=LearningPlanResponse)
async def get_plan(
    assessment_id: uuid.UUID,
):
    """Fetch the generated learning plan for an assessment."""
    plan = await storage.get_learning_plan_by_assessment(assessment_id)

    if not plan:
        raise HTTPException(
            status_code=404,
            detail="Learning plan not found. Generate one first via POST /plans/generate/{assessment_id}"
        )

    # Unfold JSON stored plan_data into response schema
    plan_data = plan["plan_data"]
    return LearningPlanResponse(
        id=plan["id"],
        assessment_id=plan["assessment_id"],
        skill_items=plan_data.get("skill_items", []),
        weekly_schedule=plan_data.get("weekly_schedule", []),
        total_hours=plan["total_hours"],
        duration_weeks=plan["duration_weeks"],
        summary=plan_data.get("summary", ""),
        assessment_report=plan_data.get("assessment_report"),
        created_at=plan["created_at"],
    )


async def _generate_plan_task(assessment_id: uuid.UUID):
    """Background task: runs plan_generator node and saves to storage."""
    from app.graph.graph import get_graph

    assessment = await storage.get_assessment(assessment_id)
    if not assessment:
        return

    try:
        await storage.update_assessment(assessment_id, status=AssessmentStatus.analyzing)

        graph = get_graph()
        config = {"configurable": {"thread_id": assessment["thread_id"]}}

        # Resume the graph — it will run gap_analyzer → plan_generator
        final_state = await graph.ainvoke(None, config=config)

        plan_data = final_state.get("learning_plan", {})
        total_hours = sum(
            item.get("estimated_hours", 0)
            for item in plan_data.get("skill_items", [])
        )
        duration_weeks = len(plan_data.get("weekly_schedule", []))

        await storage.create_learning_plan(
            assessment_id=assessment_id,
            plan_data=plan_data,
            total_hours=total_hours,
            duration_weeks=duration_weeks,
        )

        await storage.update_assessment(
            assessment_id,
            status=AssessmentStatus.completed,
            overall_score=final_state.get("overall_score", 0.0),
            skill_gaps=final_state.get("skill_gaps", []),
            strengths=final_state.get("strengths", []),
            assessment_results=final_state.get("assessment_results", []),
            completed_at=datetime.utcnow(),
        )

    except Exception as e:
        await storage.update_assessment(assessment_id, status=AssessmentStatus.failed)
        print(f"Error in _generate_plan_task: {e}")
        raise e