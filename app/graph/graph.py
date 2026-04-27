"""
app/graph/graph.py — Full LangGraph implementation
"""
from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import AgentState
from app.graph.nodes.skill_extractor import skill_extractor_node
from app.graph.nodes.assessor import assessor_node, should_continue_assessment
from app.graph.nodes.gap_analyzer import gap_analyzer_node
from app.graph.nodes.plan_generator import plan_generator_node
from app.core.config import settings
from app.memory.checkpointer import get_checkpointer


async def error_handler_node(state: AgentState) -> dict:
    error_msg = state.get("error", "An unknown error occurred")
    print(f"🚨 [error_handler] {error_msg}")
    return {
        "assessment_complete": True,
        "learning_plan": {
            "skill_items": [], "weekly_schedule": [],
            "total_hours": 0, "duration_weeks": 0,
            "summary": f"Assessment could not be completed: {error_msg}",
        },
    }


def _build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("skill_extractor", skill_extractor_node)
    builder.add_node("assessor", assessor_node)
    builder.add_node("gap_analyzer", gap_analyzer_node)
    builder.add_node("plan_generator", plan_generator_node)
    builder.add_node("error_handler", error_handler_node)

    builder.add_edge(START, "skill_extractor")
    builder.add_edge("skill_extractor", "assessor")
    builder.add_conditional_edges(
        "assessor",
        should_continue_assessment,
        {"continue": "assessor", "analyze": "gap_analyzer", "error": "error_handler"},
    )
    builder.add_edge("gap_analyzer", "plan_generator")
    builder.add_edge("plan_generator", END)
    builder.add_edge("error_handler", END)

    return builder




@lru_cache()
def get_graph():
    """
    Returns the compiled LangGraph.
    interrupt_after=["assessor"] pauses graph after assessor runs,
    enabling human-in-the-loop conversation across HTTP requests.
    """
    graph = _build_graph().compile(
        checkpointer=get_checkpointer(),
        interrupt_after=["assessor"],
    )
    print("✅ LangGraph compiled successfully")
    return graph


def print_graph_structure():
    print(get_graph().get_graph().draw_ascii())