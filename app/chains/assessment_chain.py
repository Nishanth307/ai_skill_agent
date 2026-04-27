"""
High-level orchestration helpers for assessment workflows.
"""

from app.graph.graph import get_graph


async def start_assessment_workflow(initial_state: dict, thread_id: str) -> dict:
    """
    Run the graph from the beginning until it pauses at assessor.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(initial_state, config=config)


async def continue_assessment_workflow(thread_id: str, message: str) -> dict:
    """
    Continue a paused assessment session with a user message.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
