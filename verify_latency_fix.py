import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
import uuid

# Mocking parts of the app to test the logic in isolation
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

# We'll mock storage and graph
class TestLatencyFix(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_ready_state_success(self):
        # Mock graph
        graph = MagicMock()
        graph.aget_state = AsyncMock()
        
        # Mock messages
        from langchain_core.messages import AIMessage
        ai_message = AIMessage(content="Hello? How are you?")
        
        # Snapshot values
        snapshot = MagicMock()
        snapshot.values = {
            "messages": [ai_message],
            "required_skills": [{"name": "Python"}]
        }
        
        # aget_state returns nothing first, then message
        graph.aget_state.side_effect = [None, snapshot]
        
        config = {"configurable": {"thread_id": "test-thread"}}
        
        # Import the function under test
        from app.api.routes.assessment import _wait_for_assessment_ready_state
        
        # We need to mock storage.get_assessment_by_thread too
        import app.api.routes.assessment as assessment_module
        assessment_module.storage = AsyncMock()
        assessment_module.storage.get_assessment_by_thread.return_value = {"status": "assessing"}

        # Run the function
        result = await _wait_for_assessment_ready_state(graph, config, timeout=2)
        
        self.assertEqual(result["messages"][0].content, "Hello? How are you?")
        self.assertEqual(graph.aget_state.call_count, 2)

    async def test_wait_for_ready_state_failure(self):
        # Mock graph
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=None)
        
        config = {"configurable": {"thread_id": "test-thread"}}
        
        # Import the function under test
        import app.api.routes.assessment as assessment_module
        assessment_module.storage = AsyncMock()
        # Mock storage returning failed status
        assessment_module.storage.get_assessment_by_thread.return_value = {"status": "failed", "id": "uuid"}

        # Run the function
        from app.api.routes.assessment import _wait_for_assessment_ready_state
        result = await _wait_for_assessment_ready_state(graph, config, timeout=2)
        
        # Should return {} if status is failed
        self.assertEqual(result, {})

if __name__ == "__main__":
    unittest.main()
