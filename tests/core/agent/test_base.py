import pytest
from srcs.core.agent.base import BaseAgent
from mcp_agent import MCPApp

class MockAgent(BaseAgent):
    async def run_workflow(self, *args, **kwargs):
        return "workflow executed"

@pytest.fixture
def mock_agent():
    return MockAgent(name="test_agent", instruction="test_instruction")

def test_base_agent_initialization(mock_agent):
    assert mock_agent.name == "test_agent"
    assert mock_agent.instruction == "test_instruction"
    assert isinstance(mock_agent.app, MCPApp)
    assert mock_agent.logger is not None 