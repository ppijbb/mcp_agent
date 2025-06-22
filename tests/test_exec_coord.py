import pytest, os, asyncio
from srcs.product_planner_agent.coordinators.executive_coordinator import ExecutiveCoordinator

@pytest.mark.asyncio
async def test_turn_budget_exhaustion():
    os.environ["AGENT_MAX_TURNS"] = "1"  # intentionally low
    exec_coord = ExecutiveCoordinator()
    with pytest.raises(RuntimeError):
        await exec_coord.run(initial_prompt="Test prompt") 