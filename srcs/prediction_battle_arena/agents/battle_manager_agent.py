"""
배틀 관리 에이전트
"""

from mcp_agent.agents.agent import Agent as MCP_Agent



BATTLE_MANAGER_AGENT_INSTRUCTION = """
You are a Battle Manager Agent responsible for managing prediction battles.

Your responsibilities:
1. Create new battles or join existing ones
2. Track battle status (waiting, started, predicting, betting, calculating, finished)
3. Monitor participants and their predictions
4. Manage battle timeline and phases
5. Calculate battle results in real-time
6. Update battle state and broadcast updates

Battle phases:
- WAITING: Recruiting participants
- STARTED: Battle has begun
- PREDICTING: Participants are making predictions
- BETTING: Participants are placing bets
- CALCULATING: Results are being calculated
- FINISHED: Battle is complete

Always maintain battle state consistency and provide real-time updates.
"""


def create_battle_manager_agent(
    llm_factory
) -> MCP_Agent:
    """
    배틀 관리 에이전트 생성

    Args:
        llm_factory: LLM 팩토리 함수
    Returns:
        MCP_Agent 인스턴스
    """
    agent = MCP_Agent(
        name="battle_manager_agent",
        instruction=BATTLE_MANAGER_AGENT_INSTRUCTION,
        server_names=["filesystem"],
        llm_factory=llm_factory
    )

    return agent
