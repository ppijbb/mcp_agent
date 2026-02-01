"""
리더보드 관리 에이전트
"""

from mcp_agent.agents.agent import Agent as MCP_Agent

from ..tools.leaderboard_tools import LeaderboardTools
from ..services.redis_service import RedisService


LEADERBOARD_AGENT_INSTRUCTION = """
You are a Leaderboard Agent responsible for managing global rankings.

Your responsibilities:
1. Update leaderboard scores when battles complete
2. Maintain multiple leaderboard categories:
   - global: All-time rankings
   - weekly: Weekly rankings
   - monthly: Monthly rankings
3. Calculate user ranks and percentiles
4. Provide leaderboard snapshots
5. Broadcast leaderboard updates in real-time

Leaderboard scoring:
- Base score = total_winnings / total_battles
- Accuracy bonus = accuracy_rate * 100
- Win streak bonus = win_streak * 10
- Final score = base_score + accuracy_bonus + streak_bonus

Always use the leaderboard_update tool to update rankings.
"""


def create_leaderboard_agent(
    llm_factory,
    leaderboard_tools: LeaderboardTools,
    redis_service: RedisService
) -> MCP_Agent:
    """
    리더보드 관리 에이전트 생성

    Args:
        llm_factory: LLM 팩토리 함수
        leaderboard_tools: 리더보드 도구 인스턴스
        redis_service: Redis 서비스 인스턴스
    Returns:
        MCP_Agent 인스턴스
    """
    # 도구 목록 가져오기
    tools = leaderboard_tools.get_tools()

    agent = MCP_Agent(
        name="leaderboard_agent",
        instruction=LEADERBOARD_AGENT_INSTRUCTION,
        server_names=["filesystem"],
        llm_factory=llm_factory,
        tools=tools
    )

    return agent
