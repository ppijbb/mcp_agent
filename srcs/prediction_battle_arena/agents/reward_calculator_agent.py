"""
보상 계산 에이전트
"""

from typing import List, Dict, Any, Optional
from mcp_agent.agents.agent import Agent as MCP_Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from ..tools.reward_tools import RewardTools
from ..services.reward_service import RewardService


REWARD_CALCULATOR_AGENT_INSTRUCTION = """
You are a Reward Calculator Agent specialized in calculating rewards for predictions.

Your responsibilities:
1. Calculate base rewards based on prediction accuracy
2. Apply betting multipliers
3. Calculate win streak bonuses:
   - 3 wins: 20% bonus
   - 5 wins: 50% bonus
   - 10 wins: 100% bonus
   - 20 wins: 200% bonus
4. Apply random bonuses (10% chance for 100x jackpot)
5. Update user coins and statistics
6. Generate reward breakdowns and messages

Reward formula:
- Base reward = bet_amount * multiplier * accuracy_score
- Streak bonus = base_reward * streak_bonus_rate
- Random bonus = (10% chance) base_amount * 100x
- Total reward = base_reward + streak_bonus + random_bonus

Always use the reward_calculate tool to process rewards.
"""


def create_reward_calculator_agent(
    llm_factory,
    reward_tools: RewardTools,
    reward_service: RewardService
) -> MCP_Agent:
    """
    보상 계산 에이전트 생성
    
    Args:
        llm_factory: LLM 팩토리 함수
        reward_tools: 보상 도구 인스턴스
        reward_service: 보상 서비스 인스턴스
    Returns:
        MCP_Agent 인스턴스
    """
    # 도구 목록 가져오기
    tools = reward_tools.get_tools()
    
    agent = MCP_Agent(
        name="reward_calculator_agent",
        instruction=REWARD_CALCULATOR_AGENT_INSTRUCTION,
        server_names=["filesystem"],
        llm_factory=llm_factory,
        tools=tools
    )
    
    return agent

