"""
예측 생성 에이전트
"""

from typing import List, Dict, Any, Optional
from mcp_agent.agents.agent import Agent as MCP_Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from ..tools.prediction_tools import PredictionTools


PREDICTION_AGENT_INSTRUCTION = """
You are a Prediction Generator Agent specialized in creating accurate predictions.

Your mission:
1. Analyze the given topic using MCP tools (g-search, fetch) to gather latest data
2. Create well-reasoned predictions based on:
   - Current market trends
   - Historical data patterns
   - Real-time events and news
   - Statistical analysis
3. Provide confidence scores (0.0 ~ 1.0) for each prediction
4. Include detailed reasoning and data sources

Output format: JSON with:
- prediction_text: The actual prediction
- prediction_value: Numerical value if applicable
- confidence: Confidence score (0.0 ~ 1.0)
- reasoning: Detailed explanation
- data_sources: List of sources used

Always use the prediction_create tool to save your predictions.
"""


def create_prediction_agent(
    llm_factory,
    prediction_tools: PredictionTools
) -> MCP_Agent:
    """
    예측 생성 에이전트 생성
    
    Args:
        llm_factory: LLM 팩토리 함수
        prediction_tools: 예측 도구 인스턴스
    Returns:
        MCP_Agent 인스턴스
    """
    # 도구 목록 가져오기
    tools = prediction_tools.get_tools()
    
    agent = MCP_Agent(
        name="prediction_agent",
        instruction=PREDICTION_AGENT_INSTRUCTION,
        server_names=["g-search", "fetch", "filesystem"],
        llm_factory=llm_factory,
        tools=tools
    )
    
    return agent

