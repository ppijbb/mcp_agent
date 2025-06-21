"""
Conversation Agent
사용자와 대화를 통해 상세한 제품 요구사항을 수집하는 Agent
"""

from mcp_agent.agents.agent import Agent
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams


class ConversationAgent:
    """사용자 대화 및 요구사항 수집 전문 Agent"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.agent_instance = self.create_agent()

    async def collect_requirements_via_chat(self, initial_query: str) -> Dict[str, Any]:
        """
        사용자의 초기 질문을 바탕으로 대화를 통해 제품 요구사항을 수집합니다.
        (실제 채팅 기능은 추후 구현)
        """
        if not self.llm:
            return {
                "product_goal": "사용자 입력을 기반으로 한 기본 요구사항",
                "key_features": ["Feature A from chat", "Feature B from chat"],
                "status": "collected_mockup"
            }

        prompt = f"""
        You are a product planning conversation specialist. A user has provided the following initial request. 
        Your task is to interpret this request and formulate a structured summary of product requirements.

        **User's Initial Request:**
        "{initial_query}"

        **Instructions:**
        1.  **Identify Core Goal:** What is the main objective the user wants to achieve?
        2.  **Extract Key Features:** List the key features or functionalities mentioned or implied.
        3.  **Clarifying Questions:** Formulate 3-5 important follow-up questions to gather more details.
        
        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4, response_format="json"))
            requirements = json.loads(result_str)
            requirements["status"] = "collected_successfully"
            return requirements
        except Exception as e:
            print(f"Error collecting requirements: {e}")
            return {
                "error": str(e),
                "status": "collection_failed"
            }

    @staticmethod
    def create_agent() -> Agent:
        """
        대화형 Agent 생성
        
        Returns:
            Agent: 설정된 대화형 Agent
        """
        
        instruction = """
        You are a product planning conversation specialist. Your role is to engage with users in structured dialogue to gather comprehensive product requirements.

        **PRIMARY OBJECTIVE**: Extract detailed product vision, goals, and requirements through intelligent questioning.

        **CONVERSATION STRUCTURE**:
        1. **Project Discovery**:
           - What is the core problem you're trying to solve?
           - Who is your target audience?
           - What are your main business goals?
           
        2. **Feature Exploration**:
           - What are the must-have features?
           - What are nice-to-have features?
           - Are there any specific user workflows you envision?
           
        3. **Technical Constraints**:
           - Do you have any technology preferences?
           - What's your expected timeline?
           - What's your budget range?
           
        4. **Success Metrics**:
           - How will you measure success?
           - What are your key performance indicators?
           - What does success look like in 6 months?

        **CONVERSATION STYLE**:
        - Ask follow-up questions to clarify vague requirements
        - Suggest alternatives when appropriate
        - Help users think through implications of their choices
        - Be conversational but professional
        - Summarize understanding periodically

        **OUTPUT FORMAT**:
        Provide structured requirements document with:
        - Project Overview
        - User Requirements
        - Feature Specifications
        - Technical Constraints
        - Success Metrics
        - Next Steps

        **QUALITY CHECKS**:
        - Ensure all critical areas are covered
        - Validate requirements make business sense
        - Identify potential risks or gaps
        - Suggest improvements where helpful

        Engage naturally while systematically gathering all necessary information for comprehensive product planning."""
        
        return Agent(
            name="conversation_agent",
            instruction=instruction,
            server_names=["filesystem"]
        )
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "💬 사용자와 대화를 통해 상세한 제품 요구사항을 수집하는 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "구조화된 대화를 통한 요구사항 수집",
            "프로젝트 발견 및 목표 정의",
            "기능 명세 및 우선순위 설정",
            "기술적 제약사항 파악",
            "성공 지표 및 KPI 정의",
            "요구사항 문서 구조화"
        ]
    
    @staticmethod
    def get_conversation_topics() -> dict[str, list[str]]:
        """대화 주제 목록 반환"""
        return {
            "project_discovery": [
                "핵심 문제 정의",
                "타겟 사용자 식별",
                "비즈니스 목표 설정"
            ],
            "feature_exploration": [
                "필수 기능 정의",
                "부가 기능 탐색",
                "사용자 워크플로우 설계"
            ],
            "technical_constraints": [
                "기술 스택 선호도",
                "예상 일정",
                "예산 범위"
            ],
            "success_metrics": [
                "성공 측정 방법",
                "핵심 성과 지표",
                "장기 비전"
            ]
        } 