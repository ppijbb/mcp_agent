"""
BaseAgent - 모든 Agent의 기본 클래스

이 클래스는 2025년 기준 진짜 Agent의 정의를 구현합니다:
- Perception (인식): 환경을 감지하고 데이터를 수집
- Reasoning (추론): LLM 기반 복잡한 논리적 사고와 계획 수립
- Action (행동): 목표 달성을 위한 실제 행동 실행
- Learning (학습): 경험을 통한 지속적 개선
- Autonomy (자율성): 최소한의 인간 개입으로 독립적 작업
"""

from typing import Dict, List, Any, Optional, Protocol
from abc import ABC, abstractmethod
import asyncio
import json
import uuid
from datetime import datetime


class LLMClient(Protocol):
    """LLM 클라이언트 인터페이스"""
    async def complete(self, prompt: str) -> str:
        ...


class MCPClient(Protocol):
    """MCP 클라이언트 인터페이스"""
    async def call(self, server_name: str, method: str, params: Dict) -> Dict:
        ...


class BaseAgent(ABC):
    """
    모든 Agent의 기본 클래스
    
    진짜 Agent의 4단계 사이클을 강제합니다:
    1. perceive: 환경 인식
    2. reason: LLM 기반 추론
    3. act: 행동 실행
    4. learn: 경험 학습
    """
    
    def __init__(self, llm_client: LLMClient, mcp_client: MCPClient, agent_id: str):
        """
        Args:
            llm_client: LLM 클라이언트 (Gemini 2.5 Flash Lite)
            mcp_client: MCP 서버 연결 클라이언트
            agent_id: Agent 고유 식별자
        """
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        self.agent_id = agent_id
        
        # Agent별 메모리 (학습 데이터)
        self.memory: Dict[str, Any] = {
            "experiences": [],
            "learned_patterns": {},
            "performance_metrics": {},
            "created_at": datetime.now().isoformat()
        }
        
        # Agent 상태
        self.is_active = False
        self.current_task = None
        self.error_count = 0
    
    async def run_cycle(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent의 완전한 사이클 실행
        
        이것이 진짜 Agent와 가짜 Agent의 차이점입니다:
        - 가짜: 단순한 함수 호출
        - 진짜: perceive → reason → act → learn 자율 사이클
        
        Args:
            environment: 환경 정보
            
        Returns:
            행동 결과와 학습 내용
        """
        cycle_start = datetime.now()
        
        try:
            self.is_active = True
            
            # 1. 환경 인식 (Perception)
            perception = await self.perceive(environment)
            
            # 2. 추론 (Reasoning) - 반드시 LLM 사용
            reasoning = await self.reason(perception)
            
            # 3. 행동 (Action)
            action_result = await self.act(reasoning)
            
            # 4. 학습 (Learning)
            await self.learn({
                "environment": environment,
                "perception": perception,
                "reasoning": reasoning,
                "action_result": action_result,
                "cycle_duration": (datetime.now() - cycle_start).total_seconds()
            })
            
            return {
                "agent_id": self.agent_id,
                "cycle_complete": True,
                "action_result": action_result,
                "reasoning": reasoning,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.error_count += 1
            error_info = {
                "agent_id": self.agent_id,
                "error": str(e),
                "error_count": self.error_count,
                "timestamp": datetime.now().isoformat()
            }
            
            # 에러도 학습 데이터로 활용
            await self.learn({"error": error_info})
            
            return {
                "agent_id": self.agent_id,
                "cycle_complete": False,
                "error": error_info
            }
        
        finally:
            self.is_active = False
    
    @abstractmethod
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        환경 인식 단계
        
        Agent가 현재 상황을 이해하기 위해 필요한 모든 정보를 수집합니다.
        - 게임 상태 분석
        - 외부 데이터 수집 (MCP 서버 활용)
        - 다른 Agent들의 상태 확인
        
        Args:
            environment: 환경 정보
            
        Returns:
            인식된 정보들
        """
        pass
    
    @abstractmethod
    async def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        추론 단계 - 반드시 LLM 기반이어야 함
        
        이것이 진짜 Agent의 핵심입니다:
        - 단순한 if-else가 아닌 LLM 기반 추론
        - 복잡한 상황 분석
        - 전략적 계획 수립
        - 불확실성 하에서의 의사결정
        
        Args:
            perception: 인식된 정보
            
        Returns:
            추론 결과와 행동 계획
        """
        pass
    
    @abstractmethod
    async def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        행동 실행 단계
        
        추론 결과를 바탕으로 실제 행동을 수행합니다.
        - 게임 액션 실행
        - 다른 Agent와 상호작용
        - 환경 변경
        
        Args:
            reasoning: 추론 결과
            
        Returns:
            행동 실행 결과
        """
        pass
    
    async def learn(self, experience: Dict[str, Any]) -> None:
        """
        학습 단계
        
        경험을 통해 지속적으로 개선됩니다.
        - 성공/실패 패턴 분석
        - 전략 효과성 평가
        - 상대방 행동 패턴 학습
        
        Args:
            experience: 경험 데이터
        """
        # 경험 저장
        experience_record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "data": experience
        }
        
        self.memory["experiences"].append(experience_record)
        
        # 경험이 너무 많이 쌓이면 정리 (최근 1000개만 유지)
        if len(self.memory["experiences"]) > 1000:
            self.memory["experiences"] = self.memory["experiences"][-1000:]
        
        # 패턴 분석 (간단한 버전)
        if "action_result" in experience:
            action_type = experience.get("action_result", {}).get("action_type")
            if action_type:
                if action_type not in self.memory["learned_patterns"]:
                    self.memory["learned_patterns"][action_type] = {
                        "count": 0,
                        "success_rate": 0.0,
                        "avg_confidence": 0.0
                    }
                
                pattern = self.memory["learned_patterns"][action_type]
                pattern["count"] += 1
                
                # 성공률 업데이트 (간단한 휴리스틱)
                if experience.get("action_result", {}).get("status") == "success":
                    pattern["success_rate"] = (pattern["success_rate"] * (pattern["count"] - 1) + 1.0) / pattern["count"]
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Agent의 학습 현황 요약"""
        return {
            "agent_id": self.agent_id,
            "total_experiences": len(self.memory["experiences"]),
            "learned_patterns": self.memory["learned_patterns"],
            "error_count": self.error_count,
            "is_active": self.is_active,
            "created_at": self.memory["created_at"]
        }
    
    async def validate_llm_usage(self, reasoning_result: Dict[str, Any]) -> bool:
        """
        추론 단계에서 실제로 LLM을 사용했는지 검증
        
        가짜 Agent를 방지하기 위한 검증 로직
        """
        # LLM 응답의 특징적 패턴 확인
        if not isinstance(reasoning_result, dict):
            return False
        
        # LLM 응답에는 보통 confidence, reasoning 등이 포함됨
        llm_indicators = ["confidence", "reasoning", "analysis", "strategy"]
        has_llm_indicators = any(key in reasoning_result for key in llm_indicators)
        
        return has_llm_indicators


class AgentValidationError(Exception):
    """Agent 검증 실패 시 발생하는 예외"""
    pass


def validate_agent_implementation(agent_class) -> bool:
    """
    Agent 클래스가 진짜 Agent 요구사항을 만족하는지 검증
    
    Args:
        agent_class: 검증할 Agent 클래스
        
    Returns:
        검증 통과 여부
    """
    required_methods = ["perceive", "reason", "act"]
    
    for method in required_methods:
        if not hasattr(agent_class, method):
            raise AgentValidationError(f"Agent must implement {method} method")
        
        if not asyncio.iscoroutinefunction(getattr(agent_class, method)):
            raise AgentValidationError(f"{method} must be async method")
    
    return True 