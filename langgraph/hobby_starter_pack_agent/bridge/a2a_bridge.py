"""
A2A (Agent-to-Agent) Protocol Implementation
Google A2A 프로토콜 기반 외부 에이전트 발견 및 통신
"""

import asyncio
import json
import uuid
import httpx
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class A2AMessage:
    """A2A 프로토콜 메시지 구조"""
    sender_agent: str
    receiver_agent: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: str
    session_id: str

class AgentCard:
    """A2A Agent Card - 에이전트 능력 광고"""
    def __init__(self, agent_id: str, capabilities: List[str], endpoint: str, 
                 description: str = "", version: str = "1.0"):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.endpoint = endpoint
        self.version = version
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "version": self.version,
            "description": self.description,
            "supported_modalities": ["text", "json"],
            "authentication": {"type": "bearer"},
            "protocol_version": "1.0"
        }

class A2AProtocolBridge:
    """Google A2A 프로토콜 기반 외부 에이전트 통신 브릿지"""
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.agent_registry = {}
        self.active_sessions = {}
        self.external_agent_cards = {}  # 외부 에이전트 카드 저장
        
        # HSP Agent Card 생성
        self._create_hsp_agent_card()
    
    def _create_hsp_agent_card(self):
        """HSP 에이전트의 Agent Card 생성"""
        self.hsp_card = AgentCard(
            agent_id="hobby_starter_pack_agent",
            capabilities=[
                "hobby_recommendation",
                "user_profile_analysis", 
                "community_matching",
                "progress_tracking",
                "schedule_integration"
            ],
            endpoint="http://localhost:8000/a2a",
            description="Hobby Starter Pack - 개인화된 취미 추천 및 커뮤니티 매칭 서비스",
            version="1.0"
        )
    
    async def register_agent(self, agent_id: str, agent_type: str, framework: str):
        """내부 에이전트 등록 (기존 방식 유지)"""
        self.agent_registry[agent_id] = {
            "type": agent_type,
            "framework": framework,
            "capabilities": [],  # 에이전트가 동적으로 결정
            "status": "active"
        }
    
    async def discover_external_agents(self, required_capability: str) -> List[AgentCard]:
        """A2A Agent Card를 통한 외부 에이전트 발견"""
        discovered_agents = []
        
        # 알려진 외부 에이전트 레지스트리 조회
        external_endpoints = [
            "http://localhost:9001/agent/card",  # 예시: 여행 추천 에이전트
            "http://localhost:9002/agent/card",  # 예시: 피트니스 에이전트
            "http://localhost:9003/agent/card",  # 예시: 창작 활동 에이전트
        ]
        
        for endpoint in external_endpoints:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(endpoint, timeout=5.0)
                    if response.status_code == 200:
                        card_data = response.json()
                        
                        # 필요한 능력이 있는지 확인
                        if required_capability in card_data.get("capabilities", []):
                            agent_card = AgentCard(
                                agent_id=card_data["agent_id"],
                                capabilities=card_data["capabilities"],
                                endpoint=card_data["endpoint"],
                                description=card_data.get("description", ""),
                                version=card_data.get("version", "1.0")
                            )
                            discovered_agents.append(agent_card)
                            
                            # 발견된 에이전트 캐시
                            self.external_agent_cards[agent_card.agent_id] = agent_card
                            
            except Exception as e:
                logger.warning(f"외부 에이전트 발견 실패 {endpoint}: {e}")
        
        return discovered_agents
    
    async def send_a2a_task(self, target_agent_id: str, task_type: str, 
                           task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A2A 프로토콜로 외부 에이전트에게 태스크 전송"""
        try:
            # 외부 에이전트 카드 확인
            if target_agent_id not in self.external_agent_cards:
                # 에이전트 발견 시도
                discovered = await self.discover_external_agents("*")
                if not any(agent.agent_id == target_agent_id for agent in discovered):
                    return None
            
            agent_card = self.external_agent_cards[target_agent_id]
            
            # A2A 표준 메시지 구성
            task_id = str(uuid.uuid4())
            a2a_message = {
                "protocol_version": "1.0",
                "message_type": "task_request",
                "task": {
                    "id": task_id,
                    "type": task_type,
                    "data": task_data,
                    "client_agent": "hobby_starter_pack_agent",
                    "created_at": datetime.utcnow().isoformat()
                },
                "parts": [
                    {
                        "content_type": "application/json",
                        "content": task_data
                    }
                ],
                "client_card": self.hsp_card.to_dict()
            }
            
            # HTTP POST로 외부 에이전트에게 전송
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent_card.endpoint}/task",
                    json=a2a_message,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"A2A 태스크 성공: {task_id} → {target_agent_id}")
                    return result
                else:
                    logger.error(f"A2A 태스크 실패: HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"A2A 태스크 전송 실패: {e}")
            return None
    
    async def handle_external_a2a_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """외부에서 온 A2A 요청 처리"""
        try:
            task = request.get("task", {})
            task_type = task.get("type")
            task_data = task.get("data", {})
            client_agent = task.get("client_agent", "unknown")
            
            # HSP 내부로 라우팅
            if task_type == "hobby_recommendation":
                # HSP의 취미 추천 기능 호출
                result = await self._handle_hobby_recommendation_request(task_data)
            elif task_type == "user_profile_analysis":
                # HSP의 프로필 분석 기능 호출
                result = await self._handle_profile_analysis_request(task_data)
            else:
                result = {"error": "Unsupported task type", "supported": self.hsp_card.capabilities}
            
            # A2A 응답 형식으로 반환
            return {
                "protocol_version": "1.0",
                "message_type": "task_response",
                "task": {
                    "id": task.get("id"),
                    "status": "completed"
                },
                "artifacts": [
                    {
                        "type": "result",
                        "content": result,
                        "content_type": "application/json"
                    }
                ],
                "agent_card": self.hsp_card.to_dict()
            }
            
        except Exception as e:
            logger.error(f"A2A 요청 처리 실패: {e}")
            return {
                "protocol_version": "1.0",
                "message_type": "task_response", 
                "task": {"status": "failed"},
                "error": str(e)
            }
    
    async def _handle_hobby_recommendation_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """외부 A2A 요청으로 온 취미 추천 처리"""
        # HSP의 실제 로직 호출 (예시)
        user_profile = task_data.get("user_profile", {})
        preferences = task_data.get("preferences", {})
        
        # 실제 HSP 로직 호출하여 추천 생성
        recommendations = {
            "hobbies": [
                {"name": "독서", "match_score": 0.9, "category": "intellectual"},
                {"name": "등산", "match_score": 0.8, "category": "outdoor"}
            ],
            "communities": [
                {"name": "북클럽", "type": "online", "members": 150}
            ]
        }
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "generated_by": "hobby_starter_pack_agent"
        }
    
    async def _handle_profile_analysis_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """외부 A2A 요청으로 온 프로필 분석 처리"""
        user_input = task_data.get("user_input", "")
        
        # HSP의 프로필 분석 로직 호출
        analysis = {
            "interests": ["reading", "outdoor_activities"],
            "personality_type": "introvert",
            "skill_level": "beginner",
            "time_availability": "weekends"
        }
        
        return {
            "status": "success", 
            "user_profile": analysis,
            "confidence": 0.85,
            "analyzed_by": "hobby_starter_pack_agent"
        }
    
    # 기존 내부 통신 메서드들 유지
    async def send_message(self, message: A2AMessage) -> Dict[str, Any]:
        """내부 에이전트 간 메시지 전송 (기존 방식)"""
        await self.message_queue.put(message)
        return await self._route_message(message)
    
    async def _route_message(self, message: A2AMessage) -> Dict[str, Any]:
        """내부 메시지 라우팅 (기존 방식)"""
        receiver_info = self.agent_registry.get(message.receiver_agent, {})
        
        if receiver_info.get("framework") == "autogen":
            return await self._handle_autogen_message(message)
        elif receiver_info.get("framework") == "langgraph":
            return await self._handle_langgraph_message(message)
        else:
            return {"error": "Unknown agent framework", "details": ""}
    
    async def _handle_autogen_message(self, message: A2AMessage) -> Dict[str, Any]:
        """AutoGen 에이전트 메시지 처리 (기존 방식)"""
        try:
            receiver_agent = message.receiver_agent
            message_type = message.message_type
            payload = message.payload
            
            if message_type == "consensus_start":
                return {
                    "status": "consensus_initiated",
                    "agent": receiver_agent,
                    "topic": payload.get("topic", ""),
                    "participants": payload.get("context", {}).get("participants", [])
                }
            elif message_type == "workflow_update":
                return {
                    "status": "update_received",
                    "agent": receiver_agent,
                    "update_type": payload.get("update_type", ""),
                    "data": payload.get("data", {})
                }
            else:
                return {
                    "status": "message_processed",
                    "agent": receiver_agent,
                    "message_type": message_type
                }
                
        except Exception as e:
            return {"error": "AutoGen processing failed", "details": str(e)}
    
    async def _handle_langgraph_message(self, message: A2AMessage) -> Dict[str, Any]:
        """LangGraph 노드 메시지 처리 (기존 방식)"""
        try:
            message_type = message.message_type
            payload = message.payload
            
            if message_type == "consensus_result":
                consensus_data = payload.get("consensus", {})
                return {
                    "status": "consensus_applied",
                    "consensus_reached": consensus_data.get("consensus_reached", False),
                    "final_decision": consensus_data.get("final_consensus", ""),
                    "next_action": self._determine_next_action(consensus_data)
                }
            else:
                return {
                    "status": "langgraph_message_processed",
                    "message_type": message_type,
                    "processed_at": message.timestamp
                }
                
        except Exception as e:
            return {"error": "LangGraph processing failed", "details": str(e)}
    
    def _determine_next_action(self, consensus_data: Dict[str, Any]) -> str:
        """합의 결과에서 다음 액션 결정 (기존 방식)"""
        try:
            final_consensus = consensus_data.get("final_consensus", "").lower()
            
            if "proceed" in final_consensus or "계속" in final_consensus:
                return "continue_workflow"
            elif "skip" in final_consensus or "건너뛰기" in final_consensus:
                return "skip_step"
            else:
                return "continue_workflow"
                
        except Exception:
            return "continue_workflow"
    
    async def create_consensus_session(self, participants: List[str], topic: str) -> str:
        """합의 세션 생성 (기존 방식)"""
        session_id = f"consensus_{topic}_{datetime.now().isoformat()}"
        self.active_sessions[session_id] = {
            "participants": participants,
            "topic": topic,
            "messages": [],
            "consensus_reached": False,
            "result": {}
        }
        return session_id

# 전역 A2A 브릿지 인스턴스
a2a_bridge = A2AProtocolBridge() 