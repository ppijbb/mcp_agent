import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from datetime import datetime

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HSPAutoGenAgents:
    """AutoGen 기반 전문 에이전트들"""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.llm_config = llm_config or {"model": "gpt-4", "temperature": 0.7}
        self.agents = self._initialize_agents()
        logger.info("HSPAutoGenAgents 초기화 완료")
        
    def _initialize_agents(self) -> Dict[str, AssistantAgent]:
        """에이전트 초기화 - 모든 시스템 메시지는 에이전트가 LLM으로 동적 생성"""
        
        # 사용자 프로필 분석 전문가
        profile_analyst = AssistantAgent(
            name="ProfileAnalyst",
            system_message="",  # 빈 값, LLM이 동적으로 결정
            llm_config=self.llm_config,
            description="사용자의 성격, 관심사, 라이프스타일을 분석하여 맞춤형 프로필을 생성"
        )
        
        # 취미 발견 전문가
        hobby_discoverer = AssistantAgent(
            name="HobbyDiscoverer",
            system_message="",  # 빈 값, LLM이 동적으로 결정
            llm_config=self.llm_config,
            description="개인에게 최적화된 새로운 취미 활동을 발견하고 추천"
        )
        
        # 스케줄 통합 전문가
        schedule_integrator = AssistantAgent(
            name="ScheduleIntegrator",
            system_message="",  # 빈 값, LLM이 동적으로 결정
            llm_config=self.llm_config,
            description="일상 스케줄과 취미 활동을 효율적으로 통합"
        )
        
        # 커뮤니티 매칭 전문가
        community_matcher = AssistantAgent(
            name="CommunityMatcher",
            system_message="",  # 빈 값, LLM이 동적으로 결정
            llm_config=self.llm_config,
            description="취미 기반 커뮤니티와 동료를 찾아서 연결"
        )
        
        # 진행상황 추적 전문가
        progress_tracker = AssistantAgent(
            name="ProgressTracker",
            system_message="",  # 빈 값, LLM이 동적으로 결정
            llm_config=self.llm_config,
            description="취미 활동 진행상황을 추적하고 동기부여 제공"
        )
        
        # 의사결정 중재자
        decision_moderator = AssistantAgent(
            name="DecisionModerator",
            system_message="",  # 빈 값, LLM이 동적으로 결정
            llm_config=self.llm_config,
            description="에이전트 간 의견 차이를 조율하고 최종 의사결정 지원"
        )
        
        return {
            "profile_analyst": profile_analyst,
            "hobby_discoverer": hobby_discoverer,
            "schedule_integrator": schedule_integrator,
            "community_matcher": community_matcher,
            "progress_tracker": progress_tracker,
            "decision_moderator": decision_moderator
        }
    
    def create_consensus_chat(self, relevant_agents: List[str]) -> GroupChat:
        """특정 주제에 대한 에이전트 합의 채팅 생성"""
        selected_agents = [self.agents[name] for name in relevant_agents if name in self.agents]
        
        # UserProxyAgent 추가 (실행 권한)
        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
        
        selected_agents.append(user_proxy)
        
        return GroupChat(
            agents=selected_agents,
            messages=[],
            max_round=10,  # 에이전트가 동적으로 조정
            speaker_selection_method="auto"  # 자동 선택
        )
    
    async def run_consensus(self, agents: List[str], topic: str, context: Dict[str, Any], 
                           user_profile: Optional[Dict[str, Any]] = None, 
                           bridge=None, session_id: str = None) -> Dict[str, Any]:
        """A2A 브리지를 통한 에이전트 합의 실행"""
        try:
            # A2A 브리지에 에이전트들 등록
            if bridge:
                for agent_name in agents:
                    await bridge.register_agent(
                        agent_id=agent_name,
                        agent_type=agent_name.lower(),
                        framework="autogen"
                    )
            
            # 그룹 채팅 생성
            group_chat = self.create_consensus_chat(agents)
            
            # 그룹 채팅 매니저 생성
            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=self.llm_config
            )
            
            # 초기 메시지 구성
            initial_message = self._construct_initial_message(topic, context, user_profile)
            
            # A2A 메시지로 다른 에이전트들에게 알림
            if bridge and session_id:
                from ..bridge.a2a_bridge import A2AMessage
                
                for agent_name in agents:
                    message = A2AMessage(
                        sender_agent="AutoGenConsensus",
                        receiver_agent=agent_name,
                        message_type="consensus_start",
                        payload={
                            "topic": topic,
                            "context": context,
                            "user_profile": user_profile
                        },
                        timestamp=datetime.now().isoformat(),
                        session_id=session_id
                    )
                    await bridge.send_message(message)
            
            # 그룹 채팅 시작 (비동기 실행)
            user_proxy = next(agent for agent in group_chat.agents if agent.name == "UserProxy")
            
            # 채팅 실행을 위한 비동기 wrapper
            consensus_result = await self._run_async_chat(user_proxy, manager, initial_message)
            
            # 결과를 A2A 브리지를 통해 LangGraph로 전송
            if bridge and session_id:
                result_message = A2AMessage(
                    sender_agent="AutoGenConsensus",
                    receiver_agent="LangGraphWorkflow",
                    message_type="consensus_result",
                    payload={
                        "consensus": consensus_result,
                        "participants": agents,
                        "topic": topic
                    },
                    timestamp=datetime.now().isoformat(),
                    session_id=session_id
                )
                await bridge.send_message(result_message)
            
            return consensus_result
            
        except Exception as e:
            return {"error": "Consensus execution failed", "details": str(e)}
    
    def _construct_initial_message(self, topic: str, context: Dict[str, Any], 
                                  user_profile: Optional[Dict[str, Any]]) -> str:
        """초기 합의 메시지 구성"""
        message_parts = [
            f"주제: {topic}",
            f"컨텍스트: {json.dumps(context, ensure_ascii=False, indent=2)}"
        ]
        
        if user_profile:
            message_parts.append(f"사용자 프로필: {json.dumps(user_profile, ensure_ascii=False, indent=2)}")
        
        message_parts.append("각 에이전트는 자신의 전문 영역에서 의견을 제시하고, 최종 합의안을 도출해주세요.")
        
        return "\n\n".join(message_parts)
    
    async def _run_async_chat(self, user_proxy, manager, message: str) -> Dict[str, Any]:
        """비동기 채팅 실행"""
        try:
            # AutoGen의 채팅을 비동기로 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: user_proxy.initiate_chat(manager, message=message)
            )
            
            # 채팅 결과에서 합의 내용 추출
            return self._extract_consensus_from_chat(manager.groupchat.messages)
            
        except Exception as e:
            return {"error": "Chat execution failed", "details": str(e)}
    
    def _extract_consensus_from_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """채팅 메시지에서 합의 결과 추출"""
        try:
            # 마지막 몇 개 메시지에서 합의 내용 찾기
            consensus_messages = messages[-5:] if len(messages) >= 5 else messages
            
            # 각 에이전트의 최종 의견 수집
            agent_opinions = {}
            final_consensus = ""
            
            for msg in reversed(consensus_messages):
                agent_name = msg.get("name", "unknown")
                content = msg.get("content", "")
                
                if agent_name not in agent_opinions and content:
                    agent_opinions[agent_name] = content
                
                # DecisionModerator의 최종 결론 찾기
                if agent_name == "DecisionModerator" and "최종" in content:
                    final_consensus = content
                    break
            
            return {
                "final_consensus": final_consensus,
                "agent_opinions": agent_opinions,
                "message_count": len(messages),
                "consensus_reached": bool(final_consensus)
            }
            
        except Exception as e:
            return {"error": "Consensus extraction failed", "details": str(e)} 