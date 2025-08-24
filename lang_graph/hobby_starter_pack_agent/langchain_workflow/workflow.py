"""
LangChain 기반 통합 워크플로우
메모리, 프롬프트 템플릿, 벡터 스토어, 에이전트를 활용한 최적화된 흐름
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime
import uuid

from .memory_manager import HSPMemoryManager
from .prompt_templates import HSPPromptTemplates
from .vector_store import HSPVectorStore
from .agents import HSPLangChainAgent

logger = logging.getLogger(__name__)

class HSPLangChainWorkflow:
    """LangChain 기반 통합 워크플로우"""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.llm_config = llm_config or {"model": "gpt-4", "temperature": 0.7}
        
        # LangChain 컴포넌트들 초기화
        self.memory_manager = HSPMemoryManager()
        self.prompt_templates = HSPPromptTemplates()
        self.vector_store = HSPVectorStore()
        self.langchain_agent = HSPLangChainAgent(llm_config)
        
        # 워크플로우 상태
        self.active_sessions = {}
        self.workflow_stats = {
            "total_sessions": 0,
            "completed_sessions": 0,
            "failed_sessions": 0
        }
        
        logger.info("HSP LangChain Workflow 초기화 완료")
    
    async def run_workflow(self, user_input: str, user_id: str = None, 
                          user_profile: Optional[Dict[str, Any]] = None,
                          preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """메인 워크플로우 실행"""
        try:
            # 세션 ID 생성
            session_id = user_id or str(uuid.uuid4())
            
            # 세션 시작
            await self._start_session(session_id, user_input, user_profile)
            
            # 1단계: 사용자 프로필 분석
            profile_result = await self._analyze_user_profile(session_id, user_input, user_profile)
            
            # 2단계: 취미 추천
            hobby_result = await self._recommend_hobbies(session_id, profile_result, preferences)
            
            # 3단계: 커뮤니티 매칭
            community_result = await self._match_communities(session_id, profile_result, hobby_result)
            
            # 4단계: 스케줄 통합
            schedule_result = await self._integrate_schedule(session_id, profile_result, hobby_result)
            
            # 5단계: 진행상황 추적 계획
            progress_result = await self._plan_progress_tracking(session_id, profile_result, hobby_result)
            
            # 세션 완료
            await self._complete_session(session_id)
            
            # 최종 결과 구성
            final_result = {
                "session_id": session_id,
                "success": True,
                "user_profile": profile_result,
                "hobby_recommendations": hobby_result,
                "community_matches": community_result,
                "schedule_integration": schedule_result,
                "progress_tracking": progress_result,
                "workflow_metadata": {
                    "execution_time": datetime.now().isoformat(),
                    "total_steps": 5,
                    "completed_steps": 5
                }
            }
            
            logger.info(f"워크플로우 완료: {session_id}")
            return final_result
            
        except Exception as e:
            logger.error(f"워크플로우 실행 실패: {e}")
            await self._fail_session(session_id if 'session_id' in locals() else None, str(e))
            
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id if 'session_id' in locals() else None
            }
    
    async def _start_session(self, session_id: str, user_input: str, user_profile: Optional[Dict[str, Any]]):
        """세션 시작"""
        try:
            # 메모리에 사용자 메시지 추가
            self.memory_manager.add_user_message(session_id, user_input, user_profile)
            
            # 세션 정보 저장
            self.active_sessions[session_id] = {
                "started_at": datetime.now().isoformat(),
                "user_input": user_input,
                "user_profile": user_profile or {},
                "current_step": "started",
                "steps_completed": []
            }
            
            self.workflow_stats["total_sessions"] += 1
            
            logger.info(f"세션 시작: {session_id}")
            
        except Exception as e:
            logger.error(f"세션 시작 실패: {e}")
            raise
    
    async def _analyze_user_profile(self, session_id: str, user_input: str, 
                                   user_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """사용자 프로필 분석"""
        try:
            logger.info(f"프로필 분석 시작: {session_id}")
            
            # 대화 기록 가져오기
            conversation_history = self.memory_manager.get_user_conversation_history(session_id, limit=5)
            
            # 기존 사용자 컨텍스트 가져오기
            user_context = self.memory_manager.get_hobby_context(session_id)
            
            # LangChain 에이전트로 프로필 분석
            agent_result = await self.langchain_agent.run_agent(
                f"사용자 입력을 분석하여 프로필을 생성해주세요: {user_input}",
                {
                    "conversation_history": conversation_history,
                    "user_context": user_context
                }
            )
            
            if agent_result["success"]:
                # 결과 파싱 (실제 구현에서는 더 정교한 파싱 필요)
                profile_data = {
                    "interests": ["reading", "technology"],
                    "personality_type": "introvert",
                    "skill_level": "beginner",
                    "time_availability": "weekends",
                    "budget_range": "medium",
                    "location_preference": "Seoul",
                    "confidence_score": 0.85,
                    "analysis_reasoning": agent_result["output"]
                }
                
                # 메모리에 프로필 저장
                self.memory_manager.update_hobby_context(session_id, profile_data)
                
                # AI 응답 메시지 추가
                self.memory_manager.add_ai_message(
                    session_id, 
                    f"프로필 분석 완료: {json.dumps(profile_data, ensure_ascii=False)}", 
                    "profile_analyst"
                )
                
                # 세션 상태 업데이트
                self.active_sessions[session_id]["current_step"] = "profile_analysis_completed"
                self.active_sessions[session_id]["steps_completed"].append("profile_analysis")
                
                logger.info(f"프로필 분석 완료: {session_id}")
                return profile_data
            else:
                raise Exception(f"에이전트 실행 실패: {agent_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"프로필 분석 실패: {e}")
            raise
    
    async def _recommend_hobbies(self, session_id: str, user_profile: Dict[str, Any], 
                                preferences: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """취미 추천"""
        try:
            logger.info(f"취미 추천 시작: {session_id}")
            
            # 벡터 스토어에서 유사 취미 검색
            query = f"사용자 관심사: {', '.join(user_profile.get('interests', []))}"
            similar_hobbies = self.vector_store.search_similar_hobbies(query, user_profile, top_k=10)
            
            # LangChain 에이전트로 취미 추천
            agent_result = await self.langchain_agent.get_hobby_recommendations(
                user_profile=user_profile,
                preferences=preferences.get("hobby_preferences", []) if preferences else [],
                constraints={
                    "time": user_profile.get("time_availability", "flexible"),
                    "budget": user_profile.get("budget_range", "medium"),
                    "location": user_profile.get("location_preference", "Seoul")
                }
            )
            
            # 결과 통합
            hobby_recommendations = {
                "vector_search_results": similar_hobbies,
                "agent_recommendations": agent_result.recommendations,
                "overall_reasoning": agent_result.reasoning,
                "confidence_score": agent_result.confidence_score,
                "total_recommendations": len(similar_hobbies) + len(agent_result.recommendations)
            }
            
            # AI 응답 메시지 추가
            self.memory_manager.add_ai_message(
                session_id,
                f"취미 추천 완료: {len(hobby_recommendations['total_recommendations'])}개 추천",
                "hobby_discoverer"
            )
            
            # 세션 상태 업데이트
            self.active_sessions[session_id]["current_step"] = "hobby_recommendation_completed"
            self.active_sessions[session_id]["steps_completed"].append("hobby_recommendation")
            
            logger.info(f"취미 추천 완료: {session_id}")
            return hobby_recommendations
            
        except Exception as e:
            logger.error(f"취미 추천 실패: {e}")
            raise
    
    async def _match_communities(self, session_id: str, user_profile: Dict[str, Any], 
                                hobby_result: Dict[str, Any]) -> Dict[str, Any]:
        """커뮤니티 매칭"""
        try:
            logger.info(f"커뮤니티 매칭 시작: {session_id}")
            
            # 벡터 스토어에서 커뮤니티 검색
            hobby_categories = [hobby.get("category", "") for hobby in hobby_result.get("vector_search_results", [])]
            communities = self.vector_store.search_communities(
                f"취미 카테고리: {', '.join(hobby_categories)}", 
                user_profile, 
                top_k=5
            )
            
            # LangChain 에이전트로 커뮤니티 매칭
            agent_result = await self.langchain_agent.match_communities(
                user_profile=user_profile,
                hobby_interests=[hobby.get("name", "") for hobby in hobby_result.get("vector_search_results", [])],
                location=user_profile.get("location_preference", "Seoul")
            )
            
            # 결과 통합
            community_matches = {
                "vector_search_results": communities,
                "agent_matches": agent_result.matched_communities,
                "match_scores": agent_result.match_scores,
                "total_communities": len(communities) + len(agent_result.matched_communities)
            }
            
            # AI 응답 메시지 추가
            self.memory_manager.add_ai_message(
                session_id,
                f"커뮤니티 매칭 완료: {len(community_matches['total_communities'])}개 매칭",
                "community_matcher"
            )
            
            # 세션 상태 업데이트
            self.active_sessions[session_id]["current_step"] = "community_matching_completed"
            self.active_sessions[session_id]["steps_completed"].append("community_matching")
            
            logger.info(f"커뮤니티 매칭 완료: {session_id}")
            return community_matches
            
        except Exception as e:
            logger.error(f"커뮤니티 매칭 실패: {e}")
            raise
    
    async def _integrate_schedule(self, session_id: str, user_profile: Dict[str, Any], 
                                 hobby_result: Dict[str, Any]) -> Dict[str, Any]:
        """스케줄 통합"""
        try:
            logger.info(f"스케줄 통합 시작: {session_id}")
            
            # 현재 스케줄 정보 (실제 구현에서는 MCP 서버에서 가져옴)
            current_schedule = {
                "weekdays": {"evening": ["퇴근", "저녁식사"], "night": ["휴식"]},
                "weekends": {"morning": ["아침식사"], "afternoon": ["자유시간"], "evening": ["자유시간"]}
            }
            
            # LangChain 에이전트로 스케줄 통합
            agent_result = await self.langchain_agent.integrate_schedule(
                current_schedule=current_schedule,
                hobby_activities=hobby_result.get("vector_search_results", [])[:3],  # 상위 3개만
                time_constraints={
                    "available_time": user_profile.get("time_availability", "flexible"),
                    "preferred_duration": "1-2 hours"
                }
            )
            
            # 결과 구성
            schedule_integration = {
                "original_schedule": current_schedule,
                "integrated_schedule": agent_result.integrated_schedule,
                "optimization_suggestions": agent_result.optimization_suggestions,
                "hobbies_integrated": len(hobby_result.get("vector_search_results", [])[:3])
            }
            
            # AI 응답 메시지 추가
            self.memory_manager.add_ai_message(
                session_id,
                f"스케줄 통합 완료: {len(schedule_integration['hobbies_integrated'])}개 취미 통합",
                "schedule_integrator"
            )
            
            # 세션 상태 업데이트
            self.active_sessions[session_id]["current_step"] = "schedule_integration_completed"
            self.active_sessions[session_id]["steps_completed"].append("schedule_integration")
            
            logger.info(f"스케줄 통합 완료: {session_id}")
            return schedule_integration
            
        except Exception as e:
            logger.error(f"스케줄 통합 실패: {e}")
            raise
    
    async def _plan_progress_tracking(self, session_id: str, user_profile: Dict[str, Any], 
                                     hobby_result: Dict[str, Any]) -> Dict[str, Any]:
        """진행상황 추적 계획"""
        try:
            logger.info(f"진행상황 추적 계획 시작: {session_id}")
            
            # 진행상황 추적 계획 생성
            progress_tracking_plan = {
                "tracking_methods": [
                    "주간 활동 일지",
                    "취미 달성도 체크리스트",
                    "커뮤니티 참여 기록",
                    "개인 목표 설정 및 추적"
                ],
                "milestones": [
                    {"week": 1, "goal": "첫 번째 취미 활동 시작"},
                    {"week": 4, "goal": "정기적인 활동 패턴 확립"},
                    {"week": 8, "goal": "커뮤니티 적극 참여"},
                    {"week": 12, "goal": "새로운 취미 추가 고려"}
                ],
                "motivation_strategies": [
                    "작은 성취 축하하기",
                    "진행상황 시각화",
                    "동료와의 경쟁 요소",
                    "보상 시스템 구축"
                ]
            }
            
            # AI 응답 메시지 추가
            self.memory_manager.add_ai_message(
                session_id,
                f"진행상황 추적 계획 완료: {len(progress_tracking_plan['tracking_methods'])}개 방법 제안",
                "progress_tracker"
            )
            
            # 세션 상태 업데이트
            self.active_sessions[session_id]["current_step"] = "progress_tracking_planned"
            self.active_sessions[session_id]["steps_completed"].append("progress_tracking")
            
            logger.info(f"진행상황 추적 계획 완료: {session_id}")
            return progress_tracking_plan
            
        except Exception as e:
            logger.error(f"진행상황 추적 계획 실패: {e}")
            raise
    
    async def _complete_session(self, session_id: str):
        """세션 완료"""
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["completed_at"] = datetime.now().isoformat()
                self.active_sessions[session_id]["status"] = "completed"
                
                self.workflow_stats["completed_sessions"] += 1
                
                logger.info(f"세션 완료: {session_id}")
            
        except Exception as e:
            logger.error(f"세션 완료 처리 실패: {e}")
    
    async def _fail_session(self, session_id: str, error_message: str):
        """세션 실패 처리"""
        try:
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id]["failed_at"] = datetime.now().isoformat()
                self.active_sessions[session_id]["status"] = "failed"
                self.active_sessions[session_id]["error"] = error_message
                
                self.workflow_stats["failed_sessions"] += 1
                
                logger.error(f"세션 실패: {session_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"세션 실패 처리 실패: {e}")
    
    async def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """워크플로우 상태 조회"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "세션을 찾을 수 없습니다."}
            
            session_info = self.active_sessions[session_id]
            
            return {
                "session_id": session_id,
                "status": session_info.get("status", "active"),
                "current_step": session_info.get("current_step", "unknown"),
                "steps_completed": session_info.get("steps_completed", []),
                "started_at": session_info.get("started_at"),
                "completed_at": session_info.get("completed_at"),
                "error": session_info.get("error")
            }
            
        except Exception as e:
            logger.error(f"워크플로우 상태 조회 실패: {e}")
            return {"error": str(e)}
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """워크플로우 통계 조회"""
        try:
            active_sessions = len([s for s in self.active_sessions.values() if s.get("status") == "active"])
            
            return {
                **self.workflow_stats,
                "active_sessions": active_sessions,
                "memory_stats": self.memory_manager.get_memory_stats(),
                "vector_store_stats": self.vector_store.get_vector_store_stats(),
                "agent_status": self.langchain_agent.get_agent_status()
            }
            
        except Exception as e:
            logger.error(f"워크플로우 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    async def add_sample_data(self):
        """샘플 데이터를 벡터 스토어에 추가"""
        try:
            # 샘플 취미 데이터
            sample_hobbies = [
                {
                    "id": "hobby_001",
                    "name": "독서",
                    "description": "책을 읽으며 지식과 상상력을 키우는 활동",
                    "category": "intellectual",
                    "difficulty": "beginner",
                    "benefits": ["지식 습득", "집중력 향상", "스트레스 감소"],
                    "requirements": ["책", "조용한 공간"],
                    "tips": "매일 30분씩 읽는 습관을 들이세요."
                },
                {
                    "id": "hobby_002", 
                    "name": "등산",
                    "description": "자연 속에서 체력을 기르고 스트레스를 해소하는 활동",
                    "category": "outdoor",
                    "difficulty": "beginner",
                    "benefits": ["체력 향상", "자연 접촉", "정신 건강"],
                    "requirements": ["등산화", "등산복", "식수"],
                    "tips": "처음에는 쉬운 코스부터 시작하세요."
                }
            ]
            
            # 샘플 커뮤니티 데이터
            sample_communities = [
                {
                    "id": "comm_001",
                    "name": "독서 모임",
                    "description": "매주 책을 읽고 토론하는 온라인 모임",
                    "type": "online",
                    "category": "intellectual",
                    "members": 150,
                    "activities": ["월간 독서", "온라인 토론", "독서 후기 공유"],
                    "requirements": "독서에 대한 관심"
                },
                {
                    "id": "comm_002",
                    "name": "등산 동호회",
                    "description": "주말마다 함께 등산하는 오프라인 모임",
                    "type": "offline", 
                    "category": "outdoor",
                    "members": 80,
                    "activities": ["주말 등산", "등산 기술 교육", "친목 모임"],
                    "requirements": "건강한 체력"
                }
            ]
            
            # 벡터 스토어에 데이터 추가
            hobby_success = self.vector_store.add_hobby_documents(sample_hobbies)
            community_success = self.vector_store.add_community_documents(sample_communities)
            
            if hobby_success and community_success:
                logger.info("샘플 데이터 추가 완료")
                return True
            else:
                logger.warning("일부 샘플 데이터 추가 실패")
                return False
                
        except Exception as e:
            logger.error(f"샘플 데이터 추가 실패: {e}")
            return False
    
    def clear_workflow_data(self):
        """워크플로우 데이터 초기화"""
        try:
            # 메모리 초기화
            for user_id in list(self.memory_manager.user_memories.keys()):
                self.memory_manager.clear_user_memory(user_id)
            
            # 벡터 스토어 초기화
            self.vector_store.clear_vector_store()
            
            # 에이전트 메모리 초기화
            self.langchain_agent.clear_memory()
            
            # 세션 정보 초기화
            self.active_sessions.clear()
            
            # 통계 초기화
            self.workflow_stats = {
                "total_sessions": 0,
                "completed_sessions": 0,
                "failed_sessions": 0
            }
            
            logger.info("워크플로우 데이터 초기화 완료")
            
        except Exception as e:
            logger.error(f"워크플로우 데이터 초기화 실패: {e}")
