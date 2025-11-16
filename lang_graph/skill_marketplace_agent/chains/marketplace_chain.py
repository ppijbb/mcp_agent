"""
Skill Marketplace 워크플로우 체인

LangGraph StateGraph를 사용하여 Skill Marketplace 워크플로우를 순차적으로 실행
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state_management import MarketplaceState
from ..agents.learner_profile_analyzer import LearnerProfileAnalyzerAgent
from ..agents.skill_path_recommender import SkillPathRecommenderAgent
from ..agents.instructor_matcher import InstructorMatcherAgent
from ..agents.content_recommender import ContentRecommenderAgent
from ..agents.learning_progress_tracker import LearningProgressTrackerAgent
from ..agents.marketplace_orchestrator import MarketplaceOrchestratorAgent
from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..config.marketplace_config import MarketplaceConfig
from ..utils.validators import validate_learner_profile, InputValidationError

logger = logging.getLogger(__name__)


class MarketplaceChain:
    """
    Skill Marketplace 워크플로우 체인
    
    LangGraph StateGraph를 사용하여 다음 단계를 순차적으로 실행:
    1. Validate Input
    2. Analyze Learner Profile
    3. Recommend Skill Path
    4. Match Instructor (양면 시장 핵심)
    5. Recommend Content
    6. Create Learning Plan
    7. Process Marketplace Transaction (수익화)
    8. Generate Final Report
    """
    
    def __init__(
        self,
        config: MarketplaceConfig,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None
    ):
        """
        MarketplaceChain 초기화
        
        Args:
            config: MarketplaceConfig 인스턴스
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            preferred_provider: 선호하는 Provider
        """
        self.config = config
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        self.memory = MemorySaver()

        # 에이전트 초기화
        self.learner_profile_analyzer = LearnerProfileAnalyzerAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)
        self.skill_path_recommender = SkillPathRecommenderAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)
        self.instructor_matcher = InstructorMatcherAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)
        self.content_recommender = ContentRecommenderAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)
        self.learning_progress_tracker = LearningProgressTrackerAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)
        self.marketplace_orchestrator = MarketplaceOrchestratorAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)

        self.workflow = self._setup_langgraph_workflow()
        logger.info("MarketplaceChain initialized with LangGraph workflow.")

    def _setup_langgraph_workflow(self):
        """LangGraph 워크플로우 설정"""
        workflow = StateGraph(MarketplaceState)

        # 노드 추가
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("analyze_learner_profile", self._analyze_learner_profile_node)
        workflow.add_node("recommend_skill_path", self._recommend_skill_path_node)
        workflow.add_node("match_instructor", self._match_instructor_node)
        workflow.add_node("recommend_content", self._recommend_content_node)
        workflow.add_node("create_learning_plan", self._create_learning_plan_node)
        workflow.add_node("process_transaction", self._process_transaction_node)
        workflow.add_node("generate_final_report", self._generate_final_report_node)

        # 시작점 설정
        workflow.set_entry_point("validate_input")

        # 엣지 (전환 로직) 설정
        workflow.add_edge("validate_input", "analyze_learner_profile")
        workflow.add_edge("analyze_learner_profile", "recommend_skill_path")
        workflow.add_edge("recommend_skill_path", "match_instructor")
        workflow.add_edge("match_instructor", "recommend_content")
        workflow.add_edge("recommend_content", "create_learning_plan")
        workflow.add_edge("create_learning_plan", "process_transaction")
        workflow.add_edge("process_transaction", "generate_final_report")
        workflow.add_edge("generate_final_report", END)

        return workflow.compile(checkpointer=self.memory)

    async def _validate_input_node(self, state: MarketplaceState) -> MarketplaceState:
        """입력 유효성 검사 노드"""
        logger.info("Executing node: validate_input")
        try:
            if "learner_id" not in state or not state["learner_id"]:
                raise InputValidationError("learner_id가 필요합니다.")
            
            if "learner_profile" in state and state["learner_profile"]:
                validated_profile = validate_learner_profile(state["learner_profile"])
                return {**state, "learner_profile": validated_profile, "current_step": "input_validated", "errors": []}
            
            return {**state, "current_step": "input_validated", "errors": []}
        except InputValidationError as e:
            logger.error(f"Input validation failed: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}
        except Exception as e:
            logger.error(f"Unexpected error in validate_input: {e}")
            return {**state, "errors": state.get("errors", []) + [f"Unexpected error: {e}"], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _analyze_learner_profile_node(self, state: MarketplaceState) -> MarketplaceState:
        """학습자 프로필 분석 노드"""
        logger.info("Executing node: analyze_learner_profile")
        try:
            learner_info = state.get("learner_profile", {})
            if not learner_info:
                learner_info = {"learner_id": state["learner_id"]}
            
            analysis_result = await self.learner_profile_analyzer.analyze_profile(learner_info)
            return {**state, "learner_profile": analysis_result.get("profile", {}), "current_step": "learner_profile_analyzed", "errors": []}
        except Exception as e:
            logger.error(f"Failed to analyze learner profile: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _recommend_skill_path_node(self, state: MarketplaceState) -> MarketplaceState:
        """스킬 경로 추천 노드"""
        logger.info("Executing node: recommend_skill_path")
        try:
            # 사용자 입력에서 목표 스킬 추출 (간단한 추출 로직)
            user_input = state.get("user_input", "")
            target_skill = "Python"  # 기본값, 실제로는 LLM으로 추출
            
            skill_path_result = await self.skill_path_recommender.recommend_skill_path(
                state["learner_id"],
                target_skill,
                state.get("learner_profile")
            )
            return {**state, "skill_path": skill_path_result.get("learning_path", {}), "current_step": "skill_path_recommended", "errors": []}
        except Exception as e:
            logger.error(f"Failed to recommend skill path: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _match_instructor_node(self, state: MarketplaceState) -> MarketplaceState:
        """강사 매칭 노드 (양면 시장 핵심)"""
        logger.info("Executing node: match_instructor")
        try:
            skill_path = state.get("skill_path", {})
            target_skill = skill_path.get("target_skill", "Python")  # 기본값
            
            match_result = await self.instructor_matcher.match_instructor(
                state["learner_id"],
                target_skill,
                state.get("learner_profile")
            )
            return {**state, "matched_instructors": match_result.get("matched_instructors", []), "current_step": "instructor_matched", "errors": []}
        except Exception as e:
            logger.error(f"Failed to match instructor: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _recommend_content_node(self, state: MarketplaceState) -> MarketplaceState:
        """컨텐츠 추천 노드"""
        logger.info("Executing node: recommend_content")
        try:
            skill_path = state.get("skill_path", {})
            target_skill = skill_path.get("target_skill", "Python")  # 기본값
            
            content_result = await self.content_recommender.recommend_content(
                state["learner_id"],
                target_skill,
                state.get("learner_profile")
            )
            return {**state, "recommended_content": content_result.get("recommended_content", []), "current_step": "content_recommended", "errors": []}
        except Exception as e:
            logger.error(f"Failed to recommend content: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _create_learning_plan_node(self, state: MarketplaceState) -> MarketplaceState:
        """학습 계획 생성 노드"""
        logger.info("Executing node: create_learning_plan")
        try:
            learning_plan = {
                "learner_id": state["learner_id"],
                "skill_path": state.get("skill_path", {}),
                "matched_instructors": state.get("matched_instructors", []),
                "recommended_content": state.get("recommended_content", []),
                "estimated_total_hours": state.get("skill_path", {}).get("total_estimated_hours", 0),
                "created_at": datetime.now().isoformat()
            }
            return {**state, "learning_plan": learning_plan, "current_step": "learning_plan_created", "errors": []}
        except Exception as e:
            logger.error(f"Failed to create learning plan: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _process_transaction_node(self, state: MarketplaceState) -> MarketplaceState:
        """Marketplace 거래 처리 노드 (수익화)"""
        logger.info("Executing node: process_transaction")
        try:
            matched_instructors = state.get("matched_instructors", [])
            if not matched_instructors:
                return {**state, "marketplace_transaction": {}, "current_step": "transaction_skipped", "errors": []}
            
            # 첫 번째 매칭된 강사로 세션 생성 (시뮬레이션)
            top_instructor = matched_instructors[0]
            instructor_id = top_instructor.get("instructor_id", "")
            
            # 거래 정보 생성 (실제로는 Marketplace Tools 사용)
            transaction = {
                "learner_id": state["learner_id"],
                "instructor_id": instructor_id,
                "status": "pending",
                "commission_rate": self.config.default_commission_rate,
                "created_at": datetime.now().isoformat()
            }
            
            return {**state, "marketplace_transaction": transaction, "current_step": "transaction_processed", "errors": []}
        except Exception as e:
            logger.error(f"Failed to process transaction: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _generate_final_report_node(self, state: MarketplaceState) -> MarketplaceState:
        """최종 리포트 생성 노드"""
        logger.info("Executing node: generate_final_report")
        try:
            user_request = state.get("user_input", "No specific request.")
            final_report_content = await self.marketplace_orchestrator.orchestrate_marketplace(
                user_request,
                state["learner_id"],
                {
                    "learner_profile": state.get("learner_profile"),
                    "skill_path": state.get("skill_path"),
                    "matched_instructors": state.get("matched_instructors"),
                    "recommended_content": state.get("recommended_content"),
                    "learning_plan": state.get("learning_plan"),
                    "marketplace_transaction": state.get("marketplace_transaction"),
                }
            )
            
            # 최종 리포트를 파일로 저장
            timestamp = datetime.now().strftime(self.config.report_timestamp_format)
            output_file_path = Path(self.config.output_dir) / f"marketplace_report_{state['learner_id']}_{timestamp}.json"
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_report_content, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Final marketplace report saved to {output_file_path}")
            return {**state, "final_report": {**final_report_content, "report_path": str(output_file_path)}, "current_step": "final_report_generated", "errors": []}
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def run_workflow(
        self,
        user_input: str,
        learner_id: str,
        learner_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        전체 Skill Marketplace 워크플로우를 실행합니다.
        
        Args:
            user_input: 사용자의 초기 요청
            learner_id: 학습자 ID
            learner_profile: 학습자 프로필 (선택 사항)
        
        Returns:
            최종 워크플로우 상태
        """
        initial_state: MarketplaceState = {
            "user_input": user_input,
            "learner_id": learner_id,
            "learner_profile": learner_profile or {},
            "skill_path": {},
            "matched_instructors": [],
            "recommended_content": [],
            "learning_plan": {},
            "marketplace_transaction": {},
            "final_report": {},
            "errors": [],
            "current_step": "initialized",
            "retry_count": 0
        }
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            logger.info("Marketplace workflow completed.")
            return final_state
        except Exception as e:
            logger.critical(f"Marketplace workflow terminated with an unhandled error: {e}")
            return {**initial_state, "errors": initial_state.get("errors", []) + [f"Workflow terminated unexpectedly: {e}"], "current_step": "failed"}

