"""
PetCare 워크플로우 체인

LangGraph StateGraph를 사용하여 반려동물 케어 워크플로우를 순차적으로 실행
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state_management import PetCareState
from ..agents.profile_analyzer import PetProfileAnalyzerAgent
from ..agents.health_monitor import HealthMonitorAgent
from ..agents.behavior_analyzer import BehaviorAnalyzerAgent
from ..agents.care_planner import CarePlannerAgent
from ..agents.physical_ai_controller import PhysicalAIControllerAgent
from ..agents.pet_assistant import PetAssistantAgent
from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..config.petcare_config import PetCareConfig
from ..utils.validators import validate_pet_profile, InputValidationError

logger = logging.getLogger(__name__)


class PetCareChain:
    """
    반려동물 케어 워크플로우 체인
    
    LangGraph StateGraph를 사용하여 다음 단계를 순차적으로 실행:
    1. Validate Input
    2. Profile Analysis
    3. Health Monitoring
    4. Behavior Analysis
    5. Care Plan Generation
    6. Physical AI Device Control (if needed)
    7. Final Report Generation
    """
    
    def __init__(
        self,
        config: PetCareConfig,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None
    ):
        """
        PetCareChain 초기화
        
        Args:
            config: PetCareConfig 인스턴스
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            preferred_provider: 선호하는 Provider
        """
        self.config = config
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        self.memory = MemorySaver()

        # 에이전트 초기화 (config 전달하여 최신 Physical AI 기술 사용)
        self.profile_analyzer = PetProfileAnalyzerAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)
        self.health_monitor = HealthMonitorAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)
        self.behavior_analyzer = BehaviorAnalyzerAgent(model_manager, fallback_handler, preferred_provider, config.data_dir)
        self.care_planner = CarePlannerAgent(model_manager, fallback_handler, preferred_provider, config.data_dir, config)
        self.physical_ai_controller = PhysicalAIControllerAgent(model_manager, fallback_handler, preferred_provider, config.data_dir, config)
        self.pet_assistant = PetAssistantAgent(model_manager, fallback_handler, preferred_provider, config.data_dir, config)

        self.workflow = self._setup_langgraph_workflow()
        logger.info("PetCareChain initialized with LangGraph workflow.")

    def _setup_langgraph_workflow(self):
        """LangGraph 워크플로우 설정"""
        workflow = StateGraph(PetCareState)

        # 노드 추가
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("analyze_profile", self._analyze_profile_node)
        workflow.add_node("monitor_health", self._monitor_health_node)
        workflow.add_node("analyze_behavior", self._analyze_behavior_node)
        workflow.add_node("create_care_plan", self._create_care_plan_node)
        workflow.add_node("control_devices", self._control_devices_node)
        workflow.add_node("generate_final_report", self._generate_final_report_node)

        # 시작점 설정
        workflow.set_entry_point("validate_input")

        # 엣지 (전환 로직) 설정
        workflow.add_edge("validate_input", "analyze_profile")
        workflow.add_edge("analyze_profile", "monitor_health")
        workflow.add_edge("monitor_health", "analyze_behavior")
        workflow.add_edge("analyze_behavior", "create_care_plan")
        workflow.add_edge("create_care_plan", "control_devices")
        workflow.add_edge("control_devices", "generate_final_report")
        workflow.add_edge("generate_final_report", END)

        return workflow.compile(checkpointer=self.memory)

    async def _validate_input_node(self, state: PetCareState) -> PetCareState:
        """입력 유효성 검사 노드"""
        logger.info("Executing node: validate_input")
        try:
            if "pet_id" not in state or not state["pet_id"]:
                raise InputValidationError("pet_id가 필요합니다.")
            
            if "pet_profile" in state and state["pet_profile"]:
                validated_profile = validate_pet_profile(state["pet_profile"])
                return {**state, "pet_profile": validated_profile, "current_step": "input_validated", "errors": []}
            
            return {**state, "current_step": "input_validated", "errors": []}
        except InputValidationError as e:
            logger.error(f"Input validation failed: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}
        except Exception as e:
            logger.error(f"Unexpected error in validate_input: {e}")
            return {**state, "errors": state.get("errors", []) + [f"Unexpected error: {e}"], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _analyze_profile_node(self, state: PetCareState) -> PetCareState:
        """프로필 분석 노드"""
        logger.info("Executing node: analyze_profile")
        try:
            pet_info = state.get("pet_profile", {})
            if not pet_info:
                pet_info = {"pet_id": state["pet_id"]}
            
            analysis_result = await self.profile_analyzer.analyze_profile(pet_info)
            return {**state, "pet_profile": analysis_result.get("profile", {}), "current_step": "profile_analyzed", "errors": []}
        except Exception as e:
            logger.error(f"Failed to analyze profile: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _monitor_health_node(self, state: PetCareState) -> PetCareState:
        """건강 모니터링 노드"""
        logger.info("Executing node: monitor_health")
        try:
            health_result = await self.health_monitor.monitor_health(state["pet_id"])
            return {**state, "health_status": health_result, "current_step": "health_monitored", "errors": []}
        except Exception as e:
            logger.error(f"Failed to monitor health: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _analyze_behavior_node(self, state: PetCareState) -> PetCareState:
        """행동 분석 노드"""
        logger.info("Executing node: analyze_behavior")
        try:
            behavior_result = await self.behavior_analyzer.analyze_behavior(state["pet_id"])
            return {**state, "behavior_analysis": behavior_result, "current_step": "behavior_analyzed", "errors": []}
        except Exception as e:
            logger.error(f"Failed to analyze behavior: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _create_care_plan_node(self, state: PetCareState) -> PetCareState:
        """케어 계획 생성 노드"""
        logger.info("Executing node: create_care_plan")
        try:
            care_plan_result = await self.care_planner.create_care_plan(
                state["pet_id"],
                state.get("pet_profile"),
                state.get("health_status")
            )
            return {**state, "care_plan": care_plan_result.get("care_plan", {}), "current_step": "care_plan_created", "errors": []}
        except Exception as e:
            logger.error(f"Failed to create care plan: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _control_devices_node(self, state: PetCareState) -> PetCareState:
        """Physical AI 기기 제어 노드"""
        logger.info("Executing node: control_devices")
        try:
            care_plan = state.get("care_plan", {})
            device_integration = care_plan.get("physical_ai_integration", {})
            
            device_results = []
            if device_integration:
                # 케어 계획에 따라 기기 제어
                for device_type, action_description in device_integration.items():
                    try:
                        if device_type == "robot_vacuum" and "배변" in action_description:
                            result = await self.physical_ai_controller.control_devices(
                                state["pet_id"],
                                "배변 후 청소 필요",
                                state.get("pet_profile")
                            )
                            device_results.append(result)
                        elif device_type == "smart_toy" and "활동량" in action_description:
                            result = await self.physical_ai_controller.control_devices(
                                state["pet_id"],
                                "활동량 낮음 - 놀이 필요",
                                state.get("pet_profile")
                            )
                            device_results.append(result)
                    except Exception as e:
                        logger.warning(f"Device control failed for {device_type}: {e}")
            
            return {**state, "device_control_results": device_results, "current_step": "devices_controlled", "errors": []}
        except Exception as e:
            logger.error(f"Failed to control devices: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def _generate_final_report_node(self, state: PetCareState) -> PetCareState:
        """최종 리포트 생성 노드"""
        logger.info("Executing node: generate_final_report")
        try:
            user_request = state.get("user_input", "No specific request.")
            final_report_content = await self.pet_assistant.assist_pet_care(
                user_request,
                state["pet_id"],
                {
                    "pet_profile": state.get("pet_profile"),
                    "health_status": state.get("health_status"),
                    "behavior_analysis": state.get("behavior_analysis"),
                    "care_plan": state.get("care_plan"),
                    "device_control_results": state.get("device_control_results"),
                }
            )
            
            # 최종 리포트를 파일로 저장
            timestamp = datetime.now().strftime(self.config.report_timestamp_format)
            output_file_path = Path(self.config.output_dir) / f"petcare_report_{state['pet_id']}_{timestamp}.json"
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_report_content, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Final pet care report saved to {output_file_path}")
            return {**state, "final_report": {**final_report_content, "report_path": str(output_file_path)}, "current_step": "final_report_generated", "errors": []}
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
            return {**state, "errors": state.get("errors", []) + [str(e)], "current_step": "error", "retry_count": state.get("retry_count", 0) + 1}

    async def run_workflow(
        self,
        user_input: str,
        pet_id: str,
        pet_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        전체 반려동물 케어 워크플로우를 실행합니다.
        
        Args:
            user_input: 사용자의 초기 요청
            pet_id: 반려동물 ID
            pet_profile: 반려동물 프로필 (선택 사항)
        
        Returns:
            최종 워크플로우 상태
        """
        initial_state: PetCareState = {
            "user_input": user_input,
            "pet_id": pet_id,
            "pet_profile": pet_profile or {},
            "health_status": {},
            "behavior_analysis": {},
            "care_plan": {},
            "device_control_results": [],
            "final_report": {},
            "errors": [],
            "current_step": "initialized",
            "retry_count": 0
        }
        
        try:
            # LangGraph 1.0+ requires thread_id in config when using checkpointer
            config = {"configurable": {"thread_id": f"petcare_{pet_id}"}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            logger.info("Pet care workflow completed.")
            return final_state
        except Exception as e:
            logger.critical(f"Pet care workflow terminated with an unhandled error: {e}")
            return {**initial_state, "errors": initial_state.get("errors", []) + [f"Workflow terminated unexpectedly: {e}"], "current_step": "failed"}

