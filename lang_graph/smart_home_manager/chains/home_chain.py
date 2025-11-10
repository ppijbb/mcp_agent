"""
Home Chain

LangGraph StateGraph 기반 스마트 홈 매니저 워크플로우
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state_management import HomeState
from ..agents.device_manager import DeviceManagerAgent
from ..agents.energy_optimizer import EnergyOptimizerAgent
from ..agents.security_monitor import SecurityMonitorAgent
from ..agents.maintenance_alert import MaintenanceAlertAgent
from ..agents.automation_scenario import AutomationScenarioAgent
from ..llm.model_manager import ModelManager
from ..llm.fallback_handler import FallbackHandler

logger = logging.getLogger(__name__)


class HomeChain:
    """
    스마트 홈 매니저 워크플로우 체인
    
    LangGraph StateGraph를 사용하여 다음 단계를 순차적으로 실행:
    1. Device Management
    2. Energy Optimization
    3. Security Monitoring
    4. Maintenance Alerts
    5. Automation Scenarios
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        output_dir: str = "smart_home_reports",
        data_dir: str = "home_data"
    ):
        """
        HomeChain 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            output_dir: 출력 디렉토리
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.output_dir = output_dir
        self.data_dir = data_dir
        
        # Agents 초기화
        self.device_manager = DeviceManagerAgent(
            model_manager, fallback_handler, data_dir=data_dir
        )
        self.energy_optimizer = EnergyOptimizerAgent(
            model_manager, fallback_handler, data_dir=data_dir
        )
        self.security_monitor = SecurityMonitorAgent(
            model_manager, fallback_handler, data_dir=data_dir
        )
        self.maintenance_alert = MaintenanceAlertAgent(
            model_manager, fallback_handler, data_dir=data_dir
        )
        self.automation_scenario = AutomationScenarioAgent(
            model_manager, fallback_handler, data_dir=data_dir
        )
        
        # LangGraph 워크플로우 설정
        self.memory = MemorySaver()
        self._setup_workflow()
    
    def _setup_workflow(self):
        """LangGraph 워크플로우 설정"""
        workflow = StateGraph(HomeState)
        
        # 노드 추가
        workflow.add_node("device_management", self._device_management_node)
        workflow.add_node("energy_optimization", self._energy_optimization_node)
        workflow.add_node("security_monitoring", self._security_monitoring_node)
        workflow.add_node("maintenance_alerts", self._maintenance_alerts_node)
        workflow.add_node("automation_scenarios", self._automation_scenarios_node)
        
        # 엣지 설정
        workflow.set_entry_point("device_management")
        workflow.add_edge("device_management", "energy_optimization")
        workflow.add_edge("energy_optimization", "security_monitoring")
        workflow.add_edge("security_monitoring", "maintenance_alerts")
        workflow.add_edge("maintenance_alerts", "automation_scenarios")
        workflow.add_edge("automation_scenarios", END)
        
        # 컴파일
        self.app = workflow.compile(checkpointer=self.memory)
        logger.info("Home Chain workflow initialized")
    
    def _device_management_node(self, state: HomeState) -> HomeState:
        """기기 관리 노드"""
        try:
            logger.info("Starting device management")
            result = self.device_manager.manage_devices("status")
            
            state["device_status"] = result.get("status", {})
            state["workflow_stage"] = "device_management"
            
            if not result.get("success"):
                state["errors"].append(f"Device management failed: {result.get('output', 'Unknown error')}")
            
            logger.info(f"Device management completed for home: {state['home_id']}")
        
        except Exception as e:
            logger.error(f"Device management error: {e}")
            state["errors"].append(f"Device management error: {str(e)}")
        
        return state
    
    def _energy_optimization_node(self, state: HomeState) -> HomeState:
        """에너지 최적화 노드"""
        try:
            logger.info("Starting energy optimization")
            result = self.energy_optimizer.optimize(period="week")
            
            state["energy_optimization"] = result.get("optimization", {})
            state["workflow_stage"] = "energy_optimization"
            
            logger.info(f"Energy optimization completed")
        
        except Exception as e:
            logger.error(f"Energy optimization error: {e}")
            state["errors"].append(f"Energy optimization error: {str(e)}")
        
        return state
    
    def _security_monitoring_node(self, state: HomeState) -> HomeState:
        """보안 모니터링 노드"""
        try:
            logger.info("Starting security monitoring")
            result = self.security_monitor.monitor(check_type="all")
            
            state["security_status"] = result.get("security_status", {})
            state["workflow_stage"] = "security_monitoring"
            
            logger.info(f"Security monitoring completed")
        
        except Exception as e:
            logger.error(f"Security monitoring error: {e}")
            state["errors"].append(f"Security monitoring error: {str(e)}")
        
        return state
    
    def _maintenance_alerts_node(self, state: HomeState) -> HomeState:
        """유지보수 알림 노드"""
        try:
            logger.info("Starting maintenance alerts")
            alerts = self.maintenance_alert.check_maintenance()
            
            state["maintenance_schedule"] = alerts
            state["workflow_stage"] = "maintenance_alerts"
            
            logger.info(f"Maintenance alerts completed: {len(alerts)} alerts")
        
        except Exception as e:
            logger.error(f"Maintenance alerts error: {e}")
            state["errors"].append(f"Maintenance alerts error: {str(e)}")
        
        return state
    
    def _automation_scenarios_node(self, state: HomeState) -> HomeState:
        """자동화 시나리오 노드"""
        try:
            logger.info("Starting automation scenarios")
            
            # 기본 시나리오 생성 예시
            scenario_result = self.automation_scenario.create_scenario(
                scenario_name="Evening Routine",
                description="Automatically adjust lighting and temperature in the evening",
                triggers=[{"type": "time", "value": "18:00"}],
                actions=[{"device": "lighting", "action": "dim"}, {"device": "heating", "action": "set_temperature", "value": 20}]
            )
            
            state["automation_scenarios"] = [scenario_result.get("scenario", {})]
            state["workflow_stage"] = "automation_scenarios"
            
            logger.info(f"Automation scenarios completed")
        
        except Exception as e:
            logger.error(f"Automation scenarios error: {e}")
            state["errors"].append(f"Automation scenarios error: {str(e)}")
        
        return state
    
    def run(
        self,
        user_id: str,
        home_id: str,
        devices: Optional[List[Dict[str, Any]]] = None
    ) -> HomeState:
        """
        워크플로우 실행
        
        Args:
            user_id: 사용자 ID
            home_id: 홈 ID
            devices: 기기 목록 (선택)
        
        Returns:
            최종 상태
        """
        # 초기 상태 생성
        initial_state: HomeState = {
            "user_id": user_id,
            "home_id": home_id,
            "devices": devices or [],
            "device_status": {},
            "energy_usage": {},
            "energy_optimization": {},
            "security_status": {},
            "security_alerts": [],
            "maintenance_schedule": [],
            "automation_scenarios": [],
            "timestamp": datetime.now().isoformat(),
            "workflow_stage": "initialized",
            "errors": [],
            "warnings": []
        }
        
        try:
            # 워크플로우 실행
            config = {"configurable": {"thread_id": f"home_workflow_{home_id}"}}
            final_state = self.app.invoke(initial_state, config)
            
            logger.info("Home workflow completed successfully")
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state["errors"].append(f"Workflow execution error: {str(e)}")
            return initial_state

