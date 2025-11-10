"""
Compliance Chain

LangGraph StateGraph 기반 의료기기 규제 컴플라이언스 테스트 워크플로우
"""

import logging
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state_management import ComplianceState
from ..agents.framework_analyzer import FrameworkAnalyzerAgent
from ..agents.test_generator import TestGeneratorAgent
from ..agents.test_executor import TestExecutorAgent
from ..agents.compliance_validator import ComplianceValidatorAgent
from ..agents.report_generator import ReportGeneratorAgent
from ..llm.model_manager import ModelManager
from ..llm.fallback_handler import FallbackHandler

logger = logging.getLogger(__name__)


class ComplianceChain:
    """
    의료기기 규제 컴플라이언스 테스트 워크플로우 체인
    
    LangGraph StateGraph를 사용하여 다음 단계를 순차적으로 실행:
    1. Framework Analysis
    2. Test Generation
    3. Test Execution
    4. Compliance Validation
    5. Report Generation
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        output_dir: str = "medical_device_compliance_reports"
    ):
        """
        ComplianceChain 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            output_dir: 출력 디렉토리
        """
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.output_dir = output_dir
        
        # Agents 초기화
        self.framework_analyzer = FrameworkAnalyzerAgent(
            model_manager, fallback_handler
        )
        self.test_generator = TestGeneratorAgent(
            model_manager, fallback_handler
        )
        self.test_executor = TestExecutorAgent(
            model_manager, fallback_handler
        )
        self.compliance_validator = ComplianceValidatorAgent(
            model_manager, fallback_handler
        )
        self.report_generator = ReportGeneratorAgent(
            model_manager, fallback_handler
        )
        
        # LangGraph 워크플로우 설정
        self.memory = MemorySaver()
        self._setup_workflow()
    
    def _setup_workflow(self):
        """LangGraph 워크플로우 설정"""
        workflow = StateGraph(ComplianceState)
        
        # 노드 추가
        workflow.add_node("framework_analysis", self._framework_analysis_node)
        workflow.add_node("test_generation", self._test_generation_node)
        workflow.add_node("test_execution", self._test_execution_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("report_generation", self._report_generation_node)
        
        # 엣지 설정
        workflow.set_entry_point("framework_analysis")
        workflow.add_edge("framework_analysis", "test_generation")
        workflow.add_edge("test_generation", "test_execution")
        workflow.add_edge("test_execution", "validation")
        workflow.add_edge("validation", "report_generation")
        workflow.add_edge("report_generation", END)
        
        # 컴파일
        self.app = workflow.compile(checkpointer=self.memory)
        logger.info("Compliance Chain workflow initialized")
    
    def _framework_analysis_node(self, state: ComplianceState) -> ComplianceState:
        """규제 프레임워크 분석 노드"""
        try:
            logger.info("Starting framework analysis")
            result = self.framework_analyzer.analyze(state["device_info"])
            
            state["regulatory_frameworks"] = result.get("frameworks", [])
            state["framework_requirements"] = result.get("requirements", {})
            state["workflow_stage"] = "framework_analysis"
            
            if not result.get("success"):
                state["errors"].append(f"Framework analysis failed: {result.get('analysis', 'Unknown error')}")
            
            logger.info(f"Framework analysis completed: {len(state['regulatory_frameworks'])} frameworks identified")
        
        except Exception as e:
            logger.error(f"Framework analysis error: {e}")
            state["errors"].append(f"Framework analysis error: {str(e)}")
        
        return state
    
    def _test_generation_node(self, state: ComplianceState) -> ComplianceState:
        """테스트 케이스 생성 노드"""
        try:
            logger.info("Starting test generation")
            test_cases = self.test_generator.generate(state["framework_requirements"])
            
            state["test_cases"] = test_cases
            state["workflow_stage"] = "test_generation"
            
            logger.info(f"Test generation completed: {len(test_cases)} test cases generated")
        
        except Exception as e:
            logger.error(f"Test generation error: {e}")
            state["errors"].append(f"Test generation error: {str(e)}")
        
        return state
    
    def _test_execution_node(self, state: ComplianceState) -> ComplianceState:
        """테스트 실행 노드"""
        try:
            logger.info("Starting test execution")
            test_results = self.test_executor.execute(state["test_cases"])
            
            state["test_results"] = test_results
            state["workflow_stage"] = "test_execution"
            
            logger.info(f"Test execution completed: {len(test_results)} results collected")
        
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            state["errors"].append(f"Test execution error: {str(e)}")
        
        return state
    
    def _validation_node(self, state: ComplianceState) -> ComplianceState:
        """규제 준수 검증 노드"""
        try:
            logger.info("Starting compliance validation")
            validation_result = self.compliance_validator.validate(
                state["test_results"],
                state["framework_requirements"]
            )
            
            state["compliance_status"] = validation_result.get("compliance_status", "UNKNOWN")
            state["compliance_score"] = validation_result.get("compliance_score", 0.0)
            state["risk_assessment"] = validation_result.get("risk_assessment", {})
            state["workflow_stage"] = "validation"
            
            logger.info(f"Compliance validation completed: {state['compliance_status']}")
        
        except Exception as e:
            logger.error(f"Compliance validation error: {e}")
            state["errors"].append(f"Compliance validation error: {str(e)}")
        
        return state
    
    def _report_generation_node(self, state: ComplianceState) -> ComplianceState:
        """리포트 생성 노드"""
        try:
            logger.info("Starting report generation")
            
            report_path = f"{self.output_dir}/compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            report_content = self.report_generator.generate(
                report_type="Compliance Report",
                device_info=state["device_info"],
                compliance_data={
                    "frameworks": state["regulatory_frameworks"],
                    "compliance_status": state["compliance_status"],
                    "compliance_score": state["compliance_score"],
                    "risk_assessment": state["risk_assessment"]
                },
                output_path=report_path
            )
            
            state["report"] = report_content
            state["report_path"] = report_path
            state["workflow_stage"] = "report_generation"
            
            logger.info(f"Report generation completed: {report_path}")
        
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            state["errors"].append(f"Report generation error: {str(e)}")
        
        return state
    
    def run(self, device_info: Dict[str, Any]) -> ComplianceState:
        """
        워크플로우 실행
        
        Args:
            device_info: 의료기기 정보
        
        Returns:
            최종 상태
        """
        # 초기 상태 생성
        initial_state: ComplianceState = {
            "device_info": device_info,
            "regulatory_frameworks": [],
            "framework_requirements": {},
            "test_cases": [],
            "test_results": [],
            "compliance_status": "UNKNOWN",
            "compliance_score": 0.0,
            "risk_assessment": {},
            "report": None,
            "report_path": None,
            "timestamp": datetime.now().isoformat(),
            "workflow_stage": "initialized",
            "errors": [],
            "warnings": []
        }
        
        try:
            # 워크플로우 실행
            config = {"configurable": {"thread_id": "compliance_workflow"}}
            final_state = self.app.invoke(initial_state, config)
            
            logger.info("Compliance workflow completed successfully")
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state["errors"].append(f"Workflow execution error: {str(e)}")
            return initial_state

