"""
의료기기 규제 컴플라이언스 테스트 자동화 Agent 메인 진입점
"""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

from .llm.model_manager import ModelManager, ModelProvider
from .llm.fallback_handler import FallbackHandler
from .chains.compliance_chain import ComplianceChain
from .config.compliance_config import ComplianceConfig
from .utils.validators import validate_device_info

logger = logging.getLogger(__name__)


class MedicalDeviceComplianceAgent:
    """
    의료기기 규제 컴플라이언스 테스트 자동화 Agent
    
    FDA 510(k), CE 마킹, ISO 13485 등 규제 컴플라이언스를
    자동으로 테스트하고 검증하는 Multi-Agent 시스템
    """
    
    def __init__(
        self,
        config: Optional[ComplianceConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        MedicalDeviceComplianceAgent 초기화
        
        Args:
            config: 설정 객체 (선택, 없으면 환경 변수에서 로드)
            output_dir: 출력 디렉토리 (선택)
        """
        self.config = config or ComplianceConfig.from_env()
        self.output_dir = output_dir or self.config.output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # LLM Manager 및 Fallback Handler 초기화
        self.model_manager = ModelManager(
            budget_limit=self.config.llm.budget_limit
        )
        self.fallback_handler = FallbackHandler(self.model_manager)
        
        # Compliance Chain 초기화
        self.compliance_chain = ComplianceChain(
            model_manager=self.model_manager,
            fallback_handler=self.fallback_handler,
            output_dir=self.output_dir
        )
        
        logger.info("Medical Device Compliance Agent initialized")
    
    def run_compliance_test(
        self,
        device_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        규제 컴플라이언스 테스트 실행
        
        Args:
            device_info: 의료기기 정보
                - name: 의료기기 이름
                - type: 의료기기 유형
                - classification: 의료기기 분류
                - description: 의료기기 설명 (선택)
                - intended_use: 사용 목적 (선택)
        
        Returns:
            테스트 결과
        """
        try:
            # 입력 검증
            if not validate_device_info(device_info):
                return {
                    "success": False,
                    "error": "Invalid device information",
                    "device_info": device_info
                }
            
            logger.info(f"Starting compliance test for device: {device_info.get('name')}")
            
            # 워크플로우 실행
            final_state = self.compliance_chain.run(device_info)
            
            # 결과 반환
            return {
                "success": len(final_state.get("errors", [])) == 0,
                "device_info": device_info,
                "regulatory_frameworks": final_state.get("regulatory_frameworks", []),
                "compliance_status": final_state.get("compliance_status", "UNKNOWN"),
                "compliance_score": final_state.get("compliance_score", 0.0),
                "risk_assessment": final_state.get("risk_assessment", {}),
                "report_path": final_state.get("report_path"),
                "errors": final_state.get("errors", []),
                "warnings": final_state.get("warnings", []),
                "timestamp": final_state.get("timestamp")
            }
        
        except Exception as e:
            logger.error(f"Compliance test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "device_info": device_info
            }


async def main():
    """메인 실행 함수"""
    # 예제 의료기기 정보
    device_info = {
        "name": "AI-Powered Diagnostic Device",
        "type": "Diagnostic Software",
        "classification": "Class II",
        "description": "AI-based diagnostic software for medical imaging",
        "intended_use": "Diagnostic assistance for medical professionals"
    }
    
    # Agent 초기화 및 실행
    agent = MedicalDeviceComplianceAgent()
    result = agent.run_compliance_test(device_info)
    
    print(f"Compliance test completed: {result['success']}")
    if result.get("report_path"):
        print(f"Report saved to: {result['report_path']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

