"""
스마트 홈 매니저 Agent 메인 진입점
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from .llm.model_manager import ModelManager, ModelProvider
from .llm.fallback_handler import FallbackHandler
from .chains.home_chain import HomeChain
from .config.home_config import HomeConfig
from .utils.validators import validate_home_id

logger = logging.getLogger(__name__)


class SmartHomeManager:
    """
    스마트 홈 매니저 Agent
    
    IoT 기기 통합 관리, 에너지 최적화, 보안 모니터링, 유지보수 알림,
    자동화 시나리오 생성을 제공하는 Multi-Agent 시스템
    """
    
    def __init__(
        self,
        config: Optional[HomeConfig] = None,
        output_dir: Optional[str] = None,
        data_dir: Optional[str] = None
    ):
        """
        SmartHomeManager 초기화
        
        Args:
            config: 설정 객체 (선택, 없으면 환경 변수에서 로드)
            output_dir: 출력 디렉토리 (선택)
            data_dir: 데이터 저장 디렉토리 (선택)
        """
        self.config = config or HomeConfig.from_env()
        self.output_dir = output_dir or self.config.output_dir
        self.data_dir = data_dir or self.config.data_dir
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # LLM Manager 및 Fallback Handler 초기화
        self.model_manager = ModelManager(
            budget_limit=self.config.llm.budget_limit
        )
        self.fallback_handler = FallbackHandler(self.model_manager)
        
        # Home Chain 초기화
        self.home_chain = HomeChain(
            model_manager=self.model_manager,
            fallback_handler=self.fallback_handler,
            output_dir=self.output_dir,
            data_dir=self.data_dir
        )
        
        logger.info("Smart Home Manager Agent initialized")
    
    def manage(
        self,
        user_id: str,
        home_id: str,
        devices: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        스마트 홈 관리 실행
        
        Args:
            user_id: 사용자 ID
            home_id: 홈 ID
            devices: 기기 목록 (선택)
        
        Returns:
            관리 결과
        """
        try:
            # 입력 검증
            if not validate_home_id(home_id):
                return {
                    "success": False,
                    "error": "Invalid home_id",
                    "user_id": user_id,
                    "home_id": home_id
                }
            
            logger.info(f"Starting smart home management for user: {user_id}, home: {home_id}")
            
            # 워크플로우 실행
            final_state = self.home_chain.run(user_id, home_id, devices)
            
            # 결과 반환
            return {
                "success": len(final_state.get("errors", [])) == 0,
                "user_id": user_id,
                "home_id": home_id,
                "device_status": final_state.get("device_status", {}),
                "energy_optimization": final_state.get("energy_optimization", {}),
                "security_status": final_state.get("security_status", {}),
                "security_alerts": final_state.get("security_alerts", []),
                "maintenance_schedule": final_state.get("maintenance_schedule", []),
                "automation_scenarios": final_state.get("automation_scenarios", []),
                "errors": final_state.get("errors", []),
                "warnings": final_state.get("warnings", []),
                "timestamp": final_state.get("timestamp")
            }
        
        except Exception as e:
            logger.error(f"Smart home management failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id,
                "home_id": home_id
            }


async def main():
    """메인 실행 함수"""
    # 예제 홈 정보
    user_id = "user_001"
    home_id = "home_001"
    devices = [
        {"device_id": "light_1", "type": "lighting", "name": "Living Room Light"},
        {"device_id": "thermostat_1", "type": "heating", "name": "Main Thermostat"},
        {"device_id": "camera_1", "type": "security", "name": "Front Door Camera"}
    ]
    
    # Agent 초기화 및 실행
    agent = SmartHomeManager()
    result = agent.manage(user_id, home_id, devices)
    
    print(f"Smart home management completed: {result['success']}")
    if result.get("automation_scenarios"):
        print(f"Created {len(result['automation_scenarios'])} automation scenarios")
    if result.get("maintenance_schedule"):
        print(f"Found {len(result['maintenance_schedule'])} maintenance alerts")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

