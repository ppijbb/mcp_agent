"""
Physical AI 기기 제어 도구

로봇 청소기, 스마트 장난감, 자동급식기 등 Physical AI 기기 제어
최신 기술: MQTT v5, Home Assistant API, Matter 표준 지원
"""

import logging
import json
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

from ..utils.device_connector import DeviceConnector, ConnectionProtocol
from ..config.petcare_config import PetCareConfig

logger = logging.getLogger(__name__)


class RobotVacuumControlInput(BaseModel):
    """로봇 청소기 제어 입력 스키마"""
    device_id: str = Field(description="로봇 청소기 기기 ID")
    action: str = Field(description="액션 (start, stop, pause, return_home, clean_spot)")
    location: Optional[str] = Field(default=None, description="청소할 위치 (spot cleaning 시 필요)")


class SmartToyControlInput(BaseModel):
    """스마트 장난감 제어 입력 스키마"""
    device_id: str = Field(description="스마트 장난감 기기 ID")
    action: str = Field(description="액션 (activate, deactivate, play_mode)")
    intensity: Optional[int] = Field(default=5, ge=1, le=10, description="놀이 강도 (1-10)")


class AutoFeederControlInput(BaseModel):
    """자동급식기 제어 입력 스키마"""
    device_id: str = Field(description="자동급식기 기기 ID")
    action: str = Field(description="액션 (feed, set_schedule, adjust_amount)")
    amount: Optional[float] = Field(default=None, description="급식량 (g)")
    schedule: Optional[Dict[str, Any]] = Field(default=None, description="급식 스케줄")


class SmartEnvironmentControlInput(BaseModel):
    """스마트 환경 제어 입력 스키마"""
    device_type: str = Field(description="기기 타입 (light, temperature, humidity)")
    action: str = Field(description="액션 (set, adjust)")
    value: Optional[Any] = Field(default=None, description="설정 값")


class PhysicalAITools:
    """
    Physical AI 기기 제어 도구 모음
    
    로봇 청소기, 스마트 장난감, 자동급식기, 스마트 환경 제어
    최신 기술: MQTT v5, Home Assistant REST API, Matter 표준 지원
    """
    
    def __init__(self, data_dir: str = "petcare_data", config: Optional[PetCareConfig] = None):
        """
        PhysicalAITools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
            config: PetCareConfig 인스턴스 (선택)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.devices_file = self.data_dir / "physical_ai_devices.json"
        self.config = config or PetCareConfig()
        self.tools: List[BaseTool] = []
        self.device_connector: Optional[DeviceConnector] = None
        self._initialize_connector()
        self._initialize_tools()
        self._load_devices()
    
    def _initialize_connector(self):
        """기기 연결자 초기화"""
        try:
            # Home Assistant 우선 시도
            if self.config.home_assistant_url and self.config.home_assistant_token:
                self.device_connector = DeviceConnector(
                    protocol=ConnectionProtocol.HOME_ASSISTANT,
                    home_assistant_config={
                        "url": self.config.home_assistant_url,
                        "token": self.config.home_assistant_token,
                    }
                )
                logger.info("Initialized Home Assistant connector")
            # MQTT 대체
            elif self.config.mqtt_broker_host:
                self.device_connector = DeviceConnector(
                    protocol=ConnectionProtocol.MQTT,
                    mqtt_config={
                        "host": self.config.mqtt_broker_host,
                        "port": self.config.mqtt_broker_port,
                        "username": self.config.mqtt_username,
                        "password": self.config.mqtt_password,
                        "use_tls": self.config.mqtt_use_tls,
                        "protocol_version": self.config.mqtt_protocol_version,
                    }
                )
                logger.info("Initialized MQTT connector")
            else:
                logger.warning("No device connector configured. Using mock mode.")
        except Exception as e:
            logger.warning(f"Failed to initialize device connector: {e}. Using mock mode.")
    
    def _load_devices(self):
        """기기 목록 로드"""
        if self.devices_file.exists():
            with open(self.devices_file, 'r', encoding='utf-8') as f:
                self.devices = json.load(f)
        else:
            self.devices = {
                "robot_vacuums": {},
                "smart_toys": {},
                "auto_feeders": {},
                "smart_environment": {},
            }
    
    def _save_devices(self):
        """기기 목록 저장"""
        with open(self.devices_file, 'w', encoding='utf-8') as f:
            json.dump(self.devices, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Physical AI 도구 초기화"""
        self.tools.append(self._create_robot_vacuum_control_tool())
        self.tools.append(self._create_smart_toy_control_tool())
        self.tools.append(self._create_auto_feeder_control_tool())
        self.tools.append(self._create_smart_environment_control_tool())
        logger.info(f"Initialized {len(self.tools)} Physical AI tools")
    
    def _create_robot_vacuum_control_tool(self) -> BaseTool:
        @tool("physical_ai_robot_vacuum_control", args_schema=RobotVacuumControlInput)
        def robot_vacuum_control(device_id: str, action: str, location: Optional[str] = None) -> str:
            """
            로봇 청소기를 제어합니다. (MQTT v5 / Home Assistant API 지원)
            Args:
                device_id: 로봇 청소기 기기 ID
                action: 액션 (start, stop, pause, return_home, clean_spot)
                location: 청소할 위치 (spot cleaning 시 필요)
            Returns:
                제어 결과 메시지 또는 오류 메시지
            """
            logger.info(f"Controlling robot vacuum '{device_id}': action='{action}', location='{location}'")
            
            # 실제 기기 제어 시도
            if self.device_connector:
                try:
                    params = {}
                    if location:
                        params["location"] = location
                    
                    # 동기 함수에서 비동기 호출
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 이미 실행 중인 루프가 있으면 새 태스크 생성
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.device_connector.control_device(
                                    device_id, "robot_vacuum", action, params
                                )
                            )
                            result = future.result(timeout=5)
                    else:
                        result = loop.run_until_complete(
                            self.device_connector.control_device(
                                device_id, "robot_vacuum", action, params
                            )
                        )
                    
                    if result.get("success"):
                        return f"Robot vacuum '{device_id}' executed action '{action}' at location '{location or 'entire area'}'. Status: Success."
                    else:
                        error = result.get("error", "Unknown error")
                        logger.warning(f"Device control failed: {error}")
                        return f"Robot vacuum control failed: {error}"
                except Exception as e:
                    logger.error(f"Device control error: {e}")
                    return f"Robot vacuum control error: {str(e)}"
            
            # Mock 모드 (연결자가 없을 때)
            return f"Robot vacuum '{device_id}' executed action '{action}' at location '{location or 'entire area'}'. Status: Success (Mock Mode)."
        return robot_vacuum_control
    
    def _create_smart_toy_control_tool(self) -> BaseTool:
        @tool("physical_ai_smart_toy_control", args_schema=SmartToyControlInput)
        def smart_toy_control(device_id: str, action: str, intensity: Optional[int] = 5) -> str:
            """
            스마트 장난감을 제어합니다. (MQTT v5 / Home Assistant API 지원)
            Args:
                device_id: 스마트 장난감 기기 ID
                action: 액션 (activate, deactivate, play_mode)
                intensity: 놀이 강도 (1-10)
            Returns:
                제어 결과 메시지 또는 오류 메시지
            """
            logger.info(f"Controlling smart toy '{device_id}': action='{action}', intensity={intensity}")
            
            if self.device_connector:
                try:
                    params = {"intensity": intensity}
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.device_connector.control_device(
                                    device_id, "smart_toy", action, params
                                )
                            )
                            result = future.result(timeout=5)
                    else:
                        result = loop.run_until_complete(
                            self.device_connector.control_device(
                                device_id, "smart_toy", action, params
                            )
                        )
                    
                    if result.get("success"):
                        return f"Smart toy '{device_id}' executed action '{action}' with intensity {intensity}. Status: Success."
                    else:
                        return f"Smart toy control failed: {result.get('error', 'Unknown error')}"
                except Exception as e:
                    logger.error(f"Device control error: {e}")
                    return f"Smart toy control error: {str(e)}"
            
            return f"Smart toy '{device_id}' executed action '{action}' with intensity {intensity}. Status: Success (Mock Mode)."
        return smart_toy_control
    
    def _create_auto_feeder_control_tool(self) -> BaseTool:
        @tool("physical_ai_auto_feeder_control", args_schema=AutoFeederControlInput)
        def auto_feeder_control(
            device_id: str,
            action: str,
            amount: Optional[float] = None,
            schedule: Optional[Dict[str, Any]] = None
        ) -> str:
            """
            자동급식기를 제어합니다. (MQTT v5 / Home Assistant API 지원)
            Args:
                device_id: 자동급식기 기기 ID
                action: 액션 (feed, set_schedule, adjust_amount)
                amount: 급식량 (g)
                schedule: 급식 스케줄
            Returns:
                제어 결과 메시지 또는 오류 메시지
            """
            logger.info(f"Controlling auto feeder '{device_id}': action='{action}', amount={amount}, schedule={schedule}")
            
            if self.device_connector:
                try:
                    params = {}
                    if amount is not None:
                        params["amount"] = amount
                    if schedule:
                        params["schedule"] = schedule
                    
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.device_connector.control_device(
                                    device_id, "auto_feeder", action, params
                                )
                            )
                            result = future.result(timeout=5)
                    else:
                        result = loop.run_until_complete(
                            self.device_connector.control_device(
                                device_id, "auto_feeder", action, params
                            )
                        )
                    
                    if result.get("success"):
                        return f"Auto feeder '{device_id}' executed action '{action}' with amount {amount}g. Status: Success."
                    else:
                        return f"Auto feeder control failed: {result.get('error', 'Unknown error')}"
                except Exception as e:
                    logger.error(f"Device control error: {e}")
                    return f"Auto feeder control error: {str(e)}"
            
            return f"Auto feeder '{device_id}' executed action '{action}' with amount {amount}g. Status: Success (Mock Mode)."
        return auto_feeder_control
    
    def _create_smart_environment_control_tool(self) -> BaseTool:
        @tool("physical_ai_smart_environment_control", args_schema=SmartEnvironmentControlInput)
        def smart_environment_control(device_type: str, action: str, value: Optional[Any] = None) -> str:
            """
            스마트 환경 기기(조명, 온도, 습도)를 제어합니다. (MQTT v5 / Home Assistant API 지원)
            Args:
                device_type: 기기 타입 (light, temperature, humidity)
                action: 액션 (set, adjust)
                value: 설정 값
            Returns:
                제어 결과 메시지 또는 오류 메시지
            """
            logger.info(f"Controlling smart environment '{device_type}': action='{action}', value={value}")
            
            if self.device_connector:
                try:
                    params = {"value": value, "device_type": device_type}
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.device_connector.control_device(
                                    f"smart_env_{device_type}", "smart_environment", action, params
                                )
                            )
                            result = future.result(timeout=5)
                    else:
                        result = loop.run_until_complete(
                            self.device_connector.control_device(
                                f"smart_env_{device_type}", "smart_environment", action, params
                            )
                        )
                    
                    if result.get("success"):
                        return f"Smart environment '{device_type}' executed action '{action}' with value {value}. Status: Success."
                    else:
                        return f"Smart environment control failed: {result.get('error', 'Unknown error')}"
                except Exception as e:
                    logger.error(f"Device control error: {e}")
                    return f"Smart environment control error: {str(e)}"
            
            return f"Smart environment '{device_type}' executed action '{action}' with value {value}. Status: Success (Mock Mode)."
        return smart_environment_control
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Physical AI 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Physical AI 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

