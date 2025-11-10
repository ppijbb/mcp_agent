"""
IoT 기기 제어 도구

IoT 기기 제어, 상태 조회, 설정 변경, 그룹 관리
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DeviceControlInput(BaseModel):
    """기기 제어 입력 스키마"""
    device_id: str = Field(description="기기 ID")
    action: str = Field(description="제어 액션 (on, off, set_temperature, etc.)")
    value: Optional[Any] = Field(default=None, description="액션 값")


class DeviceStatusInput(BaseModel):
    """기기 상태 조회 입력 스키마"""
    device_id: Optional[str] = Field(default=None, description="기기 ID (없으면 모든 기기)")


class DeviceConfigInput(BaseModel):
    """기기 설정 변경 입력 스키마"""
    device_id: str = Field(description="기기 ID")
    config: Dict[str, Any] = Field(description="설정 딕셔너리")


class DeviceGroupInput(BaseModel):
    """기기 그룹 관리 입력 스키마"""
    group_name: str = Field(description="그룹 이름")
    device_ids: List[str] = Field(description="기기 ID 목록")
    action: str = Field(description="액션 (create, add, remove, delete)")


class IoTTools:
    """
    IoT 기기 제어 도구 모음
    
    IoT 기기 제어, 상태 조회, 설정 변경, 그룹 관리
    """
    
    def __init__(self, data_dir: str = "home_data"):
        """
        IoTTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.devices_file = self.data_dir / "devices.json"
        self.groups_file = self.data_dir / "groups.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_devices()
        self._load_groups()
    
    def _load_devices(self):
        """기기 목록 로드"""
        if self.devices_file.exists():
            with open(self.devices_file, 'r', encoding='utf-8') as f:
                self.devices = json.load(f)
        else:
            self.devices = {}
    
    def _load_groups(self):
        """그룹 목록 로드"""
        if self.groups_file.exists():
            with open(self.groups_file, 'r', encoding='utf-8') as f:
                self.groups = json.load(f)
        else:
            self.groups = {}
    
    def _save_devices(self):
        """기기 목록 저장"""
        with open(self.devices_file, 'w', encoding='utf-8') as f:
            json.dump(self.devices, f, indent=2, ensure_ascii=False)
    
    def _save_groups(self):
        """그룹 목록 저장"""
        with open(self.groups_file, 'w', encoding='utf-8') as f:
            json.dump(self.groups, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """IoT 도구 초기화"""
        self.tools.append(self._create_device_control_tool())
        self.tools.append(self._create_device_status_tool())
        self.tools.append(self._create_device_config_tool())
        self.tools.append(self._create_device_group_tool())
        
        logger.info(f"Initialized {len(self.tools)} IoT tools")
    
    def _create_device_control_tool(self) -> BaseTool:
        """기기 제어 도구 생성"""
        
        @tool("control_device", args_schema=DeviceControlInput)
        def control_device(device_id: str, action: str, value: Optional[Any] = None) -> str:
            """
            IoT 기기 제어
            
            Args:
                device_id: 기기 ID
                action: 제어 액션 (on, off, set_temperature, set_brightness, etc.)
                value: 액션 값 (선택)
            
            Returns:
                제어 결과
            """
            try:
                logger.info(f"Controlling device {device_id}: {action} = {value}")
                
                if device_id not in self.devices:
                    return f"Error: Device {device_id} not found"
                
                device = self.devices[device_id]
                device["status"] = action
                if value is not None:
                    device["value"] = value
                
                self._save_devices()
                
                return f"Successfully controlled device {device_id}: {action} = {value}"
            
            except Exception as e:
                error_msg = f"Error controlling device {device_id}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return control_device
    
    def _create_device_status_tool(self) -> BaseTool:
        """기기 상태 조회 도구 생성"""
        
        @tool("get_device_status", args_schema=DeviceStatusInput)
        def get_device_status(device_id: Optional[str] = None) -> str:
            """
            IoT 기기 상태 조회
            
            Args:
                device_id: 기기 ID (없으면 모든 기기)
            
            Returns:
                기기 상태 정보
            """
            try:
                logger.info(f"Getting device status: {device_id or 'all devices'}")
                
                if device_id:
                    if device_id not in self.devices:
                        return f"Error: Device {device_id} not found"
                    return json.dumps(self.devices[device_id], indent=2, ensure_ascii=False)
                else:
                    return json.dumps(self.devices, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error_msg = f"Error getting device status: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return get_device_status
    
    def _create_device_config_tool(self) -> BaseTool:
        """기기 설정 변경 도구 생성"""
        
        @tool("configure_device", args_schema=DeviceConfigInput)
        def configure_device(device_id: str, config: Dict[str, Any]) -> str:
            """
            IoT 기기 설정 변경
            
            Args:
                device_id: 기기 ID
                config: 설정 딕셔너리
            
            Returns:
                설정 변경 결과
            """
            try:
                logger.info(f"Configuring device {device_id}: {config}")
                
                if device_id not in self.devices:
                    return f"Error: Device {device_id} not found"
                
                self.devices[device_id].update(config)
                self._save_devices()
                
                return f"Successfully configured device {device_id}"
            
            except Exception as e:
                error_msg = f"Error configuring device {device_id}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return configure_device
    
    def _create_device_group_tool(self) -> BaseTool:
        """기기 그룹 관리 도구 생성"""
        
        @tool("manage_device_group", args_schema=DeviceGroupInput)
        def manage_device_group(group_name: str, device_ids: List[str], action: str) -> str:
            """
            IoT 기기 그룹 관리
            
            Args:
                group_name: 그룹 이름
                device_ids: 기기 ID 목록
                action: 액션 (create, add, remove, delete)
            
            Returns:
                그룹 관리 결과
            """
            try:
                logger.info(f"Managing device group {group_name}: {action}")
                
                if action == "create":
                    self.groups[group_name] = device_ids
                elif action == "add":
                    if group_name not in self.groups:
                        self.groups[group_name] = []
                    self.groups[group_name].extend(device_ids)
                elif action == "remove":
                    if group_name in self.groups:
                        self.groups[group_name] = [d for d in self.groups[group_name] if d not in device_ids]
                elif action == "delete":
                    if group_name in self.groups:
                        del self.groups[group_name]
                else:
                    return f"Error: Unknown action {action}"
                
                self._save_groups()
                
                return f"Successfully {action}d group {group_name}"
            
            except Exception as e:
                error_msg = f"Error managing device group {group_name}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return manage_device_group
    
    def get_tools(self) -> List[BaseTool]:
        """모든 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 도구 찾기"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

