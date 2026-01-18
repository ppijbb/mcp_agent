"""
Physical AI 기기 연결 유틸리티

MQTT v5, Home Assistant API, Matter 표준을 통한 실제 기기 제어
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

logger = logging.getLogger(__name__)


class ConnectionProtocol(Enum):
    """연결 프로토콜 타입"""
    MQTT = "mqtt"
    HOME_ASSISTANT = "home_assistant"
    MATTER = "matter"
    DIRECT_API = "direct_api"


class DeviceConnector:
    """
    Physical AI 기기 연결 관리자
    
    MQTT v5, Home Assistant REST API, Matter 표준을 통한 통합 제어
    """
    
    def __init__(
        self,
        protocol: ConnectionProtocol = ConnectionProtocol.MQTT,
        mqtt_config: Optional[Dict[str, Any]] = None,
        home_assistant_config: Optional[Dict[str, Any]] = None,
    ):
        """
        DeviceConnector 초기화
        
        Args:
            protocol: 사용할 프로토콜
            mqtt_config: MQTT 설정 (host, port, username, password, use_tls, protocol_version)
            home_assistant_config: Home Assistant 설정 (url, token)
        """
        self.protocol = protocol
        self.mqtt_config = mqtt_config or {}
        self.home_assistant_config = home_assistant_config or {}
        self.mqtt_client: Optional[mqtt.Client] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        
    async def connect(self) -> bool:
        """기기 연결"""
        try:
            if self.protocol == ConnectionProtocol.MQTT:
                return await self._connect_mqtt()
            elif self.protocol == ConnectionProtocol.HOME_ASSISTANT:
                return await self._connect_home_assistant()
            elif self.protocol == ConnectionProtocol.MATTER:
                logger.warning("Matter protocol not yet fully implemented")
                return False
            else:
                logger.error(f"Unsupported protocol: {self.protocol}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def _connect_mqtt(self) -> bool:
        """MQTT 연결"""
        if not MQTT_AVAILABLE:
            logger.error("paho-mqtt not installed. Install with: pip install paho-mqtt")
            return False
        
        try:
            protocol_version = self.mqtt_config.get("protocol_version", 5)
            if protocol_version == 5:
                self.mqtt_client = mqtt.Client(
                    client_id=f"petcare_ai_{id(self)}",
                    protocol=mqtt.MQTTv5
                )
            else:
                self.mqtt_client = mqtt.Client(
                    client_id=f"petcare_ai_{id(self)}",
                    protocol=mqtt.MQTTv311
                )
            
            # 인증 설정
            username = self.mqtt_config.get("username")
            password = self.mqtt_config.get("password")
            if username and password:
                self.mqtt_client.username_pw_set(username, password)
            
            # TLS 설정
            if self.mqtt_config.get("use_tls", False):
                self.mqtt_client.tls_set()
            
            # 콜백 설정
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.on_message = self._on_mqtt_message
            
            # 연결
            host = self.mqtt_config.get("host", "localhost")
            port = self.mqtt_config.get("port", 1883)
            self.mqtt_client.connect_async(host, port, keepalive=60)
            self.mqtt_client.loop_start()
            
            # 연결 대기
            await asyncio.sleep(1)
            return self._connected
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False
    
    async def _connect_home_assistant(self) -> bool:
        """Home Assistant 연결"""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
            return False
        
        try:
            url = self.home_assistant_config.get("url")
            token = self.home_assistant_config.get("token")
            
            if not url or not token:
                logger.error("Home Assistant URL and token required")
                return False
            
            self.http_session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }
            )
            
            # 연결 테스트
            async with self.http_session.get(f"{url}/api/config") as response:
                if response.status == 200:
                    self._connected = True
                    logger.info("Connected to Home Assistant")
                    return True
                else:
                    logger.error(f"Home Assistant connection failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Home Assistant connection failed: {e}")
            return False
    
    def _on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT 연결 콜백"""
        if rc == 0:
            self._connected = True
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc, properties=None):
        """MQTT 연결 해제 콜백"""
        self._connected = False
        logger.info("Disconnected from MQTT broker")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT 메시지 수신 콜백"""
        logger.debug(f"Received MQTT message: {msg.topic} = {msg.payload.decode()}")
    
    async def control_device(
        self,
        device_id: str,
        device_type: str,
        action: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        기기 제어
        
        Args:
            device_id: 기기 ID
            device_type: 기기 타입 (robot_vacuum, smart_toy, auto_feeder, smart_environment)
            action: 액션
            params: 추가 파라미터
        
        Returns:
            제어 결과
        """
        if not self._connected:
            await self.connect()
        
        try:
            if self.protocol == ConnectionProtocol.MQTT:
                return await self._control_via_mqtt(device_id, device_type, action, params)
            elif self.protocol == ConnectionProtocol.HOME_ASSISTANT:
                return await self._control_via_home_assistant(device_id, device_type, action, params)
            else:
                return {"success": False, "error": f"Unsupported protocol: {self.protocol}"}
        except Exception as e:
            logger.error(f"Device control failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _control_via_mqtt(
        self,
        device_id: str,
        device_type: str,
        action: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """MQTT를 통한 기기 제어"""
        if not self.mqtt_client or not self._connected:
            return {"success": False, "error": "MQTT not connected"}
        
        try:
            topic = f"petcare/{device_type}/{device_id}/command"
            payload = {
                "action": action,
                "params": params or {},
            }
            
            result = self.mqtt_client.publish(
                topic,
                json.dumps(payload),
                qos=1,
                retain=False
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"MQTT command sent: {topic}")
                return {"success": True, "device_id": device_id, "action": action}
            else:
                return {"success": False, "error": f"MQTT publish failed: {result.rc}"}
                
        except Exception as e:
            logger.error(f"MQTT control error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _control_via_home_assistant(
        self,
        device_id: str,
        device_type: str,
        action: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Home Assistant API를 통한 기기 제어"""
        if not self.http_session or not self._connected:
            return {"success": False, "error": "Home Assistant not connected"}
        
        try:
            url = self.home_assistant_config.get("url")
            
            # Home Assistant 서비스 호출
            service_data = {
                "entity_id": device_id,
                **{k: v for k, v in (params or {}).items() if v is not None}
            }
            
            # 서비스 매핑
            service_map = {
                "robot_vacuum": "vacuum",
                "smart_toy": "switch",
                "auto_feeder": "switch",
                "smart_environment": "homeassistant",
            }
            
            service_domain = service_map.get(device_type, "homeassistant")
            
            # 액션을 서비스로 변환
            service = self._map_action_to_service(device_type, action)
            
            async with self.http_session.post(
                f"{url}/api/services/{service_domain}/{service}",
                json=service_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Home Assistant service called: {service_domain}.{service}")
                    return {"success": True, "device_id": device_id, "action": action, "result": result}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
                    
        except Exception as e:
            logger.error(f"Home Assistant control error: {e}")
            return {"success": False, "error": str(e)}
    
    def _map_action_to_service(self, device_type: str, action: str) -> str:
        """액션을 Home Assistant 서비스로 매핑"""
        action_map = {
            "robot_vacuum": {
                "start": "start",
                "stop": "stop",
                "pause": "pause",
                "return_home": "return_to_base",
                "clean_spot": "clean_spot",
            },
            "smart_toy": {
                "activate": "turn_on",
                "deactivate": "turn_off",
                "play_mode": "turn_on",
            },
            "auto_feeder": {
                "feed": "turn_on",
                "set_schedule": "turn_on",
                "adjust_amount": "turn_on",
            },
            "smart_environment": {
                "set": "turn_on",
                "adjust": "turn_on",
            },
        }
        
        return action_map.get(device_type, {}).get(action, "turn_on")
    
    async def disconnect(self):
        """연결 해제"""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        if self.http_session:
            await self.http_session.close()
        
        self._connected = False
        logger.info("Disconnected from device connector")
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._connected
