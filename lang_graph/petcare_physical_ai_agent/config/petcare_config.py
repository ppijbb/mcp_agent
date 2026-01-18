"""
PetCare Physical AI Agent 설정
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional


class PetCareConfig(BaseModel):
    """
    PetCare Physical AI Agent 설정 스키마
    """
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    output_dir: str = Field(default="petcare_reports", description="리포트 및 결과물 저장 디렉토리")
    data_dir: str = Field(default="petcare_data", description="반려동물 데이터 저장 디렉토리")
    
    # LLM 설정
    default_llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="기본 LLM 온도")
    default_llm_max_tokens: int = Field(default=4000, gt=0, description="기본 LLM 최대 토큰")
    
    # Physical AI 기기 설정
    robot_vacuum_enabled: bool = Field(default=True, description="로봇 청소기 연동 활성화")
    smart_toy_enabled: bool = Field(default=True, description="스마트 장난감 연동 활성화")
    auto_feeder_enabled: bool = Field(default=True, description="자동급식기 연동 활성화")
    smart_environment_enabled: bool = Field(default=True, description="스마트 환경 제어 활성화")
    
    # Physical AI 통신 프로토콜 설정
    mqtt_broker_host: Optional[str] = Field(default=None, description="MQTT 브로커 호스트 (예: localhost, mqtt.example.com)")
    mqtt_broker_port: int = Field(default=1883, description="MQTT 브로커 포트")
    mqtt_use_tls: bool = Field(default=False, description="MQTT TLS 사용 여부")
    mqtt_username: Optional[str] = Field(default=None, description="MQTT 사용자명")
    mqtt_password: Optional[str] = Field(default=None, description="MQTT 비밀번호")
    mqtt_protocol_version: int = Field(default=5, ge=3, le=5, description="MQTT 프로토콜 버전 (3 또는 5)")
    
    # Home Assistant 설정
    home_assistant_url: Optional[str] = Field(default=None, description="Home Assistant URL (예: http://homeassistant.local:8123)")
    home_assistant_token: Optional[str] = Field(default=None, description="Home Assistant Long-Lived Access Token")
    
    # Matter 설정 (향후 지원)
    matter_enabled: bool = Field(default=False, description="Matter 표준 지원 활성화")
    matter_controller_url: Optional[str] = Field(default=None, description="Matter Controller URL")
    
    # 에이전트별 프롬프트 템플릿
    profile_analyzer_prompt_template: str = Field(
        default="You are an expert pet profile analyst. Analyze pet information and create comprehensive profiles for personalized care.",
        description="Profile Analyzer Agent의 기본 프롬프트 템플릿"
    )
    health_monitor_prompt_template: str = Field(
        default="You are an expert pet health monitor. Track health metrics, detect anomalies, and provide health insights.",
        description="Health Monitor Agent의 기본 프롬프트 템플릿"
    )
    physical_ai_controller_prompt_template: str = Field(
        default="You are an expert Physical AI device controller. Control smart devices based on pet needs and behavior.",
        description="Physical AI Controller Agent의 기본 프롬프트 템플릿"
    )
    behavior_analyzer_prompt_template: str = Field(
        default="You are an expert pet behavior analyst. Analyze behavior patterns and detect anomalies.",
        description="Behavior Analyzer Agent의 기본 프롬프트 템플릿"
    )
    care_planner_prompt_template: str = Field(
        default="You are an expert pet care planner. Create personalized care plans that optimize pet health and well-being.",
        description="Care Planner Agent의 기본 프롬프트 템플릿"
    )
    pet_assistant_prompt_template: str = Field(
        default="You are a comprehensive Pet Care Assistant. Orchestrate various tools and devices to provide the best pet care experience.",
        description="Pet Assistant Agent의 기본 프롬프트 템플릿"
    )
    
    # 워크플로우 설정
    max_retries_per_step: int = Field(default=3, ge=0, description="각 워크플로우 단계별 최대 재시도 횟수")
    
    # 리포트 설정
    report_timestamp_format: str = Field(default="%Y%m%d_%H%M%S", description="리포트 파일명에 사용될 타임스탬프 형식")
    
    @classmethod
    def from_env(cls) -> "PetCareConfig":
        """환경 변수에서 설정 로드"""
        import os
        return cls(
            output_dir=os.getenv("PETCARE_OUTPUT_DIR", "petcare_reports"),
            data_dir=os.getenv("PETCARE_DATA_DIR", "petcare_data"),
            mqtt_broker_host=os.getenv("MQTT_BROKER_HOST"),
            mqtt_broker_port=int(os.getenv("MQTT_BROKER_PORT", "1883")),
            mqtt_use_tls=os.getenv("MQTT_USE_TLS", "false").lower() == "true",
            mqtt_username=os.getenv("MQTT_USERNAME"),
            mqtt_password=os.getenv("MQTT_PASSWORD"),
            mqtt_protocol_version=int(os.getenv("MQTT_PROTOCOL_VERSION", "5")),
            home_assistant_url=os.getenv("HOME_ASSISTANT_URL"),
            home_assistant_token=os.getenv("HOME_ASSISTANT_TOKEN"),
            matter_enabled=os.getenv("MATTER_ENABLED", "false").lower() == "true",
            matter_controller_url=os.getenv("MATTER_CONTROLLER_URL"),
        )

