#!/usr/bin/env python3
"""
Configuration Loader for Travel Scout
설정 파일 로더 - 하드코딩 제거
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """설정 파일 로더"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config = None
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로 반환"""
        current_dir = Path(__file__).parent
        return str(current_dir / "config" / "travel_scout.yaml")
    
    def _load_config(self) -> None:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"설정 파일 로드 완료: {self.config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"설정 파일 파싱 오류: {e}")
    
    def get_browser_config(self) -> Dict[str, Any]:
        """브라우저 설정 반환"""
        return self._config.get('browser', {})
    
    def get_mcp_server_config(self) -> Dict[str, Any]:
        """MCP 서버 설정 반환"""
        return self._config.get('mcp_server', {})
    
    def get_cities_config(self) -> Dict[str, List[str]]:
        """도시 설정 반환"""
        return self._config.get('cities', {})
    
    def get_scraping_config(self) -> Dict[str, Any]:
        """스크레이핑 설정 반환"""
        return self._config.get('scraping', {})
    
    def get_ai_config(self) -> Dict[str, Any]:
        """AI 설정 반환"""
        return self._config.get('ai', {})
    
    def get_user_defaults(self) -> Dict[str, str]:
        """사용자 기본 설정 반환"""
        return self._config.get('user_defaults', {})
    
    def get_logging_config(self) -> Dict[str, str]:
        """로깅 설정 반환"""
        return self._config.get('logging', {})
    
    def get_destination_options(self) -> List[str]:
        """목적지 옵션 반환"""
        return self.get_cities_config().get('destinations', [])
    
    def get_origin_options(self) -> List[str]:
        """출발지 옵션 반환"""
        return self.get_cities_config().get('origins', [])
    
    def get_booking_com_config(self) -> Dict[str, Any]:
        """Booking.com 스크레이핑 설정 반환"""
        return self.get_scraping_config().get('booking_com', {})
    
    def get_google_flights_config(self) -> Dict[str, Any]:
        """Google Flights 스크레이핑 설정 반환"""
        return self.get_scraping_config().get('google_flights', {})
    
    def get_ai_model_config(self) -> str:
        """AI 모델명 반환"""
        return self.get_ai_config().get('model', 'gemini-2.0-flash-exp')
    
    def get_ai_api_key(self) -> str:
        """AI API 키 반환"""
        api_key_env = self.get_ai_config().get('api_key_env', 'GOOGLE_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"환경 변수 {api_key_env}가 설정되지 않았습니다.")
        return api_key
    
    def get_analysis_prompts(self) -> Dict[str, str]:
        """AI 분석 프롬프트 반환"""
        return self.get_ai_config().get('analysis_prompts', {})
    
    def get_user_location(self) -> Dict[str, str]:
        """사용자 위치 정보 반환"""
        defaults = self.get_user_defaults()
        return {
            "origin": defaults.get("origin", "Seoul (서울)"),
            "country": defaults.get("country", "South Korea"),
            "timezone": defaults.get("timezone", "Asia/Seoul"),
            "currency": defaults.get("currency", "KRW"),
            "language": defaults.get("language", "ko"),
            "detected_method": "config_file",
            "available_origins": self.get_origin_options(),
            "available_destinations": self.get_destination_options()
        }


# 전역 설정 인스턴스
config = ConfigLoader()
