#!/usr/bin/env python3
"""
Configuration Loader for SEO Doctor
설정 파일 로더 - 하드코딩 제거
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SEOConfigLoader:
    """SEO Doctor 설정 파일 로더"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config = None
        self._load_config()

    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로 반환"""
        current_dir = Path(__file__).parent
        return str(current_dir / "config" / "seo_doctor.yaml")

    def _load_config(self) -> None:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"SEO Doctor 설정 파일 로드 완료: {self.config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"설정 파일 파싱 오류: {e}")

    def get_lighthouse_config(self, strategy: str = "mobile") -> Dict[str, Any]:
        """Lighthouse 설정 반환"""
        lighthouse_config = self._config.get('lighthouse', {})
        return lighthouse_config.get(strategy, lighthouse_config.get('mobile', {}))

    def get_thresholds(self) -> Dict[str, int]:
        """임계값 설정 반환"""
        return self._config.get('thresholds', {
            'performance': 80,
            'accessibility': 70,
            'best_practices': 70,
            'seo': 80
        })

    def get_report_config(self) -> Dict[str, Any]:
        """보고서 설정 반환"""
        return self._config.get('report', {})

    def get_mcp_servers_config(self) -> Dict[str, Any]:
        """MCP 서버 설정 반환"""
        return self._config.get('mcp_servers', {})

    def get_mcp_server_url(self, server_name: str) -> str:
        """특정 MCP 서버 URL 반환"""
        servers = self.get_mcp_servers_config()
        server = servers.get(server_name, {})
        url = server.get('url')
        if not url:
            raise ValueError(f"MCP 서버 '{server_name}'의 URL이 설정되지 않았습니다")
        return url

    def get_ai_config(self) -> Dict[str, Any]:
        """AI 설정 반환"""
        return self._config.get('ai', {})

    def get_ai_model_config(self) -> str:
        """AI 모델명 반환"""
        return self.get_ai_config().get('model', 'gemini-2.0-flash-exp')

    def get_ai_api_key(self) -> str:
        """AI API 키 반환"""
        api_key_env = self.get_ai_config().get('api_key_env', 'GOOGLE_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"환경 변수 {api_key_env}가 설정되지 않았습니다")
        return api_key

    def get_analysis_prompts(self) -> Dict[str, str]:
        """AI 분석 프롬프트 반환"""
        return self.get_ai_config().get('analysis_prompts', {})

    def get_emergency_levels(self) -> Dict[str, Any]:
        """응급 레벨 분류 반환"""
        return self._config.get('emergency_levels', {})

    def get_recovery_time_config(self) -> Dict[str, Any]:
        """회복 시간 계산 설정 반환"""
        return self._config.get('recovery_time', {})

    def get_logging_config(self) -> Dict[str, str]:
        """로깅 설정 반환"""
        return self._config.get('logging', {})

    def get_metrics_thresholds(self) -> Dict[str, Dict[str, float]]:
        """성능 메트릭 임계값 반환"""
        return self._config.get('metrics', {})

    def determine_emergency_level(self, score: float) -> Dict[str, str]:
        """점수를 기반으로 응급 레벨 결정"""
        emergency_levels = self.get_emergency_levels()

        for level_name in ['excellent', 'safe', 'caution', 'danger', 'critical']:
            level_config = emergency_levels.get(level_name, {})
            min_score = level_config.get('min_score', 0)

            if score >= min_score:
                return {
                    'level': level_name,
                    'emoji': level_config.get('emoji', ''),
                    'label': level_config.get('label', level_name)
                }

        # 기본값
        return {
            'level': 'critical',
            'emoji': '🚨',
            'label': '응급실'
        }

    def calculate_recovery_time(self, score: float, issue_count: int) -> int:
        """회복 시간 계산"""
        recovery_config = self.get_recovery_time_config()
        base_times = recovery_config.get('base', {})
        multipliers = recovery_config.get('issue_multiplier', {})

        # 응급 레벨 결정
        emergency_level = self.determine_emergency_level(score)
        level_name = emergency_level['level']

        # 기본 시간 + (문제 개수 * 배수)
        base_time = base_times.get(level_name, 30)
        multiplier = multipliers.get(level_name, 5)

        return base_time + (issue_count * multiplier)


# 전역 설정 인스턴스
seo_config = SEOConfigLoader()
