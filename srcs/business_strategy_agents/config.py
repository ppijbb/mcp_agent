"""
Configuration Management for Most Hooking Business Strategy Agent

This module handles all configuration settings, environment variables,
and system parameters for the business strategy agent.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class APIConfig:
    """API 설정"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    rate_limit: int = 100
    timeout: int = 30
    retry_count: int = 3


@dataclass
class NotionConfig:
    """Notion 설정"""
    api_key: str
    database_id: str
    workspace_id: str
    page_template: Dict[str, Any]


@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    collection_interval: int = 300  # seconds
    batch_size: int = 100
    max_concurrent_requests: int = 10
    hooking_score_threshold: float = 0.7
    sentiment_threshold: float = 0.5


@dataclass
class RegionConfig:
    """지역별 설정"""
    enabled_regions: List[str]
    timezone_mapping: Dict[str, str]
    language_codes: Dict[str, str]
    market_hours: Dict[str, Dict[str, str]]


class Config:
    """메인 설정 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._load_config()
        self._setup_api_configs()
        self._setup_monitoring_config()
        self._setup_region_config()
        self._setup_notion_config()
    
    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로"""
        return os.path.join(os.path.dirname(__file__), 'config', 'business_strategy.yaml')
    
    def _load_config(self):
        """설정 파일 로드 - 명시적 에러 처리"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.raw_config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def get_ai_model_config(self) -> Dict[str, Any]:
        """AI 모델 설정 반환"""
        return self.raw_config.get('ai_model', {})
    
    def get_mcp_servers_config(self) -> Dict[str, Any]:
        """MCP 서버 설정 반환"""
        return self.raw_config.get('mcp_servers', {})
    
    def get_keyword_categories_config(self) -> Dict[str, List[str]]:
        """키워드 카테고리 설정 반환"""
        return self.raw_config.get('keyword_categories', {})
    
    def _setup_api_configs(self):
        """API 설정 초기화 - YAML 기반"""
        self.api_configs: Dict[str, APIConfig] = {}
        
        apis = self.raw_config.get('apis', {})
        for category, api_group in apis.items():
            for api_name, api_info in api_group.items():
                full_name = f"{category}_{api_name}"
                self.api_configs[full_name] = APIConfig(
                    name=full_name,
                    base_url=api_info['base_url'],
                    api_key=os.getenv(f"{full_name.upper()}_API_KEY"),
                    secret_key=os.getenv(f"{full_name.upper()}_SECRET_KEY"),
                    rate_limit=api_info.get('rate_limit', 100),
                    timeout=api_info.get('timeout', 30),
                    retry_count=api_info.get('retry_count', 3)
                )
    
    def _setup_monitoring_config(self):
        """모니터링 설정 초기화 - YAML 기반"""
        monitoring = self.raw_config.get('monitoring', {})
        self.monitoring = MonitoringConfig(
            collection_interval=monitoring.get('collection_interval', 300),
            batch_size=monitoring.get('batch_size', 100),
            max_concurrent_requests=monitoring.get('max_concurrent_requests', 10),
            hooking_score_threshold=monitoring.get('hooking_score_threshold', 0.7),
            sentiment_threshold=monitoring.get('sentiment_threshold', 0.5)
        )
        
        self.monitoring_keywords = monitoring.get('keywords', [])
    
    def _setup_region_config(self):
        """지역 설정 초기화 - YAML 기반"""
        regions = self.raw_config.get('regions', {})
        self.region = RegionConfig(
            enabled_regions=regions.get('enabled', []),
            timezone_mapping=regions.get('timezones', {}),
            language_codes=regions.get('languages', {}),
            market_hours=regions.get('market_hours', {})
        )
    
    def _setup_notion_config(self):
        """Notion 설정 초기화 - YAML 기반"""
        notion_data = self.raw_config.get('notion', {})
        
        # 환경변수에서 Notion 설정 로드
        api_key = os.getenv('NOTION_API_KEY', '')
        database_id = os.getenv('NOTION_DATABASE_ID', '')
        workspace_id = os.getenv('NOTION_WORKSPACE_ID', '')
        
        if bool(api_key) and database_id:
            self.notion = NotionConfig(
                api_key=api_key,
                database_id=database_id,
                workspace_id=workspace_id,
                page_template=notion_data.get('database_templates', {})
            )
        else:
            self.notion = None
    
    def get_api_config(self, api_name: str) -> Optional[APIConfig]:
        """API 설정 반환"""
        return self.api_configs.get(api_name)
    
    def get_keywords_by_category(self, category: str) -> List[str]:
        """카테고리별 키워드 반환 - YAML 기반"""
        keyword_categories = self.get_keyword_categories_config()
        return keyword_categories.get(category, [])
    
    def update_config(self, key_path: str, value: Any):
        """설정 업데이트 - YAML 기반"""
        keys = key_path.split('.')
        current = self.raw_config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        # YAML 파일로 저장
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.raw_config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            'name': self.raw_config['system']['name'],
            'version': self.raw_config['system']['version'],
            'environment': self.raw_config['system']['environment'],
            'enabled_regions': self.region.enabled_regions,
            'api_count': len(self.api_configs),
            'keywords_count': len(self.monitoring_keywords)
        }


# 환경변수 헬퍼 함수들
def get_env_or_default(key: str, default: Any = None) -> Any:
    """환경변수 또는 기본값 반환"""
    return os.getenv(key, default)


def require_env(key: str) -> str:
    """필수 환경변수 반환 (없으면 에러)"""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


def setup_environment():
    """환경 설정"""
    # 로깅 설정
    import logging
    
    log_level = get_env_or_default('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/business_strategy_agent.log')
        ]
    )
    
    # 로그 디렉토리 생성
    os.makedirs('logs', exist_ok=True)


# 글로벌 설정 인스턴스
config = Config()


def get_config() -> Config:
    """설정 인스턴스 반환"""
    return config


# 설정 검증
def validate_config() -> List[str]:
    """설정 검증 및 문제점 반환"""
    issues = []
    
    # API 키 검증
    required_apis = ['news_reuters', 'social_twitter', 'trends_google_trends']
    for api_name in required_apis:
        api_config = config.get_api_config(api_name)
        if not api_config or not api_config.api_key:
            issues.append(f"Missing API key for {api_name}")
    
    # Notion 설정 검증
    if not config.notion:
        issues.append("Notion configuration is incomplete")
    
    # 지역 설정 검증
    if not config.region.enabled_regions:
        issues.append("No regions enabled for monitoring")
    
    return issues 

def classify_keywords_by_category(keywords: List[str]) -> Dict[str, List[str]]:
    """카테고리별 키워드 분류 로직 구현 - YAML 기반"""
    
    # 설정에서 카테고리 패턴 로드
    config_instance = get_config()
    category_patterns = config_instance.get_keyword_categories_config()
    
    # 결과 딕셔너리 초기화
    categorized = {category: [] for category in category_patterns.keys()}
    categorized["기타"] = []
    
    # 각 키워드를 카테고리별로 분류
    for keyword in keywords:
        keyword_lower = keyword.lower().strip()
        classified = False
        
        # 각 카테고리의 패턴과 매칭
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if pattern.lower() in keyword_lower or keyword_lower in pattern.lower():
                    categorized[category].append(keyword)
                    classified = True
                    break
            if classified:
                break
        
        # 어떤 카테고리에도 속하지 않는 경우 "기타"로 분류
        if not classified:
            categorized["기타"].append(keyword)
    
    # 빈 카테고리 제거
    return {k: v for k, v in categorized.items() if v} 