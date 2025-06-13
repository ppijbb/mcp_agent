"""
Configuration Management for Most Hooking Business Strategy Agent

This module handles all configuration settings, environment variables,
and system parameters for the business strategy agent.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json


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
        return os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'business_strategy.json')
    
    def _load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.raw_config = json.load(f)
            else:
                self.raw_config = self._get_default_config()
                self._save_config()
        except Exception as e:
            print(f"Failed to load config: {e}")
            self.raw_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "system": {
                "name": "Most Hooking Business Strategy Agent",
                "version": "1.0.0",
                "environment": "development",
                "log_level": "INFO",
                "data_retention_days": 30
            },
            "monitoring": {
                "collection_interval": 300,
                "batch_size": 100,
                "max_concurrent_requests": 10,
                "hooking_score_threshold": 0.7,
                "sentiment_threshold": 0.5,
                "keywords": [
                    "AI", "artificial intelligence", "machine learning", "deep learning",
                    "startup", "unicorn", "IPO", "funding", "investment", "venture capital",
                    "technology", "fintech", "healthtech", "edtech", "climate tech",
                    "market trend", "consumer behavior", "digital transformation",
                    "e-commerce", "social commerce", "creator economy",
                    "Web3", "blockchain", "cryptocurrency", "NFT", "metaverse",
                    "sustainability", "ESG", "green technology", "renewable energy"
                ]
            },
            "regions": {
                "enabled": ["east_asia", "north_america"],
                "timezones": {
                    "east_asia": "Asia/Seoul",
                    "north_america": "America/New_York"
                },
                "languages": {
                    "east_asia": ["ko", "ja", "zh"],
                    "north_america": ["en"]
                },
                "market_hours": {
                    "east_asia": {
                        "open": "09:00",
                        "close": "18:00"
                    },
                    "north_america": {
                        "open": "09:30",
                        "close": "16:00"
                    }
                }
            },
            "apis": {
                "news": {
                    "reuters": {
                        "base_url": "https://api.reuters.com/v1/",
                        "rate_limit": 100
                    },
                    "bloomberg": {
                        "base_url": "https://api.bloomberg.com/v1/",
                        "rate_limit": 50
                    },
                    "naver": {
                        "base_url": "https://openapi.naver.com/v1/search/",
                        "rate_limit": 25000
                    }
                },
                "social": {
                    "twitter": {
                        "base_url": "https://api.twitter.com/2/",
                        "rate_limit": 300
                    },
                    "linkedin": {
                        "base_url": "https://api.linkedin.com/v2/",
                        "rate_limit": 100
                    },
                    "weibo": {
                        "base_url": "https://api.weibo.com/2/",
                        "rate_limit": 150
                    }
                },
                "community": {
                    "reddit": {
                        "base_url": "https://www.reddit.com/api/v1/",
                        "rate_limit": 60
                    },
                    "hackernews": {
                        "base_url": "https://hacker-news.firebaseio.com/v0/",
                        "rate_limit": 1000
                    }
                },
                "trends": {
                    "google_trends": {
                        "base_url": "https://trends.googleapis.com/trends/api/",
                        "rate_limit": 100
                    }
                },
                "business": {
                    "crunchbase": {
                        "base_url": "https://api.crunchbase.com/api/v4/",
                        "rate_limit": 200
                    },
                    "pitchbook": {
                        "base_url": "https://api.pitchbook.com/v1/",
                        "rate_limit": 100
                    }
                }
            },
            "notion": {
                "database_templates": {
                    "daily_insights": {
                        "properties": {
                            "Title": {"title": {}},
                            "Date": {"date": {}},
                            "Region": {"select": {"options": []}},
                            "Category": {"select": {"options": []}},
                            "Hooking Score": {"number": {"format": "percent"}},
                            "Business Opportunity": {"select": {"options": []}},
                            "Key Insights": {"rich_text": {}},
                            "Action Items": {"rich_text": {}},
                            "ROI Prediction": {"rich_text": {}},
                            "Status": {"select": {"options": []}}
                        }
                    }
                }
            },
            "output": {
                "notion": {
                    "enabled": True,
                    "update_frequency": "daily"
                },
                "slack": {
                    "enabled": True,
                    "critical_alerts": True,
                    "daily_summary": True
                },
                "email": {
                    "enabled": False,
                    "recipients": []
                }
            }
        }
    
    def _save_config(self):
        """설정 저장"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.raw_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def _setup_api_configs(self):
        """API 설정 초기화"""
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
        """모니터링 설정 초기화"""
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
        """지역 설정 초기화"""
        regions = self.raw_config.get('regions', {})
        self.region = RegionConfig(
            enabled_regions=regions.get('enabled', []),
            timezone_mapping=regions.get('timezones', {}),
            language_codes=regions.get('languages', {}),
            market_hours=regions.get('market_hours', {})
        )
    
    def _setup_notion_config(self):
        """Notion 설정 초기화"""
        notion_data = self.raw_config.get('notion', {})
        
        # 환경변수에서 Notion 설정 로드
        api_key = os.getenv('NOTION_API_KEY', '')
        database_id = os.getenv('NOTION_DATABASE_ID', '')
        workspace_id = os.getenv('NOTION_WORKSPACE_ID', '')
        
        if api_key and database_id:
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
        """카테고리별 키워드 반환"""
        # TODO: 카테고리별 키워드 분류 로직 구현
        return self.monitoring_keywords
    
    def update_config(self, key_path: str, value: Any):
        """설정 업데이트"""
        keys = key_path.split('.')
        current = self.raw_config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        self._save_config()
    
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