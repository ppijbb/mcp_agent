"""
Configuration settings for Urban Hive system.

This module centralizes all configuration parameters that were previously hardcoded,
making the system more maintainable and flexible.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API endpoint configuration."""
    public_data_api_key: str = os.getenv("PUBLIC_DATA_API_KEY") or ""
    timeout_seconds: int = 10
    max_retries: int = 3

    # Korean public data API endpoints
    endpoints: Dict[str, str] = None

    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = {
                "illegal_dumping": "https://api.data.go.kr/openapi/tn_pubr_public_illegal_dump_api",
                "traffic_accidents": "https://api.data.go.kr/openapi/tn_pubr_public_trfcacdnt_api",
                "air_quality": "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty",
                "population": "https://kosis.kr/openapi/statisticsData.do",
                "admin_districts": "https://api.data.go.kr/openapi/tn_pubr_public_admin_district_api",
                "seoul_districts": "https://openapi.seoul.go.kr:8088/6d4d776b466c656533356a4b4b5872/json/ListNecessaryStoresInformationOfSupermarket/1/25",
                "geographic_codes": "https://sgis.kostat.go.kr/openapi/service/administrative/code",
                "crime_stats": "https://api.data.go.kr/openapi/tn_pubr_public_crime_occurrence_api",
            }


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    cache_duration_hours: int = 24
    max_cache_size: int = 1000
    district_cache_hours: int = 24


@dataclass
class DataGenerationConfig:
    """Data generation configuration."""
    locale: str = "ko_KR"
    default_region: str = "seoul"
    randomization_seed: Optional[int] = None

    # Data quality settings
    verification_rate: float = 0.75  # 75% of community members are verified
    free_resource_rate: float = 0.4  # 40% of resources are free

    # Realistic distribution weights
    activity_level_weights: Dict[str, float] = None
    urgency_level_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.activity_level_weights is None:
            self.activity_level_weights = {"high": 0.3, "medium": 0.5, "low": 0.2}

        if self.urgency_level_weights is None:
            self.urgency_level_weights = {"낮음": 0.3, "보통": 0.4, "높음": 0.25, "매우높음": 0.05}


@dataclass
class UrbanDataConfig:
    """Urban data analysis configuration."""
    # Seoul administrative code
    seoul_administrative_code: str = "11"

    # Traffic analysis settings
    rush_hour_morning: List[int] = None
    rush_hour_evening: List[int] = None
    high_congestion_threshold: int = 75

    # Safety analysis settings
    crime_rate_thresholds: Dict[str, float] = None
    safety_response_time_max: int = 8  # minutes

    # Community settings
    max_group_participants_default: int = 20
    min_group_members: int = 5

    def __post_init__(self):
        if self.rush_hour_morning is None:
            self.rush_hour_morning = [7, 8, 9]

        if self.rush_hour_evening is None:
            self.rush_hour_evening = [17, 18, 19]

        if self.crime_rate_thresholds is None:
            self.crime_rate_thresholds = {
                "매우낮음": 2.0,
                "낮음": 3.0,
                "보통": 4.0,
                "높음": 5.0
            }


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    provider: str = os.getenv("LLM_PROVIDER", "google")
    model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    api_key: str = os.getenv("GOOGLE_API_KEY", "")

    def __post_init__(self):
        if not self.api_key and self.provider == "google":
            raise ValueError("GOOGLE_API_KEY environment variable is required for Google LLM provider")


@dataclass
class SystemConfig:
    """Overall system configuration."""
    api: APIConfig = None
    cache: CacheConfig = None
    data_generation: DataGenerationConfig = None
    urban_data: UrbanDataConfig = None
    llm: LLMConfig = None

    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # MCP server settings
    mcp_server_url: str = "urban-hive://"
    provider_base_url: str = "http://127.0.0.1:8001"

    def __post_init__(self):
        if self.api is None:
            self.api = APIConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.data_generation is None:
            self.data_generation = DataGenerationConfig()
        if self.urban_data is None:
            self.urban_data = UrbanDataConfig()
        if self.llm is None:
            self.llm = LLMConfig()


# Global configuration instance
config = SystemConfig()


# Convenience functions for accessing configuration
def get_api_config() -> APIConfig:
    """Get API configuration."""
    return config.api


def get_cache_config() -> CacheConfig:
    """Get cache configuration."""
    return config.cache


def get_data_generation_config() -> DataGenerationConfig:
    """Get data generation configuration."""
    return config.data_generation


def get_urban_data_config() -> UrbanDataConfig:
    """Get urban data configuration."""
    return config.urban_data


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return config.llm


def update_config_from_env():
    """Update configuration from environment variables."""
    # API settings
    if os.getenv("PUBLIC_DATA_API_KEY"):
        config.api.public_data_api_key = os.getenv("PUBLIC_DATA_API_KEY")

    if os.getenv("API_TIMEOUT"):
        try:
            config.api.timeout_seconds = int(os.getenv("API_TIMEOUT"))
        except ValueError:
            pass

    # Cache settings
    if os.getenv("CACHE_DURATION_HOURS"):
        try:
            config.cache.cache_duration_hours = int(os.getenv("CACHE_DURATION_HOURS"))
        except ValueError:
            pass

    # Data generation settings
    if os.getenv("DEFAULT_REGION"):
        config.data_generation.default_region = os.getenv("DEFAULT_REGION")

    if os.getenv("RANDOMIZATION_SEED"):
        try:
            config.data_generation.randomization_seed = int(os.getenv("RANDOMIZATION_SEED"))
        except ValueError:
            pass

    # MCP settings
    if os.getenv("MCP_SERVER_URL"):
        config.mcp_server_url = os.getenv("MCP_SERVER_URL")

    if os.getenv("PROVIDER_BASE_URL"):
        config.provider_base_url = os.getenv("PROVIDER_BASE_URL")

    # LLM settings
    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.getenv("LLM_PROVIDER")

    if os.getenv("LLM_MODEL"):
        config.llm.model = os.getenv("LLM_MODEL")

    if os.getenv("LLM_TEMPERATURE"):
        try:
            config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        except ValueError:
            pass

    if os.getenv("LLM_MAX_TOKENS"):
        try:
            config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
        except ValueError:
            pass

    if os.getenv("GOOGLE_API_KEY"):
        config.llm.api_key = os.getenv("GOOGLE_API_KEY")


def get_region_specific_config(region: str = "seoul") -> Dict:
    """Get region-specific configuration."""
    region_configs = {
        "seoul": {
            "administrative_code": "11",
            "districts_count": 25,
            "population_density": "high",
            "main_business_districts": ["강남구", "서초구", "중구", "영등포구"],
            "cultural_districts": ["종로구", "중구", "마포구", "홍대"],
            "residential_districts": ["노원구", "도봉구", "은평구", "강북구"]
        },
        "busan": {
            "administrative_code": "21",
            "districts_count": 16,
            "population_density": "medium",
            "main_business_districts": ["해운대구", "부산진구", "동래구"],
            "cultural_districts": ["중구", "서구", "부산진구"],
            "residential_districts": ["북구", "사상구", "강서구"]
        },
        "incheon": {
            "administrative_code": "23",
            "districts_count": 10,
            "population_density": "medium",
            "main_business_districts": ["남동구", "연수구"],
            "cultural_districts": ["중구", "동구"],
            "residential_districts": ["부평구", "계양구", "서구"]
        }
    }

    return region_configs.get(region.lower(), region_configs["seoul"])


# Load configuration from environment on import
update_config_from_env()
