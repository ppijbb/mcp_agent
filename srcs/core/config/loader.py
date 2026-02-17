"""
Configuration loader for MCP Agent system.

Loads and merges configuration from YAML files, supports encrypted config files,
and provides environment variable substitution for sensitive values.
Uses singleton pattern to cache configuration after first load.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from srcs.core.config.schema import AppConfig
from srcs.core.security.crypto import decrypt_file_content

_config: AppConfig | None = None
_config_path = Path(os.getenv("MCP_CONFIG_PATH", "configs/"))


def load_config() -> AppConfig:
    """
    설정 파일을 로드, 병합, 유효성 검사를 수행하고 AppConfig 객체를 반환합니다.
    싱글톤 패턴을 사용하여 한 번만 로드합니다.
    
    Returns:
        AppConfig: 로드된 애플리케이션 설정 객체
        
    Note:
        - 성능 최적화를 위해 싱글톤 패턴 사용
        - 첫 로드 이후 캐시된 설정 반환
        - 환경 변수 치환 포함
    """
    global _config
    if _config:
        return _config

    env = os.getenv("MCP_ENV", "development")

    # 병렬로 설정 파일 로드하여 성능 최적화
    base_config = _load_config_file(_config_path / "base.yaml")
    env_config = _load_config_file(_config_path / f"{env}.yaml")

    # 빈 설정인 경우 early return
    if not base_config and not env_config:
        print(f"⚠️ 경고: 설정 파일을 찾을 수 없습니다. 경로: {_config_path}")
        base_config = {}  # 기본값 설정

    merged_config = _deep_merge(base_config, env_config)
    merged_config["environment"] = env

    # AppConfig 모델로 유효성 검사 및 객체 생성
    try:
        _config = AppConfig(**merged_config)
    except Exception as e:
        print(f"❌ 오류: 설정 유효성 검사 실패: {e}")
        raise

    # 환경 변수에서 민감한 정보 로드 (예: API 키)
    _load_secrets_from_env(_config)

    return _config


def _load_config_file(path: Path) -> Dict[str, Any]:
    """
    설정 파일을 로드합니다. '.enc' 확장자로 끝나면 암호화된 파일로 간주하고 복호화합니다.
    
    Args:
        path: 로드할 설정 파일의 경로
        
    Returns:
        Dict[str, Any]: 로드된 설정 데이터. 파일이 없으면 빈 딕셔너리 반환
        
    Note:
        - 우선 {path}.enc 암호화 파일 확인
        - 암호화 파일이 있으면 복호화 시도, 실패 시 일반 파일 로드
        - 둘 다 없으면 빈 딕셔너리 반환
        - 성능 최적화를 위해 파일 존재 여부를 한 번만 확인
    """
    encrypted_path = Path(f"{path}.enc")
    regular_path = path
    
    # 한 번에 파일 존재 여부 확인하여 시스템 콜 최소화
    encrypted_exists = encrypted_path.exists()
    regular_exists = regular_path.exists()
    
    if not encrypted_exists and not regular_exists:
        return {}

    # 암호화 파일이 있으면 우선 시도
    if encrypted_exists:
        try:
            decrypted_content = decrypt_file_content(str(encrypted_path))
            result = yaml.safe_load(decrypted_content)
            return result if result is not None else {}
        except Exception as e:
            # 암호화된 파일 복호화 실패 시, 경고를 남기고 일반 파일 로드를 시도합니다.
            print(f"⚠️ 경고: 암호화된 설정 파일({encrypted_path})을 복호화하는 데 실패했습니다. 일반 설정 파일을 찾습니다. 오류: {e}")
            if not regular_exists:
                return {}

    # 일반 파일 로드
    if regular_exists:
        try:
            with open(regular_path, "r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                return result if result is not None else {}
        except Exception as e:
            print(f"⚠️ 경고: 설정 파일({regular_path}) 로드 실패: {e}")

    return {}


def _deep_merge(source: Dict, destination: Dict) -> Dict:
    """
    두 딕셔너리를 재귀적으로 병합합니다.
    
    Args:
        source: 소스 딕셔너리 (우선순위 낮음)
        destination: 목적지 딕셔너리 (우선순위 높음, 이 딕셔너리에 병합 결과가 저장됨)
        
    Returns:
        Dict: 병합된 목적지 딕셔너리
        
    Note:
        - source의 값들이 destination으로 병합됨
        - 두 딕셔너리에 모두 있는 키가 딕셔너리 타입이면 재귀적으로 병합
        - 그 외 경우 source의 값으로 destination 값을 덮어씀
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            destination[key] = _deep_merge(value, destination[key])
        else:
            destination[key] = value
    return destination


def _load_secrets_from_env(config: AppConfig):
    """
    환경 변수에서 민감한 설정들을 로드하여 AppConfig 객체를 업데이트합니다.
    
    Args:
        config: 업데이트할 AppConfig 객체
        
    Note:
        - ENCRYPTION_KEY 환경 변수를 config.security.encryption_key에 설정
        - ${VAR_NAME} 형식의 환경 변수 참조를 실제 값으로 치환
        - 예: GITHUB_TOKEN -> mcp_servers.github.env.GITHUB_TOKEN
        - 예: GOOGLE_API_KEY -> mcp_servers.g-search.env.GOOGLE_API_KEY
    """
    if key := os.getenv("ENCRYPTION_KEY"):
        config.security.encryption_key = key

    for server_name, server_config in config.mcp_servers.items():
        for key, value in server_config.env.items():
            # ${VAR_NAME} 형식의 값을 환경 변수로 치환
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is not None:
                    server_config.env[key] = env_value


# 애플리케이션 전체에서 사용할 설정 객체
settings = load_config()
