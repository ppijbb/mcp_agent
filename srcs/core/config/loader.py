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
    """
    global _config
    if _config:
        return _config

    env = os.getenv("MCP_ENV", "development")
    
    base_config = _load_config_file(_config_path / "base.yaml")
    env_config = _load_config_file(_config_path / f"{env}.yaml")
    
    merged_config = _deep_merge(base_config, env_config)
    merged_config["environment"] = env

    # AppConfig 모델로 유효성 검사 및 객체 생성
    _config = AppConfig(**merged_config)
    
    # 환경 변수에서 민감한 정보 로드 (예: API 키)
    _load_secrets_from_env(_config)
    
    return _config

def _load_config_file(path: Path) -> Dict[str, Any]:
    """
    설정 파일을 로드합니다. '.enc' 확장자로 끝나면 암호화된 파일로 간주하고 복호화합니다.
    """
    encrypted_path = Path(f"{path}.enc")

    if encrypted_path.exists():
        try:
            decrypted_content = decrypt_file_content(str(encrypted_path))
            return yaml.safe_load(decrypted_content) or {}
        except Exception as e:
            # 암호화된 파일 복호화 실패 시, 경고를 남기고 일반 파일 로드를 시도합니다.
            print(f"⚠️ 경고: 암호화된 설정 파일({encrypted_path})을 복호화하는 데 실패했습니다. 일반 설정 파일을 찾습니다. 오류: {e}")
            pass # 아래의 일반 파일 로드로 넘어갑니다.

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
            
    return {}

def _deep_merge(source: Dict, destination: Dict) -> Dict:
    """두 딕셔너리를 재귀적으로 병합합니다."""
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            destination[key] = _deep_merge(value, destination[key])
        else:
            destination[key] = value
    return destination

def _load_secrets_from_env(config: AppConfig):
    """
    환경 변수에서 민감한 설정들을 로드하여 AppConfig 객체를 업데이트합니다.
    - GITHUB_TOKEN -> mcp_servers.github.env.GITHUB_TOKEN
    - GOOGLE_API_KEY -> mcp_servers.g-search.env.GOOGLE_API_KEY
    """
    if key := os.getenv("ENCRYPTION_KEY"):
        config.security.encryption_key = key

    for server_name, server_config in config.mcp_servers.items():
        for key, value in server_config.env.items():
            # ${VAR_NAME} 형식의 값을 환경 변수로 치환
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                server_config.env[key] = os.getenv(env_var)


# 애플리케이션 전체에서 사용할 설정 객체
settings = load_config() 