import os
import json
from typing import Dict, Any, List, Optional


def _parse_args(args_str: Optional[str]) -> List[str]:
    if not args_str:
        return []
    return [token for token in args_str.split(" ") if token]


def _maybe_json_env(env_str: Optional[str]) -> Dict[str, str]:
    if not env_str:
        return {}
    try:
        data = json.loads(env_str)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_external_server_config(server_name: str) -> Optional[Dict[str, Any]]:
    """
    환경 변수에서 외부 MCP 서버 실행 정보를 읽어 mcp_agent 설정 형식으로 반환합니다.

    필수:
      - <NAME>_MCP_CMD
    선택:
      - <NAME>_MCP_ARGS           (공백 구분 인자 문자열)
      - <NAME>_MCP_TIMEOUT_MS     (정수, 기본 30000)
      - <NAME>_MCP_TRUST          ("true"/"false", 기본 true)
      - <NAME>_MCP_ENV_JSON       (JSON 문자열 {"KEY":"VAL",...})
    """
    key = server_name.upper().replace("-", "_")
    cmd = os.getenv(f"{key}_MCP_CMD")
    if not cmd:
        return None

    args = _parse_args(os.getenv(f"{key}_MCP_ARGS", ""))
    timeout_raw = os.getenv(f"{key}_MCP_TIMEOUT_MS", "30000")
    trust_raw = os.getenv(f"{key}_MCP_TRUST", "true").lower()
    env_json = _maybe_json_env(os.getenv(f"{key}_MCP_ENV_JSON"))

    try:
        timeout = int(timeout_raw)
    except ValueError:
        timeout = 30000

    trust = trust_raw != "false"

    return {
        "command": cmd,
        "args": args,
        "timeout": timeout,
        "trust": trust,
        **({"env": env_json} if env_json else {}),
    }


def configure_external_servers(context, candidates: List[str]) -> List[str]:
    """
    주어진 후보 이름들에 대해 환경변수가 존재하는 서버만 context.config.mcp.servers에 등록한다.
    이미 존재하는 이름은 덮어쓰지 않고 건너뜀.
    반환값은 추가된 서버 이름 리스트.
    """
    added: List[str] = []
    server_map = getattr(context.config.mcp, "servers", {})

    for name in candidates:
        if name in server_map:
            continue
        cfg = load_external_server_config(name)
        if cfg:
            server_map[name] = cfg
            added.append(name)

    # context.config.mcp.servers는 dict와 유사한 객체일 수 있음. 직접 재할당로 안전하게 반영.
    context.config.mcp.servers = server_map
    return added


