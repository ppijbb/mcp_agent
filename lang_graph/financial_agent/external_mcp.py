import os
from typing import List, Optional
from mcp.client.stdio import StdioServerParameters


def parse_args(args_str: Optional[str]) -> List[str]:
    """
    공백 분리 인자 문자열을 안전하게 리스트로 변환합니다.
    JSON이 아닌 단순 공백 구분을 사용합니다.
    """
    if not args_str:
        return []
    # 간단한 분리: 따옴표로 감싼 인자를 고려하지 않음. 필요 시 확장.
    return [token for token in args_str.split(" ") if token]


def get_server_params(server_name: str) -> StdioServerParameters:
    """
    환경변수에서 외부 MCP 서버 실행 정보를 읽어 StdioServerParameters를 생성합니다.

    필수 환경 변수:
      - <SERVER>_MCP_CMD   (예: OPENAPI_MCP_CMD=/usr/bin/node or python)
      - <SERVER>_MCP_ARGS  (예: OPENAPI_MCP_ARGS="/path/to/server.js --flag value")

    사용 예:
      export OPENAPI_MCP_CMD="node"
      export OPENAPI_MCP_ARGS="/opt/mcp/openapi-server.js --spec /opt/specs/polygon.yaml"
    """
    key_prefix = server_name.upper()
    cmd = os.getenv(f"{key_prefix}_MCP_CMD")
    args_str = os.getenv(f"{key_prefix}_MCP_ARGS", "")

    if not cmd:
        raise ValueError(
            f"환경변수 {key_prefix}_MCP_CMD 가 설정되지 않았습니다. 서버 '{server_name}'를 초기화할 수 없습니다."
        )

    args = parse_args(args_str)
    return StdioServerParameters(command=cmd, args=args, env=None)


