#!/usr/bin/env python3
"""
Smithery MCP 서버 연결 테스트 스크립트

기존 시스템과 독립적으로 Smithery MCP 서버들의 연결 상태를 테스트합니다.
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import ListToolsResult, TextContent
    from mcp.shared.exceptions import McpError
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.error("MCP package not available. Install with: pip install mcp")
    sys.exit(1)


class SmitheryMCPTester:
    """Smithery MCP 서버 연결 테스트"""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, Any] = {}
        
        # Smithery 서버 목록
        self.smithery_servers = {
            "fetch": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@smithery-ai/fetch",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", "")
                ]
            },
            "docfork": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@docfork/mcp",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", "")
                ]
            },
            "context7-mcp": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@upstash/context7-mcp",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", "")
                ]
            },
            "parallel-search": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@parallel/search",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", "")
                ]
            },
            "tavily-mcp": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@Jeetanshu18/tavily-mcp",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", "")
                ]
            },
            "WebSearch-MCP": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@mnhlt/WebSearch-MCP",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", "")
                ]
            },
            "semantic_scholar": {
                "type": "http",
                "httpUrl": "https://server.smithery.ai/@hamid-vakilzadeh/mcpsemanticscholar/mcp",
                "params": {
                    "api_key": os.getenv("SMITHERY_API_KEY", ""),
                    "profile": os.getenv("SMITHERY_PROFILE", "")
                }
            }
        }
    
    async def test_server(self, server_name: str, config: Dict[str, Any], timeout: float = 15.0) -> Dict[str, Any]:
        """단일 서버 연결 테스트"""
        result = {
            "server_name": server_name,
            "success": False,
            "connection_time": None,
            "tools_count": 0,
            "tools": [],
            "error": None,
            "error_type": None,
            "config_type": config.get("type", "stdio")
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Testing {server_name} ({result['config_type']})...")
            
            # 환경 변수 치환
            if result['config_type'] == "stdio":
                args = []
                for arg in config.get("args", []):
                    if arg.startswith("${") and arg.endswith("}"):
                        env_var = arg[2:-1]
                        args.append(os.getenv(env_var, ""))
                    else:
                        args.append(arg)
                
                # 빈 API 키 체크
                if "--key" in args:
                    key_idx = args.index("--key")
                    if key_idx + 1 < len(args) and not args[key_idx + 1]:
                        result["error"] = "SMITHERY_API_KEY not set"
                        result["error_type"] = "missing_api_key"
                        return result
                
                server_params = StdioServerParameters(
                    command=config["command"],
                    args=args
                )
                
                # stdio 연결
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        
                        # 도구 목록 가져오기
                        tools_result = await asyncio.wait_for(
                            session.list_tools(),
                            timeout=timeout
                        )
                        
                        tools = tools_result.tools if hasattr(tools_result, 'tools') else []
                        result["tools_count"] = len(tools)
                        result["tools"] = [tool.name for tool in tools]
                        result["success"] = True
                        
            else:
                # HTTP 연결
                http_url = config.get("httpUrl") or config.get("url")
                if not http_url:
                    result["error"] = "No URL provided"
                    result["error_type"] = "missing_url"
                    return result
                
                # URL 파라미터 추가
                params = config.get("params", {})
                if params:
                    from urllib.parse import urlencode
                    url_params = {}
                    for key, value in params.items():
                        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                            env_var = value[2:-1]
                            url_params[key] = os.getenv(env_var, "")
                        else:
                            url_params[key] = value
                    
                    if url_params:
                        http_url = f"{http_url}?{urlencode(url_params)}"
                
                # streamable HTTP 연결
                async with streamablehttp_client(http_url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        
                        # 도구 목록 가져오기
                        tools_result = await asyncio.wait_for(
                            session.list_tools(),
                            timeout=timeout
                        )
                        
                        tools = tools_result.tools if hasattr(tools_result, 'tools') else []
                        result["tools_count"] = len(tools)
                        result["tools"] = [tool.name for tool in tools]
                        result["success"] = True
            
            connection_time = (datetime.now() - start_time).total_seconds()
            result["connection_time"] = connection_time
            logger.info(f"✅ {server_name}: Connected in {connection_time:.2f}s, {result['tools_count']} tools")
            
        except asyncio.TimeoutError:
            result["error"] = f"Connection timeout after {timeout}s"
            result["error_type"] = "timeout"
            logger.error(f"❌ {server_name}: Timeout")
        except McpError as e:
            result["error"] = str(e)
            result["error_type"] = "mcp_error"
            error_code = getattr(e.error, 'code', None) if hasattr(e, 'error') else None
            if error_code:
                result["error"] += f" (code: {error_code})"
            logger.error(f"❌ {server_name}: MCP Error - {result['error']}")
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            logger.error(f"❌ {server_name}: {result['error_type']} - {result['error']}")
        
        return result
    
    async def test_all_servers(self, timeout: float = 15.0, max_concurrency: int = 3):
        """모든 Smithery 서버 테스트 (병렬)"""
        logger.info("=" * 80)
        logger.info("Smithery MCP 서버 연결 테스트 시작")
        logger.info("=" * 80)
        
        # API 키 확인
        api_key = os.getenv("SMITHERY_API_KEY")
        if not api_key:
            logger.warning("⚠️ SMITHERY_API_KEY 환경 변수가 설정되지 않았습니다")
            logger.info("   일부 서버는 API 키 없이도 테스트할 수 있습니다")
        else:
            logger.info(f"✅ SMITHERY_API_KEY 설정됨 (길이: {len(api_key)})")
        
        profile = os.getenv("SMITHERY_PROFILE")
        if profile:
            logger.info(f"✅ SMITHERY_PROFILE 설정됨: {profile}")
        
        logger.info(f"타임아웃: {timeout}초, 최대 동시 연결: {max_concurrency}")
        logger.info("")
        
        # 병렬 테스트
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def test_with_semaphore(server_name: str, config: Dict[str, Any]):
            async with semaphore:
                return await self.test_server(server_name, config, timeout)
        
        tasks = [
            asyncio.create_task(test_with_semaphore(name, config))
            for name, config in self.smithery_servers.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 수집
        for i, result in enumerate(results):
            server_name = list(self.smithery_servers.keys())[i]
            if isinstance(result, Exception):
                self.results[server_name] = {
                    "server_name": server_name,
                    "success": False,
                    "error": str(result),
                    "error_type": type(result).__name__
                }
            else:
                self.results[server_name] = result
        
        # 결과 출력
        self.print_results()
    
    def print_results(self):
        """테스트 결과 출력"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("테스트 결과 요약")
        logger.info("=" * 80)
        
        successful = []
        failed = []
        
        for server_name, result in self.results.items():
            if result.get("success"):
                successful.append(server_name)
            else:
                failed.append(server_name)
        
        logger.info(f"✅ 성공: {len(successful)}/{len(self.results)}")
        logger.info(f"❌ 실패: {len(failed)}/{len(self.results)}")
        logger.info("")
        
        # 성공한 서버
        if successful:
            logger.info("✅ 성공한 서버:")
            for server_name in successful:
                result = self.results[server_name]
                logger.info(f"  - {server_name}: {result['tools_count']} tools ({result.get('connection_time', 0):.2f}s)")
                if result.get('tools'):
                    logger.info(f"    도구: {', '.join(result['tools'][:5])}")
                    if len(result['tools']) > 5:
                        logger.info(f"    ... 외 {len(result['tools']) - 5}개")
        
        logger.info("")
        
        # 실패한 서버
        if failed:
            logger.info("❌ 실패한 서버:")
            for server_name in failed:
                result = self.results[server_name]
                logger.info(f"  - {server_name}: {result.get('error_type', 'unknown')}")
                logger.info(f"    에러: {result.get('error', 'Unknown error')}")
        
        logger.info("")
        logger.info("=" * 80)
        
        # 상세 결과 JSON 출력
        logger.info("상세 결과 (JSON):")
        print(json.dumps(self.results, indent=2, ensure_ascii=False, default=str))


async def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smithery MCP 서버 연결 테스트")
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="서버당 연결 타임아웃 (초, 기본값: 15)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="최대 동시 연결 수 (기본값: 3)"
    )
    parser.add_argument(
        "--server",
        type=str,
        help="특정 서버만 테스트 (예: fetch, docfork)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 형식으로만 출력"
    )
    
    args = parser.parse_args()
    
    tester = SmitheryMCPTester()
    
    if args.server:
        # 특정 서버만 테스트
        if args.server not in tester.smithery_servers:
            logger.error(f"서버 '{args.server}'를 찾을 수 없습니다")
            logger.info(f"사용 가능한 서버: {', '.join(tester.smithery_servers.keys())}")
            return
        
        config = tester.smithery_servers[args.server]
        result = await tester.test_server(args.server, config, args.timeout)
        tester.results[args.server] = result
        
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            tester.print_results()
    else:
        # 모든 서버 테스트
        await tester.test_all_servers(args.timeout, args.concurrency)
        
        if args.json:
            print(json.dumps(tester.results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())

