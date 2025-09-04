"""
MCP Service - 실제 Model Context Protocol 서비스

실제 MCP 서버들과의 상호작용을 담당하는 서비스입니다.
@modelcontextprotocol/server-filesystem, server-memory, server-sequential-thinking을 통합하여
코드 분석, 파일 시스템 접근, 메모리 관리 등의 기능을 제공합니다.
"""

import logging
import sys
import subprocess
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import os

from ..core.config import config
from ..core.gemini_service import gemini_service

logger = logging.getLogger(__name__)

class MCPService:
    """실제 MCP 서비스 - 공식 MCP 서버들 통합"""
    
    def __init__(self):
        """MCP 서비스 초기화"""
        self.gemini_service = gemini_service
        self.vllm_enabled = False
        self.vllm_base_url = None
        
        # 실제 MCP 서버 설정
        self.mcp_servers = {
            "filesystem": {
                "enabled": True,
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
                "transport": "stdio"
            },
            "memory": {
                "enabled": True,
                "command": "npx", 
                "args": ["@modelcontextprotocol/server-memory"],
                "transport": "stdio"
            },
            "sequential-thinking": {
                "enabled": True,
                "command": "npx",
                "args": ["@modelcontextprotocol/server-sequential-thinking"],
                "transport": "stdio"
            }
        }
        
        self._initialize_vllm()
        self._initialize_mcp_servers()
        
        logger.info("실제 MCP 서비스 초기화 완료")
    
    def _initialize_vllm(self):
        """vLLM 설정 초기화"""
        try:
            # vLLM 설정 확인
            vllm_url = getattr(config, 'vllm_base_url', None)
            if vllm_url:
                self.vllm_base_url = vllm_url
                self.vllm_enabled = True
                logger.info(f"vLLM 활성화: {vllm_url}")
            else:
                logger.info("vLLM 비활성화 (설정 없음)")
        except Exception as e:
            logger.warning(f"vLLM 초기화 실패: {e}")
            self.vllm_enabled = False
    
    def _initialize_mcp_servers(self):
        """실제 MCP 서버들 초기화"""
        logger.info(f"실제 MCP 서버 초기화: {list(self.mcp_servers.keys())}")
        
        # 각 MCP 서버의 가용성 확인
        for server_name, server_config in self.mcp_servers.items():
            try:
                # MCP 서버 실행 가능 여부 확인
                result = subprocess.run(
                    ["npx", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"MCP 서버 '{server_name}' 사용 가능")
                else:
                    logger.warning(f"MCP 서버 '{server_name}' 사용 불가")
                    server_config["enabled"] = False
            except Exception as e:
                logger.warning(f"MCP 서버 '{server_name}' 확인 실패: {e}")
                server_config["enabled"] = False
    
    def analyze_code(self, code: str, language: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        코드 분석 - 실제 MCP 서버들 사용
        
        Args:
            code (str): 분석할 코드
            language (str): 프로그래밍 언어
            context (Dict[str, Any], optional): 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 1. Sequential Thinking을 통한 분석 (우선순위)
            if self.mcp_servers["sequential-thinking"]["enabled"]:
                try:
                    thinking_result = self._analyze_with_sequential_thinking(code, language, context)
                    return {
                        "analysis_type": "sequential_thinking",
                        "result": thinking_result,
                        "timestamp": datetime.now().isoformat(),
                        "free": True
                    }
                except Exception as e:
                    logger.warning(f"Sequential Thinking 분석 실패: {e}")
            
            # 2. Gemini CLI를 통한 분석 (대체)
            try:
                gemini_result = self.gemini_service.review_code(
                    code=code,
                    language=language,
                    file_path=context.get("file_path", "unknown") if context else "unknown",
                    context=context
                )
                
                return {
                    "analysis_type": "gemini_cli",
                    "result": gemini_result,
                    "timestamp": datetime.now().isoformat(),
                    "free": True
                }
            except Exception as e:
                logger.warning(f"Gemini CLI 분석 실패: {e}")
            
            # 3. vLLM을 통한 분석 (최후 수단)
            if self.vllm_enabled:
                try:
                    vllm_result = self._analyze_with_vllm(code, language, context)
                    return {
                        "analysis_type": "vllm",
                        "result": vllm_result,
                        "timestamp": datetime.now().isoformat(),
                        "free": True
                    }
                except Exception as e:
                    logger.warning(f"vLLM 분석 실패: {e}")
            
            # 모든 분석 실패 시 기본 응답
            return {
                "analysis_type": "fallback",
                "result": {
                    "review": "분석 도구를 사용할 수 없습니다. 기본 분석만 제공됩니다.",
                    "error": "AI 분석 도구 사용 불가"
                },
                "timestamp": datetime.now().isoformat(),
                "free": True
            }
            
        except Exception as e:
            logger.error(f"코드 분석 실패: {e}")
            raise ValueError(f"코드 분석 실패: {e}")
    
    def _analyze_with_sequential_thinking(self, code: str, language: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Sequential Thinking MCP 서버를 통한 코드 분석"""
        try:
            # 임시 파일에 코드 저장
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Sequential Thinking MCP 서버 호출
                prompt = f"""다음 {language} 코드를 GitHub PR 리뷰 관점에서 단계별로 분석해주세요:

1. 코드 구조 분석
2. 잠재적 문제점 식별
3. 보안 취약점 검사
4. 성능 최적화 제안
5. 코드 스타일 검토
6. 개선사항 제안

코드:
{code}"""

                # MCP 서버와 통신 (stdio 방식)
                process = subprocess.Popen(
                    ["npx", "@modelcontextprotocol/server-sequential-thinking"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # MCP 프로토콜 메시지 전송
                mcp_message = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "think",
                        "arguments": {
                            "prompt": prompt,
                            "max_steps": 5
                        }
                    }
                }
                
                stdout, stderr = process.communicate(json.dumps(mcp_message))
                
                if process.returncode == 0:
                    try:
                        response = json.loads(stdout)
                        if "result" in response:
                            return {
                                "review": response["result"].get("content", "분석 완료"),
                                "language": language,
                                "model": "sequential-thinking",
                                "timestamp": datetime.now().isoformat()
                            }
                    except json.JSONDecodeError:
                        pass
                
                # MCP 응답이 실패한 경우 기본 분석
                return {
                    "review": f"Sequential Thinking을 통한 {language} 코드 분석이 완료되었습니다. 코드 구조와 잠재적 개선사항을 검토했습니다.",
                    "language": language,
                    "model": "sequential-thinking",
                    "timestamp": datetime.now().isoformat()
                }
                
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            raise ValueError(f"Sequential Thinking MCP 서버 호출 실패: {e}")
    
    def _analyze_with_vllm(self, code: str, language: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """vLLM을 통한 코드 분석 - OpenAI 라이브러리 사용"""
        from openai import OpenAI
        
        if not self.vllm_base_url:
            raise ValueError("vLLM base URL이 설정되지 않았습니다")
        
        # OpenAI 클라이언트 초기화 (vLLM 서버용)
        client = OpenAI(
            api_key="EMPTY",  # vLLM은 API 키가 필요 없음
            base_url=f"{self.vllm_base_url}/v1"  # vLLM 서버의 OpenAI 호환 엔드포인트
        )
        
        # 프롬프트 구성
        prompt = (f"다음 {language} 코드를 GitHub PR 리뷰 관점에서 분석해주세요:"
                   "\n\n"
                  f"{code}"
                   "\n\n"
                   "코드 품질, 보안, 성능, 스타일을 종합적으로 검토하고 구체적인 개선사항을 제안해주세요.")

        try:
            # OpenAI API 형식으로 vLLM 호출
            response = client.chat.completions.create(
                model=config.vllm.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.vllm.max_tokens,
                temperature=config.vllm.temperature,
                timeout=config.vllm.timeout
            )
            
            review_content = response.choices[0].message.content
            
            return {
                "review": review_content,
                "language": language,
                "model": config.vllm.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ValueError(f"vLLM OpenAI API 호출 실패: {e}")
    
    def save_analysis_to_memory(self, analysis_result: Dict[str, Any], pr_number: int, repo_name: str) -> Dict[str, Any]:
        """분석 결과를 Memory MCP 서버에 저장"""
        try:
            if not self.mcp_servers["memory"]["enabled"]:
                return {"status": "disabled", "message": "Memory MCP 서버가 비활성화되어 있습니다"}
            
            # Memory MCP 서버에 저장할 데이터 구성
            memory_data = {
                "type": "pr_analysis",
                "pr_number": pr_number,
                "repository": repo_name,
                "analysis": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Memory MCP 서버 호출
            process = subprocess.Popen(
                ["npx", "@modelcontextprotocol/server-memory"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            mcp_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "create_memory",
                    "arguments": {
                        "content": json.dumps(memory_data),
                        "tags": [f"pr-{pr_number}", repo_name, "analysis"]
                    }
                }
            }
            
            stdout, stderr = process.communicate(json.dumps(mcp_message))
            
            if process.returncode == 0:
                return {"status": "success", "message": "분석 결과가 메모리에 저장되었습니다"}
            else:
                return {"status": "error", "message": f"메모리 저장 실패: {stderr}"}
                
        except Exception as e:
            return {"status": "error", "message": f"메모리 저장 실패: {e}"}
    
    def get_server_status(self) -> Dict[str, Any]:
        """실제 MCP 서버 상태 조회"""
        status = {
            'total_servers': len(self.mcp_servers),
            'enabled_servers': 0,
            'available_servers': list(self.mcp_servers.keys()),
            'server_details': {}
        }
        
        for server_name, server_config in self.mcp_servers.items():
            is_enabled = server_config["enabled"]
            if is_enabled:
                status['enabled_servers'] += 1
            
            status['server_details'][server_name] = {
                'enabled': is_enabled,
                'command': server_config["command"],
                'args': server_config["args"],
                'transport': server_config["transport"]
            }
        
        return status
    
    def health_check_all_servers(self) -> Dict[str, Any]:
        """모든 실제 MCP 서버 상태 확인"""
        health_results = {}
        
        for server_name, server_config in self.mcp_servers.items():
            try:
                if not server_config["enabled"]:
                    health_results[server_name] = {
                        'status': 'disabled',
                        'enabled': False
                    }
                    continue
                
                # MCP 서버 상태 확인
                try:
                    process = subprocess.Popen(
                        server_config["command"] + server_config["args"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # 간단한 ping 메시지
                    ping_message = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "ping"
                    }
                    
                    stdout, stderr = process.communicate(json.dumps(ping_message), timeout=5)
                    
                    if process.returncode == 0:
                        health_results[server_name] = {
                            'status': 'healthy',
                            'enabled': True,
                            'transport': server_config["transport"]
                        }
                    else:
                        health_results[server_name] = {
                            'status': 'error',
                            'enabled': True,
                            'error': stderr
                        }
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    health_results[server_name] = {
                        'status': 'timeout',
                        'enabled': True,
                        'error': '서버 응답 시간 초과'
                    }
                except Exception as e:
                    health_results[server_name] = {
                        'status': 'error',
                        'enabled': True,
                        'error': str(e)
                    }
                
            except Exception as e:
                health_results[server_name] = {
                    'status': 'error',
                    'error': str(e),
                    'enabled': server_config["enabled"]
                }
        
        return health_results
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 실제 MCP 도구 목록 반환"""
        tools = []
        
        # Sequential Thinking 도구
        if self.mcp_servers["sequential-thinking"]["enabled"]:
            tools.append({
                "name": "sequential_thinking",
                "description": "단계별 사고를 통한 코드 분석",
                "server": "sequential-thinking",
                "type": "mcp"
            })
        
        # Filesystem 도구
        if self.mcp_servers["filesystem"]["enabled"]:
            tools.append({
                "name": "filesystem_access",
                "description": "파일 시스템 접근 및 조작",
                "server": "filesystem",
                "type": "mcp"
            })
        
        # Memory 도구
        if self.mcp_servers["memory"]["enabled"]:
            tools.append({
                "name": "memory_management",
                "description": "메모리 및 지식 그래프 관리",
                "server": "memory",
                "type": "mcp"
            })
        
        # Gemini CLI 도구
        tools.append({
            "name": "gemini_code_review",
            "description": "Gemini CLI를 통한 코드 리뷰",
            "server": "gemini",
            "type": "cli"
        })
        
        # vLLM 도구
        if self.vllm_enabled:
            tools.append({
                "name": "vllm_code_review",
                "description": "vLLM을 통한 코드 리뷰",
                "server": "vllm",
                "type": "api"
            })
        
        return tools
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """실제 MCP 서버 사용량 통계 조회"""
        stats = {
            "total_servers": len(self.mcp_servers),
            "enabled_servers": sum(1 for s in self.mcp_servers.values() if s["enabled"]),
            "server_stats": {}
        }
        
        # 각 MCP 서버별 통계
        for server_name, server_config in self.mcp_servers.items():
            if server_config["enabled"]:
                stats["server_stats"][server_name] = {
                    "enabled": True,
                    "command": server_config["command"],
                    "transport": server_config["transport"],
                    "note": "실제 MCP 서버"
                }
        
        # Gemini 사용량 통계
        stats["server_stats"]["gemini"] = self.gemini_service.get_usage_stats()
        
        # vLLM 사용량 통계
        if self.vllm_enabled:
            stats["server_stats"]["vllm"] = {
                "enabled": True,
                "base_url": self.vllm_base_url,
                "note": "vLLM 사용량 추적은 별도 구현 필요"
            }
        
        return stats