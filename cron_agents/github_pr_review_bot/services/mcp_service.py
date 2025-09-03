"""
MCP Service - Model Context Protocol 서비스

MCP 서버들과의 상호작용을 담당하는 서비스입니다.
gemini-cli, vLLM, 그리고 기타 MCP 서버들을 통합하여 코드 분석, 검색, 파일 시스템 접근 등의 기능을 제공합니다.
"""

import logging
import sys
import subprocess
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.config import config
from ..core.gemini_service import gemini_service

logger = logging.getLogger(__name__)

class MCPService:
    """MCP 서비스 - gemini-cli 및 vLLM 통합"""
    
    def __init__(self):
        """MCP 서비스 초기화"""
        self.gemini_service = gemini_service
        self.vllm_enabled = False
        self.vllm_base_url = None
        
        self._initialize_vllm()
        self._initialize_mcp_servers()
        
        logger.info("MCP 서비스 초기화 완료 (gemini-cli + vLLM 통합)")
    
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
        """MCP 서버들 초기화"""
        self.mcp_servers = {
            "gemini": {
                "enabled": True,
                "type": "cli",
                "service": self.gemini_service
            },
            "vllm": {
                "enabled": self.vllm_enabled,
                "type": "api",
                "base_url": self.vllm_base_url
            }
        }
        
        logger.info(f"MCP 서버 초기화 완료: {list(self.mcp_servers.keys())}")
    
    def analyze_code(self, code: str, language: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        코드 분석 - gemini-cli 또는 vLLM 사용
        
        Args:
            code (str): 분석할 코드
            language (str): 프로그래밍 언어
            context (Dict[str, Any], optional): 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # Gemini CLI를 통한 분석 (우선순위)
            if self.mcp_servers["gemini"]["enabled"]:
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
            
            # vLLM을 통한 분석 (대체)
            if self.mcp_servers["vllm"]["enabled"]:
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
                    "review": "분석 도구를 사용할 수 없습니다. 로컬 분석만 제공됩니다.",
                    "error": "AI 분석 도구 사용 불가"
                },
                "timestamp": datetime.now().isoformat(),
                "free": True
            }
            
        except Exception as e:
            logger.error(f"코드 분석 실패: {e}")
            raise ValueError(f"코드 분석 실패: {e}")
    
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
        prompt = f"""다음 {language} 코드를 GitHub PR 리뷰 관점에서 분석해주세요:

{code}

코드 품질, 보안, 성능, 스타일을 종합적으로 검토하고 구체적인 개선사항을 제안해주세요."""

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
    
    def search_best_practices(self, language: str, topic: str = "code review") -> Dict[str, Any]:
        """모범 사례 검색 - gemini-cli 사용"""
        try:
            if self.mcp_servers["gemini"]["enabled"]:
                query = f"{language} {topic} best practices"
                result = self.gemini_service.review_code(
                    code=f"# {language} {topic} 모범 사례에 대해 설명해주세요",
                    language=language,
                    file_path="best_practices.md",
                    context={"type": "best_practices", "topic": topic}
                )
                return {"result": result, "query": query, "source": "gemini_cli"}
            else:
                raise ValueError("Gemini CLI가 비활성화되어 있습니다")
        except Exception as e:
            raise ValueError(f"모범 사례 검색 실패: {e}")
    
    def fetch_external_data(self, url: str) -> Dict[str, Any]:
        """외부 데이터 가져오기 - gemini-cli를 통한 웹 검색"""
        try:
            if self.mcp_servers["gemini"]["enabled"]:
                # Gemini CLI를 통해 웹 검색 시뮬레이션
                result = self.gemini_service.review_code(
                    code=f"# 다음 URL의 내용을 요약해주세요: {url}",
                    language="markdown",
                    file_path="external_data.md",
                    context={"type": "web_search", "url": url}
                )
                return {"result": result, "url": url, "source": "gemini_cli"}
            else:
                raise ValueError("Gemini CLI가 비활성화되어 있습니다")
        except Exception as e:
            raise ValueError(f"외부 데이터 가져오기 실패: {e}")
    
    def get_server_status(self) -> Dict[str, Any]:
        """서버 상태 조회"""
        status = {
            'total_servers': len(self.mcp_servers),
            'enabled_servers': 0,
            'available_servers': list(self.mcp_servers.keys()),
            'server_details': {}
        }
        
        for server_name, server_info in self.mcp_servers.items():
            is_enabled = server_info["enabled"]
            if is_enabled:
                status['enabled_servers'] += 1
            
            if server_name == "gemini":
                health = self.gemini_service.health_check()
                status['server_details'][server_name] = {
                    'enabled': is_enabled,
                    'type': server_info["type"],
                    'health': health
                }
            elif server_name == "vllm":
                status['server_details'][server_name] = {
                    'enabled': is_enabled,
                    'type': server_info["type"],
                    'base_url': server_info["base_url"]
                }
        
        return status
    
    def health_check_all_servers(self) -> Dict[str, Any]:
        """모든 MCP 서버 상태 확인"""
        health_results = {}
        
        for server_name, server_info in self.mcp_servers.items():
            try:
                if not server_info["enabled"]:
                    health_results[server_name] = {
                        'status': 'disabled',
                        'enabled': False
                    }
                    continue
                
                if server_name == "gemini":
                    health = self.gemini_service.health_check()
                    health_results[server_name] = {
                        'status': health.get('status', 'unknown'),
                        'enabled': True,
                        'details': health
                    }
                elif server_name == "vllm":
                    # vLLM 상태 확인 - OpenAI 라이브러리 사용
                    try:
                        from openai import OpenAI
                        client = OpenAI(
                            api_key="EMPTY",
                            base_url=f"{server_info['base_url']}/v1"
                        )
                        # 간단한 모델 목록 요청으로 상태 확인
                        models = client.models.list()
                        health_results[server_name] = {
                            'status': 'healthy',
                            'enabled': True,
                            'available_models': len(models.data) if models.data else 0
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
                    'enabled': server_info["enabled"]
                }
        
        return health_results
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록 반환"""
        tools = []
        
        if self.mcp_servers["gemini"]["enabled"]:
            tools.append({
                "name": "gemini_code_review",
                "description": "Gemini CLI를 통한 코드 리뷰",
                "server": "gemini",
                "type": "cli"
            })
            tools.append({
                "name": "gemini_best_practices",
                "description": "Gemini CLI를 통한 모범 사례 검색",
                "server": "gemini",
                "type": "cli"
            })
        
        if self.mcp_servers["vllm"]["enabled"]:
            tools.append({
                "name": "vllm_code_review",
                "description": "vLLM을 통한 코드 리뷰",
                "server": "vllm",
                "type": "api"
            })
        
        return tools
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """도구 호출"""
        try:
            if tool_name == "gemini_code_review":
                return self.gemini_service.review_code(**kwargs)
            elif tool_name == "gemini_best_practices":
                return self.search_best_practices(**kwargs)
            elif tool_name == "vllm_code_review":
                return self._analyze_with_vllm(**kwargs)
            else:
                raise ValueError(f"알 수 없는 도구: {tool_name}")
        except Exception as e:
            raise ValueError(f"도구 호출 실패 ({tool_name}): {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """사용량 통계 조회"""
        stats = {
            "total_servers": len(self.mcp_servers),
            "enabled_servers": sum(1 for s in self.mcp_servers.values() if s["enabled"]),
            "server_stats": {}
        }
        
        # Gemini 사용량 통계
        if self.mcp_servers["gemini"]["enabled"]:
            stats["server_stats"]["gemini"] = self.gemini_service.get_usage_stats()
        
        # vLLM 사용량 통계 (간단한 구현)
        if self.mcp_servers["vllm"]["enabled"]:
            stats["server_stats"]["vllm"] = {
                "enabled": True,
                "base_url": self.vllm_base_url,
                "note": "vLLM 사용량 추적은 별도 구현 필요"
            }
        
        return stats
