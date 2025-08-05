"""
MCP Integration - 공식 Python MCP SDK 사용

이 모듈은 공식 Python MCP SDK를 사용하여 코드 리뷰를 강화합니다.
웹 검색 결과에서 확인된 공식 SDK를 활용합니다.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# 공식 Python MCP SDK 사용
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client import Client
    from mcp.server import Server
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.tools import tool
    from mcp.server.fastmcp.resources import resource
    from mcp.server.fastmcp.prompts import prompt
except ImportError:
    logging.error("공식 MCP SDK가 설치되지 않았습니다. 'pip install mcp'를 실행하세요.")
    raise

from .config import config

logger = logging.getLogger(__name__)

class MCPIntegrationManager:
    """공식 Python MCP SDK를 사용한 통합 관리자"""
    
    def __init__(self):
        # 공식 MCP SDK 클라이언트 초기화
        self.client = None
        self.servers = {}
        
        # 사용 가능한 MCP 서버들 (공식 SDK 기반)
        self.available_servers = {
            'code-analysis': {
                'name': 'Code Analysis',
                'description': 'Neo4j 기반 코드 분석 및 구조적 인사이트',
                'command': ['python', '-m', 'mcp.servers.code_analysis'],
                'port': 8001
            },
            'code-expert-review': {
                'name': 'Code Expert Review', 
                'description': 'Martin Fowler, Uncle Bob 등 전문가 시뮬레이션',
                'command': ['python', '-m', 'mcp.servers.code_expert_review'],
                'port': 8002
            },
            'language-server': {
                'name': 'Language Server',
                'description': '다중 언어 코드 분석 및 조작',
                'command': ['python', '-m', 'mcp.servers.language_server'],
                'port': 8003
            },
            'tree-sitter': {
                'name': 'Tree-sitter',
                'description': '구조적 코드 이해 및 조작',
                'command': ['python', '-m', 'mcp.servers.tree_sitter'],
                'port': 8004
            },
            'sonarcloud': {
                'name': 'SonarCloud',
                'description': '코드 품질 및 보안 분석',
                'command': ['python', '-m', 'mcp.servers.sonarcloud'],
                'port': 8005
            },
            'codeql': {
                'name': 'CodeQL',
                'description': '정적 보안 분석 엔진',
                'command': ['python', '-m', 'mcp.servers.codeql'],
                'port': 8006
            }
        }
        
        logger.info(f"MCP Integration Manager initialized with {len(self.available_servers)} available servers")
    
    async def connect_to_server(self, server_name: str) -> bool:
        """
        MCP 서버에 연결 (공식 SDK 사용)
        
        Args:
            server_name (str): 서버 이름
            
        Returns:
            bool: 연결 성공 여부
        """
        if server_name not in self.available_servers:
            logger.error(f"Unknown MCP server: {server_name}")
            return False
        
        server_info = self.available_servers[server_name]
        
        try:
            # 공식 MCP SDK를 사용한 연결
            server_params = StdioServerParameters(
                command=server_info['command']
            )
            
            async with ClientSession(server_params) as session:
                # 서버 정보 확인
                server_info_result = await session.list_tools()
                if server_info_result:
                    self.servers[server_name] = {
                        'session': session,
                        'info': server_info,
                        'tools': server_info_result
                    }
                    logger.info(f"Connected to MCP server: {server_name}")
                    return True
                else:
                    logger.warning(f"Failed to connect to {server_name}: No tools available")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to {server_name}: {e}")
            return False
    
    async def call_mcp_tool(self, server_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        MCP 서버의 도구 호출 (공식 SDK 사용)
        
        Args:
            server_name (str): 서버 이름
            tool_name (str): 도구 이름
            **kwargs: 도구 매개변수
            
        Returns:
            Dict[str, Any]: 도구 실행 결과
        """
        if server_name not in self.servers:
            if not await self.connect_to_server(server_name):
                return {"error": f"Failed to connect to {server_name}"}
        
        server_data = self.servers[server_name]
        session = server_data['session']
        
        try:
            # 공식 SDK를 사용한 도구 호출
            result = await session.call_tool(tool_name, kwargs)
            return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            return {"error": str(e)}
    
    async def get_comprehensive_review(self, code: str, language: str, 
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        종합적인 코드 리뷰 (공식 MCP SDK 활용)
        
        Args:
            code (str): 분석할 코드
            language (str): 프로그래밍 언어
            context (Dict[str, Any], optional): 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 종합 리뷰 결과
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'mcp_analyses': {},
            'summary': {}
        }
        
        # 1. 기본 코드 분석
        if await self.connect_to_server('code-analysis'):
            analysis_result = await self.call_mcp_tool(
                'code-analysis', 'analyze_codebase',
                code=code, language=language
            )
            results['mcp_analyses']['code_analysis'] = analysis_result
        
        # 2. 언어별 분석
        if await self.connect_to_server('language-server'):
            language_result = await self.call_mcp_tool(
                'language-server', 'code_completion',
                code=code, language=language
            )
            results['mcp_analyses']['language_analysis'] = language_result
        
        # 3. 보안 분석
        if await self.connect_to_server('codeql'):
            security_result = await self.call_mcp_tool(
                'codeql', 'security_scan',
                code=code, language=language
            )
            results['mcp_analyses']['security_analysis'] = security_result
        
        # 4. 성능 분석
        if await self.connect_to_server('sonarcloud'):
            performance_result = await self.call_mcp_tool(
                'sonarcloud', 'quality_analysis',
                code=code, language=language
            )
            results['mcp_analyses']['performance_analysis'] = performance_result
        
        # 5. 전문가 리뷰
        if await self.connect_to_server('code-expert-review'):
            expert_result = await self.call_mcp_tool(
                'code-expert-review', 'expert_review',
                code=code, language=language,
                expert='martin_fowler'  # 또는 'uncle_bob'
            )
            results['mcp_analyses']['expert_review'] = expert_result
        
        # 6. 구조적 분석
        if await self.connect_to_server('tree-sitter'):
            structure_result = await self.call_mcp_tool(
                'tree-sitter', 'parse_code',
                code=code, language=language
            )
            results['mcp_analyses']['structural_analysis'] = structure_result
        
        # 7. 종합 요약 생성
        results['summary'] = self._generate_comprehensive_summary(results['mcp_analyses'])
        
        return results
    
    def _generate_comprehensive_summary(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """종합 분석 요약 생성"""
        summary = {
            'total_analyses': len(analyses),
            'critical_issues': 0,
            'high_priority_issues': 0,
            'medium_priority_issues': 0,
            'recommendations': [],
            'expert_insights': [],
            'security_findings': [],
            'performance_insights': []
        }
        
        for analysis_name, analysis_result in analyses.items():
            if 'error' in analysis_result:
                continue
            
            # 보안 분석 결과 처리
            if 'security_analysis' in analysis_name:
                security_data = analysis_result.get('vulnerabilities', [])
                summary['security_findings'] = security_data
                summary['critical_issues'] += len([v for v in security_data if v.get('severity') == 'critical'])
                summary['high_priority_issues'] += len([v for v in security_data if v.get('severity') == 'high'])
            
            # 성능 분석 결과 처리
            elif 'performance_analysis' in analysis_name:
                performance_data = analysis_result.get('issues', [])
                summary['performance_insights'] = performance_data
                summary['critical_issues'] += len([i for i in performance_data if i.get('impact') == 'critical'])
                summary['high_priority_issues'] += len([i for i in performance_data if i.get('impact') == 'high'])
            
            # 전문가 리뷰 결과 처리
            elif 'expert_review' in analysis_name:
                expert_data = analysis_result.get('recommendations', [])
                summary['expert_insights'] = expert_data
                summary['recommendations'].extend(expert_data)
            
            # 일반 권장사항 수집
            if 'recommendations' in analysis_result:
                summary['recommendations'].extend(analysis_result['recommendations'])
        
        # 중복 제거
        summary['recommendations'] = list(set(summary['recommendations']))
        summary['expert_insights'] = list(set(summary['expert_insights']))
        
        return summary
    
    async def get_specialized_analysis(self, analysis_type: str, code: str, 
                                     language: str, **kwargs) -> Dict[str, Any]:
        """
        특화된 분석 수행 (공식 SDK 사용)
        
        Args:
            analysis_type (str): 분석 유형
            code (str): 분석할 코드
            language (str): 프로그래밍 언어
            **kwargs: 추가 매개변수
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        analysis_mapping = {
            'security': ('codeql', 'security_scan'),
            'performance': ('sonarcloud', 'quality_analysis'),
            'quality': ('code-analysis', 'analyze_codebase'),
            'expert': ('code-expert-review', 'expert_review'),
            'structure': ('tree-sitter', 'parse_code'),
            'language': ('language-server', 'code_completion')
        }
        
        if analysis_type not in analysis_mapping:
            return {"error": f"Unknown analysis type: {analysis_type}"}
        
        server_name, tool_name = analysis_mapping[analysis_type]
        
        return await self.call_mcp_tool(
            server_name, tool_name,
            code=code, language=language, **kwargs
        )
    
    def get_available_servers(self) -> List[Dict[str, Any]]:
        """사용 가능한 MCP 서버 목록 반환"""
        return [
            {
                'name': name,
                'info': info,
                'connected': name in self.servers
            }
            for name, info in self.available_servers.items()
        ]
    
    def get_server_status(self) -> Dict[str, Any]:
        """서버 상태 정보 반환"""
        return {
            'total_servers': len(self.available_servers),
            'connected_servers': len(self.servers),
            'available_servers': list(self.available_servers.keys()),
            'active_connections': list(self.servers.keys())
        }
    
    async def health_check_all_servers(self) -> Dict[str, Any]:
        """모든 MCP 서버 상태 확인"""
        health_results = {}
        
        for server_name in self.available_servers.keys():
            try:
                is_connected = await self.connect_to_server(server_name)
                health_results[server_name] = {
                    'status': 'healthy' if is_connected else 'unhealthy',
                    'connected': is_connected
                }
            except Exception as e:
                health_results[server_name] = {
                    'status': 'error',
                    'error': str(e),
                    'connected': False
                }
        
        return health_results

# 전역 인스턴스
mcp_integration_manager = MCPIntegrationManager() 