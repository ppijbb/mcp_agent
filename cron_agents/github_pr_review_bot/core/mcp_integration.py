"""
MCP Integration - 다양한 MCP 서버들과의 통합

이 모듈은 코드 리뷰를 강화하기 위한 다양한 MCP 서버들과의 통합을 제공합니다.
웹 검색 결과에서 확인된 관련 MCP 서버들을 활용합니다.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
import json

from .config import config

logger = logging.getLogger(__name__)

class MCPIntegrationManager:
    """MCP 서버 통합 관리자"""
    
    def __init__(self):
        # 웹 검색 결과에서 확인된 관련 MCP 서버들
        self.available_mcp_servers = {
            # 코드 분석 관련
            'code_analysis': {
                'name': 'Code Analysis',
                'description': 'Neo4j 기반 코드 분석 및 구조적 인사이트',
                'endpoint': 'http://localhost:8001',
                'tools': ['analyze_codebase', 'get_quality_metrics', 'structural_insights']
            },
            'language_server': {
                'name': 'Language Server',
                'description': '다중 언어 코드 분석 및 조작',
                'endpoint': 'http://localhost:8002',
                'tools': ['code_completion', 'diagnostics', 'refactoring']
            },
            'tree_sitter': {
                'name': 'Tree-sitter',
                'description': '구조적 코드 이해 및 조작',
                'endpoint': 'http://localhost:8003',
                'tools': ['parse_code', 'extract_symbols', 'code_navigation']
            },
            
            # 보안 관련
            'codeql': {
                'name': 'CodeQL',
                'description': '정적 보안 분석 엔진',
                'endpoint': 'http://localhost:8004',
                'tools': ['security_scan', 'vulnerability_detection', 'query_evaluation']
            },
            'sonarcloud': {
                'name': 'SonarCloud',
                'description': '코드 품질 및 보안 분석',
                'endpoint': 'http://localhost:8005',
                'tools': ['quality_analysis', 'security_scan', 'code_smells']
            },
            
            # 성능 관련
            'performance_analyzer': {
                'name': 'Performance Analyzer',
                'description': '코드 성능 분석 및 최적화',
                'endpoint': 'http://localhost:8006',
                'tools': ['performance_analysis', 'bottleneck_detection', 'optimization_suggestions']
            },
            
            # 문서화 관련
            'autodocument': {
                'name': 'Autodocument',
                'description': '자동 문서 생성 및 코드 리뷰',
                'endpoint': 'http://localhost:8007',
                'tools': ['generate_docs', 'create_test_plans', 'code_review']
            },
            
            # 전문가 리뷰
            'code_expert_review': {
                'name': 'Code Expert Review',
                'description': 'Martin Fowler, Uncle Bob 등 전문가 시뮬레이션',
                'endpoint': 'http://localhost:8008',
                'tools': ['expert_review', 'refactoring_suggestions', 'clean_code_recommendations']
            },
            
            # 다중 LLM 검증
            'multi_llm_cross_check': {
                'name': 'Multi-LLM Cross-Check',
                'description': '다중 LLM 동시 쿼리 및 비교',
                'endpoint': 'http://localhost:8009',
                'tools': ['cross_check', 'fact_checking', 'perspective_comparison']
            },
            
            # 코드베이스 분석
            'codebase_insight': {
                'name': 'Codebase Insight',
                'description': '심층 코드베이스 분석 및 패턴 감지',
                'endpoint': 'http://localhost:8010',
                'tools': ['pattern_detection', 'architecture_analysis', 'knowledge_management']
            }
        }
        
        self.active_connections = {}
        logger.info(f"MCP Integration Manager initialized with {len(self.available_mcp_servers)} available servers")
    
    async def connect_to_server(self, server_name: str) -> bool:
        """
        MCP 서버에 연결
        
        Args:
            server_name (str): 서버 이름
            
        Returns:
            bool: 연결 성공 여부
        """
        if server_name not in self.available_mcp_servers:
            logger.error(f"Unknown MCP server: {server_name}")
            return False
        
        server_info = self.available_mcp_servers[server_name]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{server_info['endpoint']}/health")
                if response.status_code == 200:
                    self.active_connections[server_name] = server_info
                    logger.info(f"Connected to MCP server: {server_name}")
                    return True
                else:
                    logger.warning(f"Failed to connect to {server_name}: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to {server_name}: {e}")
            return False
    
    async def call_mcp_tool(self, server_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        MCP 서버의 도구 호출
        
        Args:
            server_name (str): 서버 이름
            tool_name (str): 도구 이름
            **kwargs: 도구 매개변수
            
        Returns:
            Dict[str, Any]: 도구 실행 결과
        """
        if server_name not in self.active_connections:
            if not await self.connect_to_server(server_name):
                return {"error": f"Failed to connect to {server_name}"}
        
        server_info = self.active_connections[server_name]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{server_info['endpoint']}/tools/{tool_name}",
                    json=kwargs,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"Tool call failed: {response.status_code}"}
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            return {"error": str(e)}
    
    async def get_comprehensive_review(self, code: str, language: str, 
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        종합적인 코드 리뷰 (다중 MCP 서버 활용)
        
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
        if await self.connect_to_server('code_analysis'):
            analysis_result = await self.call_mcp_tool(
                'code_analysis', 'analyze_codebase',
                code=code, language=language
            )
            results['mcp_analyses']['code_analysis'] = analysis_result
        
        # 2. 언어별 분석
        if await self.connect_to_server('language_server'):
            language_result = await self.call_mcp_tool(
                'language_server', 'code_completion',
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
        if await self.connect_to_server('performance_analyzer'):
            performance_result = await self.call_mcp_tool(
                'performance_analyzer', 'performance_analysis',
                code=code, language=language
            )
            results['mcp_analyses']['performance_analysis'] = performance_result
        
        # 5. 전문가 리뷰
        if await self.connect_to_server('code_expert_review'):
            expert_result = await self.call_mcp_tool(
                'code_expert_review', 'expert_review',
                code=code, language=language,
                expert='martin_fowler'  # 또는 'uncle_bob'
            )
            results['mcp_analyses']['expert_review'] = expert_result
        
        # 6. 다중 LLM 검증
        if await self.connect_to_server('multi_llm_cross_check'):
            cross_check_result = await self.call_mcp_tool(
                'multi_llm_cross_check', 'cross_check',
                code=code, language=language,
                providers=['openai', 'anthropic', 'google']
            )
            results['mcp_analyses']['multi_llm_verification'] = cross_check_result
        
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
        특화된 분석 수행
        
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
            'performance': ('performance_analyzer', 'performance_analysis'),
            'quality': ('code_analysis', 'get_quality_metrics'),
            'expert': ('code_expert_review', 'expert_review'),
            'documentation': ('autodocument', 'generate_docs'),
            'architecture': ('codebase_insight', 'architecture_analysis')
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
                'connected': name in self.active_connections
            }
            for name, info in self.available_mcp_servers.items()
        ]
    
    def get_server_status(self) -> Dict[str, Any]:
        """서버 상태 정보 반환"""
        return {
            'total_servers': len(self.available_mcp_servers),
            'connected_servers': len(self.active_connections),
            'available_servers': list(self.available_mcp_servers.keys()),
            'active_connections': list(self.active_connections.keys())
        }
    
    async def health_check_all_servers(self) -> Dict[str, Any]:
        """모든 MCP 서버 상태 확인"""
        health_results = {}
        
        for server_name in self.available_mcp_servers.keys():
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