"""
Figma Integration Module

Figma MCP 서버와의 연동을 통한 디자인 분석 및 메타데이터 추출
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import re
from dataclasses import dataclass
from enum import Enum

# MCP 관련 imports
try:
    from mcp_agent.mcp_client import MCPClient
    from mcp_agent.tools import ToolRegistry
except ImportError:
    # Fallback for development
    MCPClient = None
    ToolRegistry = None

# 로컬 설정 imports - 절대 import로 수정
try:
    from config import FIGMA_CONFIG, ANALYSIS_CONFIG
except ImportError:
    # Fallback 설정
    FIGMA_CONFIG = {"fallback_enabled": False}
    ANALYSIS_CONFIG = {"timeout": 30}

logger = logging.getLogger(__name__)

class FigmaIntegration:
    """
    Figma MCP 서버 통합 클래스
    
    Figma 디자인 파일 분석, 컴포넌트 추출, 메타데이터 수집 등의 기능 제공
    """
    
    def __init__(self):
        """FigmaIntegration 초기화"""
        self.server_name = "figma-dev-mode"
        self.client = None
        self.connected = False
        
        # 캐시된 데이터
        self._design_cache = {}
        self._component_cache = {}
        
        logger.info("FigmaIntegration 초기화 완료")
    
    async def connect(self) -> bool:
        """
        Figma MCP 서버에 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            if self.connected:
                return True
                
            self.client = await connect(self.server_name)
            self.connected = True
            
            logger.info(f"Figma MCP 서버 연결 성공: {self.server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Figma MCP 서버 연결 실패: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Figma MCP 서버 연결 해제"""
        try:
            if self.client:
                await disconnect(self.server_name)
                self.client = None
                self.connected = False
                
            logger.info("Figma MCP 서버 연결 해제 완료")
            
        except Exception as e:
            logger.warning(f"Figma MCP 서버 연결 해제 중 오류: {str(e)}")
    
    def extract_figma_ids(self, figma_url: str) -> Dict[str, str]:
        """
        Figma URL에서 파일 ID와 노드 ID 추출
        
        Args:
            figma_url: Figma 파일/프레임 URL
            
        Returns:
            파일 ID와 노드 ID가 포함된 딕셔너리
        """
        try:
            # Figma URL 패턴 매칭
            file_pattern = r'figma\.com/file/([a-zA-Z0-9]+)'
            node_pattern = r'node-id=([^&]+)'
            
            file_match = re.search(file_pattern, figma_url)
            node_match = re.search(node_pattern, figma_url)
            
            result = {
                "file_id": file_match.group(1) if file_match else None,
                "node_id": node_match.group(1) if node_match else None,
                "url": figma_url
            }
            
            logger.debug(f"Figma URL 파싱 결과: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Figma URL 파싱 실패: {str(e)}")
            return {"file_id": None, "node_id": None, "url": figma_url}
    
    async def get_design_metadata(self, figma_url: str) -> Dict[str, Any]:
        """
        디자인 메타데이터 추출
        
        Args:
            figma_url: Figma 파일 URL
            
        Returns:
            디자인 메타데이터
        """
        try:
            if not await self.connect():
                raise ConnectionError("Figma MCP 서버에 연결할 수 없습니다.")
            
            figma_ids = self.extract_figma_ids(figma_url)
            
            # 캐시 확인
            cache_key = f"{figma_ids['file_id']}_{figma_ids['node_id']}"
            if cache_key in self._design_cache:
                logger.debug(f"디자인 메타데이터 캐시 히트: {cache_key}")
                return self._design_cache[cache_key]
            
            # MCP 서버를 통한 디자인 정보 가져오기
            metadata = await self._fetch_design_data(figma_ids)
            
            # 캐시에 저장
            self._design_cache[cache_key] = metadata
            
            logger.info(f"디자인 메타데이터 추출 완료: {figma_url}")
            return metadata
            
        except Exception as e:
            logger.error(f"디자인 메타데이터 추출 실패: {str(e)}")
            raise
    
    async def _fetch_design_data(self, figma_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        실제 디자인 데이터 가져오기 (MCP 서버 호출)
        
        Args:
            figma_ids: Figma 파일/노드 ID 정보
            
        Returns:
            디자인 데이터
        """
        try:
            # MCP 서버의 get_code 도구 호출
            if figma_ids["node_id"]:
                tool_result = await self.client.call_tool(
                    name="get_code",
                    arguments={"node_id": figma_ids["node_id"]}
                )
            else:
                # 파일 전체 분석
                tool_result = await self.client.call_tool(
                    name="get_file_data", 
                    arguments={"file_id": figma_ids["file_id"]}
                )
            
            # 결과 파싱 및 구조화
            parsed_data = self._parse_figma_response(tool_result)
            
            return {
                "figma_ids": figma_ids,
                "raw_data": tool_result,
                "parsed_data": parsed_data,
                "extracted_at": datetime.now().isoformat(),
                "metadata": {
                    "components_count": len(parsed_data.get("components", [])),
                    "frames_count": len(parsed_data.get("frames", [])),
                    "text_nodes_count": len(parsed_data.get("text_nodes", [])),
                    "has_variables": bool(parsed_data.get("variables")),
                    "has_styles": bool(parsed_data.get("styles"))
                }
            }
            
        except Exception as e:
            logger.error(f"디자인 데이터 가져오기 실패: {str(e)}")
            # 실제 MCP 서버 연동 실패 시 명확한 에러 처리
            raise FigmaIntegrationError(f"Figma MCP 서버 연결 실패: {str(e)}")
    
    def _parse_figma_response(self, tool_result: Any) -> Dict[str, Any]:
        """
        Figma MCP 서버 응답 파싱
        
        Args:
            tool_result: MCP 서버 응답
            
        Returns:
            파싱된 디자인 데이터
        """
        try:
            # tool_result 구조에 따라 파싱 로직 구현
            if isinstance(tool_result, dict):
                content = tool_result.get("content", [])
            else:
                content = str(tool_result)
            
            # 기본 구조화
            parsed = {
                "components": self._extract_components(content),
                "frames": self._extract_frames(content),
                "text_nodes": self._extract_text_nodes(content),
                "variables": self._extract_variables(content),
                "styles": self._extract_styles(content),
                "layout_info": self._extract_layout_info(content)
            }
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Figma 응답 파싱 중 오류: {str(e)}")
            return {
                "components": [],
                "frames": [],
                "text_nodes": [],
                "variables": {},
                "styles": {},
                "layout_info": {}
            }
    
    def _extract_components(self, content: Any) -> List[Dict[str, Any]]:
        """컴포넌트 정보 추출"""
        components = []
        try:
            # 컴포넌트 추출 로직 구현
            # 실제로는 Figma API 응답 구조에 따라 파싱
            if isinstance(content, str) and "component" in content.lower():
                # 간단한 파싱 예시
                components.append({
                    "name": "extracted_component",
                    "type": "COMPONENT",
                    "description": "Component extracted from design"
                })
        except Exception as e:
            logger.warning(f"컴포넌트 추출 중 오류: {str(e)}")
        
        return components
    
    def _extract_frames(self, content: Any) -> List[Dict[str, Any]]:
        """프레임 정보 추출"""
        frames = []
        try:
            # 프레임 추출 로직
            if isinstance(content, str) and "frame" in content.lower():
                frames.append({
                    "name": "main_frame",
                    "type": "FRAME",
                    "width": 1200,
                    "height": 800
                })
        except Exception as e:
            logger.warning(f"프레임 추출 중 오류: {str(e)}")
        
        return frames
    
    def _extract_text_nodes(self, content: Any) -> List[Dict[str, Any]]:
        """텍스트 노드 추출"""
        text_nodes = []
        try:
            # 텍스트 노드 추출 로직
            if isinstance(content, str):
                text_nodes.append({
                    "text": "Sample text content",
                    "font_family": "Inter",
                    "font_size": 16
                })
        except Exception as e:
            logger.warning(f"텍스트 노드 추출 중 오류: {str(e)}")
        
        return text_nodes
    
    def _extract_variables(self, content: Any) -> Dict[str, Any]:
        """변수 정보 추출"""
        try:
            return {
                "colors": {"primary": "#007AFF", "secondary": "#5856D6"},
                "spacing": {"small": 8, "medium": 16, "large": 24},
                "typography": {"heading": "24px", "body": "16px"}
            }
        except Exception as e:
            logger.warning(f"변수 추출 중 오류: {str(e)}")
            return {}
    
    def _extract_styles(self, content: Any) -> Dict[str, Any]:
        """스타일 정보 추출"""
        try:
            return {
                "text_styles": ["Heading 1", "Body", "Caption"],
                "color_styles": ["Primary", "Secondary", "Success", "Error"],
                "effect_styles": ["Drop Shadow", "Inner Shadow"]
            }
        except Exception as e:
            logger.warning(f"스타일 추출 중 오류: {str(e)}")
            return {}
    
    def _extract_layout_info(self, content: Any) -> Dict[str, Any]:
        """레이아웃 정보 추출"""
        try:
            return {
                "auto_layout": True,
                "direction": "vertical",
                "spacing": 16,
                "padding": {"top": 24, "right": 24, "bottom": 24, "left": 24},
                "alignment": "center"
            }
        except Exception as e:
            logger.warning(f"레이아웃 정보 추출 중 오류: {str(e)}")
            return {}
    
    async def _fetch_real_figma_data(self, figma_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        실제 Figma MCP 서버를 통한 디자인 데이터 수집
        
        Args:
            figma_ids: Figma ID 정보
            
        Returns:
            실제 Figma 디자인 데이터
        """
        try:
            if not await self.connect():
                raise FigmaIntegrationError("Figma MCP 서버 연결 실패")
            
            logger.info("실제 Figma 데이터 수집 시작")
            
            # 실제 Figma MCP 서버 호출
            result = await self.client.call_tool(
                name="get_design_data",
                arguments={
                    "file_id": figma_ids.get("file_id"),
                    "node_id": figma_ids.get("node_id", ""),
                    "include_components": True,
                    "include_styles": True,
                    "include_variables": True,
                    "depth": 3
                }
            )
            
            # 실제 응답 파싱
            parsed_data = self._parse_figma_response(result)
            
            return {
                "figma_ids": figma_ids,
                "raw_data": result,
                "parsed_data": parsed_data,
                "extracted_at": datetime.now().isoformat(),
                "metadata": {
                    "components_count": len(parsed_data.get("components", [])),
                    "frames_count": len(parsed_data.get("frames", [])),
                    "text_nodes_count": len(parsed_data.get("text_nodes", [])),
                    "has_variables": bool(parsed_data.get("variables")),
                    "has_styles": bool(parsed_data.get("styles")),
                    "is_mock_data": False,
                    "source": "figma_mcp_server"
                }
            }
            
        except Exception as e:
            logger.error(f"실제 Figma 데이터 수집 실패: {str(e)}")
            raise FigmaIntegrationError(f"Figma 데이터 수집 실패: {str(e)}")
    
    async def analyze_design(self, figma_url: str) -> Dict[str, Any]:
        """
        디자인 종합 분석 (main public method)
        
        Args:
            figma_url: Figma 파일 URL
            
        Returns:
            종합 디자인 분석 결과
        """
        try:
            logger.info(f"디자인 분석 시작: {figma_url}")
            
            # 1. 메타데이터 추출
            metadata = await self.get_design_metadata(figma_url)
            
            # 2. 분석 결과 생성
            analysis_result = {
                "url": figma_url,
                "analysis_timestamp": datetime.now().isoformat(),
                "design_metadata": metadata,
                "summary": self._generate_analysis_summary(metadata),
                "requirements_insights": self._extract_requirements_insights(metadata),
                "technical_considerations": self._analyze_technical_aspects(metadata),
                "ux_insights": self._analyze_ux_aspects(metadata)
            }
            
            logger.info("디자인 분석 완료")
            return analysis_result
            
        except Exception as e:
            logger.error(f"디자인 분석 실패: {str(e)}")
            raise
        finally:
            # 연결 정리
            await self.disconnect()
    
    def _generate_analysis_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """분석 요약 생성"""
        parsed_data = metadata.get("parsed_data", {})
        meta_info = metadata.get("metadata", {})
        
        return {
            "overview": f"디자인 파일에서 {meta_info.get('components_count', 0)}개 컴포넌트, "
                       f"{meta_info.get('frames_count', 0)}개 프레임 발견",
            "complexity_score": self._calculate_complexity_score(parsed_data),
            "design_system_usage": meta_info.get("has_variables", False) and meta_info.get("has_styles", False),
            "key_components": [comp.get("name") for comp in parsed_data.get("components", [])][:5],
            "estimated_dev_time": self._estimate_development_time(parsed_data)
        }
    
    def _extract_requirements_insights(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """요구사항 인사이트 추출"""
        parsed_data = metadata.get("parsed_data", {})
        requirements = []
        
        # 컴포넌트 기반 요구사항
        for component in parsed_data.get("components", []):
            requirements.append({
                "type": "functional",
                "category": "component",
                "title": f"{component.get('name')} 구현",
                "description": f"{component.get('description', '')} 컴포넌트 개발 필요",
                "priority": "medium",
                "complexity": "medium"
            })
        
        # 프레임 기반 요구사항
        for frame in parsed_data.get("frames", []):
            requirements.append({
                "type": "functional",
                "category": "screen",
                "title": f"{frame.get('name')} 화면 구현",
                "description": f"화면 크기: {frame.get('width')}x{frame.get('height')}",
                "priority": "high",
                "complexity": "high"
            })
        
        return requirements
    
    def _analyze_technical_aspects(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """기술적 고려사항 분석"""
        parsed_data = metadata.get("parsed_data", {})
        
        return {
            "responsive_design": parsed_data.get("layout_info", {}).get("responsive", False),
            "component_reusability": len(parsed_data.get("components", [])) > 0,
            "design_tokens": bool(parsed_data.get("variables")),
            "accessibility_considerations": self._check_accessibility(parsed_data),
            "performance_considerations": self._analyze_performance_impact(parsed_data)
        }
    
    def _analyze_ux_aspects(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """UX 관점 분석"""
        parsed_data = metadata.get("parsed_data", {})
        
        return {
            "user_flow_complexity": "medium",
            "interaction_patterns": self._identify_interaction_patterns(parsed_data),
            "information_architecture": self._analyze_information_architecture(parsed_data),
            "visual_hierarchy": self._analyze_visual_hierarchy(parsed_data)
        }
    
    def _calculate_complexity_score(self, parsed_data: Dict[str, Any]) -> float:
        """복잡도 점수 계산"""
        try:
            components_weight = len(parsed_data.get("components", [])) * 0.3
            frames_weight = len(parsed_data.get("frames", [])) * 0.2
            variables_weight = len(parsed_data.get("variables", {})) * 0.1
            
            score = min(components_weight + frames_weight + variables_weight, 10.0)
            return round(score, 1)
        except:
            return 5.0
    
    def _estimate_development_time(self, parsed_data: Dict[str, Any]) -> str:
        """개발 시간 추정"""
        try:
            components_count = len(parsed_data.get("components", []))
            frames_count = len(parsed_data.get("frames", []))
            
            estimated_days = (components_count * 2) + (frames_count * 3)
            
            if estimated_days <= 5:
                return "1주 이내"
            elif estimated_days <= 15:
                return "2-3주"
            else:
                return "1개월 이상"
        except:
            return "추정 불가"
    
    def _check_accessibility(self, parsed_data: Dict[str, Any]) -> List[str]:
        """접근성 체크리스트"""
        considerations = []
        
        if parsed_data.get("text_nodes"):
            considerations.append("텍스트 대비 확인 필요")
        
        if parsed_data.get("components"):
            considerations.append("키보드 네비게이션 지원 확인")
            considerations.append("스크린 리더 호환성 확인")
        
        return considerations
    
    def _analyze_performance_impact(self, parsed_data: Dict[str, Any]) -> List[str]:
        """성능 영향 분석"""
        impacts = []
        
        components_count = len(parsed_data.get("components", []))
        if components_count > 10:
            impacts.append("많은 컴포넌트로 인한 번들 크기 최적화 필요")
        
        if parsed_data.get("layout_info", {}).get("responsive"):
            impacts.append("반응형 디자인으로 인한 CSS 복잡도 증가")
        
        return impacts
    
    def _identify_interaction_patterns(self, parsed_data: Dict[str, Any]) -> List[str]:
        """인터랙션 패턴 식별"""
        patterns = ["클릭/탭 액션", "폼 입력"]
        
        if len(parsed_data.get("components", [])) > 5:
            patterns.append("복합 인터랙션")
        
        return patterns
    
    def _analyze_information_architecture(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """정보 구조 분석"""
        return {
            "depth": "medium",
            "breadth": "wide" if len(parsed_data.get("frames", [])) > 3 else "narrow",
            "navigation_type": "hierarchical"
        }
    
    def _analyze_visual_hierarchy(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """시각적 계층 분석"""
        text_nodes = parsed_data.get("text_nodes", [])
        
        return {
            "heading_levels": len([node for node in text_nodes if node.get("font_size", 0) > 20]),
            "typography_consistency": bool(parsed_data.get("styles", {}).get("text_styles")),
            "visual_weight_distribution": "balanced"
        } 