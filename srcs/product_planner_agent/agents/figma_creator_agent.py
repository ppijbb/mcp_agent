"""
Figma Creator Agent
PRD 요구사항을 바탕으로 Figma에서 직접 디자인을 생성하는 Agent
"""
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from srcs.product_planner_agent.integrations.figma_integration import FigmaIntegration, FigmaComponent
from srcs.product_planner_agent.utils.logger import get_product_planner_logger
from srcs.product_planner_agent.agents.base_agent_simple import BaseAgentSimple as BaseAgent

logger = get_product_planner_logger("agents.figma_creator")

@dataclass
class UIComponentSpec:
    """UI 컴포넌트 명세"""
    type: str
    content: Optional[str] = None
    width: float = 100
    height: float = 100
    style: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None

class FigmaCreatorAgent(BaseAgent):
    """Figma 컴포넌트 생성 에이전트"""
    
    def __init__(self):
        super().__init__()
        self.figma_integration = FigmaIntegration()
        self.logger = logger
    
    async def run_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Figma 컴포넌트 생성 워크플로우 실행
        
        Args:
            input_data: 입력 데이터 (PRD 내용, 컴포넌트 명세 등)
            
        Returns:
            생성된 컴포넌트 정보
        """
        try:
            self.logger.info("Figma 컴포넌트 생성 워크플로우 시작")
            
            # PRD에서 UI 컴포넌트 추출
            prd_content = input_data.get("prd_content", "")
            components = await self._extract_components_from_prd(prd_content)
            
            # 컴포넌트 레이아웃 생성
            layout_result = await self._create_component_layout(components)
            
            # 결과 반환 (spec-only)
            result = {
                "status": "success",
                "components_spec_count": len(components),
                "layout_spec": layout_result,
                "components_spec": components
            }
            
            self.logger.info(f"Figma 컴포넌트 생성 완료: {len(components)}개 컴포넌트")
            return result
            
        except Exception as e:
            self.logger.error(f"Figma 컴포넌트 생성 실패: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "components_created": 0
            }
    
    async def _extract_components_from_prd(self, prd_content: str) -> List[UIComponentSpec]:
        """PRD 내용에서 UI 컴포넌트 추출"""
        components = []
        
        # 기본 UI 컴포넌트 패턴 매칭
        import re
        
        # 버튼 패턴
        button_patterns = [
            r'버튼[:\s]*([^\n]+)',
            r'button[:\s]*([^\n]+)',
            r'클릭[:\s]*([^\n]+)',
            r'submit[:\s]*([^\n]+)',
            r'확인[:\s]*([^\n]+)',
            r'취소[:\s]*([^\n]+)'
        ]
        
        for pattern in button_patterns:
            matches = re.findall(pattern, prd_content, re.IGNORECASE)
            for match in matches:
                components.append(UIComponentSpec(
                    type="button",
                    content=match.strip(),
                    width=120,
                    height=40,
                    style={
                        "bg_color": "#007AFF",
                        "text_color": "#FFFFFF",
                        "corner_radius": 8
                    }
                ))
        
        # 입력 필드 패턴
        input_patterns = [
            r'입력[:\s]*([^\n]+)',
            r'input[:\s]*([^\n]+)',
            r'텍스트[:\s]*([^\n]+)',
            r'검색[:\s]*([^\n]+)',
            r'이름[:\s]*([^\n]+)',
            r'이메일[:\s]*([^\n]+)',
            r'비밀번호[:\s]*([^\n]+)'
        ]
        
        for pattern in input_patterns:
            matches = re.findall(pattern, prd_content, re.IGNORECASE)
            for match in matches:
                components.append(UIComponentSpec(
                    type="input",
                    content=match.strip(),
                    width=200,
                    height=40,
                    style={
                        "border_color": "#CCCCCC",
                        "bg_color": "#FFFFFF"
                    }
                ))
        
        # 텍스트 패턴
        text_patterns = [
            r'제목[:\s]*([^\n]+)',
            r'title[:\s]*([^\n]+)',
            r'설명[:\s]*([^\n]+)',
            r'description[:\s]*([^\n]+)',
            r'라벨[:\s]*([^\n]+)',
            r'label[:\s]*([^\n]+)'
        ]
        
        for pattern in text_patterns:
            matches = re.findall(pattern, prd_content, re.IGNORECASE)
            for match in matches:
                components.append(UIComponentSpec(
                    type="text",
                    content=match.strip(),
                    width=len(match.strip()) * 12,
                    height=20,
                    style={
                        "font_size": 16,
                        "color": "#000000",
                        "font_family": "Inter"
                    }
                ))
        
        # 카드 패턴
        card_patterns = [
            r'카드[:\s]*([^\n]+)',
            r'card[:\s]*([^\n]+)',
            r'아이템[:\s]*([^\n]+)',
            r'item[:\s]*([^\n]+)'
        ]
        
        for pattern in card_patterns:
            matches = re.findall(pattern, prd_content, re.IGNORECASE)
            for match in matches:
                components.append(UIComponentSpec(
                    type="card",
                    content=match.strip(),
                    width=300,
                    height=200,
                    style={
                        "bg_color": "#FFFFFF",
                        "shadow": True
                    }
                ))
        
        # 컴포넌트가 없는 경우에도 더미를 생성하지 않습니다 (no-fallback policy)
        
        self.logger.info(f"PRD에서 {len(components)}개 컴포넌트 추출")
        return components
    
    async def _create_component_layout(self, components: List[UIComponentSpec]) -> Dict[str, Any]:
        """컴포넌트들을 레이아웃으로 배치하여 생성"""
        try:
            # UIComponentSpec을 FigmaComponent로 변환
            figma_components = []
            
            for i, comp in enumerate(components):
                figma_comp = FigmaComponent(
                    type=comp.type,
                    x=0,  # 레이아웃에서 자동 계산
                    y=0,
                    width=comp.width,
                    height=comp.height,
                    content=comp.content,
                    style=comp.style,
                    properties=comp.properties
                )
                figma_components.append(figma_comp)
            
            # 레이아웃 생성
            layout_result = await self.figma_integration.create_layout(
                components=figma_components,
                start_x=50,
                start_y=50,
                spacing=20
            )
            
            return layout_result
            
        except Exception as e:
            self.logger.error(f"컴포넌트 레이아웃 생성 실패: {str(e)}")
            raise
    
    async def create_specific_components(self, component_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """특정 컴포넌트 명세에 따라 컴포넌트 생성"""
        try:
            components = []
            
            for spec in component_specs:
                component = UIComponentSpec(
                    type=spec.get("type", "rectangle"),
                    content=spec.get("content"),
                    width=spec.get("width", 100),
                    height=spec.get("height", 100),
                    style=spec.get("style", {}),
                    properties=spec.get("properties", {})
                )
                components.append(component)
            
            layout_result = await self._create_component_layout(components)
            
            return {
                "status": "success",
                "components_spec_count": len(components),
                "layout_spec": layout_result
            }
            
        except Exception as e:
            self.logger.error(f"특정 컴포넌트 생성 실패: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def create_mobile_app_layout(self, app_name: str, features: List[str]) -> Dict[str, Any]:
        """모바일 앱 레이아웃 생성"""
        try:
            components = []
            
            # 앱 제목
            components.append(UIComponentSpec(
                type="text",
                content=app_name,
                width=300,
                height=40,
                style={
                    "font_size": 24,
                    "color": "#000000",
                    "font_family": "Inter"
                }
            ))
            
            # 기능 버튼들
            for i, feature in enumerate(features[:5]):  # 최대 5개 기능
                components.append(UIComponentSpec(
                    type="button",
                    content=feature,
                    width=200,
                    height=50,
                    style={
                        "bg_color": "#007AFF",
                        "text_color": "#FFFFFF",
                        "corner_radius": 25
                    }
                ))
            
            # 검색 입력
            components.append(UIComponentSpec(
                type="input",
                content="검색어를 입력하세요",
                width=250,
                height=40,
                style={
                    "border_color": "#CCCCCC",
                    "bg_color": "#FFFFFF"
                }
            ))
            
            layout_result = await self._create_component_layout(components)
            
            return {
                "status": "success",
                "app_name": app_name,
                "components_spec_count": len(components),
                "layout_spec": layout_result
            }
            
        except Exception as e:
            self.logger.error(f"모바일 앱 레이아웃 생성 실패: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def create_web_dashboard_layout(self, dashboard_title: str, widgets: List[str]) -> Dict[str, Any]:
        """웹 대시보드 레이아웃 생성"""
        try:
            components = []
            
            # 대시보드 제목
            components.append(UIComponentSpec(
                type="text",
                content=dashboard_title,
                width=400,
                height=50,
                style={
                    "font_size": 28,
                    "color": "#000000",
                    "font_family": "Inter"
                }
            ))
            
            # 위젯 카드들
            for i, widget in enumerate(widgets[:6]):  # 최대 6개 위젯
                components.append(UIComponentSpec(
                    type="card",
                    content=widget,
                    width=250,
                    height=150,
                    style={
                        "bg_color": "#FFFFFF",
                        "shadow": True
                    }
                ))
            
            # 네비게이션 버튼
            components.append(UIComponentSpec(
                type="button",
                content="새로고침",
                width=100,
                height=35,
                style={
                    "bg_color": "#28A745",
                    "text_color": "#FFFFFF",
                    "corner_radius": 6
                }
            ))
            
            layout_result = await self._create_component_layout(components)
            
            return {
                "status": "success",
                "dashboard_title": dashboard_title,
                "components_spec_count": len(components),
                "layout_spec": layout_result
            }
            
        except Exception as e:
            self.logger.error(f"웹 대시보드 레이아웃 생성 실패: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 