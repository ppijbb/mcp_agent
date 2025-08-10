"""
Figma Integration
Read operations use the Figma REST API where applicable. "Creation" helpers below
produce spec-only structures describing intended nodes/layouts. They do not attempt
to write to Figma (the Figma REST API does not support arbitrary node creation).
"""

import aiohttp
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class FigmaComponent:
    """Figma 컴포넌트 데이터 클래스"""
    type: str  # 'rectangle', 'text', 'button', 'input'
    x: float
    y: float
    width: float
    height: float
    content: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None

class FigmaIntegration:
    """Figma REST API 통합 클래스"""
    
    def __init__(self):
        self.access_token = os.getenv('FIGMA_ACCESS_TOKEN')
        self.file_key = os.getenv('FIGMA_FILE_KEY')
        self.base_url = "https://api.figma.com/v1"
        
        # 환경변수가 없어도 스펙 전용 모드로 동작 (쓰기 호출은 수행되지 않음)
        if not self.access_token:
            print("⚠️ FIGMA_ACCESS_TOKEN 환경변수가 설정되지 않았습니다. 스펙 전용 모드로 동작합니다.")
        if not self.file_key:
            print("⚠️ FIGMA_FILE_KEY 환경변수가 설정되지 않았습니다. 스펙 전용 모드로 동작합니다.")
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Figma API 요청 공통 메서드"""
        headers = {
            "X-Figma-Token": self.access_token,
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url, headers=headers) as response:
                    return await response.json()
            elif method.upper() == "POST":
                async with session.post(url, headers=headers, json=data) as response:
                    return await response.json()
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
    
    async def get_file_info(self) -> Dict:
        """Figma 파일 정보 조회"""
        return await self._make_request("GET", f"/files/{self.file_key}")
    
    async def get_file_nodes(self, node_ids: List[str]) -> Dict:
        """특정 노드 정보 조회"""
        node_ids_str = ",".join(node_ids)
        return await self._make_request("GET", f"/files/{self.file_key}/nodes?ids={node_ids_str}")
    
    async def create_rectangle(self, x: float, y: float, width: float, height: float, 
                             fill_color: str = "#E1E5E9", corner_radius: float = 0) -> Dict:
        """사각형 노드 스펙 생성 (Figma에 기록하지 않음)"""
        node_data = {
            "name": "Rectangle",
            "type": "RECTANGLE",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "fills": [{"type": "SOLID", "color": self._hex_to_rgb(fill_color)}],
            "cornerRadius": corner_radius
        }
        
        return await self._create_node(node_data)
    
    async def create_text(self, x: float, y: float, content: str, 
                         font_size: float = 14, font_family: str = "Inter",
                         color: str = "#000000", width: Optional[float] = None) -> Dict:
        """텍스트 노드 스펙 생성 (Figma에 기록하지 않음)"""
        node_data = {
            "name": "Text",
            "type": "TEXT",
            "x": x,
            "y": y,
            "width": width or len(content) * font_size * 0.6,  # 텍스트 길이에 따른 자동 너비
            "height": font_size * 1.2,
            "characters": content,
            "style": {
                "fontFamily": font_family,
                "fontSize": font_size,
                "fontWeight": 400,
                "textAlignHorizontal": "LEFT",
                "textAlignVertical": "TOP"
            },
            "fills": [{"type": "SOLID", "color": self._hex_to_rgb(color)}]
        }
        
        return await self._create_node(node_data)
    
    async def create_button(self, x: float, y: float, text: str, 
                           width: float = 120, height: float = 40,
                           bg_color: str = "#007AFF", text_color: str = "#FFFFFF",
                           corner_radius: float = 8) -> Dict:
        """버튼 컴포넌트 스펙 생성 (배경 + 텍스트, Figma에 기록하지 않음)"""
        # 배경 사각형 생성
        bg_rect = await self.create_rectangle(
            x=x, y=y, width=width, height=height,
            fill_color=bg_color, corner_radius=corner_radius
        )
        
        # 텍스트 생성 (버튼 중앙에 배치)
        text_x = x + (width - len(text) * 12) / 2  # 텍스트 중앙 정렬
        text_y = y + (height - 14) / 2
        
        text_node = await self.create_text(
            x=text_x, y=text_y, content=text,
            font_size=14, color=text_color, width=len(text) * 12
        )
        
        return {
            "background": bg_rect,
            "text": text_node,
            "type": "button",
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
    
    async def create_input_field(self, x: float, y: float, placeholder: str = "입력하세요",
                                width: float = 200, height: float = 40,
                                border_color: str = "#CCCCCC", bg_color: str = "#FFFFFF") -> Dict:
        """입력 필드 컴포넌트 스펙 생성 (Figma에 기록하지 않음)"""
        # 배경 사각형 (테두리 포함)
        border_rect = await self.create_rectangle(
            x=x, y=y, width=width, height=height,
            fill_color=bg_color, corner_radius=4
        )
        
        # 테두리 효과 (stroke 추가)
        border_data = {
            "name": "Input Border",
            "type": "RECTANGLE",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "fills": [],
            "strokes": [{"type": "SOLID", "color": self._hex_to_rgb(border_color)}],
            "strokeWeight": 1,
            "cornerRadius": 4
        }
        
        border_node = await self._create_node(border_data)
        
        # 플레이스홀더 텍스트
        text_x = x + 12  # 좌측 패딩
        text_y = y + (height - 14) / 2
        
        placeholder_text = await self.create_text(
            x=text_x, y=text_y, content=placeholder,
            font_size=14, color="#999999", width=width - 24
        )
        
        return {
            "background": border_rect,
            "border": border_node,
            "placeholder": placeholder_text,
            "type": "input",
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
    
    async def create_card(self, x: float, y: float, title: str, content: str,
                         width: float = 300, height: float = 200,
                         bg_color: str = "#FFFFFF", shadow: bool = True) -> Dict:
        """카드 컴포넌트 스펙 생성 (Figma에 기록하지 않음)"""
        # 카드 배경
        card_bg = await self.create_rectangle(
            x=x, y=y, width=width, height=height,
            fill_color=bg_color, corner_radius=8
        )
        
        # 제목 텍스트
        title_text = await self.create_text(
            x=x + 16, y=y + 16, content=title,
            font_size=18, font_family="Inter", color="#000000",
            width=width - 32
        )
        
        # 내용 텍스트
        content_text = await self.create_text(
            x=x + 16, y=y + 50, content=content,
            font_size=14, font_family="Inter", color="#666666",
            width=width - 32
        )
        
        return {
            "background": card_bg,
            "title": title_text,
            "content": content_text,
            "type": "card",
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
    
    async def create_layout(self, components: List[FigmaComponent], 
                           start_x: float = 0, start_y: float = 0,
                           spacing: float = 20) -> Dict:
        """컴포넌트들을 레이아웃 스펙으로 배치 (쓰기 호출 없음)"""
        created_components = []
        current_x = start_x
        current_y = start_y
        max_height_in_row = 0
        
        for component in components:
            # 컴포넌트 타입에 따른 생성
            if component.type == "rectangle":
                created = await self.create_rectangle(
                    x=current_x, y=current_y,
                    width=component.width, height=component.height,
                    fill_color=component.style.get("fill_color", "#E1E5E9") if component.style else "#E1E5E9",
                    corner_radius=component.style.get("corner_radius", 0) if component.style else 0
                )
            elif component.type == "text":
                created = await self.create_text(
                    x=current_x, y=current_y,
                    content=component.content or "",
                    font_size=component.style.get("font_size", 14) if component.style else 14,
                    color=component.style.get("color", "#000000") if component.style else "#000000",
                    width=component.width
                )
            elif component.type == "button":
                created = await self.create_button(
                    x=current_x, y=current_y,
                    text=component.content or "Button",
                    width=component.width, height=component.height,
                    bg_color=component.style.get("bg_color", "#007AFF") if component.style else "#007AFF",
                    text_color=component.style.get("text_color", "#FFFFFF") if component.style else "#FFFFFF"
                )
            elif component.type == "input":
                created = await self.create_input_field(
                    x=current_x, y=current_y,
                    placeholder=component.content or "입력하세요",
                    width=component.width, height=component.height
                )
            else:
                # 기본 사각형으로 생성
                created = await self.create_rectangle(
                    x=current_x, y=current_y,
                    width=component.width, height=component.height
                )
            
            created_components.append(created)
            
            # 다음 컴포넌트 위치 계산
            current_x += component.width + spacing
            max_height_in_row = max(max_height_in_row, component.height)
            
            # 줄바꿈 (너무 길어지면)
            if current_x > start_x + 800:  # 최대 너비
                current_x = start_x
                current_y += max_height_in_row + spacing
                max_height_in_row = 0
        
        return {
            "components": created_components,
            "layout": {
                "start_x": start_x,
                "start_y": start_y,
                "total_width": current_x - start_x,
                "total_height": current_y + max_height_in_row - start_y
            }
        }
    
    async def _create_node(self, node_data: Dict) -> Dict:
        """노드 스펙 생성 (Figma에 기록하지 않음; ID를 생성하지 않음)"""
        return {
            "spec_only": True,
            "name": node_data.get("name", "Node"),
            "type": node_data.get("type", "RECTANGLE"),
            "position": {"x": node_data.get("x", 0), "y": node_data.get("y", 0)},
            "size": {"width": node_data.get("width", 100), "height": node_data.get("height", 100)},
            "data": node_data,
        }
    
    def _hex_to_rgb(self, hex_color: str) -> Dict[str, float]:
        """16진수 색상을 RGB로 변환"""
        hex_color = hex_color.lstrip('#')
        return {
            "r": int(hex_color[0:2], 16) / 255,
            "g": int(hex_color[2:4], 16) / 255,
            "b": int(hex_color[4:6], 16) / 255
        }

# 기존 함수들 (하위 호환성 유지)
async def create_rectangles_on_canvas(rectangles_data: List[Dict]) -> Dict:
    """사각형들을 캔버스에 생성 (기존 함수)"""
    figma = FigmaIntegration()
    components = []
    
    for rect_data in rectangles_data:
        component = FigmaComponent(
            type="rectangle",
            x=rect_data.get("x", 0),
            y=rect_data.get("y", 0),
            width=rect_data.get("width", 100),
            height=rect_data.get("height", 100),
            style={"fill_color": rect_data.get("fill_color", "#E1E5E9")}
        )
        components.append(component)
    
    return await figma.create_layout(components)

async def create_ui_components(components_data: List[Dict]) -> Dict:
    """UI 컴포넌트들을 생성"""
    figma = FigmaIntegration()
    components = []
    
    for comp_data in components_data:
        component = FigmaComponent(
            type=comp_data.get("type", "rectangle"),
            x=comp_data.get("x", 0),
            y=comp_data.get("y", 0),
            width=comp_data.get("width", 100),
            height=comp_data.get("height", 100),
            content=comp_data.get("content"),
            style=comp_data.get("style", {}),
            properties=comp_data.get("properties", {})
        )
        components.append(component)
    
    return await figma.create_layout(components) 