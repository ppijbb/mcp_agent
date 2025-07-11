"""
Figma Integration
Figma REST API와 통신하여 디자인을 생성, 수정, 조회하는 기능을 담당합니다.
"""

import requests
import uuid
from typing import List, Dict, Any, TypedDict

from srcs.product_planner_agent.utils import env_settings
from srcs.product_planner_agent.utils.logger import get_product_planner_logger
from srcs.core.errors import ExternalServiceError

logger = get_product_planner_logger("integration.figma")

# 환경 변수에서 API 키를 가져옵니다.
FIGMA_API_KEY = env_settings.get("FIGMA_API_KEY", required=True)
FIGMA_API_BASE_URL = "https://api.figma.com/v1"

class RGB(TypedDict):
    """Figma 색상 값을 위한 타입 정의"""
    r: float
    g: float
    b: float

class RectangleParams(TypedDict):
    """사각형 생성을 위한 파라미터 타입 정의"""
    name: str
    x: int
    y: int
    width: int
    height: int
    color: RGB

def create_rectangles_on_canvas(file_key: str, parent_node_id: str, rectangles: List[RectangleParams]) -> Dict[str, Any]:
    """
    Figma 캔버스에 여러 개의 사각형 노드를 생성합니다.

    Args:
        file_key: 대상 Figma 파일의 키
        parent_node_id: 사각형을 추가할 부모 노드의 ID
        rectangles: 생성할 사각형들의 속성 리스트

    Returns:
        Figma API의 JSON 응답
    
    Raises:
        ExternalServiceError: API 키가 없거나 API 요청이 실패했을 때 발생
    """
    if not FIGMA_API_KEY:
        raise ExternalServiceError("FIGMA_API_KEY가 환경 변수에 설정되지 않았습니다.")

    headers = {
        "X-Figma-Token": FIGMA_API_KEY,
        "Content-Type": "application/json",
    }

    nodes_to_create = []
    for rect in rectangles:
        node = {
            "id": str(uuid.uuid4()),
            "type": "RECTANGLE",
            "name": rect["name"],
            "x": rect["x"],
            "y": rect["y"],
            "width": rect["width"],
            "height": rect["height"],
            "fills": [{
                "type": "SOLID",
                "color": {**rect["color"], "a": 1},
            }],
        }
        nodes_to_create.append(node)

    payload = {
        "nodes": nodes_to_create,
        "parentNode": parent_node_id,
    }

    url = f"{FIGMA_API_BASE_URL}/files/{file_key}/nodes"
    
    try:
        logger.info(f"Figma API에 사각형 생성을 요청합니다: POST {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # 2xx 응답이 아닐 경우 예외 발생
        
        logger.info(f"{len(rectangles)}개의 사각형을 Figma 파일({file_key})에 성공적으로 생성했습니다.")
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = f"Figma API 요청 실패: {e}"
        if e.response is not None:
            error_message += f" - 응답: {e.response.text}"
        logger.error(error_message)
        raise ExternalServiceError(error_message) from e 