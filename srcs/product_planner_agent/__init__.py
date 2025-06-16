"""
Product Planner Agent Package

Figma와 Notion을 연동하여 프로덕트 기획 업무를 자동화하는 AI Agent 패키지
"""

from .product_planner_agent import ProductPlannerAgent
from .config import (
    PRODUCT_PLANNER_SERVERS,
    PRD_TEMPLATE_CONFIG,
    ROADMAP_CONFIG,
    NOTION_DATABASE_SCHEMAS
)

__version__ = "0.1.0"
__author__ = "MCP Agent Development Team"

__all__ = [
    "ProductPlannerAgent",
    "PRODUCT_PLANNER_SERVERS",
    "PRD_TEMPLATE_CONFIG", 
    "ROADMAP_CONFIG",
    "NOTION_DATABASE_SCHEMAS"
] 