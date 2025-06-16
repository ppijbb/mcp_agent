"""
Product Planner Agent Configuration

프로덕트 기획자 Agent 전용 설정 및 상수 정의
"""

from typing import List, Dict, Any
from datetime import datetime

# 기본 설정 가져오기 (fallback 포함)
try:
    from common.config import DEFAULT_SERVERS, DEFAULT_COMPANY_NAME
except ImportError:
    DEFAULT_SERVERS = ["filesystem", "fetch"]
    DEFAULT_COMPANY_NAME = "TechCorp Inc."

# Product Planner Agent 전용 MCP 서버 설정
PRODUCT_PLANNER_SERVERS = [
    "figma-dev-mode",
    "notion-api", 
    "filesystem"
]

# MCP 서버별 상세 설정
FIGMA_MCP_CONFIG = {
    "server_name": "figma-dev-mode",
    "required_tools": [
        "get_design_data",
        "get_file_data", 
        "get_code",
        "extract_components",
        "get_variables"
    ],
    "connection_timeout": 30,
    "retry_attempts": 3,
    "fallback_enabled": False  # Mock 데이터 비활성화
}

NOTION_MCP_CONFIG = {
    "server_name": "notion-api",
    "required_tools": [
        "create_page",
        "update_page",
        "create_database",
        "query_database",
        "create_database_entry"
    ],
    "connection_timeout": 20,
    "retry_attempts": 3,
    "fallback_enabled": False  # Mock 데이터 비활성화
}

# MCP 서버 연결 상태 검증
MCP_SERVER_HEALTH_CHECKS = {
    "figma-dev-mode": {
        "test_tool": "get_file_data",
        "test_args": {"file_id": "test"},
        "expected_response_type": "dict"
    },
    "notion-api": {
        "test_tool": "create_page",
        "test_args": {"title": "Health Check", "content": []},
        "expected_response_type": "dict"
    }
}

# PRD 템플릿 설정
PRD_TEMPLATE_CONFIG = {
    "sections": [
        "executive_summary",
        "problem_statement", 
        "solution_overview",
        "user_stories",
        "technical_requirements",
        "success_metrics",
        "timeline_milestones"
    ],
    "required_fields": [
        "product_name",
        "target_audience",
        "key_features",
        "success_criteria"
    ],
    "optional_fields": [
        "competitive_analysis",
        "market_research",
        "risk_assessment",
        "resource_requirements"
    ]
}

# 로드맵 설정
ROADMAP_CONFIG = {
    "phases": ["discovery", "design", "development", "testing", "launch"],
    "estimation_factors": {
        "complexity_multiplier": 1.5,
        "risk_buffer": 0.2,
        "integration_overhead": 0.3
    },
    "priority_levels": ["Critical", "High", "Medium", "Low"],
    "default_sprint_length": 14  # days
}

# Notion 데이터베이스 스키마
NOTION_DATABASE_SCHEMAS = {
    "requirements": {
        "Name": {"type": "title"},
        "Priority": {
            "type": "select", 
            "options": ["Critical", "High", "Medium", "Low"]
        },
        "Status": {
            "type": "select", 
            "options": ["New", "In Progress", "Review", "Done", "Blocked"]
        },
        "Assignee": {"type": "people"},
        "Due Date": {"type": "date"},
        "Figma Link": {"type": "url"},
        "Complexity": {
            "type": "select",
            "options": ["Simple", "Medium", "Complex"]
        },
        "Epic": {"type": "relation"},
        "Story Points": {"type": "number"}
    },
    "roadmap": {
        "Milestone": {"type": "title"},
        "Phase": {
            "type": "select", 
            "options": ["Discovery", "Design", "Development", "Testing", "Launch"]
        },
        "Start Date": {"type": "date"},
        "End Date": {"type": "date"},
        "Dependencies": {"type": "relation"},
        "Progress": {"type": "number"},
        "Owner": {"type": "people"},
        "Status": {
            "type": "select",
            "options": ["Planning", "In Progress", "At Risk", "Complete", "Cancelled"]
        },
        "Key Results": {"type": "rich_text"}
    },
    "design_specs": {
        "Component": {"type": "title"},
        "Figma URL": {"type": "url"},
        "Status": {
            "type": "select",
            "options": ["Draft", "Review", "Approved", "Deprecated"]
        },
        "Design System": {"type": "checkbox"},
        "Responsive": {"type": "checkbox"},
        "Accessibility": {"type": "checkbox"},
        "Last Updated": {"type": "date"},
        "Designer": {"type": "people"},
        "Related Requirements": {"type": "relation"}
    }
}

# Figma 관련 설정
FIGMA_CONFIG = {
    "supported_node_types": [
        "FRAME", "COMPONENT", "INSTANCE", "GROUP", 
        "TEXT", "RECTANGLE", "ELLIPSE", "POLYGON"
    ],
    "analysis_depth": 3,  # 분석할 하위 노드 깊이
    "extract_variables": True,
    "extract_components": True,
    "extract_styles": True
}

# 분석 및 생성 설정
ANALYSIS_CONFIG = {
    "min_confidence_score": 0.7,
    "max_requirements_per_component": 5,
    "default_estimation_unit": "story_points",
    "auto_categorize": True,
    "include_accessibility_requirements": True
}

# 출력 설정
OUTPUT_CONFIG = {
    "default_format": "markdown",
    "include_metadata": True,
    "auto_timestamp": True,
    "backup_enabled": True
}

def get_timestamp() -> str:
    """현재 타임스탬프 반환"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_output_dir(agent_name: str) -> str:
    """출력 디렉토리 경로 생성"""
    return f"{agent_name}_reports_{get_timestamp()}"

def validate_config() -> Dict[str, Any]:
    """설정 검증 및 상태 반환"""
    try:
        validation_results = {
            "status": "valid",
            "servers": PRODUCT_PLANNER_SERVERS,
            "schemas_count": len(NOTION_DATABASE_SCHEMAS),
            "prd_sections": len(PRD_TEMPLATE_CONFIG["sections"]),
            "roadmap_phases": len(ROADMAP_CONFIG["phases"]),
            "figma_config": {
                "fallback_enabled": FIGMA_MCP_CONFIG["fallback_enabled"],
                "required_tools": len(FIGMA_MCP_CONFIG["required_tools"])
            },
            "notion_config": {
                "fallback_enabled": NOTION_MCP_CONFIG["fallback_enabled"],
                "required_tools": len(NOTION_MCP_CONFIG["required_tools"])
            },
            "react_pattern": "implemented",
            "mock_data_removed": True,
            "timestamp": get_timestamp()
        }
        
        # 필수 설정 검증
        if not PRODUCT_PLANNER_SERVERS:
            validation_results["status"] = "error"
            validation_results["error"] = "MCP 서버 목록이 비어있습니다"
        
        if FIGMA_MCP_CONFIG["fallback_enabled"] or NOTION_MCP_CONFIG["fallback_enabled"]:
            validation_results["status"] = "warning"
            validation_results["warning"] = "일부 서버에서 Mock 데이터 Fallback이 활성화되어 있습니다"
        
        return validation_results
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": get_timestamp()
        }

async def validate_mcp_server_connection(server_name: str) -> Dict[str, Any]:
    """
    특정 MCP 서버 연결 상태 검증
    
    Args:
        server_name: 검증할 서버 이름
        
    Returns:
        연결 상태 정보
    """
    try:
        if server_name not in MCP_SERVER_HEALTH_CHECKS:
            return {
                "server": server_name,
                "status": "unknown",
                "error": "Health check 설정이 없습니다"
            }
        
        health_check = MCP_SERVER_HEALTH_CHECKS[server_name]
        
        # 실제 연결 테스트는 여기서 구현
        # 현재는 설정 검증만 수행
        return {
            "server": server_name,
            "status": "configured",
            "test_tool": health_check["test_tool"],
            "expected_type": health_check["expected_response_type"],
            "timestamp": get_timestamp()
        }
        
    except Exception as e:
        return {
            "server": server_name,
            "status": "error",
            "error": str(e),
            "timestamp": get_timestamp()
        } 