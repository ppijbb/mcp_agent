"""
표준 A2A Page 헬퍼 함수

모든 pages에서 일관된 방식으로 agent를 호출하기 위한 표준화된 헬퍼 함수들
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from srcs.common.agent_interface import AgentType
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a, _normalize_entry_point, _detect_agent_type

logger = logging.getLogger(__name__)


def create_standard_agent_metadata(
    agent_id: str,
    agent_name: str,
    entry_point: str,
    agent_type: AgentType,
    capabilities: List[str],
    description: str
) -> Dict[str, Any]:
    """
    표준화된 agent_metadata 생성
    
    Args:
        agent_id: Agent 고유 ID
        agent_name: Agent 이름
        entry_point: 실행 경로 (모듈 경로 또는 CLI 명령)
        agent_type: Agent 타입 (AgentType enum)
        capabilities: Agent 능력 목록
        description: Agent 설명
        
    Returns:
        표준화된 agent_metadata 딕셔너리
    """
    # entry_point 정규화
    normalized_entry_point = _normalize_entry_point(entry_point)
    
    return {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "entry_point": normalized_entry_point,
        "agent_type": agent_type,  # AgentType enum 사용
        "capabilities": capabilities,
        "description": description
    }


def create_standard_input_data(
    agent_type: AgentType,
    entry_point: str,
    result_json_path: Optional[str] = None,
    class_name: Optional[str] = None,
    method_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Agent 타입에 따라 표준화된 input_data 생성
    
    Args:
        agent_type: Agent 타입 (AgentType enum)
        entry_point: 실행 경로
        result_json_path: 결과 JSON 파일 경로 (선택)
        class_name: 클래스 이름 (MCP Agent 클래스 기반인 경우)
        method_name: 메서드 이름 (MCP Agent 클래스/함수 기반인 경우)
        **kwargs: Agent별 추가 파라미터
        
    Returns:
        표준화된 input_data 딕셔너리
    """
    input_data = {}
    
    # entry_point 정규화
    normalized_entry_point = _normalize_entry_point(entry_point)
    
    if agent_type == AgentType.MCP_AGENT:
        # MCP Agent는 클래스 기반 또는 함수 기반
        if class_name:
            # 클래스 기반
            input_data = {
                "module_path": normalized_entry_point,
                "class_name": class_name,
                "method_name": method_name or "main",
                **kwargs
            }
        elif method_name:
            # 함수 기반
            input_data = {
                "module_path": normalized_entry_point,
                "method_name": method_name,
                **kwargs
            }
        else:
            # entry_point만 있는 경우 (기존 방식 호환)
            input_data = {
                **kwargs
            }
    elif agent_type == AgentType.LANGGRAPH_AGENT:
        # LangGraph Agent는 messages와 query 중심
        input_data = {
            **kwargs
        }
    elif agent_type == AgentType.SPARKLEFORGE_AGENT:
        # SparkleForge Agent는 query와 context 중심
        input_data = {
            **kwargs
        }
    elif agent_type == AgentType.CRON_AGENT:
        # Cron Agent는 스케줄 정보 중심
        input_data = {
            **kwargs
        }
    else:
        # 기타 Agent 타입
        input_data = {
            **kwargs
        }
    
    # result_json_path 추가 (제공된 경우)
    if result_json_path:
        input_data["result_json_path"] = result_json_path
    
    return input_data


def execute_standard_agent_via_a2a(
    placeholder,
    agent_id: str,
    agent_name: str,
    entry_point: str,
    agent_type: AgentType,
    capabilities: List[str],
    description: str,
    input_params: Dict[str, Any],
    class_name: Optional[str] = None,
    method_name: Optional[str] = None,
    result_json_path: Optional[Path] = None,
    use_a2a: bool = True
) -> Optional[Dict[str, Any]]:
    """
    표준화된 방식으로 A2A를 통해 agent 실행
    
    Args:
        placeholder: Streamlit placeholder 컨테이너
        agent_id: Agent 고유 ID
        agent_name: Agent 이름
        entry_point: 실행 경로
        agent_type: Agent 타입 (AgentType enum)
        capabilities: Agent 능력 목록
        description: Agent 설명
        input_params: Agent에 전달할 입력 파라미터
        class_name: 클래스 이름 (MCP Agent 클래스 기반인 경우)
        method_name: 메서드 이름 (MCP Agent 클래스/함수 기반인 경우)
        result_json_path: 결과 JSON 파일 경로 (선택)
        use_a2a: A2A 사용 여부 (기본값: True)
        
    Returns:
        성공 시 결과 데이터(dict), 실패 시 None
    """
    # 표준화된 agent_metadata 생성
    agent_metadata = create_standard_agent_metadata(
        agent_id=agent_id,
        agent_name=agent_name,
        entry_point=entry_point,
        agent_type=agent_type,
        capabilities=capabilities,
        description=description
    )
    
    # 표준화된 input_data 생성
    result_json_path_str = str(result_json_path) if result_json_path else None
    
    # input_params에서 result_json_path가 있으면 제거하여 중복 전달 방지
    params = input_params.copy()
    if "result_json_path" in params:
        params.pop("result_json_path")
        
    input_data = create_standard_input_data(
        agent_type=agent_type,
        entry_point=entry_point,
        result_json_path=result_json_path_str,
        class_name=class_name,
        method_name=method_name,
        **params
    )
    
    # A2A를 통해 agent 실행
    result = run_agent_via_a2a(
        placeholder=placeholder,
        agent_metadata=agent_metadata,
        input_data=input_data,
        result_json_path=result_json_path,
        use_a2a=use_a2a
    )
    
    return result


def process_standard_agent_result(
    result: Optional[Dict[str, Any]],
    agent_id: str,
    check_success: bool = True,
    check_data: bool = True
) -> Dict[str, Any]:
    """
    표준화된 방식으로 agent 실행 결과 처리
    
    Args:
        result: run_agent_via_a2a에서 반환된 결과
        agent_id: Agent ID (에러 메시지용)
        check_success: success 필드 확인 여부
        check_data: data 필드 확인 여부
        
    Returns:
        처리된 결과 딕셔너리
        {
            "success": bool,
            "data": Any,
            "error": Optional[str],
            "has_data": bool
        }
    """
    if result is None:
        return {
            "success": False,
            "data": None,
            "error": f"Agent {agent_id} 실행 결과가 없습니다.",
            "has_data": False
        }
    
    # result가 dict인 경우
    if isinstance(result, dict):
        # success 필드 확인
        if check_success:
            success = result.get("success", False)
            if not success:
                error_msg = result.get("error", "Unknown error")
                return {
                    "success": False,
                    "data": result.get("data"),
                    "error": error_msg,
                    "has_data": "data" in result and result.get("data") is not None
                }
        
        # data 필드 확인
        if check_data:
            if "data" not in result:
                return {
                    "success": result.get("success", True),
                    "data": None,
                    "error": f"Agent {agent_id} 결과에 'data' 필드가 없습니다.",
                    "has_data": False
                }
        
        return {
            "success": result.get("success", True),
            "data": result.get("data"),
            "error": result.get("error"),
            "has_data": "data" in result and result.get("data") is not None,
            "execution_time": result.get("execution_time"),
            "metadata": result.get("metadata", {})
        }
    else:
        # result가 dict가 아닌 경우 (예: 문자열, 리스트 등)
        return {
            "success": True,
            "data": result,
            "error": None,
            "has_data": result is not None
        }


def detect_agent_type_from_entry_point(entry_point: str) -> AgentType:
    """
    entry_point로부터 agent_type 자동 감지
    
    Args:
        entry_point: 실행 경로
        
    Returns:
        감지된 AgentType enum
    """
    detected_type_str = _detect_agent_type(entry_point)
    
    # 문자열을 AgentType enum으로 변환
    try:
        return AgentType(detected_type_str)
    except ValueError:
        # 알 수 없는 타입인 경우 MCP_AGENT로 기본값
        return AgentType.MCP_AGENT

