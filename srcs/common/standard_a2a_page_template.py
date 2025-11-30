"""
표준 A2A Page 템플릿

모든 pages에서 일관된 UI 패턴과 구조를 제공하는 표준 템플릿
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from srcs.common.page_utils import create_agent_page
from srcs.common.standard_a2a_page_helper import (
    execute_standard_agent_via_a2a,
    process_standard_agent_result,
    detect_agent_type_from_entry_point
)
from srcs.common.agent_interface import AgentType
from configs.settings import get_reports_path

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader
except ImportError:
    result_reader = None


def create_standard_a2a_page(
    agent_id: str,
    agent_name: str,
    page_icon: str,
    page_type: str,
    title: str,
    subtitle: str,
    entry_point: str,
    agent_type: Optional[AgentType] = None,
    capabilities: Optional[List[str]] = None,
    description: Optional[str] = None,
    form_fields: Optional[List[Dict[str, Any]]] = None,
    display_results_func: Optional[Callable[[Dict[str, Any]], None]] = None,
    result_category: Optional[str] = None
):
    """
    표준화된 A2A Page 생성
    
    Args:
        agent_id: Agent 고유 ID
        agent_name: Agent 이름
        page_icon: 페이지 아이콘
        page_type: 페이지 타입
        title: 페이지 제목
        subtitle: 페이지 부제목
        entry_point: 실행 경로
        agent_type: Agent 타입 (None이면 자동 감지)
        capabilities: Agent 능력 목록
        description: Agent 설명
        form_fields: 폼 필드 정의 리스트
            [
                {
                    "type": "text_area" | "text_input" | "selectbox" | "slider" | "number_input",
                    "key": "field_key",
                    "label": "Field Label",
                    "default": default_value,
                    "options": [...],  # selectbox인 경우
                    "min_value": 0,  # slider/number_input인 경우
                    "max_value": 10,
                    "help": "Help text"
                },
                ...
            ]
        display_results_func: 결과 표시 함수 (선택)
        result_category: 결과 카테고리 (result_reader용)
    """
    # 페이지 기본 설정
    create_agent_page(
        agent_name=agent_name,
        page_icon=page_icon,
        page_type=page_type,
        title=title,
        subtitle=subtitle,
        module_path=entry_point
    )
    
    # Agent 타입 자동 감지 (제공되지 않은 경우)
    if agent_type is None:
        agent_type = detect_agent_type_from_entry_point(entry_point)
    
    # 기본값 설정
    if capabilities is None:
        capabilities = []
    if description is None:
        description = subtitle
    
    # 결과 placeholder
    result_placeholder = st.empty()
    
    # 폼 생성
    with st.form(f"{agent_id}_form"):
        st.subheader(f"📝 {agent_name} 설정")
        
        form_data = {}
        
        if form_fields:
            for field in form_fields:
                field_type = field.get("type", "text_input")
                field_key = field.get("key")
                field_label = field.get("label", field_key)
                field_default = field.get("default", "")
                field_help = field.get("help", "")
                
                if field_type == "text_area":
                    form_data[field_key] = st.text_area(
                        field_label,
                        value=field_default,
                        height=field.get("height", 150),
                        help=field_help
                    )
                elif field_type == "text_input":
                    form_data[field_key] = st.text_input(
                        field_label,
                        value=field_default,
                        help=field_help
                    )
                elif field_type == "selectbox":
                    form_data[field_key] = st.selectbox(
                        field_label,
                        options=field.get("options", []),
                        index=field.get("default_index", 0),
                        help=field_help
                    )
                elif field_type == "slider":
                    form_data[field_key] = st.slider(
                        field_label,
                        min_value=field.get("min_value", 0),
                        max_value=field.get("max_value", 100),
                        value=field.get("default", field.get("min_value", 0)),
                        help=field_help
                    )
                elif field_type == "number_input":
                    form_data[field_key] = st.number_input(
                        field_label,
                        min_value=field.get("min_value", 0),
                        max_value=field.get("max_value", 100),
                        value=field.get("default", 0),
                        help=field_help
                    )
        
        submitted = st.form_submit_button(f"🚀 {agent_name} 실행", use_container_width=True)
    
    # 폼 제출 처리
    if submitted:
        # 필수 필드 검증
        required_fields = [f.get("key") for f in (form_fields or []) if f.get("required", False)]
        missing_fields = [f for f in required_fields if not form_data.get(f) or not str(form_data.get(f)).strip()]
        
        if missing_fields:
            st.warning(f"다음 필드를 입력해주세요: {', '.join(missing_fields)}")
        else:
            # 결과 경로 설정
            reports_path = Path(get_reports_path(agent_id))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id=agent_id,
                agent_name=agent_name,
                entry_point=entry_point,
                agent_type=agent_type,
                capabilities=capabilities,
                description=description,
                input_params=form_data,
                result_json_path=result_json_path,
                use_a2a=True
            )
            
            # 결과 처리
            processed_result = process_standard_agent_result(result, agent_id)
            
            if processed_result["success"] and processed_result["has_data"]:
                if display_results_func:
                    display_results_func(processed_result["data"])
                else:
                    _default_display_results(processed_result["data"])
            elif not processed_result["success"]:
                st.error(f"❌ {agent_name} 실행 실패: {processed_result.get('error', 'Unknown error')}")
    
    # 최신 결과 확인
    if result_reader and result_category:
        _display_latest_results(agent_id, result_category, agent_name)


def _default_display_results(result_data: Dict[str, Any]):
    """기본 결과 표시 함수"""
    st.markdown("---")
    st.subheader("📊 실행 결과")
    
    if isinstance(result_data, dict):
        st.json(result_data)
    else:
        st.write(result_data)


def _display_latest_results(agent_id: str, result_category: str, agent_name: str):
    """최신 결과 표시"""
    st.markdown("---")
    st.markdown(f"## 📊 최신 {agent_name} 결과")
    
    if result_reader:
        latest_result = result_reader.get_latest_result(agent_id, result_category)
        
        if latest_result:
            with st.expander(f"🤖 최신 {agent_name} 실행 결과", expanded=False):
                st.subheader(f"✈️ 최근 {agent_name} 실행 결과")
                
                if isinstance(latest_result, dict):
                    st.json(latest_result)
                else:
                    st.write(latest_result)
        else:
            st.info(f"💡 아직 {agent_name}의 결과가 없습니다. 위에서 작업을 실행해보세요.")


def create_simple_a2a_page(
    agent_id: str,
    agent_name: str,
    page_icon: str,
    entry_point: str,
    agent_type: Optional[AgentType] = None,
    form_config: Optional[Dict[str, Any]] = None,
    display_func: Optional[Callable[[Dict[str, Any]], None]] = None
):
    """
    간단한 A2A Page 생성 (최소 설정)
    
    Args:
        agent_id: Agent 고유 ID
        agent_name: Agent 이름
        page_icon: 페이지 아이콘
        entry_point: 실행 경로
        agent_type: Agent 타입 (None이면 자동 감지)
        form_config: 폼 설정 딕셔너리
        display_func: 결과 표시 함수
    """
    create_standard_a2a_page(
        agent_id=agent_id,
        agent_name=agent_name,
        page_icon=page_icon,
        page_type=agent_id,
        title=agent_name,
        subtitle=f"{agent_name} 실행 페이지",
        entry_point=entry_point,
        agent_type=agent_type,
        form_fields=form_config.get("fields", []) if form_config else [],
        display_results_func=display_func,
        result_category=form_config.get("result_category") if form_config else None
    )

