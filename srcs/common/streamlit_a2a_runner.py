"""
Streamlit용 A2A Agent 실행 헬퍼

Streamlit 페이지에서 A2A를 통해 agent를 실행하는 공통 함수
"""

import streamlit as st
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from srcs.common.standard_agent_runner import StandardAgentRunner
from srcs.common.agent_interface import AgentType, AgentMetadata
from srcs.common.a2a_integration import get_global_registry

logger = logging.getLogger(__name__)


def _detect_agent_type(entry_point: str) -> str:
    """entry_point를 분석하여 agent 타입 자동 판단"""
    if entry_point.startswith("lang_graph.") or "lang_graph/" in entry_point:
        return AgentType.LANGGRAPH_AGENT.value
    elif entry_point.startswith("cron_agents.") or "cron_agents/" in entry_point:
        return AgentType.CRON_AGENT.value
    elif entry_point.startswith("sparkleforge.") or "sparkleforge/" in entry_point:
        return AgentType.SPARKLEFORGE_AGENT.value
    else:
        return AgentType.MCP_AGENT.value


def _normalize_entry_point(entry_point: str) -> str:
    """entry_point를 정규화 (CLI 명령에서 모듈 경로 추출)"""
    # "python -m srcs.enterprise_agents.esg_carbon_neutral_agent" -> "srcs.enterprise_agents.esg_carbon_neutral_agent"
    if entry_point.startswith("python -m "):
        return entry_point.replace("python -m ", "")
    # "srcs/enterprise_agents/esg_carbon_neutral_agent.py" -> "srcs.enterprise_agents.esg_carbon_neutral_agent"
    if entry_point.endswith(".py"):
        entry_point = entry_point.replace(".py", "")
    entry_point = entry_point.replace("/", ".")
    return entry_point


def run_agent_via_a2a(
    placeholder,
    agent_metadata: Dict[str, Any],
    input_data: Dict[str, Any],
    result_json_path: Optional[Path] = None,
    use_a2a: bool = True,
    log_expander_title: str = "🤖 A2A Agent 실행 중..."
) -> Optional[Dict[str, Any]]:
    """
    A2A를 통해 agent를 실행하는 Streamlit 헬퍼 함수
    
    Args:
        placeholder: Streamlit placeholder 컨테이너
        agent_metadata: Agent 메타데이터 딕셔너리
            - agent_id: Agent ID (필수)
            - agent_name: Agent 이름 (필수)
            - entry_point: 실행 경로 (필수, 모듈 경로 또는 CLI 명령)
            - agent_type: Agent 타입 (선택, 자동 판단됨)
            - capabilities: Agent 능력 목록 (선택)
            - description: 설명 (선택)
        input_data: Agent에 전달할 입력 데이터
        result_json_path: 결과를 저장할 JSON 파일 경로 (선택)
        use_a2a: A2A 사용 여부 (기본값: True)
        log_expander_title: 로그 제목
        
    Returns:
        성공 시 결과 데이터(dict), 실패 시 None
    """
    if placeholder is None:
        st.error("결과를 표시할 UI 컨테이너가 지정되지 않았습니다.")
        return None
    
    # 필수 필드 검증
    agent_id = agent_metadata.get("agent_id")
    agent_name = agent_metadata.get("agent_name")
    entry_point = agent_metadata.get("entry_point")
    
    if not agent_id or not agent_name or not entry_point:
        st.error("❌ agent_metadata에 agent_id, agent_name, entry_point가 필요합니다.")
        return None
    
    # entry_point 정규화
    entry_point = _normalize_entry_point(entry_point)
    
    # agent_type 자동 판단 (지정되지 않은 경우)
    agent_type = agent_metadata.get("agent_type")
    if not agent_type:
        agent_type = _detect_agent_type(entry_point)
    
    # AgentMetadata 객체 생성
    try:
        agent_type_enum = AgentType(agent_type)
    except ValueError:
        st.error(f"❌ 잘못된 agent_type: {agent_type}")
        return None
    
    metadata = AgentMetadata(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type_enum,
        description=agent_metadata.get("description", ""),
        capabilities=agent_metadata.get("capabilities", []),
        entry_point=entry_point,
    )
    
    # Streamlit 세션 상태에서 runner 초기화
    if "a2a_runner" not in st.session_state:
        st.session_state.a2a_runner = StandardAgentRunner()
    
    runner = st.session_state.a2a_runner
    registry = get_global_registry()
    
    with placeholder.container():
        with st.spinner(log_expander_title):
            try:
                # Agent를 레지스트리에 등록 (아직 등록되지 않은 경우)
                async def register_and_run():
                    # Agent가 이미 등록되어 있는지 확인
                    existing_agent = await registry.get_agent(agent_id)
                    
                    if not existing_agent:
                        # Agent 등록 (A2A adapter 없이 먼저 등록)
                        await registry.register_agent(
                            agent_id=agent_id,
                            agent_type=agent_type,
                            metadata=metadata.to_dict(),
                            a2a_adapter=None  # 실행 시점에 생성
                        )
                        logger.info(f"Agent registered: {agent_id}")
                    
                    # Agent 실행
                    result = await runner.run_agent(
                        agent_id=agent_id,
                        input_data=input_data,
                        use_a2a=use_a2a
                    )
                    
                    return result
                
                # 비동기 함수 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(register_and_run())
                finally:
                    loop.close()
                
                # 결과 처리
                if result.success:
                    st.success("✅ Agent 실행이 성공적으로 완료되었습니다!")
                    
                    # 결과 데이터 준비
                    result_data = {
                        "success": True,
                        "data": result.data,
                        "execution_time": result.execution_time,
                        "metadata": result.metadata,
                        "timestamp": result.timestamp.isoformat(),
                    }
                    
                    # 결과를 JSON 파일로 저장
                    if result_json_path:
                        result_json_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(result_json_path, 'w', encoding='utf-8') as f:
                            json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
                        logger.info(f"Result saved to: {result_json_path}")
                    
                    # 결과 표시
                    if result.data:
                        st.json(result.data)
                    
                    return result_data
                else:
                    st.error(f"❌ Agent 실행 실패: {result.error}")
                    
                    # 에러 결과 저장
                    error_result = {
                        "success": False,
                        "error": result.error,
                        "execution_time": result.execution_time,
                        "metadata": result.metadata,
                        "timestamp": result.timestamp.isoformat(),
                    }
                    
                    if result_json_path:
                        result_json_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(result_json_path, 'w', encoding='utf-8') as f:
                            json.dump(error_result, f, indent=2, ensure_ascii=False, default=str)
                    
                    return None
                    
            except Exception as e:
                error_msg = f"Agent 실행 중 오류 발생: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(f"❌ {error_msg}")
                
                # 에러 결과 저장
                if result_json_path:
                    error_result = {
                        "success": False,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat(),
                    }
                    result_json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(result_json_path, 'w', encoding='utf-8') as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False, default=str)
                
                return None
    
    return None


def get_registered_agents() -> List[Dict[str, Any]]:
    """
    등록된 agent 목록 조회 (동기 함수)
    
    Returns:
        등록된 agent 목록
    """
    if "a2a_runner" not in st.session_state:
        return []
    
    registry = get_global_registry()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        agents = loop.run_until_complete(registry.list_agents())
        return agents
    finally:
        loop.close()


def send_a2a_message(
    source_agent_id: str,
    target_agent_id: str,
    message_type: str,
    payload: Dict[str, Any]
) -> bool:
    """
    A2A 메시지 전송 (동기 함수)
    
    Args:
        source_agent_id: 소스 agent ID
        target_agent_id: 타겟 agent ID (빈 문자열이면 브로드캐스트)
        message_type: 메시지 타입
        payload: 메시지 페이로드
        
    Returns:
        전송 성공 여부
    """
    if "a2a_runner" not in st.session_state:
        st.error("A2A runner가 초기화되지 않았습니다.")
        return False
    
    from srcs.common.a2a_adapter import CommonAgentA2AWrapper
    from srcs.common.a2a_integration import get_global_broker, A2AMessage, MessagePriority
    
    registry = get_global_registry()
    broker = get_global_broker()
    
    async def send():
        source_agent = await registry.get_agent(source_agent_id)
        if not source_agent:
            logger.error(f"Source agent not found: {source_agent_id}")
            return False
        
        # 임시 wrapper 생성하여 메시지 전송
        wrapper = CommonAgentA2AWrapper(
            agent_id=source_agent_id,
            agent_metadata=source_agent.get("metadata", {})
        )
        
        return await wrapper.send_message(
            target_agent=target_agent_id,
            message_type=message_type,
            payload=payload,
            priority=MessagePriority.MEDIUM.value
        )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(send())
    finally:
        loop.close()

