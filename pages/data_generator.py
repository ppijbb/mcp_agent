"""
📊 Data Generator Page

AI 기반 데이터 생성 및 분석 도구
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime
import asyncio
import re
import json
import pandas as pd
import plotly.express as px
from srcs.common.streamlit_log_handler import setup_streamlit_logging
from srcs.advanced_agents.enhanced_data_generator import SyntheticDataAgent
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('data_generator')
except (ImportError, ModuleNotFoundError):
    st.error("❌ 설정 파일을 찾을 수 없습니다. `configs/settings.py`를 확인해주세요.")
    # Fallback path
    REPORTS_PATH = os.path.join(project_root, "reports", "data_generator")
    os.makedirs(REPORTS_PATH, exist_ok=True)

# AI Data Generation Agent 임포트 - 필수 의존성
try:
    from srcs.basic_agents.data_generator import AIDataGenerationAgent as agent
except ImportError as e:
    st.error(f"❌ AI 에이전트를 불러올 수 없습니다: {e}")
    st.error("시스템 요구사항: `AIDataGenerationAgent`와 `SyntheticDataAgent`가 필수입니다.")
    st.info("에이전트 모듈을 설치하고 다시 시도해주세요: `srcs/basic_agents/`")
    st.stop()

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# 페이지 설정
try:
    st.set_page_config(
        page_title="🤖 AI 데이터 생성기",
        page_icon="🤖",
        layout="wide",
    )
except Exception:
    pass

st.title("🤖 AI 데이터 생성기")
st.caption("🚀 필요한 모든 종류의 합성 데이터를 AI로 생성하세요")

# --- Real-time Log Display ---
log_expander = st.expander("실시간 실행 로그", expanded=False)
log_container = log_expander.empty()
# Capture logs from the root mcp_agent logger
setup_streamlit_logging(["mcp_agent", "synthetic_data_orchestrator", "ai_data_generation_agent"], log_container)
# --- End Log Display ---

# --- Session State for results ---
if 'detailed_generator_result_placeholder' not in st.session_state:
    st.session_state.detailed_generator_result_placeholder = None

def parse_request(prompt: str) -> tuple[str | None, int | None]:
    """
    Parses a user prompt to extract data type and record count.
    Example: "고객 데이터 100개 만들어줘" -> ("고객", 100)
    """
    # Regex to find a number and the text preceding "데이터" or a similar keyword
    match = re.search(r"(.+?)(?: 데이터|)\s*(\d+)\s*개", prompt)
    if match:
        data_type = match.group(1).strip()
        record_count = int(match.group(2))
        return data_type, record_count
    return None, None

def execute_chat_data_agent_process(data_type: str, record_count: int) -> str:
    """실제 AI를 사용한 스마트 데이터 생성"""
    
    try:
        with st.spinner("AI가 지능적으로 데이터를 생성 중입니다..."):
            
            # config 생성
            config = {
                'type': data_type,
                'count': record_count,
                'purpose': f"{data_type} 데이터 생성",
                'quality': '고품질'
            }
            
            # 실제 에이전트 호출
            result = agent.generate_smart_data(config)
            
            if not result:
                raise Exception("에이전트가 유효한 결과를 반환하지 않았습니다.")
            
            # 결과 포맷팅 (간단한 형태)
            agent_output = f"생성된 {data_type} 데이터 {record_count}개:\n{str(result)}"
            
            st.session_state['ai_generated_data'] = {
                'agent_output': agent_output,
                'config': config,
                'raw_result': result
            }
            
            # 파일 저장 처리 (선택적)
            save_to_file = st.checkbox("결과를 파일로 저장", value=False)
            if save_to_file:
                file_saved, output_path = save_data_generator_results(agent_output, config)
                if file_saved:
                    st.success(f"💾 데이터가 파일로 저장되었습니다: {output_path}")
                else:
                    st.error("파일 저장 중 오류가 발생했습니다.")
            
            st.success("✅ AI 스마트 데이터 생성이 완료되었습니다!")
            st.rerun()
            
    except Exception as e:
        st.error(f"AI 데이터 생성 실패: {e}")
        st.error("에이전트의 generate_smart_data 메서드를 확인해주세요.")


def render_chat_generator():
    """Renders the chat-based data generator using SyntheticDataAgent."""
    st.header("💬 채팅으로 간단하게 생성하기")
    st.info("Meta의 Synthetic Data Kit을 활용하여 더 복잡한 데이터셋(Q&A, CoT 등)을 생성합니다.")

    # Use the state management pattern for the agent
    if 'enhanced_data_agent' not in st.session_state:
        st.session_state.enhanced_data_agent = SyntheticDataAgent(output_dir="generated_data")
    agent = st.session_state.enhanced_data_agent

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "안녕하세요! 어떤 종류의 데이터를 몇 개나 생성해 드릴까요?\n\n예시: `고객 데이터 100개 생성해줘`"}]

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("어떤 데이터를 생성할까요?"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        data_type, record_count = parse_request(prompt)

        if data_type and record_count:
            with st.chat_message("assistant"):
                with st.spinner(f"'{data_type}' 데이터 {record_count}개를 생성하는 중... 잠시만 기다려주세요."):
                    try:
                        response = asyncio.run(agent.run(data_type=data_type, record_count=record_count))
                        st.markdown(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"데이터 생성 중 오류 발생: {e}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

        else:
            response = "요청을 이해하지 못했어요. `[데이터 종류] [숫자]개` 형식으로 말씀해주세요. (예: `제품 150개`)"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

def render_detailed_generator():
    """Renders the detailed, form-based data generator using AIDataGenerationAgent."""
    st.header("⚙️ 상세 설정으로 생성하기")
    st.info("Orchestrator-based AI Agent를 활용하여 특정 비즈니스 요구사항에 맞는 데이터를 생성합니다.")
    
    # 결과 표시를 위한 플레이스홀더
    st.session_state.detailed_generator_result_placeholder = st.container()

    try:
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 AI 스마트 데이터 생성", 
            "📊 AI 맞춤형 데이터셋", 
            "👥 AI 고객 프로필", 
            "📈 AI 시계열 예측"
        ])
        
        with tab1:
            render_ai_smart_data_generation()
        
        with tab2:
            render_ai_custom_datasets()
        
        with tab3:
            render_ai_customer_profiles()
        
        with tab4:
            render_ai_timeseries_prediction()
            
    except Exception as e:
        st.error(f"❌ AI Data Generation Agent 초기화 실패: {e}")
        st.error("AIDataGenerationAgent 구현을 확인해주세요.")
        st.stop()

def render_ai_smart_data_generation():
    """AI 기반 스마트 데이터 생성"""
    
    st.markdown("### 🤖 AI 스마트 데이터 생성")
    st.info("AI가 요구사항을 분석하여 지능적으로 데이터를 생성합니다.")
    
    with st.form("smart_data_form"):
        st.markdown("#### ⚙️ AI 데이터 생성 설정")
        
        data_purpose = st.text_area(
            "데이터 사용 목적",
            placeholder="예: 고객 행동 분석을 위한 샘플 데이터",
            help="AI가 목적에 맞는 데이터를 생성합니다"
        )
        
        data_type = st.text_input("데이터 유형", value="고객", help="생성할 데이터의 유형을 입력하세요 (예: 제품, 거래내역).")
        records_count = st.number_input("레코드 수", min_value=10, max_value=10000, value=100)
        
        submitted = st.form_submit_button("🚀 AI 스마트 데이터 생성", width='stretch')

        if submitted:
            if not data_purpose.strip() or not data_type.strip():
                st.error("데이터 사용 목적과 유형을 모두 입력해주세요.")
            else:
                config = {
                    'purpose': data_purpose,
                    'type': data_type,
                    'count': records_count,
                    'quality': '고품질', # Simplified for this form
                    'relationships': True,
                    'patterns': True
                }
                execute_detailed_data_agent_process('generate_smart_data', config)
    
def render_ai_custom_datasets():
    """AI 맞춤형 데이터셋 생성 UI"""
    st.markdown("### 📊 AI 맞춤형 데이터셋")
    st.info("특정 도메인과 요구사항에 맞는 맞춤형 데이터셋을 생성합니다.")

    with st.form("custom_dataset_form"):
        domain = st.selectbox("데이터 도메인", options=load_domains(), index=0)
        description = st.text_area("데이터셋 상세 설명", placeholder="예: 온라인 게임 사용자의 3개월간 아이템 구매 패턴 데이터")
        records_count = st.number_input("레코드 수", min_value=10, max_value=5000, value=50)
        
        submitted = st.form_submit_button("📊 맞춤 데이터셋 생성", width='stretch')
        if submitted:
            if not description.strip():
                st.error("데이터셋 상세 설명을 입력해주세요.")
            else:
                config = {
                    'domain': domain,
                    'description': description,
                    'count': records_count
                }
                execute_detailed_data_agent_process('create_custom_dataset', config)

def render_ai_customer_profiles():
    """AI 고객 프로필 생성 UI"""
    st.markdown("### 👥 AI 고객 프로필")
    st.info("다양한 비즈니스 유형과 고객 세그먼트에 맞는 가상 고객 프로필을 생성합니다.")

    with st.form("customer_profiles_form"):
        business_type = st.selectbox("비즈니스 유형", options=load_business_types())
        target_segment = st.text_input("타겟 고객 세그먼트", placeholder="예: 20대 대학생, IT 업계 종사자")
        records_count = st.number_input("생성할 프로필 수", min_value=5, max_value=1000, value=10)
        
        submitted = st.form_submit_button("👥 고객 프로필 생성", width='stretch')
        if submitted:
            if not target_segment.strip():
                st.error("타겟 고객 세그먼트를 입력해주세요.")
            else:
                config = {
                    'business_type': business_type,
                    'target_segment': target_segment,
                    'count': records_count
                }
                execute_detailed_data_agent_process('generate_customer_profiles', config)

def render_ai_timeseries_prediction():
    """AI 시계열 예측 데이터 생성 UI"""
    st.markdown("### 📈 AI 시계열 예측 데이터")
    st.info("과거 데이터 패턴을 학습하여 미래 시점의 데이터를 예측 생성합니다.")

    with st.form("timeseries_form"):
        series_type = st.selectbox("시계열 데이터 종류", options=load_series_types())
        time_period = st.selectbox("예측 기간", options=load_time_periods())
        frequency = st.selectbox("데이터 빈도", options=load_frequencies())
        
        submitted = st.form_submit_button("📈 시계열 데이터 생성", width='stretch')
        if submitted:
            config = {
                'type': series_type,
                'period': time_period,
                'frequency': frequency
            }
            execute_detailed_data_agent_process('generate_timeseries_data', config)

def execute_detailed_data_agent_process(agent_method: str, config: dict):
    """상세 데이터 에이전트를 별도 프로세스로 실행하고 결과를 표시합니다."""
    
    placeholder = st.session_state.detailed_generator_result_placeholder
    if placeholder is None:
        st.error("결과를 표시할 위치를 찾을 수 없습니다. 페이지를 새로고침해주세요.")
        return

    with placeholder.container():
        with st.spinner(f"AI가 데이터를 생성 중입니다... (Method: {agent_method})"):
            reports_path = Path(get_reports_path('data_generator'))
            reports_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_json_path = reports_path / f"detailed_data_result_{agent_method}_{timestamp}.json"
            
            # 입력 파라미터 준비
            input_data = {
                "agent_method": agent_method,
                "config": config,
                "result_json_path": str(result_json_path)
            }
            
            # 결과 표시용 placeholder 생성
            result_placeholder = st.empty()
            
            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id=f"detailed_data_agent_{agent_method}",
                agent_name=f"Detailed Data Agent ({agent_method})",
                entry_point="srcs.basic_agents.data_generator",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["data_generation", "synthetic_data", "custom_datasets"],
                description="AI 기반 데이터 생성 및 분석",
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )
            
            if result and result.get("success"):
                display_detailed_data_results(result.get("data", {}), config)
            elif result and result.get("error"):
                st.error(f"❌ 실행은 완료되었지만 오류가 보고되었습니다: {result.get('error', '알 수 없는 오류')}")

def display_detailed_data_results(result: dict, config: dict):
    """상세 생성기 결과를 포맷하여 표시합니다."""
    st.markdown("#### 📊 AI 생성 데이터")
    
    data_content = result.get('agent_output', '')
    if isinstance(data_content, (list, dict)):
        data_content = json.dumps(data_content, indent=2, ensure_ascii=False)

    st.text_area(
        "생성된 데이터 결과",
        value=data_content,
        height=300,
        disabled=True,
        key=f"result_{datetime.now().timestamp()}" # To avoid duplicate key error
    )
    
    if st.download_button("📥 데이터 다운로드 (.json)", data=data_content, file_name=f"generated_data_{config.get('type', 'custom')}.json", width='stretch'):
        st.toast("다운로드가 시작되었습니다!")

    with st.expander("🔍 품질 측정 항목 보기"):
        st.info("품질 측정 항목은 데이터 생성 완료 후 표시됩니다.")


# 아래 함수들은 기존 로직을 그대로 사용하거나, 더미 데이터를 반환합니다.
# 실제 구현에서는 데이터베이스나 설정 파일에서 이 값들을 로드해야 합니다.

def load_data_types():
    """데이터 유형을 동적으로 로드"""
    try:
        return ["비즈니스 데이터", "고객 데이터", "금융 데이터", "의료 데이터", "교육 데이터", "기술 데이터"]
    except Exception:
        return None

def load_quality_levels():
    """품질 수준을 동적으로 로드"""
    try:
        return ["기본", "고품질", "프리미엄", "엔터프라이즈"]
    except Exception:
        return None

def load_domains():
    """도메인을 동적으로 로드"""
    try:
        return ["전자상거래", "금융서비스", "헬스케어", "교육", "부동산", "제조업"]
    except Exception:
        return None

def load_complexity_levels():
    """복잡도 수준을 동적으로 로드"""
    try:
        return ["단순", "중간", "복잡", "고급"]
    except Exception:
        return None

def load_business_types():
    """비즈니스 유형을 동적으로 로드"""
    try:
        return ["B2C 전자상거래", "B2B 서비스", "SaaS", "금융서비스", "교육", "헬스케어"]
    except Exception:
        return None

def load_series_types():
    """시계열 유형을 동적으로 로드"""
    try:
        return ["매출 예측", "주가 변동", "트래픽 패턴", "센서 데이터", "날씨 데이터"]
    except Exception:
        return None

def load_time_periods():
    """기간을 동적으로 로드"""
    try:
        return ["1개월", "3개월", "6개월", "1년", "2년"]
    except Exception:
        return None

def load_frequencies():
    """주기를 동적으로 로드"""
    try:
        return ["시간별", "일별", "주별", "월별"]
    except Exception:
        return None

def save_data_generator_results(data_text, config):
    """Data Generator 결과를 파일로 저장"""
    
    try:
        os.makedirs(REPORTS_PATH, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_generation_result_{timestamp}.md"
        filepath = os.path.join(REPORTS_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# AI Data Generator 결과 보고서\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**요청 설정**: \n```json\n{json.dumps(config, indent=2, ensure_ascii=False)}\n```\n\n")
            f.write("---\n\n")
            f.write(data_text)
            f.write("\n\n---\n")
            f.write("*본 보고서는 AI Data Generator Agent에 의해 자동 생성되었습니다.*\n")
        
        return True, filepath
        
    except Exception as e:
        st.error(f"파일 저장 중 오류: {e}")
        return False, None

def render_results_viewer():
    """결과 확인 탭 렌더링"""
    st.header("📊 생성된 데이터 결과")
    st.caption("Data Generator Agent가 생성한 데이터를 확인하세요")
    
    # Data Generator Agent의 최신 결과 확인
    latest_result = result_reader.get_latest_result("data_generator_agent", "data_generation")
    
    if latest_result:
        st.success("✅ 최신 생성된 데이터를 찾았습니다!")
        
        # 결과 표시
        if isinstance(latest_result, dict) and 'generated_data' in latest_result:
            st.subheader("📋 생성된 데이터")
            
            # 데이터 표시
            if isinstance(latest_result['generated_data'], list):
                df = pd.DataFrame(latest_result['generated_data'])
                st.dataframe(df, width='stretch')
                
                # 다운로드 버튼
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv,
                    file_name=f"generated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # 데이터 시각화
                if not df.empty:
                    st.subheader("📈 데이터 시각화")
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            x_col = st.selectbox("X축 선택", numeric_cols, key="viz_x")
                            y_col = st.selectbox("Y축 선택", [col for col in numeric_cols if col != x_col], key="viz_y")
                            if x_col and y_col:
                                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                                st.plotly_chart(fig, width='stretch')
                        
                        with col2:
                            hist_col = st.selectbox("히스토그램 컬럼", numeric_cols, key="viz_hist")
                            if hist_col:
                                fig = px.histogram(df, x=hist_col, title=f"{hist_col} 분포")
                                st.plotly_chart(fig, width='stretch')
            
            # 품질 메트릭 표시
            if 'quality_metrics' in latest_result:
                st.subheader("📊 품질 메트릭")
                metrics = latest_result['quality_metrics']
                cols = st.columns(len(metrics))
                for i, (key, value) in enumerate(metrics.items()):
                    with cols[i]:
                        st.metric(key, value)
            
            # 설정 정보 표시
            if 'config' in latest_result:
                with st.expander("⚙️ 생성 설정", expanded=False):
                    st.json(latest_result['config'])
        
        else:
            st.write("결과 데이터 형식이 예상과 다릅니다.")
    
    else:
        st.warning("📭 아직 생성된 데이터가 없습니다.")
        st.info("💡 '채팅으로 생성' 또는 '상세 설정으로 생성' 탭에서 데이터를 생성해보세요.")
        
        # 기존 결과 목록 표시 (있다면)
        agent_results = result_reader.get_agent_results("data_generator_agent")
        if agent_results["results"]:
            st.subheader("📋 이전 생성 결과")
            selected_result = result_display.display_result_selector("data_generator_agent")
            if selected_result:
                result_data = result_reader.load_result(selected_result["file_path"])
                result_display.display_result(result_data, selected_result.get("metadata"))

# --- Main App Structure ---
tab1, tab2, tab3 = st.tabs(["💬 채팅으로 생성 (Enhanced SDK)", "⚙️ 상세 설정으로 생성 (Orchestrator)", "📊 결과 확인"])

with tab1:
    render_chat_generator()

with tab2:
    render_detailed_generator()

with tab3:
    render_results_viewer() 