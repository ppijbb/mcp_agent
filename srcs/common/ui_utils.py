import streamlit as st
import sys
import json
from pathlib import Path
from datetime import datetime
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process

def run_agent_process(
    placeholder: st.container,
    command: list[str], 
    process_key_prefix: str,
    log_expander_title: str = "실시간 실행 로그"
) -> dict | None:
    """
    에이전트 실행 프로세스를 시작하고 UI에서 모니터링하는 공통 함수

    Args:
        placeholder: 결과를 표시할 Streamlit 컨테이너
        command: 실행할 커맨드 리스트
        process_key_prefix: 중복을 피하기 위한 spm 프로세스 키 접두사
        log_expander_title: 로그 Expander의 제목

    Returns:
        성공 시 결과 데이터(dict), 실패 시 None
    """
    if placeholder is None:
        st.error("결과를 표시할 UI 컨테이너가 지정되지 않았습니다.")
        return None

    with placeholder.container():
        with st.spinner("🤖 에이전트가 작업을 수행 중입니다..."):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            process_key = f"{process_key_prefix}_{timestamp}"
            
            process = Process(command, key=process_key).start()

            log_expander = st.expander(log_expander_title, expanded=True)
            with log_expander:
                spm.st_process_monitor(process, key=f"monitor_{process_key}").loop_until_finished()
                
            if process.get_return_code() == 0:
                # The command should have included a --result-json-path argument.
                # We need to find it to read the result file.
                result_json_path_str = None
                try:
                    # 커맨드에서 --result-json-path 인자의 위치를 찾아 그 다음 값을 가져옴
                    idx = command.index("--result-json-path")
                    result_json_path_str = command[idx + 1]
                except (ValueError, IndexError):
                    st.error("❌ 내부 오류: 실행 커맨드에서 결과 파일 경로를 찾을 수 없습니다.")
                    return None

                try:
                    with open(result_json_path_str, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    
                    if "success" in result and not result.get("success"):
                        st.error(f"❌ 작업은 완료되었지만 오류가 보고되었습니다: {result.get('error', '알 수 없는 오류')}")
                        return None

                    st.success("✅ 작업이 성공적으로 완료되었습니다!")
                    # 'data' 키가 있으면 해당 값을, 없으면 전체 result 객체를 반환
                    return result.get("data", result)

                except FileNotFoundError:
                    st.error(f"❌ 결과 파일({result_json_path_str})을 찾을 수 없습니다.")
                    return None
                except Exception as e:
                    st.error(f"결과 파일을 읽거나 처리하는 중 오류가 발생했습니다: {e}")
                    return None
            else:
                st.error(f"❌ 에이전트 실행에 실패했습니다. (Return Code: {process.get_return_code()})")
                return None 