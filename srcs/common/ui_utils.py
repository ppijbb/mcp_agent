import streamlit as st
import sys
import json
from pathlib import Path
from datetime import datetime
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process

def run_agent_process(
    placeholder,
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
            
            process = Process(command, output_file=process_key).start()
            
            # expander 중첩 문제를 피하기 위해 직접 process monitor 사용
            st.info(f"🔄 {log_expander_title}")
            spm.st_process_monitor(process, label=f"monitor_{process_key}").loop_until_finished()
                
            if process.get_return_code() == 0:
                # 먼저 --result-json-path 인자를 찾아보고, 없으면 --result-txt-path를 찾아봄
                result_path_str = None
                is_json_format = False
                
                try:
                    # JSON 파일 경로 찾기
                    idx = command.index("--result-json-path")
                    result_path_str = command[idx + 1]
                    is_json_format = True
                except (ValueError, IndexError):
                    try:
                        # 텍스트 파일 경로 찾기
                        idx = command.index("--result-txt-path")
                        result_path_str = command[idx + 1]
                        is_json_format = False
                    except (ValueError, IndexError):
                        st.error("❌ 내부 오류: 실행 커맨드에서 결과 파일 경로를 찾을 수 없습니다.")
                        return None

                try:
                    with open(result_path_str, 'r', encoding='utf-8') as f:
                        if is_json_format:
                            result = json.load(f)
                            
                            if "success" in result and not result.get("success"):
                                st.error(f"❌ 작업은 완료되었지만 오류가 보고되었습니다: {result.get('error', '알 수 없는 오류')}")
                                return None

                            st.success("✅ 작업이 성공적으로 완료되었습니다!")
                            # 'data' 키가 있으면 해당 값을, 없으면 전체 result 객체를 반환
                            return result.get("data", result)
                        else:
                            # 텍스트 파일 처리
                            result_text = f.read()
                            
                            if result_text.strip():
                                st.success("✅ 작업이 성공적으로 완료되었습니다!")
                                return {"result_text": result_text}
                            else:
                                st.error("❌ 결과 파일이 비어있습니다.")
                                return None

                except FileNotFoundError:
                    st.error(f"❌ 결과 파일({result_path_str})을 찾을 수 없습니다.")
                    return None
                except Exception as e:
                    st.error(f"결과 파일을 읽거나 처리하는 중 오류가 발생했습니다: {e}")
                    return None
            else:
                st.error(f"❌ 에이전트 실행에 실패했습니다. (Return Code: {process.get_return_code()})")
                return None 