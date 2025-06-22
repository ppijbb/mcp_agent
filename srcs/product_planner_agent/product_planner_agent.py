#!/usr/bin/env python3
"""
Product Planner Agent - Streamlined Main Runner
"""

import asyncio
import os
import sys
import re
from urllib.parse import unquote
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 👉 새 계층형 아키텍처: ExecutiveCoordinator 사용
from srcs.product_planner_agent.coordinators.executive_coordinator import ExecutiveCoordinator
from srcs.product_planner_agent.utils.status_logger import STATUS_FILE

# Centralized env helper
from srcs.product_planner_agent.utils import env_settings as env

def parse_figma_url(url: str) -> tuple[str | None, str | None]:
    """
    Figma URL에서 file_id와 node_id를 추출합니다.
    예: https://www.figma.com/file/FILE_ID/File-Name?node-id=NODE_ID
    """
    # file_id: /file/ 다음에 오는 문자열
    file_id_match = re.search(r'figma\.com/file/([^/]+)', url)
    file_id = file_id_match.group(1) if file_id_match else None
    
    # node-id: 쿼리 파라미터에서 추출
    node_id_match = re.search(r'node-id=([^&]+)', url)
    node_id = unquote(node_id_match.group(1)) if node_id_match else None
    
    return file_id, node_id

def get_input_params() -> tuple[str, str]:
    """
    커맨드 라인 인자와 환경 변수에서 필요한 파라미터를 가져옵니다.
    """
    if len(sys.argv) < 2:
        print("❌ 사용법: python -m srcs.product_planner_agent.product_planner_agent <figma_url>")
        print("예시: python -m srcs.product_planner_agent.product_planner_agent \"https://www.figma.com/file/abc/Project?node-id=1-2\"")
        sys.exit(1)
        
    figma_url = sys.argv[1]
    file_id, node_id = parse_figma_url(figma_url)
    
    if not file_id or not node_id:
        print("❌ 유효하지 않은 Figma URL입니다. URL에 file_id와 node-id가 모두 포함되어 있는지 확인하세요.")
        sys.exit(1)
        
    figma_api_key = env.get("FIGMA_API_KEY", required=True)
    
    return figma_url, figma_api_key

async def run_agent_workflow(figma_url: str, figma_api_key: str) -> bool:
    """Product Planner Agent 워크플로우 실행"""
    file_id, node_id = parse_figma_url(figma_url)
    
    if not file_id or not node_id:
        print("❌ 유효하지 않은 Figma URL입니다. URL에 file_id와 node-id가 모두 포함되어 있는지 확인하세요.")
        return False
        
    if not figma_api_key:
        print("❌ FIGMA_API_KEY가 제공되지 않았습니다.")
        return False

    print("=" * 60)
    print("🚀 Product Planner Agent v3.0 (Streamlined)")
    print(f"📊 분석 대상 Figma URL: {figma_url}")
    print("=" * 60)

    # 이전 상태 파일 삭제
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
        print(f"🧹 이전 상태 파일({STATUS_FILE})을 삭제했습니다.")

    print("\n" + "="*60)
    print("📈 실시간 진행 현황 모니터링")
    print("새 터미널을 열고 아래 명령어를 실행하세요:")
    print(f" streamlit run pages/product_planner.py")
    print("="*60 + "\n")
    
    success = False
    try:
        executive = ExecutiveCoordinator()
        print("🚀 계층형 워크플로우를 시작합니다... (자세한 진행 상황은 Streamlit 페이지를 확인하세요)")

        # ExecutiveCoordinator는 단일 문자열 initial_prompt를 입력으로 받도록 설계됨
        initial_prompt = (
            f"Analyze the Figma design and create a comprehensive product plan.\n"
            f"Figma URL: {figma_url}\n"
            f"(file_id={file_id}, node_id={node_id})"
        )

        result = await executive.run(initial_prompt=initial_prompt)
        if result:
            print("\n✅ 워크플로우가 성공적으로 완료되었습니다.")
            print("📄 최종 보고서가 'planning' 디렉토리에 생성되었습니다.")
            success = True
        else:
            print("\n⚠️ 워크플로우는 완료되었지만, 결과가 없습니다.")
            
    except Exception as e:
        print(f"\n❌ 워크플로우 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        success = False
        
    return success

async def main():
    """Product Planner Agent 메인 실행 함수 (CLI용)"""
    figma_url, figma_api_key = get_input_params()
    
    # CLI에서 실행 시에는 Streamlit 안내 메시지를 약간 다르게 표시할 수 있습니다.
    # 이 부분은 run_agent_workflow 내부와 중복되므로, 필요에 따라 조정이 가능합니다.
    # 여기서는 run_agent_workflow의 출력에 의존합니다.
    
    return await run_agent_workflow(figma_url, figma_api_key)

if __name__ == "__main__":
    # mcp-agent 라이브러리 관련 경고 메시지 무시
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    try:
        is_successful = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        is_successful = False
    except Exception as e:
        print(f"\n❌ 예측하지 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        is_successful = False

    if is_successful:
        print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n💥 워크플로우 실행이 실패했거나 중단되었습니다.")
    
    sys.exit(0 if is_successful else 1) 