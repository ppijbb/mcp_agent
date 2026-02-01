import argparse
import asyncio
import sys
from pathlib import Path
import json
import re

# 프로젝트 루트 설정
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from srcs.product_planner_agent.product_planner_agent import ProductPlannerAgent
from srcs.product_planner_agent.utils.logger import get_product_planner_logger

# Setup unified logger for this script
logger = get_product_planner_logger("run_script")


async def main():
    """
    Product Planner Agent 실행 스크립트.
    이 스크립트는 에이전트 워크플로우를 시작하는 역할만 합니다.
    모든 MCP/LLM 설정 및 실행 로직은 BaseAgent 아키텍처에 의해 처리됩니다.
    """
    parser = argparse.ArgumentParser(description="Run the Product Planner Agent workflow.")
    parser.add_argument("--product-concept", required=True, help="The high-level concept for the product.")
    parser.add_argument("--user-persona", required=True, help="A description of the target user persona.")
    parser.add_argument("--figma-file-id", help="The file ID of the Figma design (manual override).")
    parser.add_argument("--figma-url", help="Full Figma URL. The file ID will be extracted from this.")
    parser.add_argument("--result-json-path", help="Path to save the final report JSON file.")

    args = parser.parse_args()

    # Determine figma_file_id from URL if provided
    figma_file_id = args.figma_file_id
    if args.figma_url:
        match = re.search(r'file/([a-zA-Z0-9_-]+)', args.figma_url)
        if match:
            figma_file_id_from_url = match.group(1)
            if figma_file_id and figma_file_id != figma_file_id_from_url:
                logger.warning(f"Both --figma-file-id ('{figma_file_id}') and --figma-url (extracted '{figma_file_id_from_url}') were provided. Using the ID from --figma-file-id.")
            elif not figma_file_id:
                figma_file_id = figma_file_id_from_url
                logger.info(f"Extracted Figma File ID '{figma_file_id}' from URL.")
        else:
            logger.warning(f"Could not extract Figma File ID from URL: {args.figma_url}")

    logger.info("🚀 Initializing Product Planner Agent...")

    # 에이전트 인스턴스 생성. BaseAgent.__init__이 MCPApp 설정을 처리합니다.
    product_planner = ProductPlannerAgent()

    logger.info("🚀 Starting Product Planner Workflow...")
    logger.info(f"   - Product Concept: {args.product_concept[:100]}...")
    logger.info(f"   - User Persona: {args.user_persona[:100]}...")
    logger.info(f"   - Figma File ID: {figma_file_id or 'Not provided'}")
    logger.info("-" * 30)

    try:
        # 워크플로우를 위한 초기 컨텍스트 딕셔너리 생성
        initial_context = {
            "product_concept": args.product_concept,
            "user_persona": args.user_persona,
            "figma_file_id": figma_file_id,
        }

        # 에이전트의 run 메서드를 직접 호출.
        # BaseAgent.run이 오류 처리, 재시도, 서킷 브레이커를 포함합니다.
        final_report = await product_planner.run(initial_context)

        logger.info("✅ Workflow finished successfully.")

        # Save the result to a file if path is provided
        if args.result_json_path:
            logger.info(f"💾 Saving final report to {args.result_json_path}")
            try:
                with open(args.result_json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_report, f, indent=2, ensure_ascii=False)
                logger.info("✅ Report saved successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to save report to {args.result_json_path}: {e}")

        # Print to console for debugging or direct execution
        logger.info("Final Report Summary (first 500 chars):")
        # 최종 리포트를 보기 좋게 출력합니다.
        print(json.dumps(final_report, indent=2, ensure_ascii=False)[:500] + "...")


    except Exception as e:
        logger.critical(f"❌ An error occurred during agent execution: {e}", exc_info=True)
        # 에러 발생 시에도 결과 파일에 실패 상태 저장
        if args.result_json_path:
            error_report = {"success": False, "error": str(e)}
            with open(args.result_json_path, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=2, ensure_ascii=False)
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
