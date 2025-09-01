"""
GitHub PR Review Bot - 메인 실행 파일

이 모듈은 GitHub PR 리뷰 봇의 메인 진입점입니다.
Fallback 없이 오류 발생 시 즉시 종료됩니다.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

from .core.pr_review_server import GitHubPRReviewServer
from .core.config import config

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def main():
    """
    메인 함수 - 모든 오류는 즉시 종료됩니다 (NO FALLBACK)
    """
    try:
        # 필수 환경 변수 확인 - 없으면 즉시 종료
        if not config.github.token:
            logger.error("GITHUB_TOKEN 환경 변수가 설정되지 않았습니다.")
            sys.exit(1)
        
        # LLM API 키 확인 - 없으면 즉시 종료
        if not any([
            config.llm.openai_api_key,
            config.llm.anthropic_api_key,
            config.llm.google_api_key
        ]):
            logger.error("LLM API 키가 설정되지 않았습니다. (OpenAI, Anthropic, Google 중 하나 필요)")
            sys.exit(1)
        
        # 기본 PR 리뷰 비활성화 확인
        if config.github.auto_review_enabled:
            logger.warning("자동 PR 리뷰가 활성화되어 있습니다. 보안을 위해 비활성화를 권장합니다.")
        
        logger.info("GitHub PR Review Bot 시작 - NO FALLBACK MODE")
        logger.info(f"Auto Review Enabled: {config.github.auto_review_enabled}")
        logger.info(f"Fail Fast on Error: {config.github.fail_fast_on_error}")
        
        # MCP 서버 시작 - 오류 발생 시 즉시 종료
        server = GitHubPRReviewServer()
        await server.run()
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"치명적 오류 발생: {e}")
        logger.error("Fallback 없이 즉시 종료됩니다.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 