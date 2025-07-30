"""
GitHub PR Review Bot - 메인 실행 파일

이 모듈은 GitHub PR 리뷰 봇의 메인 진입점입니다.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from .pr_review_server import GitHubPRReviewServer

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """메인 함수"""
    # 필수 환경 변수 확인
    if not os.environ.get("GITHUB_TOKEN"):
        logger.error("GITHUB_TOKEN 환경 변수가 설정되지 않았습니다.")
        return
    
    # MCP 서버 시작
    server = GitHubPRReviewServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 