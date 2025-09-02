"""
GitHub PR Review Bot - Modular Architecture

이 모듈은 GitHub PR 리뷰 봇의 메인 진입점입니다.
모듈화된 아키텍처를 사용하여 관리하기 쉽고 확장 가능한 구조를 제공합니다.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

from .core.config import config

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def validate_environment():
    """환경 변수 검증"""
    required_vars = [
        'GITHUB_TOKEN',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not getattr(config, var.lower(), None):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        sys.exit(1)
    
    logger.info("환경 변수 검증 완료")

async def main():
    """메인 함수"""
    try:
        logger.info("GitHub PR Review Bot 시작 - Modular Architecture")
        
        # 환경 변수 검증
        validate_environment()
        
        # 설정 정보 로그
        logger.info(f"GitHub 자동 리뷰: {'활성화' if config.github.auto_review_enabled else '비활성화'}")
        logger.info(f"명시적 리뷰 요청 필요: {'예' if config.github.require_explicit_review_request else '아니오'}")
        logger.info(f"즉시 실패 모드: {'활성화' if config.github.fail_fast_on_error else '비활성화'}")
        
        # 웹훅 서버 시작
        from .webhook_server import app
        import uvicorn
        
        logger.info("웹훅 서버 시작 중...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(0)
    except Exception as e:
        logger.error(f"애플리케이션 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())