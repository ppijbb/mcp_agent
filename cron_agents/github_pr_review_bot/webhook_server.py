"""
GitHub Webhook Server - Modular Architecture

이 서버는 GitHub 웹훅 이벤트를 받아서 PR 리뷰를 자동으로 수행합니다.
모듈화된 아키텍처를 사용하여 관리하기 쉽고 확장 가능한 구조를 제공합니다.
"""

import logging
import sys
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .core.config import config
from .application import PRReviewApp

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="GitHub PR Review Bot - Modular Architecture",
    description="GitHub PR 리뷰 자동화 봇 (모듈화된 구조)",
    version="3.0.0"
)

# 애플리케이션 인스턴스
pr_review_app = None

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    global pr_review_app
    try:
        pr_review_app = PRReviewApp()
        logger.info("PR 리뷰 애플리케이션 시작 완료")
    except Exception as e:
        logger.error(f"애플리케이션 시작 실패: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise HTTPException(status_code=500, detail="애플리케이션 시작 실패")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리"""
    logger.info("PR 리뷰 애플리케이션 종료")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "GitHub PR Review Bot - Modular Architecture",
        "version": "3.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    if not pr_review_app:
        raise HTTPException(status_code=503, detail="애플리케이션이 초기화되지 않음")
    
    try:
        health_status = pr_review_app.get_health_status()
        return health_status
    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        raise HTTPException(status_code=500, detail=f"헬스 체크 실패: {e}")

@app.get("/info")
async def get_info():
    """서비스 정보 조회"""
    if not pr_review_app:
        raise HTTPException(status_code=503, detail="애플리케이션이 초기화되지 않음")
    
    try:
        service_info = pr_review_app.get_service_info()
        return {
            "application": "GitHub PR Review Bot",
            "version": "3.0.0",
            "architecture": "modular",
            "services": service_info
        }
    except Exception as e:
        logger.error(f"서비스 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"서비스 정보 조회 실패: {e}")

@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """GitHub 웹훅 엔드포인트"""
    if not pr_review_app:
        raise HTTPException(status_code=503, detail="애플리케이션이 초기화되지 않음")
    
    try:
        # 요청 데이터 가져오기
        payload = await request.body()
        signature = request.headers.get("X-Hub-Signature-256", "")
        event_type = request.headers.get("X-GitHub-Event", "")
        
        if not signature:
            logger.warning("웹훅 서명이 없습니다.")
            raise HTTPException(status_code=400, detail="웹훅 서명이 필요합니다.")
        
        if not event_type:
            logger.warning("GitHub 이벤트 타입이 없습니다.")
            raise HTTPException(status_code=400, detail="GitHub 이벤트 타입이 필요합니다.")
        
        logger.info(f"웹훅 이벤트 수신: {event_type}")
        
        # 웹훅 이벤트 처리
        result = pr_review_app.process_webhook(
            event_type=event_type,
            payload=payload,
            signature=signature
        )
        
        logger.info(f"웹훅 처리 완료: {result.get('status', 'unknown')}")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"웹훅 처리 실패: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise HTTPException(status_code=500, detail=f"웹훅 처리 실패: {e}")

@app.post("/review/{repo_owner}/{repo_name}/{pr_number}")
async def manual_review(repo_owner: str, repo_name: str, pr_number: int):
    """수동 PR 리뷰 엔드포인트"""
    if not pr_review_app:
        raise HTTPException(status_code=503, detail="애플리케이션이 초기화되지 않음")
    
    try:
        repo_full_name = f"{repo_owner}/{repo_name}"
        
        logger.info(f"수동 PR 리뷰 요청: {repo_full_name}#{pr_number}")
        
        # PR 리뷰 수행
        result = pr_review_app.review_pr(
            repo_full_name=repo_full_name,
            pr_number=pr_number
        )
        
        logger.info(f"수동 PR 리뷰 완료: {repo_full_name}#{pr_number}")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"수동 PR 리뷰 실패: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise HTTPException(status_code=500, detail=f"수동 PR 리뷰 실패: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # 서버 실행
    uvicorn.run(
        "webhook_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )