"""
GitHub Webhook Server - NO FALLBACK MODE

이 모듈은 GitHub Webhook 이벤트를 수신하고 처리하는 서버를 구현합니다.
기본적으로 PR 리뷰는 비활성화되어 있으며, 오류 발생 시 즉시 종료됩니다.
"""

import os
import hmac
import hashlib
import json
import logging
import sys
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse

from .core.pr_review_server import GitHubPRReviewServer
from .core.config import config

logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="GitHub PR Review Webhook Server - NO FALLBACK MODE",
    description="PR 리뷰는 명시적 요청이 있을 때만 실행됩니다."
)
pr_review_server = GitHubPRReviewServer()

def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    """
    GitHub Webhook 서명 검증 - 실패 시 즉시 False 반환 (NO FALLBACK)
    """
    if not signature_header:
        logger.error("서명 헤더가 없습니다.")
        return False
    
    webhook_secret = config.github.webhook_secret
    if not webhook_secret:
        logger.error("Webhook 시크릿이 설정되지 않았습니다.")
        return False
    
    webhook_secret_bytes = webhook_secret.encode()
    signature_parts = signature_header.split("=")
    
    if len(signature_parts) != 2:
        logger.error("잘못된 서명 형식입니다.")
        return False
    
    algorithm, signature = signature_parts
    if algorithm != "sha256":
        logger.error(f"지원하지 않는 해시 알고리즘: {algorithm}")
        return False
    
    mac = hmac.new(webhook_secret_bytes, msg=payload_body, digestmod=hashlib.sha256)
    expected_signature = mac.hexdigest()
    
    if not hmac.compare_digest(expected_signature, signature):
        logger.error("서명 검증 실패")
        return False
    
    return True

def should_process_pr_event(payload: Dict[str, Any]) -> bool:
    """
    PR 이벤트 처리 여부 결정 - 기본적으로 비활성화
    """
    # 자동 리뷰가 비활성화된 경우
    if not config.github.auto_review_enabled:
        logger.info("자동 PR 리뷰가 비활성화되어 있습니다.")
        
        # 명시적 리뷰 요청이 필요한 경우 확인
        if config.github.require_explicit_review_request:
            pr_body = payload.get("pull_request", {}).get("body", "")
            # PR 설명에 특정 키워드가 있는 경우만 처리
            review_keywords = ["@review-bot", "[REVIEW]", "[리뷰요청]"]
            if not any(keyword in pr_body for keyword in review_keywords):
                logger.info("명시적 리뷰 요청이 없습니다. PR 처리를 건너뜁니다.")
                return False
            
            logger.info("명시적 리뷰 요청이 확인되었습니다. PR 처리를 진행합니다.")
            return True
    
    return True

async def process_pull_request(payload: Dict[str, Any]):
    """
    PR 이벤트 처리 - 오류 발생 시 즉시 종료 (NO FALLBACK)
    """
    try:
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repository = payload.get("repository", {})
        repo_full_name = repository.get("full_name")
        pr_number = pr.get("number")
        
        # 필수 정보 검증
        if not repo_full_name:
            raise ValueError("저장소 정보가 없습니다.")
        if not pr_number:
            raise ValueError("PR 번호가 없습니다.")
        
        # PR 이벤트 처리 여부 확인
        if not should_process_pr_event(payload):
            return
        
        # 처리할 액션인지 확인
        if action not in ["opened", "synchronize"]:
            logger.info(f"처리하지 않는 액션입니다: {action}")
            return
        
        logger.info(f"PR 이벤트 처리 시작: {repo_full_name}#{pr_number}, 액션: {action}")
        
        # PR 리뷰 생성 - 실패 시 즉시 종료
        review_args = {
            "repository": repo_full_name,
            "pr_number": pr_number,
            "review_type": "detailed"
        }
        
        review_result = await pr_review_server._review_pull_request(review_args)
        
        # 리뷰 결과 검증
        if not review_result or not hasattr(review_result, "json"):
            raise ValueError("리뷰 생성에 실패했습니다.")
        
        review_data = review_result.json
        if not review_data or "review" not in review_data:
            raise ValueError("유효한 리뷰 데이터가 생성되지 않았습니다.")
        
        # GitHub에 리뷰 등록
        review = review_data["review"]
        review_body = review.get("review")
        
        if not review_body:
            raise ValueError("리뷰 내용이 비어있습니다.")
        
        submit_args = {
            "repository": repo_full_name,
            "pr_number": pr_number,
            "review_body": review_body,
            "event": "COMMENT"
        }
        
        submit_result = await pr_review_server._submit_review(submit_args)
        
        if not submit_result:
            raise ValueError("리뷰 등록에 실패했습니다.")
        
        logger.info(f"PR 리뷰가 성공적으로 등록되었습니다: {repo_full_name}#{pr_number}")
        
    except Exception as e:
        logger.error(f"PR 처리 중 치명적 오류 발생: {e}")
        if config.github.fail_fast_on_error:
            logger.error("Fail fast 모드로 인해 프로세스를 종료합니다.")
            sys.exit(1)
        else:
            # 오류를 상위로 전파 (fallback 없음)
            raise

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks,
                       x_hub_signature_256: Optional[str] = Header(None)):
    """
    GitHub Webhook 엔드포인트 - NO FALLBACK MODE
    """
    try:
        payload_body = await request.body()
        
        # 서명 검증 (설정된 경우)
        if config.github.webhook_secret:
            if not verify_signature(payload_body, x_hub_signature_256):
                raise HTTPException(status_code=401, detail="Invalid signature")
        
        # JSON 파싱
        try:
            payload = json.loads(payload_body)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # 이벤트 타입 확인
        event_type = request.headers.get("X-GitHub-Event")
        if event_type == "pull_request":
            # 백그라운드에서 PR 처리
            background_tasks.add_task(process_pull_request, payload)
            return JSONResponse({"status": "Processing pull request event"})
        
        logger.info(f"무시된 이벤트 타입: {event_type}")
        return JSONResponse({"status": "Ignored event"})
        
    except HTTPException:
        # HTTP 예외는 그대로 전파
        raise
    except Exception as e:
        logger.error(f"웹훅 처리 중 오류 발생: {e}")
        if config.github.fail_fast_on_error:
            logger.error("Fail fast 모드로 인해 오류를 HTTP 예외로 변환합니다.")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """
    상태 확인 엔드포인트
    """
    return {
        "status": "healthy",
        "mode": "NO FALLBACK",
        "auto_review_enabled": config.github.auto_review_enabled,
        "fail_fast_on_error": config.github.fail_fast_on_error
    }

@app.get("/config")
async def get_config():
    """
    현재 설정 정보 반환 (민감한 정보 제외)
    """
    return {
        "github": {
            "auto_review_enabled": config.github.auto_review_enabled,
            "require_explicit_review_request": config.github.require_explicit_review_request,
            "fail_fast_on_error": config.github.fail_fast_on_error,
            "max_retry_attempts": config.github.max_retry_attempts
        },
        "llm": {
            "default_provider": config.llm.default_provider,
            "fail_on_llm_error": config.llm.fail_on_llm_error,
            "require_valid_response": config.llm.require_valid_response
        }
    }
