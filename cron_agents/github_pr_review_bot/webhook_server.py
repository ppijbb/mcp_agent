"""
GitHub Webhook Server

이 모듈은 GitHub Webhook 이벤트를 수신하고 처리하는 서버를 구현합니다.
"""

import os
import hmac
import hashlib
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse

from .core.pr_review_server import GitHubPRReviewServer

logger = logging.getLogger(__name__)
app = FastAPI(title="GitHub PR Review Webhook Server")
pr_review_server = GitHubPRReviewServer()

def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    """GitHub Webhook 서명 검증"""
    if not signature_header:
        return False
    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "").encode()
    if not webhook_secret:
        return False
    signature_parts = signature_header.split("=")
    if len(signature_parts) != 2:
        return False
    algorithm, signature = signature_parts
    if algorithm != "sha256":
        return False
    mac = hmac.new(webhook_secret, msg=payload_body, digestmod=hashlib.sha256)
    expected_signature = mac.hexdigest()
    return hmac.compare_digest(expected_signature, signature)

async def process_pull_request(payload: Dict[str, Any]):
    """PR 이벤트 처리"""
    try:
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repository = payload.get("repository", {})
        repo_full_name = repository.get("full_name")
        pr_number = pr.get("number")
        
        if not repo_full_name or not pr_number:
            return
        
        if action in ["opened", "synchronize"]:
            logger.info(f"PR 이벤트 감지: {repo_full_name}#{pr_number}, 액션: {action}")
            
            # PR 리뷰 생성
            review_args = {
                "repository": repo_full_name,
                "pr_number": pr_number,
                "review_type": "detailed"
            }
            review_result = await pr_review_server._review_pull_request(review_args)
            
            # GitHub에 리뷰 등록
            if hasattr(review_result, "json") and review_result.json:
                review_data = review_result.json
                if "review" in review_data:
                    review = review_data["review"]
                    review_body = review.get("review", "자동 생성된 코드 리뷰")
                    
                    submit_args = {
                        "repository": repo_full_name,
                        "pr_number": pr_number,
                        "review_body": review_body,
                        "event": "COMMENT"
                    }
                    await pr_review_server._submit_review(submit_args)
    except Exception as e:
        logger.error(f"PR 처리 중 오류 발생: {e}")

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks,
                       x_hub_signature_256: Optional[str] = Header(None)):
    """GitHub Webhook 엔드포인트"""
    payload_body = await request.body()
    
    if os.environ.get("GITHUB_WEBHOOK_SECRET"):
        if not verify_signature(payload_body, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        payload = json.loads(payload_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    event_type = request.headers.get("X-GitHub-Event")
    if event_type == "pull_request":
        background_tasks.add_task(process_pull_request, payload)
        return JSONResponse({"status": "Processing pull request event"})
    
    return JSONResponse({"status": "Ignored event"})

@app.get("/health")
async def health_check():
    """상태 확인 엔드포인트"""
    return {"status": "healthy"}
