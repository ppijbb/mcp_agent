"""
GitHub PR Review Bot - Ultra Compact Version

이 모듈은 GitHub PR 리뷰 봇의 메인 진입점입니다.
웹훅 서버와 봇 로직이 모두 통합된 ultra compact한 구조를 제공합니다.
"""

import os
import sys
import asyncio
import logging
import hmac
import hashlib
import json
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

from .core.config import config
from .core.ai import gemini_service, MCPIntegrationManager
from .core.api import GitHubClient

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="GitHub PR Review Bot - Ultra Compact",
    description="GitHub PR 리뷰 자동화 봇 (ultra compact한 구조)",
    version="5.0.0"
)

# 데이터 모델
@dataclass
class PRInfo:
    """PR 정보 - 단일 데이터 모델"""
    pr_number: int
    repo_full_name: str
    pr_title: str
    pr_body: str
    pr_diff: str
    language: str
    files: List[str]
    author: str
    stats: Dict[str, int]
    head_ref: str
    base_ref: str
    action: str
    detailed_changes: Optional[Dict[str, Any]] = None
    line_by_line_changes: Optional[Dict[str, Any]] = None

# 봇 인스턴스
github_client = None
gemini_service_instance = None
mcp_manager = None

def validate_environment():
    """환경 변수 검증"""
    required_vars = [
        'GITHUB_TOKEN'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not getattr(config, var.lower(), None):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        sys.exit(1)
    
    logger.info("환경 변수 검증 완료")

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    global github_client, gemini_service_instance, mcp_manager
    try:
        github_client = GitHubClient()
        gemini_service_instance = gemini_service
        mcp_manager = MCPIntegrationManager()
        logger.info("GitHub PR Review Bot 시작 완료")
    except Exception as e:
        logger.error(f"봇 시작 실패: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise HTTPException(status_code=500, detail="봇 시작 실패")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리"""
    logger.info("GitHub PR Review Bot 종료")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "GitHub PR Review Bot - Ultra Compact",
        "version": "5.0.0",
        "status": "running"
    }

# 봇 로직 함수들
def verify_signature(payload: bytes, signature: str) -> bool:
    """웹훅 서명 검증"""
    if not config.github.webhook_secret:
        logger.error("웹훅 시크릿이 설정되지 않았습니다.")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise ValueError("웹훅 시크릿이 필요합니다.")
    
    try:
        expected_signature = hmac.new(
            config.github.webhook_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        expected_signature = f"sha256={expected_signature}"
        is_valid = hmac.compare_digest(signature, expected_signature)
        
        if is_valid:
            logger.info("웹훅 서명 검증 성공")
        else:
            logger.warning("웹훅 서명 검증 실패")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"웹훅 서명 검증 중 오류: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise ValueError(f"웹훅 서명 검증 실패: {e}")

def parse_payload(payload: bytes) -> Dict[str, Any]:
    """웹훅 페이로드 파싱"""
    try:
        data = json.loads(payload.decode('utf-8'))
        logger.info(f"웹훅 페이로드 파싱 성공: {data.get('action', 'unknown')} 이벤트")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"웹훅 페이로드 JSON 파싱 실패: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise ValueError(f"웹훅 페이로드 파싱 실패: {e}")
    except Exception as e:
        logger.error(f"웹훅 페이로드 파싱 중 오류: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise ValueError(f"웹훅 페이로드 파싱 실패: {e}")

def extract_pr_info(event_data: Dict[str, Any]) -> PRInfo:
    """PR 정보 추출 - 향상된 변경사항 추적"""
    pr = event_data.get('pull_request', {})
    repository = event_data.get('repository', {})
    
    if not pr or not repository:
        raise ValueError("PR 또는 저장소 정보가 없습니다.")
    
    # PR 파일 정보 조회 (향상된 버전)
    pr_files = github_client.get_pr_files(repository['full_name'], pr['number'])
    pr_diff = github_client.get_pr_diff(repository['full_name'], pr['number'])
    
    # 상세한 변경사항 분석
    detailed_changes = github_client.get_detailed_changes(repository['full_name'], pr['number'])
    
    # 라인별 변경사항 분석
    line_by_line_changes = github_client.get_line_by_line_changes(repository['full_name'], pr['number'])
    
    # 언어 감지
    language = detect_language(pr_files)
    
    return PRInfo(
        pr_number=pr['number'],
        repo_full_name=repository['full_name'],
        pr_title=pr.get('title', ''),
        pr_body=pr.get('body', ''),
        pr_diff=pr_diff,
        language=language,
        files=[f['filename'] for f in pr_files],  # 딕셔너리 형태로 변경
        author=pr.get('user', {}).get('login', 'unknown'),
        stats={
            'additions': pr.get('additions', 0),
            'deletions': pr.get('deletions', 0),
            'changed_files': pr.get('changed_files', 0)
        },
        head_ref=pr.get('head', {}).get('ref', ''),
        base_ref=pr.get('base', {}).get('ref', ''),
        action=event_data.get('action', ''),
        detailed_changes=detailed_changes,  # 상세 변경사항 추가
        line_by_line_changes=line_by_line_changes  # 라인별 변경사항 추가
    )

def detect_language(pr_files: List[Any]) -> str:
    """프로그래밍 언어 감지"""
    if not pr_files:
        return "unknown"
    
    extensions = {}
    for file in pr_files:
        # 딕셔너리 형태와 객체 형태 모두 지원
        if isinstance(file, dict):
            filename = file.get('filename', '')
        else:
            filename = getattr(file, 'filename', '')
            
        if '.' in filename:
            ext = filename.split('.')[-1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
    
    if extensions:
        most_common_ext = max(extensions, key=extensions.get)
        language_map = {
            'py': 'python', 'js': 'javascript', 'ts': 'typescript',
            'java': 'java', 'cpp': 'cpp', 'c': 'c', 'cs': 'csharp',
            'go': 'go', 'rs': 'rust', 'php': 'php', 'rb': 'ruby',
            'swift': 'swift', 'kt': 'kotlin', 'scala': 'scala',
            'r': 'r', 'm': 'matlab', 'sh': 'bash', 'sql': 'sql',
            'html': 'html', 'css': 'css', 'xml': 'xml', 'json': 'json',
            'yaml': 'yaml', 'yml': 'yaml', 'md': 'markdown'
        }
        return language_map.get(most_common_ext, most_common_ext)
    
    return "unknown"

def should_review_pr(pr_info: PRInfo) -> bool:
    """PR 리뷰 여부 결정"""
    if not config.github.auto_review_enabled:
        return False
    
    # Draft PR 스킵
    if config.github.skip_draft_prs and pr_info.action == "opened":
        # Draft PR 확인을 위해 PR 객체 조회
        pr = github_client.get_pull_request(pr_info.repo_full_name, pr_info.pr_number)
        if pr.draft:
            logger.info("드래프트 PR이므로 리뷰 스킵")
            return False
    
    # PR 크기 체크
    total_changes = pr_info.stats['additions'] + pr_info.stats['deletions']
    if total_changes < config.github.min_pr_size_threshold:
        logger.info(f"PR 크기가 너무 작아서 리뷰 스킵: {total_changes} < {config.github.min_pr_size_threshold}")
        return False
    
    if total_changes > config.github.max_pr_size_threshold:
        logger.info(f"PR 크기가 너무 커서 리뷰 스킵: {total_changes} > {config.github.max_pr_size_threshold}")
        return False
    
    if not config.github.require_explicit_review_request:
        return True
    
    # 명시적 리뷰 요청 키워드 확인
    review_keywords = ["@review-bot", "[REVIEW]", "[리뷰요청]"]
    return any(keyword in pr_info.pr_body for keyword in review_keywords)

def generate_review(pr_info: PRInfo) -> str:
    """리뷰 생성 - 향상된 변경사항 추적 기반"""
    try:
        # MCP 서비스를 통한 향상된 코드 분석
        mcp_result = mcp_manager.analyze_code(
            code=pr_info.pr_diff,
            language=pr_info.language,
            context={
                "pr_number": pr_info.pr_number,
                "pr_title": pr_info.pr_title,
                "pr_body": pr_info.pr_body,
                "author": pr_info.author,
                "files": pr_info.files,
                "repo_full_name": pr_info.repo_full_name,
                "file_path": f"{pr_info.repo_full_name}#{pr_info.pr_number}",
                "detailed_changes": pr_info.detailed_changes,
                "line_by_line_changes": pr_info.line_by_line_changes
            }
        )
        
        # 리뷰 포맷팅
        review_parts = []
        
        # 헤더
        review_parts.append(f"## 🔍 PR #{pr_info.pr_number} 리뷰 (향상된 변경사항 추적)")
        review_parts.append("")
        
        # 기본 정보
        review_parts.append("### 📋 기본 정보")
        review_parts.append(f"- **저장소**: {pr_info.repo_full_name}")
        review_parts.append(f"- **브랜치**: {pr_info.head_ref} → {pr_info.base_ref}")
        review_parts.append(f"- **언어**: {pr_info.language}")
        review_parts.append(f"- **변경된 파일**: {len(pr_info.files)}개")
        review_parts.append(f"- **변경 통계**: +{pr_info.stats['additions']}/-{pr_info.stats['deletions']}")
        review_parts.append("")
        
        # 상세 변경사항 분석 (새로운 기능)
        if pr_info.detailed_changes:
            review_parts.append("### 🔍 상세 변경사항 분석")
            change_summary = pr_info.detailed_changes.get('summary', {})
            review_parts.append(f"- **총 파일 수**: {change_summary.get('total_files', 0)}개")
            review_parts.append(f"- **총 변경 라인**: {change_summary.get('total_changes', 0)}줄")
            review_parts.append(f"- **커밋 수**: {change_summary.get('commits_count', 0)}개")
            review_parts.append("")
            
            # 변경 카테고리 분석
            categories = pr_info.detailed_changes.get('change_categories', {})
            if categories.get('new_files'):
                review_parts.append("#### 📁 새로 추가된 파일")
                for file_info in categories['new_files']:
                    review_parts.append(f"- `{file_info['filename']}` ({file_info['changes']}줄)")
                review_parts.append("")
            
            if categories.get('deleted_files'):
                review_parts.append("#### 🗑️ 삭제된 파일")
                for file_info in categories['deleted_files']:
                    review_parts.append(f"- `{file_info['filename']}` ({file_info['changes']}줄)")
                review_parts.append("")
            
            if categories.get('critical_files'):
                review_parts.append("#### ⚠️ 중요 파일 변경")
                for file_info in categories['critical_files']:
                    review_parts.append(f"- `{file_info['filename']}` ({file_info['change_type']}, {file_info['changes']}줄)")
                review_parts.append("")
            
            # 영향도 분석
            impact_analysis = pr_info.detailed_changes.get('impact_analysis', {})
            if impact_analysis.get('api_changes'):
                review_parts.append("#### 🔌 API 변경사항")
                for change in impact_analysis['api_changes']:
                    review_parts.append(f"- `{change['file']}`: {change['type']}")
                review_parts.append("")
            
            if impact_analysis.get('breaking_changes'):
                review_parts.append("#### 💥 잠재적 Breaking Changes")
                for change in impact_analysis['breaking_changes']:
                    review_parts.append(f"- `{change['file']}`: {change['type']}")
                review_parts.append("")
            
            if impact_analysis.get('dependency_changes'):
                review_parts.append("#### 📦 의존성 변경사항")
                for change in impact_analysis['dependency_changes']:
                    review_parts.append(f"- `{change['file']}`: {change['type']}")
                review_parts.append("")
            
            # 의미적 변경사항 분석
            semantic_changes = pr_info.detailed_changes.get('semantic_changes', {})
            if any(semantic_changes.values()):
                review_parts.append("#### 🎯 의미적 변경사항")
                for category, changes in semantic_changes.items():
                    if changes:
                        category_name = {
                            'feature_additions': '새 기능 추가',
                            'bug_fixes': '버그 수정',
                            'refactoring': '리팩토링',
                            'performance_improvements': '성능 개선',
                            'security_updates': '보안 업데이트'
                        }.get(category, category)
                        review_parts.append(f"- **{category_name}**: {len(changes)}개 파일")
                review_parts.append("")
        
        # 라인별 변경사항 분석 (새로운 기능)
        if pr_info.line_by_line_changes:
            review_parts.append("### 📝 라인별 변경사항 분석")
            line_summary = pr_info.line_by_line_changes.get('summary', {})
            review_parts.append(f"- **분석된 파일**: {line_summary.get('total_files_analyzed', 0)}개")
            review_parts.append(f"- **추가된 라인**: {line_summary.get('total_lines_added', 0)}줄")
            review_parts.append(f"- **삭제된 라인**: {line_summary.get('total_lines_removed', 0)}줄")
            review_parts.append("")
            
            # 중요 변경사항 표시
            critical_changes = []
            function_changes = []
            import_changes = []
            
            for file_change in pr_info.line_by_line_changes.get('file_changes', []):
                critical_changes.extend(file_change.get('critical_changes', []))
                function_changes.extend(file_change.get('function_changes', []))
                import_changes.extend(file_change.get('import_changes', []))
            
            if critical_changes:
                review_parts.append("#### ⚠️ 중요 변경사항 감지")
                for change in critical_changes[:5]:  # 최대 5개만 표시
                    review_parts.append(f"- `{change['line'][:50]}...` ({change['type']})")
                review_parts.append("")
            
            if function_changes:
                review_parts.append("#### 🔧 함수 변경사항")
                for change in function_changes[:5]:  # 최대 5개만 표시
                    review_parts.append(f"- `{change['function_name']}` ({change['type']})")
                review_parts.append("")
            
            if import_changes:
                review_parts.append("#### 📦 Import 변경사항")
                for change in import_changes[:5]:  # 최대 5개만 표시
                    review_parts.append(f"- `{change['module']}` ({change['type']})")
                review_parts.append("")
        
        # AI 분석 결과
        analysis_type = mcp_result.get('analysis_type', 'unknown')
        if analysis_type == 'mcp_enhanced_gemini_with_external_context':
            review_parts.append("### 🤖 MCP 연동 AI 분석 결과 (외부 코드베이스 포함)")
            review_parts.append("**GitHub 메타데이터, 댓글 분석, 외부 코드베이스 조회 포함**")
        elif analysis_type == 'mcp_enhanced_gemini':
            review_parts.append("### 🤖 MCP 연동 AI 분석 결과 (무료)")
            review_parts.append("**GitHub 메타데이터 및 댓글 분석 포함**")
        else:
            review_parts.append("### 🤖 Gemini AI 분석 결과 (무료)")
        
        result = mcp_result.get('result', {})
        if result.get('review'):
            review_parts.append(result['review'])
        else:
            review_parts.append("AI 분석 결과를 가져올 수 없습니다.")
        review_parts.append("")
        
        # MCP 메타데이터 정보
        if mcp_result.get('github_metadata', {}).get('status') == 'success':
            review_parts.append("### 📊 GitHub 메타데이터")
            review_parts.append("- ✅ PR 상세 정보 수집 완료")
            review_parts.append("")
        
        if mcp_result.get('comments_analysis', {}).get('status') == 'success':
            review_parts.append("### 💬 PR 댓글 분석")
            review_parts.append("- ✅ 기존 댓글 및 피드백 분석 완료")
            review_parts.append("")
        
        # 비용 정보
        review_parts.append("### 💰 비용 정보")
        review_parts.append("- ✅ **완전 무료**: API 호출 없이 로컬 Gemini CLI 사용")
        review_parts.append("- ✅ **AI 리뷰**: Google Gemini AI를 통한 고품질 리뷰")
        review_parts.append("")
        
        # 권장사항
        review_parts.append("### ✅ 권장사항")
        review_parts.append("- 코드 스타일 가이드를 준수하세요")
        review_parts.append("- 테스트 코드를 추가하세요")
        review_parts.append("- 문서화를 업데이트하세요")
        review_parts.append("- 보안 취약점을 확인하세요")
        review_parts.append("")
        
        # 푸터
        review_parts.append("---")
        review_parts.append(f"*리뷰 생성 시간: {datetime.now().isoformat()}*")
        review_parts.append("*무료 AI 리뷰 - Gemini CLI 기반*")
        
        return "\n".join(review_parts)
        
    except Exception as e:
        logger.error(f"리뷰 생성 실패: {e}")
        return f"## ⚠️ 리뷰 생성 실패\n\nAI 분석 중 오류가 발생했습니다: {e}\n\n기본 리뷰를 제공합니다."

def process_pr_review(pr_info: PRInfo) -> Dict[str, Any]:
    """PR 리뷰 처리"""
    try:
        # 리뷰 여부 확인
        if not should_review_pr(pr_info):
            logger.info(f"PR #{pr_info.pr_number} 리뷰 스킵: 필터링 조건에 해당")
            return {"status": "skipped", "reason": "filtered_out"}
        
        # 리뷰 생성
        review_content = generate_review(pr_info)
        
        # 리뷰 게시
        posted_review = github_client.create_review(
            repo_full_name=pr_info.repo_full_name,
            pr_number=pr_info.pr_number,
            body=review_content,
            event="COMMENT"
        )
        
        return {
            "status": "completed",
            "pr_number": pr_info.pr_number,
            "repository": pr_info.repo_full_name,
            "review_id": posted_review.id,
            "free_ai_review": True
        }
        
    except Exception as e:
        logger.error(f"PR 리뷰 처리 실패: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise ValueError(f"PR 리뷰 처리 실패: {e}")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    if not github_client or not gemini_service_instance or not mcp_manager:
        raise HTTPException(status_code=503, detail="봇이 초기화되지 않음")
    
    try:
        mcp_health = mcp_manager.health_check_all_servers()
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "github": "connected",
                "gemini": "available",
                "mcp": mcp_health
            }
        }
        return health_status
    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        raise HTTPException(status_code=500, detail=f"헬스 체크 실패: {e}")

@app.get("/info")
async def get_info():
    """서비스 정보 조회"""
    if not github_client or not gemini_service_instance or not mcp_manager:
        raise HTTPException(status_code=503, detail="봇이 초기화되지 않음")
    
    try:
        return {
            "application": "GitHub PR Review Bot",
            "version": "5.0.0",
            "architecture": "ultra_compact",
            "free_ai_review": True,
            "cost": "$0.00 (완전 무료)"
        }
    except Exception as e:
        logger.error(f"서비스 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"서비스 정보 조회 실패: {e}")

@app.get("/stats")
async def get_stats():
    """사용량 통계 조회"""
    if not github_client or not gemini_service_instance or not mcp_manager:
        raise HTTPException(status_code=503, detail="봇이 초기화되지 않음")
    
    try:
        usage_stats = {
            "gemini_stats": gemini_service_instance.get_usage_stats(),
            "mcp_stats": mcp_manager.get_usage_stats(),
            "timestamp": datetime.now().isoformat()
        }
        return {
            "usage_stats": usage_stats,
            "free_ai_review": True,
            "cost": "$0.00 (완전 무료)"
        }
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {e}")

@app.get("/mcp")
async def get_mcp_info():
    """MCP 사용량 정보 조회"""
    if not github_client or not gemini_service_instance or not mcp_manager:
        raise HTTPException(status_code=503, detail="봇이 초기화되지 않음")
    
    try:
        usage_stats = {
            "gemini_stats": gemini_service_instance.get_usage_stats(),
            "mcp_stats": mcp_manager.get_usage_stats(),
            "timestamp": datetime.now().isoformat()
        }
        available_tools = mcp_manager.get_available_tools()
        mcp_health = mcp_manager.health_check_all_servers()
        
        return {
            "usage_stats": usage_stats,
            "available_tools": available_tools,
            "mcp_health": mcp_health,
            "free_ai_review": True,
            "cost": "$0.00 (완전 무료)",
            "mcp_integration": True
        }
    except Exception as e:
        logger.error(f"MCP 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 정보 조회 실패: {e}")

@app.get("/optimization")
async def get_optimization_status():
    """최적화 상태 조회"""
    try:
        return {
            "gemini_model": config.gemini.gemini_model,
            "gemini_cli_path": config.gemini.gemini_cli_path,
            "aggressive_caching": config.optimization.enable_aggressive_caching,
            "batch_processing": config.optimization.enable_batch_processing,
            "max_requests_per_day": config.gemini.max_requests_per_day,
            "pr_size_thresholds": {
                "min": config.github.min_pr_size_threshold,
                "max": config.github.max_pr_size_threshold
            },
            "skip_draft_prs": config.github.skip_draft_prs,
            "skip_auto_merge_prs": config.github.skip_auto_merge_prs,
            "free_ai_review": True,
            "cost": "$0.00 (완전 무료)"
        }
    except Exception as e:
        logger.error(f"최적화 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"최적화 상태 조회 실패: {e}")

@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """GitHub 웹훅 엔드포인트"""
    if not github_client or not gemini_service_instance or not mcp_manager:
        raise HTTPException(status_code=503, detail="봇이 초기화되지 않음")
    
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
        if event_type == "pull_request":
            # 1. 서명 검증
            if not verify_signature(payload, signature):
                raise ValueError("웹훅 서명 검증 실패")
            
            # 2. 페이로드 파싱
            event_data = parse_payload(payload)
            
            # 3. PR 정보 추출
            pr_info = extract_pr_info(event_data)
            
            # 4. PR 리뷰 처리
            result = process_pr_review(pr_info)
            
            logger.info(f"웹훅 처리 완료: {result.get('status', 'unknown')}")
            return JSONResponse(content=result)
        else:
            logger.info(f"이벤트 타입 '{event_type}'은 처리하지 않음")
            return JSONResponse(content={"status": "ignored", "reason": "unsupported_event_type"})
        
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
    if not github_client or not gemini_service_instance or not mcp_manager:
        raise HTTPException(status_code=503, detail="봇이 초기화되지 않음")
    
    try:
        repo_full_name = f"{repo_owner}/{repo_name}"
        
        logger.info(f"수동 PR 리뷰 요청: {repo_full_name}#{pr_number}")
        
        # PR 정보 조회 및 리뷰 수행
        pr = github_client.get_pull_request(repo_full_name, pr_number)
        pr_files = github_client.get_pr_files(repo_full_name, pr_number)
        pr_diff = github_client.get_pr_diff(repo_full_name, pr_number)
        detailed_changes = github_client.get_detailed_changes(repo_full_name, pr_number)
        line_by_line_changes = github_client.get_line_by_line_changes(repo_full_name, pr_number)
        
        # PR 정보 생성
        pr_info = PRInfo(
            pr_number=pr_number,
            repo_full_name=repo_full_name,
            pr_title=pr.title,
            pr_body=pr.body or '',
            pr_diff=pr_diff,
            language=detect_language(pr_files),
            files=[f['filename'] for f in pr_files],  # 딕셔너리 형태로 변경
            author=pr.user.login,
            stats={
                'additions': pr.additions,
                'deletions': pr.deletions,
                'changed_files': pr.changed_files
            },
            head_ref=pr.head.ref,
            base_ref=pr.base.ref,
            action='manual_review',
            detailed_changes=detailed_changes,
            line_by_line_changes=line_by_line_changes
        )
        
        # 리뷰 처리
        result = process_pr_review(pr_info)
        
        logger.info(f"수동 PR 리뷰 완료: {repo_full_name}#{pr_number}")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"수동 PR 리뷰 실패: {e}")
        if config.github.fail_fast_on_error:
            sys.exit(1)
        raise HTTPException(status_code=500, detail=f"수동 PR 리뷰 실패: {e}")

async def main():
    """메인 함수"""
    try:
        logger.info("GitHub PR Review Bot 시작 - Ultra Compact Version")
        
        # 환경 변수 검증
        validate_environment()
        
        # 설정 정보 로그
        logger.info(f"GitHub 자동 리뷰: {'활성화' if config.github.auto_review_enabled else '비활성화'}")
        logger.info(f"명시적 리뷰 요청 필요: {'예' if config.github.require_explicit_review_request else '아니오'}")
        logger.info(f"즉시 실패 모드: {'활성화' if config.github.fail_fast_on_error else '비활성화'}")
        logger.info("무료 AI 리뷰: Gemini CLI 사용")
        
        # 웹훅 서버 시작
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