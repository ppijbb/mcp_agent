"""
Review Service - Gemini CLI 기반 무료 PR 리뷰 서비스

PR 리뷰 생성 및 관리를 담당하는 서비스입니다.
로컬 분석과 Gemini CLI를 통해 무료로 코드 리뷰를 수행합니다.
"""

import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

from .github_service import GitHubService
from .mcp_service import MCPService
from ..core.config import config

logger = logging.getLogger(__name__)

class ReviewService:
    """리뷰 서비스 - Gemini CLI 기반 무료 AI"""
    
    def __init__(self, github_service: GitHubService, mcp_service: MCPService):
        """리뷰 서비스 초기화 - MCP 통합 (Gemini CLI + vLLM)"""
        self.github_service = github_service
        self.mcp_service = mcp_service
        logger.info("MCP 통합 리뷰 서비스 초기화 완료 (Gemini CLI + vLLM)")
    
    def should_review_pr(self, pr_body: str, pr_data: Dict[str, Any] = None) -> bool:
        """PR 리뷰 여부 결정 - 스마트 필터링"""
        if not config.github.auto_review_enabled:
            return False
        
        # 드래프트 PR 스킵
        if config.github.skip_draft_prs and pr_data and pr_data.get("draft", False):
            logger.info("드래프트 PR이므로 리뷰 스킵")
            return False
        
        # 자동 머지 PR 스킵
        if config.github.skip_auto_merge_prs and pr_data and pr_data.get("auto_merge", False):
            logger.info("자동 머지 PR이므로 리뷰 스킵")
            return False
        
        # PR 크기 체크
        if pr_data:
            additions = pr_data.get("additions", 0)
            deletions = pr_data.get("deletions", 0)
            total_changes = additions + deletions
            
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
        return any(keyword in pr_body for keyword in review_keywords)
    
    def generate_comprehensive_review(self, repo_full_name: str, pr_number: int, github_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """종합적인 PR 리뷰 생성 - GitHub 앱 정보 활용"""
        try:
            # PR 정보 조회
            pr = self.github_service.get_pull_request(repo_full_name, pr_number)
            pr_files = self.github_service.get_pr_files(repo_full_name, pr_number)
            pr_diff = self.github_service.get_pr_diff(repo_full_name, pr_number)
            
            # 언어 감지
            language = self._detect_language(pr_files)
            
            # GitHub 앱에서 제공하는 풍부한 컨텍스트 정보 활용
            enhanced_context = self._build_enhanced_context(
                pr_number=pr_number,
                repo_full_name=repo_full_name,
                pr_files=pr_files,
                github_context=github_context
            )
            
            # MCP 서비스를 통한 AI 분석 (GitHub 앱 정보 활용)
            try:
                mcp_result = self.mcp_service.analyze_code(
                    code=pr_diff,
                    language=language,
                    context=enhanced_context
                )
                
                analysis_result = {
                    "mcp_analysis": mcp_result,
                    "free_ai_review": True
                }
            except Exception as e:
                logger.warning(f"MCP 분석 실패: {e}")
                analysis_result = {
                    "mcp_analysis": None,
                    "error": str(e),
                    "free_ai_review": False
                }
            
            # 리뷰 생성
            review_content = self._format_review(analysis_result, pr, pr_files)
            
            return {
                "pr_number": pr_number,
                "repository": repo_full_name,
                "language": language,
                "analysis": analysis_result,

                "review_content": review_content,
                "timestamp": datetime.now().isoformat(),
                "free_ai_review": True
            }
            
        except Exception as e:
            logger.error(f"종합 리뷰 생성 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"종합 리뷰 생성 실패: {e}")
    
    def _detect_language(self, pr_files: List[Any]) -> str:
        """프로그래밍 언어 감지"""
        if not pr_files:
            return "unknown"
        
        # 파일 확장자 기반 언어 감지
        extensions = {}
        for file in pr_files:
            filename = file.filename
            if '.' in filename:
                ext = filename.split('.')[-1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        # 가장 많은 파일 확장자로 언어 결정
        if extensions:
            most_common_ext = max(extensions, key=extensions.get)
            language_map = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'java': 'java',
                'cpp': 'cpp',
                'c': 'c',
                'cs': 'csharp',
                'go': 'go',
                'rs': 'rust',
                'php': 'php',
                'rb': 'ruby',
                'swift': 'swift',
                'kt': 'kotlin',
                'scala': 'scala',
                'r': 'r',
                'm': 'matlab',
                'sh': 'bash',
                'sql': 'sql',
                'html': 'html',
                'css': 'css',
                'xml': 'xml',
                'json': 'json',
                'yaml': 'yaml',
                'yml': 'yaml',
                'md': 'markdown'
            }
            return language_map.get(most_common_ext, most_common_ext)
        
        return "unknown"
    
    def _build_enhanced_context(self, pr_number: int, repo_full_name: str, pr_files: List[Any], github_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """GitHub 앱 정보를 활용한 향상된 컨텍스트 구축"""
        context = {
            "pr_number": pr_number,
            "repo_full_name": repo_full_name,
            "files": [f.filename for f in pr_files],
            "file_path": f"{repo_full_name}#{pr_number}"
        }
        
        # GitHub 앱에서 제공하는 풍부한 정보 추가
        if github_context:
            # PR 기본 정보
            context.update({
                "pr_title": github_context.get("pr_title"),
                "pr_body": github_context.get("pr_body"),
                "pr_state": github_context.get("pr_state"),
                "pr_draft": github_context.get("pr_draft"),
                "head_ref": github_context.get("head_ref"),
                "base_ref": github_context.get("base_ref"),
                "action": github_context.get("action")
            })
            
            # GitHub 앱에서 제공하는 상세 정보들
            if github_context.get("author"):
                context["author"] = github_context["author"]
            
            if github_context.get("sender"):
                context["sender"] = github_context["sender"]
            
            if github_context.get("repository"):
                context["repository"] = github_context["repository"]
            
            if github_context.get("pr_details"):
                context["pr_details"] = github_context["pr_details"]
            
            if github_context.get("branches"):
                context["branches"] = github_context["branches"]
            
            if github_context.get("labels"):
                context["labels"] = github_context["labels"]
            
            if github_context.get("milestone"):
                context["milestone"] = github_context["milestone"]
            
            if github_context.get("reviewers"):
                context["reviewers"] = github_context["reviewers"]
            
            if github_context.get("stats"):
                context["stats"] = github_context["stats"]
        
        return context
    
    def _format_review(self, analysis_result: Dict[str, Any], pr: Any, pr_files: List[Any]) -> str:
        """리뷰 내용 포맷팅 - MCP 통합 (Gemini CLI + vLLM)"""
        review_parts = []
        
        # 헤더
        review_parts.append(f"## 🔍 PR #{pr.number} 리뷰 (무료 AI 리뷰)")
        review_parts.append("")
        
        # 기본 정보
        review_parts.append("### 📋 기본 정보")
        review_parts.append(f"- **저장소**: {pr.head.repo.full_name}")
        review_parts.append(f"- **브랜치**: {pr.head.ref}")
        review_parts.append(f"- **언어**: {analysis_result.get('language', 'unknown')}")
        review_parts.append(f"- **변경된 파일**: {len(pr_files)}개")
        review_parts.append("")
        
        # MCP AI 분석 결과 (무료)
        if analysis_result.get('mcp_analysis') and not analysis_result.get('error'):
            mcp_analysis = analysis_result['mcp_analysis']
            analysis_type = mcp_analysis.get('analysis_type', 'unknown')
            
            if analysis_type == 'gemini_cli':
                review_parts.append("### 🤖 Gemini AI 분석 결과 (무료)")
            elif analysis_type == 'vllm':
                review_parts.append("### 🚀 vLLM AI 분석 결과 (무료)")
            else:
                review_parts.append("### 🤖 AI 분석 결과 (무료)")
            
            result = mcp_analysis.get('result', {})
            if result.get('review'):
                review_parts.append(result['review'])
            review_parts.append("")
        elif analysis_result.get('error'):
            review_parts.append("### ⚠️ AI 분석 실패")
            review_parts.append(f"MCP 서비스 호출 중 오류가 발생했습니다: {analysis_result['error']}")
            review_parts.append("")
        
        # 무료 AI 리뷰 정보
        if analysis_result.get('free_ai_review'):
            review_parts.append("### 💰 비용 정보")
            if analysis_result.get('mcp_analysis'):
                mcp_analysis = analysis_result['mcp_analysis']
                analysis_type = mcp_analysis.get('analysis_type', 'unknown')
                if analysis_type == 'gemini_cli':
                    review_parts.append("- ✅ **완전 무료**: API 호출 없이 로컬 Gemini CLI 사용")
                    review_parts.append("- ✅ **AI 리뷰**: Google Gemini AI를 통한 고품질 리뷰")
                elif analysis_type == 'vllm':
                    review_parts.append("- ✅ **완전 무료**: 로컬 vLLM 서버 사용")
                    review_parts.append("- ✅ **AI 리뷰**: vLLM을 통한 고품질 리뷰")
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
        review_parts.append(f"*리뷰 생성 시간: {analysis_result.get('timestamp', 'unknown')}*")
        if analysis_result.get('mcp_analysis'):
            mcp_analysis = analysis_result['mcp_analysis']
            analysis_type = mcp_analysis.get('analysis_type', 'unknown')
            if analysis_type == 'gemini_cli':
                review_parts.append("*무료 AI 리뷰 - Gemini CLI 기반*")
            elif analysis_type == 'vllm':
                review_parts.append("*무료 AI 리뷰 - vLLM 기반*")
            else:
                review_parts.append("*무료 AI 리뷰 - MCP 통합*")
        else:
            review_parts.append("*무료 AI 리뷰 - MCP 통합*")
        
        return "\n".join(review_parts)
    
    def post_review(self, repo_full_name: str, pr_number: int, review_content: str) -> Any:
        """리뷰 게시"""
        try:
            review = self.github_service.create_review(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                body=review_content,
                event="COMMENT"
            )
            logger.info(f"리뷰 게시 성공: PR #{pr_number}")
            return review
        except Exception as e:
            logger.error(f"리뷰 게시 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"리뷰 게시 실패: {e}")
    
    def process_pr_review(self, repo_full_name: str, pr_number: int, github_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """PR 리뷰 처리 (전체 프로세스) - GitHub 앱 정보 활용"""
        try:
            # PR 정보 조회
            pr = self.github_service.get_pull_request(repo_full_name, pr_number)
            
            # PR 데이터 준비
            pr_data = {
                "draft": pr.draft,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "auto_merge": getattr(pr, 'auto_merge', False)
            }
            
            # 리뷰 여부 확인
            if not self.should_review_pr(pr.body or "", pr_data):
                logger.info(f"PR #{pr_number} 리뷰 스킵: 필터링 조건에 해당")
                return {"status": "skipped", "reason": "filtered_out"}
            
            # 종합 리뷰 생성 (GitHub 앱 정보 활용)
            review_data = self.generate_comprehensive_review(repo_full_name, pr_number, github_context)
            
            # 리뷰 게시
            posted_review = self.post_review(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                review_content=review_data["review_content"]
            )
            
            return {
                "status": "completed",
                "pr_number": pr_number,
                "repository": repo_full_name,
                "review_id": posted_review.id,
                "review_data": review_data,
                "free_ai_review": True
            }
            
        except Exception as e:
            logger.error(f"PR 리뷰 처리 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"PR 리뷰 처리 실패: {e}")
    
    def get_mcp_usage_stats(self) -> Dict[str, Any]:
        """MCP 사용량 통계 조회"""
        return self.mcp_service.get_usage_stats()
    
    def get_mcp_health(self) -> Dict[str, Any]:
        """MCP 서비스 상태 확인"""
        return self.mcp_service.health_check_all_servers()
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록 조회"""
        return self.mcp_service.get_available_tools()