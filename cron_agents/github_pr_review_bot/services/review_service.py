"""
Review Service

PR 리뷰 생성 및 관리를 담당하는 서비스입니다.
코드 분석, 리뷰 생성, 품질 평가 등의 기능을 제공합니다.
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
    """리뷰 서비스"""
    
    def __init__(self, github_service: GitHubService, mcp_service: MCPService):
        """리뷰 서비스 초기화"""
        self.github_service = github_service
        self.mcp_service = mcp_service
        logger.info("리뷰 서비스 초기화 완료")
    
    def should_review_pr(self, pr_body: str) -> bool:
        """PR 리뷰 여부 결정"""
        if not config.github.auto_review_enabled:
            return False
        
        if not config.github.require_explicit_review_request:
            return True
        
        # 명시적 리뷰 요청 키워드 확인
        review_keywords = ["@review-bot", "[REVIEW]", "[리뷰요청]"]
        return any(keyword in pr_body for keyword in review_keywords)
    
    def generate_comprehensive_review(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """종합적인 PR 리뷰 생성"""
        try:
            # PR 정보 조회
            pr = self.github_service.get_pull_request(repo_full_name, pr_number)
            pr_files = self.github_service.get_pr_files(repo_full_name, pr_number)
            pr_diff = self.github_service.get_pr_diff(repo_full_name, pr_number)
            
            # 언어 감지
            language = self._detect_language(pr_files)
            
            # 코드 분석
            analysis_result = self.mcp_service.analyze_code(
                code=pr_diff,
                language=language,
                context={
                    "pr_number": pr_number,
                    "repository": repo_full_name,
                    "files": [f.filename for f in pr_files]
                }
            )
            
            # 리뷰 생성
            review_content = self._format_review(analysis_result, pr, pr_files)
            
            return {
                "pr_number": pr_number,
                "repository": repo_full_name,
                "language": language,
                "analysis": analysis_result,
                "review_content": review_content,
                "timestamp": datetime.now().isoformat()
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
    
    def _format_review(self, analysis_result: Dict[str, Any], pr: Any, pr_files: List[Any]) -> str:
        """리뷰 내용 포맷팅"""
        review_parts = []
        
        # 헤더
        review_parts.append(f"## 🔍 PR #{pr.number} 리뷰")
        review_parts.append("")
        
        # 기본 정보
        review_parts.append("### 📋 기본 정보")
        review_parts.append(f"- **저장소**: {pr.head.repo.full_name}")
        review_parts.append(f"- **브랜치**: {pr.head.ref}")
        review_parts.append(f"- **언어**: {analysis_result.get('language', 'unknown')}")
        review_parts.append(f"- **변경된 파일**: {len(pr_files)}개")
        review_parts.append("")
        
        # 분석 결과
        if analysis_result.get('analysis_results'):
            review_parts.append("### 🔬 분석 결과")
            for analysis_type, result in analysis_result['analysis_results'].items():
                review_parts.append(f"#### {analysis_type.replace('_', ' ').title()}")
                if isinstance(result, dict) and 'result' in result:
                    review_parts.append(str(result['result'])[:500] + "...")
                else:
                    review_parts.append(str(result)[:500] + "...")
                review_parts.append("")
        
        # 최종 리뷰
        if analysis_result.get('final_review'):
            review_parts.append("### 💡 종합 리뷰")
            review_parts.append(analysis_result['final_review'])
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
    
    def process_pr_review(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """PR 리뷰 처리 (전체 프로세스)"""
        try:
            # PR 정보 조회
            pr = self.github_service.get_pull_request(repo_full_name, pr_number)
            
            # 리뷰 여부 확인
            if not self.should_review_pr(pr.body or ""):
                logger.info(f"PR #{pr_number} 리뷰 스킵: 명시적 요청 없음")
                return {"status": "skipped", "reason": "no_explicit_request"}
            
            # 종합 리뷰 생성
            review_data = self.generate_comprehensive_review(repo_full_name, pr_number)
            
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
                "review_data": review_data
            }
            
        except Exception as e:
            logger.error(f"PR 리뷰 처리 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"PR 리뷰 처리 실패: {e}")
