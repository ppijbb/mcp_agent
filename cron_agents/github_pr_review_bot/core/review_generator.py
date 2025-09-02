"""
Review Generator - MCP 통합을 통한 코드 리뷰 생성 (NO FALLBACK MODE)

이 모듈은 MCP 통합을 통해 종합적인 코드 리뷰를 생성합니다.
모든 오류는 fallback 없이 즉시 상위로 전파됩니다.
"""

import os
import logging
import json
import sys
from typing import Dict, List, Any, Optional, Tuple

from .config import config
from .mcp_integration import mcp_integration_manager

logger = logging.getLogger(__name__)

class ReviewGenerator:
    """MCP 통합을 사용하여 코드 리뷰를 생성하는 클래스 - NO FALLBACK MODE"""
    
    def __init__(self):
        """
        리뷰 생성기 초기화 - 실패 시 즉시 종료
        """
        try:
            self.mcp_manager = mcp_integration_manager
            if not self.mcp_manager:
                raise ValueError("MCP 통합 관리자 초기화에 실패했습니다.")
            
            logger.info("ReviewGenerator가 MCP 통합으로 초기화되었습니다 (NO FALLBACK MODE)")
        except Exception as e:
            logger.error(f"ReviewGenerator 초기화 중 치명적 오류 발생: {e}")
            if config.llm.fail_on_llm_error:
                sys.exit(1)
            raise
    
    async def generate_review(self, diff_content: str, 
                             pr_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        MCP 통합을 통해 종합적인 코드 리뷰를 생성합니다 - NO FALLBACK
        
        Args:
            diff_content (str): PR의 diff 내용
            pr_metadata (Dict[str, Any], optional): PR 메타데이터 (제목, 설명 등)
            
        Returns:
            Dict[str, Any]: 생성된 리뷰 정보
            
        Raises:
            ValueError: 필수 파라미터가 없거나 리뷰 생성에 실패한 경우
        """
        # 필수 파라미터 검증
        if not diff_content:
            raise ValueError("diff_content가 비어있습니다.")
        
        # PR 메타데이터 준비
        metadata = pr_metadata or {}
        pr_title = metadata.get("title", "")
        pr_description = metadata.get("description", "")
        
        logger.info(f"MCP 통합 리뷰 생성 시작: PR 제목={pr_title[:50]}...")
        
        # 파일 확장자 추출하여 언어 감지
        language = self._detect_language_from_diff(diff_content)
        
        # MCP 통합을 통한 종합 리뷰 생성
        context = {
            "pr_title": pr_title,
            "pr_description": pr_description,
            "diff_content": diff_content
        }
        
        comprehensive_review = await self.mcp_manager.get_comprehensive_review(
            code=diff_content,
            language=language,
            context=context
        )
        
        # 응답 검증
        if not comprehensive_review:
            raise ValueError("MCP 통합으로부터 응답을 받지 못했습니다.")
        
        if "mcp_analyses" not in comprehensive_review:
            raise ValueError("유효한 MCP 분석 데이터가 생성되지 않았습니다.")
        
        if not comprehensive_review["mcp_analyses"]:
            raise ValueError("MCP 분석 결과가 비어있습니다.")
        
        # 리뷰 요약 생성
        review_summary = self._generate_review_summary(comprehensive_review)
        
        result = {
            "review": review_summary,
            "summary": comprehensive_review.get("summary", {}),
            "mcp_analyses": comprehensive_review.get("mcp_analyses", {}),
            "language": language,
            "timestamp": comprehensive_review.get("timestamp")
        }
        
        logger.info("MCP 통합 리뷰 생성 완료")
        return result
    
    async def generate_file_review(self, file_patch: str, 
                                  file_path: str) -> List[Dict[str, Any]]:
        """
        MCP 통합을 통해 특정 파일의 변경사항에 대한 라인별 리뷰를 생성합니다 - NO FALLBACK
        
        Args:
            file_patch (str): 파일의 patch/diff 내용
            file_path (str): 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 라인별 리뷰 코멘트 목록
            
        Raises:
            ValueError: 필수 파라미터가 없거나 파일 리뷰 생성에 실패한 경우
        """
        # 필수 파라미터 검증
        if not file_patch:
            raise ValueError("file_patch가 비어있습니다.")
        if not file_path:
            raise ValueError("file_path가 비어있습니다.")
        
        # 파일 확장자 추출
        file_extension = file_path.split(".")[-1] if "." in file_path else ""
        language = self._get_language_from_extension(file_extension)
        
        logger.info(f"MCP 통합 파일 리뷰 생성 시작: {file_path}")
        
        # MCP 통합을 통한 파일 분석
        file_analysis = await self.mcp_manager.get_specialized_analysis(
            analysis_type='filesystem',
            code=file_patch,
            language=language,
            file_path=file_path
        )
        
        # 응답 검증
        if not file_analysis:
            raise ValueError(f"파일 리뷰 생성 실패: {file_path} - MCP 통합 응답 없음")
        
        if "error" in file_analysis:
            raise ValueError(f"파일 리뷰 생성 실패: {file_path} - {file_analysis['error']}")
        
        if not isinstance(file_analysis, dict):
            raise ValueError(f"파일 분석 결과가 딕셔너리 형태가 아닙니다: {file_path}")
        
        # 라인별 코멘트 생성
        comments = self._generate_line_comments(file_analysis, file_path)
        
        logger.info(f"MCP 통합 파일 리뷰 생성 완료: {file_path}, 코멘트 수: {len(comments)}")
        return comments
    
    async def analyze_code_quality(self, code_content: str, 
                                  file_path: str) -> Dict[str, Any]:
        """
        MCP 통합을 통해 코드 품질을 분석합니다 - NO FALLBACK
        
        Args:
            code_content (str): 코드 내용
            file_path (str): 파일 경로
            
        Returns:
            Dict[str, Any]: 코드 품질 분석 결과
            
        Raises:
            ValueError: 필수 파라미터가 없거나 코드 품질 분석에 실패한 경우
        """
        # 필수 파라미터 검증
        if not code_content:
            raise ValueError("code_content가 비어있습니다.")
        if not file_path:
            raise ValueError("file_path가 비어있습니다.")
        
        # 파일 확장자 추출
        file_extension = file_path.split(".")[-1] if "." in file_path else ""
        language = self._get_language_from_extension(file_extension)
        
        logger.info(f"MCP 통합 코드 품질 분석 시작: {file_path}")
        
        # MCP 통합을 통한 종합 품질 분석
        quality_analysis = await self.mcp_manager.get_specialized_analysis(
            analysis_type='filesystem',
            code=code_content,
            language=language,
            file_path=file_path
        )
        
        # 응답 검증
        if not quality_analysis:
            raise ValueError(f"코드 품질 분석 실패: {file_path} - MCP 통합 응답 없음")
        
        if "error" in quality_analysis:
            raise ValueError(f"코드 품질 분석 실패: {file_path} - {quality_analysis['error']}")
        
        if not isinstance(quality_analysis, dict):
            raise ValueError(f"품질 분석 결과가 딕셔너리 형태가 아닙니다: {file_path}")
        
        # 품질 점수 계산
        quality_score = self._calculate_quality_score(quality_analysis)
        
        result = {
            "quality_score": quality_score,
            "issues": quality_analysis.get("issues", []),
            "recommendations": quality_analysis.get("recommendations", []),
            "language": language,
            "file_path": file_path,
            "analysis_details": quality_analysis
        }
        
        logger.info(f"MCP 통합 코드 품질 분석 완료: {file_path}, 품질 점수: {quality_score}")
        return result
    
    async def generate_summary_review(self, pr_files: List[Dict[str, Any]], 
                                     pr_metadata: Dict[str, Any] = None) -> str:
        """
        MCP 통합을 통해 PR 전체에 대한 요약 리뷰를 생성합니다 - NO FALLBACK
        
        Args:
            pr_files (List[Dict[str, Any]]): PR의 파일 변경사항 목록
            pr_metadata (Dict[str, Any], optional): PR 메타데이터 (제목, 설명 등)
            
        Returns:
            str: 생성된 요약 리뷰
            
        Raises:
            ValueError: 필수 파라미터가 없거나 요약 리뷰 생성에 실패한 경우
        """
        # 필수 파라미터 검증
        if not pr_files:
            raise ValueError("pr_files가 비어있습니다.")
        if not isinstance(pr_files, list):
            raise ValueError("pr_files가 리스트 형태가 아닙니다.")
        
        # PR 메타데이터 준비
        metadata = pr_metadata or {}
        pr_title = metadata.get("title", "")
        pr_description = metadata.get("description", "")
        
        logger.info(f"MCP 통합 요약 리뷰 생성 시작: 파일 수={len(pr_files)}")
        
        # 모든 파일의 변경사항을 하나의 코드로 결합
        combined_code = self._combine_file_changes(pr_files)
        
        # MCP 통합을 통한 종합 분석
        comprehensive_analysis = await self.mcp_manager.get_comprehensive_review(
            code=combined_code,
            language="mixed",  # 여러 언어가 섞여있을 수 있음
            context={
                "pr_title": pr_title,
                "pr_description": pr_description,
                "file_count": len(pr_files),
                "total_changes": sum(f.get("changes", 0) for f in pr_files)
            }
        )
        
        # 응답 검증
        if not comprehensive_analysis:
            raise ValueError("요약 리뷰 생성 실패 - MCP 통합 응답 없음")
        
        if "summary" not in comprehensive_analysis:
            raise ValueError("요약 리뷰 생성 실패 - 유효한 요약 데이터 없음")
        
        if not isinstance(comprehensive_analysis["summary"], dict):
            raise ValueError("요약 데이터가 딕셔너리 형태가 아닙니다.")
        
        # 요약 리뷰 생성
        summary = self._generate_pr_summary(comprehensive_analysis, pr_files, metadata)
        
        if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
            raise ValueError("요약 리뷰 생성 실패 - 요약 내용이 비어있거나 유효하지 않습니다.")
        
        logger.info("MCP 통합 요약 리뷰 생성 완료")
        return summary
    
    # 헬퍼 메서드들
    def _detect_language_from_diff(self, diff_content: str) -> str:
        """diff 내용에서 언어를 감지합니다."""
        # 간단한 언어 감지 로직
        if "def " in diff_content or "import " in diff_content:
            return "python"
        elif "function " in diff_content or "const " in diff_content:
            return "javascript"
        elif "public class" in diff_content or "private " in diff_content:
            return "java"
        elif "#include" in diff_content or "int main" in diff_content:
            return "cpp"
        elif "package " in diff_content or "func " in diff_content:
            return "go"
        else:
            return "unknown"
    
    def _get_language_from_extension(self, extension: str) -> str:
        """파일 확장자에서 언어를 추출합니다."""
        language_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "go": "go",
            "rs": "rust",
            "php": "php",
            "rb": "ruby",
            "swift": "swift",
            "kt": "kotlin",
            "scala": "scala",
            "sh": "bash",
            "sql": "sql"
        }
        return language_map.get(extension.lower(), "unknown")
    
    def _generate_review_summary(self, comprehensive_review: Dict[str, Any]) -> str:
        """종합 리뷰에서 요약을 생성합니다."""
        summary = comprehensive_review.get("summary", {})
        
        review_parts = []
        
        # 전체 분석 결과
        total_analyses = summary.get("total_analyses", 0)
        review_parts.append(f"## 종합 코드 리뷰 결과")
        review_parts.append(f"총 {total_analyses}개의 분석 도구를 사용하여 검토했습니다.\n")
        
        # 중요 이슈
        critical_issues = summary.get("critical_issues", 0)
        high_priority_issues = summary.get("high_priority_issues", 0)
        
        if critical_issues > 0 or high_priority_issues > 0:
            review_parts.append("### 🚨 중요 이슈")
            if critical_issues > 0:
                review_parts.append(f"- 치명적 이슈: {critical_issues}개")
            if high_priority_issues > 0:
                review_parts.append(f"- 높은 우선순위 이슈: {high_priority_issues}개")
            review_parts.append("")
        
        # 보안 발견사항
        security_findings = summary.get("security_findings", [])
        if security_findings:
            review_parts.append("### 🔒 보안 분석")
            for finding in security_findings[:3]:  # 상위 3개만
                review_parts.append(f"- {finding.get('description', '보안 이슈 발견')}")
            review_parts.append("")
        
        # 성능 인사이트
        performance_insights = summary.get("performance_insights", [])
        if performance_insights:
            review_parts.append("### ⚡ 성능 분석")
            for insight in performance_insights[:3]:  # 상위 3개만
                review_parts.append(f"- {insight.get('description', '성능 개선 제안')}")
            review_parts.append("")
        
        # 전문가 인사이트
        expert_insights = summary.get("expert_insights", [])
        if expert_insights:
            review_parts.append("### 👨‍💻 전문가 리뷰")
            for insight in expert_insights[:3]:  # 상위 3개만
                review_parts.append(f"- {insight}")
            review_parts.append("")
        
        # 권장사항
        recommendations = summary.get("recommendations", [])
        if recommendations:
            review_parts.append("### 💡 권장사항")
            for rec in recommendations[:5]:  # 상위 5개만
                review_parts.append(f"- {rec}")
            review_parts.append("")
        
        return "\n".join(review_parts)
    
    def _generate_line_comments(self, file_analysis: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
        """파일 분석 결과에서 라인별 코멘트를 생성합니다."""
        comments = []
        
        issues = file_analysis.get("issues", [])
        if not isinstance(issues, list):
            raise ValueError(f"파일 분석 결과의 issues가 리스트 형태가 아닙니다: {file_path}")
        
        for i, issue in enumerate(issues[:10]):  # 최대 10개 코멘트
            if not isinstance(issue, dict):
                raise ValueError(f"이슈 {i}가 딕셔너리 형태가 아닙니다: {file_path}")
            
            if "line" not in issue:
                raise ValueError(f"이슈 {i}에 line 정보가 없습니다: {file_path}")
            
            if "message" not in issue:
                raise ValueError(f"이슈 {i}에 message 정보가 없습니다: {file_path}")
            
            comment = {
                "path": file_path,
                "position": issue["line"],
                "body": f"**{issue.get('severity', 'INFO').upper()}**: {issue['message']}"
            }
            comments.append(comment)
        
        return comments
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> int:
        """분석 결과에서 품질 점수를 계산합니다."""
        issues = analysis.get("issues", [])
        
        if not isinstance(issues, list):
            raise ValueError("분석 결과의 issues가 리스트 형태가 아닙니다.")
        
        if not issues:
            return 100
        
        # 심각도별 점수 차감
        score = 100
        for i, issue in enumerate(issues):
            if not isinstance(issue, dict):
                raise ValueError(f"이슈 {i}가 딕셔너리 형태가 아닙니다.")
            
            if "severity" not in issue:
                raise ValueError(f"이슈 {i}에 severity 정보가 없습니다.")
            
            severity = issue["severity"].lower()
            if severity == "critical":
                score -= 20
            elif severity == "high":
                score -= 10
            elif severity == "medium":
                score -= 5
            elif severity == "low":
                score -= 2
            else:
                raise ValueError(f"알 수 없는 심각도: {severity}")
        
        return max(0, score)
    
    def _combine_file_changes(self, pr_files: List[Dict[str, Any]]) -> str:
        """PR 파일들의 변경사항을 하나의 코드로 결합합니다."""
        combined = []
        
        for i, file_info in enumerate(pr_files):
            if not isinstance(file_info, dict):
                raise ValueError(f"파일 정보 {i}가 딕셔너리 형태가 아닙니다.")
            
            if "filename" not in file_info:
                raise ValueError(f"파일 정보 {i}에 filename이 없습니다.")
            
            if "patch" not in file_info:
                raise ValueError(f"파일 정보 {i}에 patch가 없습니다.")
            
            filename = file_info["filename"]
            patch = file_info["patch"]
            
            if not filename:
                raise ValueError(f"파일 정보 {i}의 filename이 비어있습니다.")
            
            if not patch:
                raise ValueError(f"파일 정보 {i}의 patch가 비어있습니다.")
            
            combined.append(f"=== {filename} ===")
            combined.append(patch)
            combined.append("")
        
        if not combined:
            raise ValueError("결합할 파일 변경사항이 없습니다.")
        
        return "\n".join(combined)
    
    def _generate_pr_summary(self, analysis: Dict[str, Any], pr_files: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """PR 전체 요약을 생성합니다."""
        if not isinstance(analysis, dict):
            raise ValueError("분석 결과가 딕셔너리 형태가 아닙니다.")
        
        if "summary" not in analysis:
            raise ValueError("분석 결과에 summary가 없습니다.")
        
        summary = analysis["summary"]
        if not isinstance(summary, dict):
            raise ValueError("summary가 딕셔너리 형태가 아닙니다.")
        
        if not isinstance(metadata, dict):
            raise ValueError("메타데이터가 딕셔너리 형태가 아닙니다.")
        
        if not isinstance(pr_files, list):
            raise ValueError("PR 파일 목록이 리스트 형태가 아닙니다.")
        
        summary_parts = []
        
        # PR 제목 검증
        title = metadata.get("title", "")
        if not title:
            raise ValueError("PR 제목이 비어있습니다.")
        
        summary_parts.append(f"## PR 요약: {title}")
        summary_parts.append("")
        
        # 파일 변경 통계
        total_files = len(pr_files)
        if total_files == 0:
            raise ValueError("변경된 파일이 없습니다.")
        
        total_additions = 0
        total_deletions = 0
        
        for i, f in enumerate(pr_files):
            if not isinstance(f, dict):
                raise ValueError(f"파일 정보 {i}가 딕셔너리 형태가 아닙니다.")
            
            additions = f.get("additions", 0)
            deletions = f.get("deletions", 0)
            
            if not isinstance(additions, int) or additions < 0:
                raise ValueError(f"파일 {i}의 additions가 유효하지 않습니다: {additions}")
            
            if not isinstance(deletions, int) or deletions < 0:
                raise ValueError(f"파일 {i}의 deletions가 유효하지 않습니다: {deletions}")
            
            total_additions += additions
            total_deletions += deletions
        
        summary_parts.append(f"### 📊 변경 통계")
        summary_parts.append(f"- 변경된 파일: {total_files}개")
        summary_parts.append(f"- 추가된 라인: {total_additions}줄")
        summary_parts.append(f"- 삭제된 라인: {total_deletions}줄")
        summary_parts.append("")
        
        # 분석 결과 요약
        total_analyses = summary.get("total_analyses", 0)
        critical_issues = summary.get("critical_issues", 0)
        
        if not isinstance(total_analyses, int) or total_analyses < 0:
            raise ValueError(f"total_analyses가 유효하지 않습니다: {total_analyses}")
        
        if not isinstance(critical_issues, int) or critical_issues < 0:
            raise ValueError(f"critical_issues가 유효하지 않습니다: {critical_issues}")
        
        summary_parts.append(f"### 🔍 분석 결과")
        summary_parts.append(f"- 사용된 분석 도구: {total_analyses}개")
        summary_parts.append(f"- 발견된 중요 이슈: {critical_issues}개")
        summary_parts.append("")
        
        # 권장사항
        recommendations = summary.get("recommendations", [])
        if not isinstance(recommendations, list):
            raise ValueError("recommendations가 리스트 형태가 아닙니다.")
        
        if recommendations:
            summary_parts.append("### 💡 주요 권장사항")
            for i, rec in enumerate(recommendations[:3]):
                if not isinstance(rec, str):
                    raise ValueError(f"권장사항 {i}가 문자열이 아닙니다: {rec}")
                if not rec.strip():
                    raise ValueError(f"권장사항 {i}가 비어있습니다.")
                summary_parts.append(f"- {rec}")
            summary_parts.append("")
        
        result = "\n".join(summary_parts)
        if not result.strip():
            raise ValueError("생성된 요약이 비어있습니다.")
        
        return result 