"""
Review Enhancer - 고도화된 코드 리뷰 강화 도구

이 모듈은 코드 리뷰의 성능을 크게 향상시키는 다양한 분석 도구들을 제공합니다.
웹 검색, 보안 분석, 성능 분석, 아키텍처 분석 등을 통합합니다.
"""

import logging
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import httpx
from bs4 import BeautifulSoup
import ast
import subprocess
import tempfile
import os

from .config import config

logger = logging.getLogger(__name__)

class SecurityAnalyzer:
    """보안 취약점 분석기"""
    
    def __init__(self):
        self.vulnerability_patterns = {
            'sql_injection': [
                r'execute\s*\(\s*[\'"][^\'"]*[\'"]\s*\+\s*\w+',
                r'cursor\.execute\s*\(\s*[\'"][^\'"]*[\'"]\s*\+\s*\w+',
                r'f\s*[\'"][^\'"]*\{[^}]*\}\s*[\'"]',
            ],
            'xss': [
                r'innerHTML\s*=\s*\w+',
                r'document\.write\s*\(\s*\w+',
                r'\.html\s*\(\s*\w+',
            ],
            'path_traversal': [
                r'open\s*\(\s*\w+\s*\+\s*[\'"]\.\.',
                r'file_get_contents\s*\(\s*\w+\s*\+\s*[\'"]\.\.',
            ],
            'command_injection': [
                r'os\.system\s*\(\s*\w+',
                r'subprocess\.run\s*\(\s*\w+',
                r'exec\s*\(\s*\w+',
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*[\'"][^\'"]{8,}[\'"]',
                r'api_key\s*=\s*[\'"][^\'"]{20,}[\'"]',
                r'secret\s*=\s*[\'"][^\'"]{10,}[\'"]',
            ]
        }
    
    def analyze_security_vulnerabilities(self, code: str, language: str) -> Dict[str, Any]:
        """보안 취약점 분석"""
        vulnerabilities = []
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    vulnerabilities.append({
                        'type': vuln_type,
                        'severity': self._get_severity(vuln_type),
                        'line': line_num,
                        'code': match.group(),
                        'description': self._get_vulnerability_description(vuln_type),
                        'recommendation': self._get_recommendation(vuln_type)
                    })
        
        return {
            'vulnerabilities': vulnerabilities,
            'total_count': len(vulnerabilities),
            'high_severity': len([v for v in vulnerabilities if v['severity'] == 'high']),
            'medium_severity': len([v for v in vulnerabilities if v['severity'] == 'medium']),
            'low_severity': len([v for v in vulnerabilities if v['severity'] == 'low'])
        }
    
    def _get_severity(self, vuln_type: str) -> str:
        """취약점 심각도 반환"""
        severity_map = {
            'sql_injection': 'high',
            'xss': 'high',
            'command_injection': 'high',
            'path_traversal': 'medium',
            'hardcoded_secrets': 'medium'
        }
        return severity_map.get(vuln_type, 'low')
    
    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """취약점 설명 반환"""
        descriptions = {
            'sql_injection': 'SQL injection vulnerability detected',
            'xss': 'Cross-site scripting vulnerability detected',
            'path_traversal': 'Path traversal vulnerability detected',
            'command_injection': 'Command injection vulnerability detected',
            'hardcoded_secrets': 'Hardcoded secrets detected'
        }
        return descriptions.get(vuln_type, 'Security vulnerability detected')
    
    def _get_recommendation(self, vuln_type: str) -> str:
        """보안 권장사항 반환"""
        recommendations = {
            'sql_injection': 'Use parameterized queries or ORM',
            'xss': 'Sanitize user input and use proper encoding',
            'path_traversal': 'Validate and sanitize file paths',
            'command_injection': 'Avoid dynamic command execution',
            'hardcoded_secrets': 'Use environment variables or secure vaults'
        }
        return recommendations.get(vuln_type, 'Review and fix security issue')

class PerformanceAnalyzer:
    """성능 분석기"""
    
    def __init__(self):
        self.performance_issues = {
            'n_plus_one': [
                r'for\s+\w+\s+in\s+\w+\.all\(\):',
                r'for\s+\w+\s+in\s+\w+\.filter\(\):',
            ],
            'inefficient_queries': [
                r'\.all\(\)\.filter\(',
                r'\.filter\(\)\.all\(\)',
            ],
            'memory_leaks': [
                r'global\s+\w+',
                r'\.append\(\)\s+in\s+loop',
            ],
            'expensive_operations': [
                r'\.sort\(\)\s+in\s+loop',
                r'\.reverse\(\)\s+in\s+loop',
            ]
        }
    
    def analyze_performance_issues(self, code: str, language: str) -> Dict[str, Any]:
        """성능 이슈 분석"""
        issues = []
        
        for issue_type, patterns in self.performance_issues.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    issues.append({
                        'type': issue_type,
                        'line': line_num,
                        'code': match.group(),
                        'description': self._get_performance_description(issue_type),
                        'impact': self._get_performance_impact(issue_type),
                        'optimization': self._get_optimization_suggestion(issue_type)
                    })
        
        return {
            'issues': issues,
            'total_count': len(issues),
            'critical': len([i for i in issues if i['impact'] == 'critical']),
            'high': len([i for i in issues if i['impact'] == 'high']),
            'medium': len([i for i in issues if i['impact'] == 'medium'])
        }
    
    def _get_performance_description(self, issue_type: str) -> str:
        """성능 이슈 설명"""
        descriptions = {
            'n_plus_one': 'N+1 query problem detected',
            'inefficient_queries': 'Inefficient database query detected',
            'memory_leaks': 'Potential memory leak detected',
            'expensive_operations': 'Expensive operation in loop detected'
        }
        return descriptions.get(issue_type, 'Performance issue detected')
    
    def _get_performance_impact(self, issue_type: str) -> str:
        """성능 영향도"""
        impact_map = {
            'n_plus_one': 'critical',
            'inefficient_queries': 'high',
            'memory_leaks': 'high',
            'expensive_operations': 'medium'
        }
        return impact_map.get(issue_type, 'low')
    
    def _get_optimization_suggestion(self, issue_type: str) -> str:
        """최적화 제안"""
        suggestions = {
            'n_plus_one': 'Use select_related() or prefetch_related()',
            'inefficient_queries': 'Optimize query with proper indexing',
            'memory_leaks': 'Use local variables instead of globals',
            'expensive_operations': 'Move expensive operations outside loops'
        }
        return suggestions.get(issue_type, 'Review and optimize code')

class CodeQualityAnalyzer:
    """코드 품질 분석기"""
    
    def __init__(self):
        self.quality_metrics = {
            'complexity': self._analyze_complexity,
            'maintainability': self._analyze_maintainability,
            'readability': self._analyze_readability,
            'test_coverage': self._analyze_test_coverage
        }
    
    def analyze_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """코드 품질 종합 분석"""
        results = {}
        
        for metric_name, analyzer_func in self.quality_metrics.items():
            results[metric_name] = analyzer_func(code, language)
        
        return results
    
    def _analyze_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """복잡도 분석"""
        lines = code.split('\n')
        complexity_score = 0
        
        for line in lines:
            # 조건문, 반복문, 중첩 등 복잡도 요소 계산
            if re.search(r'\bif\b|\bfor\b|\bwhile\b|\btry\b', line):
                complexity_score += 1
            if re.search(r'\.\w+\(.*\)\.\w+\(', line):  # 메서드 체이닝
                complexity_score += 0.5
        
        return {
            'score': complexity_score,
            'level': 'high' if complexity_score > 10 else 'medium' if complexity_score > 5 else 'low',
            'description': f'Cyclomatic complexity: {complexity_score}'
        }
    
    def _analyze_maintainability(self, code: str, language: str) -> Dict[str, Any]:
        """유지보수성 분석"""
        maintainability_issues = []
        
        # 긴 함수 검출
        functions = re.findall(r'def\s+\w+\([^)]*\):.*?(?=def|\Z)', code, re.DOTALL)
        for func in functions:
            lines = func.split('\n')
            if len(lines) > 50:
                maintainability_issues.append({
                    'type': 'long_function',
                    'description': f'Function with {len(lines)} lines detected'
                })
        
        # 중복 코드 검출
        lines = code.split('\n')
        line_counts = {}
        for line in lines:
            clean_line = line.strip()
            if len(clean_line) > 10:
                line_counts[clean_line] = line_counts.get(clean_line, 0) + 1
        
        duplicates = {line: count for line, count in line_counts.items() if count > 3}
        if duplicates:
            maintainability_issues.append({
                'type': 'duplicate_code',
                'description': f'{len(duplicates)} duplicate code blocks detected'
            })
        
        return {
            'issues': maintainability_issues,
            'score': max(0, 10 - len(maintainability_issues)),
            'level': 'low' if len(maintainability_issues) > 5 else 'medium' if len(maintainability_issues) > 2 else 'high'
        }
    
    def _analyze_readability(self, code: str, language: str) -> Dict[str, Any]:
        """가독성 분석"""
        readability_issues = []
        
        # 긴 라인 검출
        lines = code.split('\n')
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            readability_issues.append({
                'type': 'long_lines',
                'lines': long_lines,
                'description': f'{len(long_lines)} lines exceed 120 characters'
            })
        
        # 주석 비율
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_ratio = comment_lines / max(code_lines, 1)
        
        if comment_ratio < 0.1:
            readability_issues.append({
                'type': 'low_comments',
                'description': f'Comment ratio: {comment_ratio:.1%} (recommended: >10%)'
            })
        
        return {
            'issues': readability_issues,
            'comment_ratio': comment_ratio,
            'score': max(0, 10 - len(readability_issues)),
            'level': 'low' if len(readability_issues) > 3 else 'medium' if len(readability_issues) > 1 else 'high'
        }
    
    def _analyze_test_coverage(self, code: str, language: str) -> Dict[str, Any]:
        """테스트 커버리지 분석"""
        # 간단한 테스트 파일 존재 여부 확인
        test_indicators = [
            'test_', 'Test', 'unittest', 'pytest', 'assert'
        ]
        
        test_score = 0
        for indicator in test_indicators:
            if indicator in code:
                test_score += 1
        
        return {
            'score': test_score,
            'level': 'high' if test_score >= 3 else 'medium' if test_score >= 1 else 'low',
            'description': f'Test coverage indicators: {test_score}/4'
        }

class WebSearchEnhancer:
    """웹 검색 기반 리뷰 강화"""
    
    def __init__(self):
        self.search_apis = {
            'stackoverflow': 'https://api.stackexchange.com/2.3/search',
            'github': 'https://api.github.com/search',
            'npm': 'https://registry.npmjs.org',
            'pypi': 'https://pypi.org/pypi'
        }
    
    async def search_best_practices(self, language: str, framework: str = None) -> List[Dict[str, Any]]:
        """모범 사례 검색"""
        results = []
        
        # Stack Overflow 검색
        async with httpx.AsyncClient() as client:
            query = f"{language} best practices"
            if framework:
                query += f" {framework}"
            
            try:
                response = await client.get(
                    self.search_apis['stackoverflow'],
                    params={
                        'order': 'desc',
                        'sort': 'votes',
                        'tagged': language,
                        'intitle': query,
                        'site': 'stackoverflow'
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', [])[:5]:
                        results.append({
                            'source': 'Stack Overflow',
                            'title': item['title'],
                            'url': item['link'],
                            'score': item['score'],
                            'tags': item.get('tags', [])
                        })
            except Exception as e:
                logger.error(f"Stack Overflow search error: {e}")
        
        return results
    
    async def search_security_vulnerabilities(self, language: str, library: str = None) -> List[Dict[str, Any]]:
        """보안 취약점 정보 검색"""
        results = []
        
        # GitHub Security Advisories 검색
        async with httpx.AsyncClient() as client:
            query = f"{language} security vulnerability"
            if library:
                query += f" {library}"
            
            try:
                response = await client.get(
                    f"{self.search_apis['github']}/repositories",
                    params={'q': query, 'sort': 'updated'},
                    headers={'Accept': 'application/vnd.github.v3+json'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', [])[:3]:
                        results.append({
                            'source': 'GitHub',
                            'name': item['name'],
                            'url': item['html_url'],
                            'description': item['description'],
                            'language': item['language']
                        })
            except Exception as e:
                logger.error(f"GitHub search error: {e}")
        
        return results
    
    async def search_alternatives(self, library: str) -> List[Dict[str, Any]]:
        """대안 라이브러리 검색"""
        results = []
        
        # npm/pypi 검색
        async with httpx.AsyncClient() as client:
            try:
                if library.endswith('.js') or 'npm' in library:
                    # npm 검색
                    response = await client.get(f"{self.search_apis['npm']}/{library}")
                    if response.status_code == 200:
                        data = response.json()
                        results.append({
                            'source': 'npm',
                            'name': data['name'],
                            'version': data.get('dist-tags', {}).get('latest'),
                            'description': data.get('description', ''),
                            'dependencies': len(data.get('dependencies', {}))
                        })
                else:
                    # PyPI 검색
                    response = await client.get(f"{self.search_apis['pypi']}/{library}/json")
                    if response.status_code == 200:
                        data = response.json()
                        results.append({
                            'source': 'PyPI',
                            'name': data['info']['name'],
                            'version': data['info']['version'],
                            'description': data['info'].get('summary', ''),
                            'dependencies': len(data['info'].get('requires_dist', []))
                        })
            except Exception as e:
                logger.error(f"Package search error: {e}")
        
        return results

class ReviewEnhancer:
    """코드 리뷰 강화 통합 클래스"""
    
    def __init__(self):
        self.security_analyzer = SecurityAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.quality_analyzer = CodeQualityAnalyzer()
        self.web_searcher = WebSearchEnhancer()
    
    async def enhance_review(self, code: str, language: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        종합적인 코드 리뷰 강화
        
        Args:
            code (str): 분석할 코드
            language (str): 프로그래밍 언어
            context (Dict[str, Any], optional): 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 강화된 리뷰 결과
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'enhancements': {}
        }
        
        # 1. 보안 분석
        results['enhancements']['security'] = self.security_analyzer.analyze_security_vulnerabilities(code, language)
        
        # 2. 성능 분석
        results['enhancements']['performance'] = self.performance_analyzer.analyze_performance_issues(code, language)
        
        # 3. 코드 품질 분석
        results['enhancements']['quality'] = self.quality_analyzer.analyze_code_quality(code, language)
        
        # 4. 웹 검색 기반 정보
        if context and context.get('enable_web_search', True):
            results['enhancements']['web_search'] = {
                'best_practices': await self.web_searcher.search_best_practices(language),
                'security_info': await self.web_searcher.search_security_vulnerabilities(language),
                'alternatives': await self.web_searcher.search_alternatives(context.get('library', ''))
            }
        
        # 5. 종합 점수 계산
        results['overall_score'] = self._calculate_overall_score(results['enhancements'])
        
        return results
    
    def _calculate_overall_score(self, enhancements: Dict[str, Any]) -> Dict[str, Any]:
        """종합 점수 계산"""
        scores = {
            'security': 10,
            'performance': 10,
            'quality': 10
        }
        
        # 보안 점수 계산
        security = enhancements.get('security', {})
        if security.get('total_count', 0) > 0:
            high_vulns = security.get('high_severity', 0)
            medium_vulns = security.get('medium_severity', 0)
            scores['security'] = max(0, 10 - (high_vulns * 3 + medium_vulns * 2))
        
        # 성능 점수 계산
        performance = enhancements.get('performance', {})
        if performance.get('total_count', 0) > 0:
            critical_issues = performance.get('critical', 0)
            high_issues = performance.get('high', 0)
            scores['performance'] = max(0, 10 - (critical_issues * 3 + high_issues * 2))
        
        # 품질 점수 계산
        quality = enhancements.get('quality', {})
        if quality:
            maintainability = quality.get('maintainability', {})
            readability = quality.get('readability', {})
            scores['quality'] = (maintainability.get('score', 10) + readability.get('score', 10)) // 2
        
        # 종합 점수
        overall_score = sum(scores.values()) / len(scores)
        
        return {
            'overall': round(overall_score, 1),
            'breakdown': scores,
            'grade': 'A' if overall_score >= 8 else 'B' if overall_score >= 6 else 'C' if overall_score >= 4 else 'D'
        }
    
    def generate_enhanced_review_summary(self, enhancement_results: Dict[str, Any]) -> str:
        """강화된 리뷰 요약 생성"""
        summary_parts = []
        
        # 보안 요약
        security = enhancement_results.get('security', {})
        if security.get('total_count', 0) > 0:
            summary_parts.append(f"🔒 Security: {security['total_count']} vulnerabilities found")
            if security.get('high_severity', 0) > 0:
                summary_parts.append(f"⚠️ {security['high_severity']} high-severity issues")
        
        # 성능 요약
        performance = enhancement_results.get('performance', {})
        if performance.get('total_count', 0) > 0:
            summary_parts.append(f"⚡ Performance: {performance['total_count']} issues detected")
            if performance.get('critical', 0) > 0:
                summary_parts.append(f"🚨 {performance['critical']} critical performance issues")
        
        # 품질 요약
        quality = enhancement_results.get('quality', {})
        if quality:
            maintainability = quality.get('maintainability', {})
            readability = quality.get('readability', {})
            summary_parts.append(f"📊 Quality: Maintainability {maintainability.get('score', 10)}/10, Readability {readability.get('score', 10)}/10")
        
        # 종합 점수
        overall_score = enhancement_results.get('overall_score', {})
        if overall_score:
            summary_parts.append(f"🎯 Overall Score: {overall_score.get('overall', 0)}/10 ({overall_score.get('grade', 'N/A')})")
        
        return "\n".join(summary_parts) if summary_parts else "No significant issues detected."

# 전역 인스턴스
review_enhancer = ReviewEnhancer() 