"""
Playwright-Lighthouse를 사용한 실제 SEO 분석
Medium 글의 helpers.ts 파일을 Python으로 변환
"""

import asyncio
import json
import tempfile
import os
from typing import Dict, Any
from datetime import datetime
from .config_loader import seo_config


class PlaywrightLighthouseAnalyzer:
    """Playwright-Lighthouse를 사용한 웹사이트 분석기"""

    def __init__(self):
        self.port = 9222
        self.temp_dir = tempfile.mkdtemp()

    async def analyze_website(self, url: str, strategy: str = "mobile") -> Dict[str, Any]:
        """
        웹사이트 SEO 분석 수행

        Args:
            url: 분석할 웹사이트 URL
            strategy: "mobile" 또는 "desktop"

        Returns:
            분석 결과 딕셔너리
        """
        try:
            if not url:
                raise ValueError("분석할 URL이 필요합니다")

            if strategy not in ["mobile", "desktop"]:
                raise ValueError(f"잘못된 strategy: {strategy}. 'mobile' 또는 'desktop'만 가능합니다")

            # 설정 파일에서 Lighthouse 설정 로드
            config = seo_config.get_lighthouse_config(strategy)

            # Node.js 스크립트로 Lighthouse 실행
            lighthouse_script = self._create_lighthouse_script(url, config, strategy)

            # Node.js 스크립트 실행
            result = await self._run_lighthouse_script(lighthouse_script)

            # 결과 파싱 및 정리
            analyzed_data = self._parse_lighthouse_result(result, url)

            return analyzed_data

        except Exception as e:
            error_msg = f"Lighthouse 분석 실패: {str(e)}"
            raise Exception(error_msg)

    def _create_lighthouse_script(self, url: str, config: Dict, strategy: str) -> str:
        """Node.js Lighthouse 스크립트 생성"""

        script_content = f"""
const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');

async function runLighthouse() {{
    const chrome = await chromeLauncher.launch({{
        chromeFlags: ['--headless', '--no-sandbox', '--disable-dev-shm-usage']
    }});

    const options = {{
        logLevel: 'info',
        output: 'json',
        port: chrome.port,
        emulatedFormFactor: '{strategy}',
        onlyCategories: ['performance', 'seo', 'accessibility', 'best-practices']
    }};

    const runnerResult = await lighthouse('{url}', options);

    await chrome.kill();

    console.log(JSON.stringify(runnerResult.lighthouseResult));
}}

runLighthouse().catch(console.error);
"""

        # 임시 스크립트 파일 생성
        script_path = os.path.join(self.temp_dir, f"lighthouse_{strategy}_{int(datetime.now().timestamp())}.js")
        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path

    async def _run_lighthouse_script(self, script_path: str) -> Dict[str, Any]:
        """Node.js 스크립트 실행"""

        try:
            # Node.js로 스크립트 실행
            process = await asyncio.create_subprocess_exec(
                'node', script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                result = json.loads(stdout.decode())
                return result
            else:
                raise Exception(f"Lighthouse 실행 실패: {stderr.decode()}")

        except Exception as e:
            raise Exception(f"스크립트 실행 오류: {str(e)}")
        finally:
            # 임시 파일 정리
            if os.path.exists(script_path):
                os.remove(script_path)

    def _parse_lighthouse_result(self, result: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Lighthouse 결과 파싱 및 SEO Doctor 형식으로 변환"""

        try:
            categories = result.get('categories', {})
            audits = result.get('audits', {})

            # 점수 추출
            performance_score = int(categories.get('performance', {}).get('score', 0) * 100)
            seo_score = int(categories.get('seo', {}).get('score', 0) * 100)
            accessibility_score = int(categories.get('accessibility', {}).get('score', 0) * 100)
            best_practices_score = int(categories.get('best-practices', {}).get('score', 0) * 100)

            # 전체 점수 계산 (성능과 SEO 가중 평균)
            overall_score = int((performance_score * 0.4 + seo_score * 0.4 +
                               accessibility_score * 0.1 + best_practices_score * 0.1))

            # 주요 메트릭 추출
            metrics = self._extract_key_metrics(audits)

            # 문제점 및 개선사항 추출
            issues = self._extract_issues(audits, categories)

            # 복구 시간 예측
            recovery_days = self._estimate_recovery_time(overall_score, len(issues))

            # 응급 레벨 결정
            emergency_level = self._determine_emergency_level(overall_score)

            return {
                "url": url,
                "overall_score": overall_score,
                "scores": {
                    "performance": performance_score,
                    "seo": seo_score,
                    "accessibility": accessibility_score,
                    "best_practices": best_practices_score
                },
                "metrics": metrics,
                "issues": issues,
                "recovery_days": recovery_days,
                "emergency_level": emergency_level,
                "improvement_potential": min(40, 100 - overall_score),
                "timestamp": datetime.now().isoformat(),
                "raw_lighthouse_result": result  # 원본 데이터 보존
            }

        except Exception as e:
            raise Exception(f"결과 파싱 오류: {str(e)}")

    def _extract_key_metrics(self, audits: Dict[str, Any]) -> Dict[str, Any]:
        """주요 성능 메트릭 추출"""

        metrics = {}

        # Core Web Vitals
        if 'largest-contentful-paint' in audits:
            lcp = audits['largest-contentful-paint'].get('numericValue', 0)
            metrics['lcp'] = f"{lcp/1000:.2f}초"

        if 'first-contentful-paint' in audits:
            fcp = audits['first-contentful-paint'].get('numericValue', 0)
            metrics['fcp'] = f"{fcp/1000:.2f}초"

        if 'cumulative-layout-shift' in audits:
            cls = audits['cumulative-layout-shift'].get('numericValue', 0)
            metrics['cls'] = f"{cls:.3f}"

        if 'total-blocking-time' in audits:
            tbt = audits['total-blocking-time'].get('numericValue', 0)
            metrics['tbt'] = f"{tbt}ms"

        if 'speed-index' in audits:
            si = audits['speed-index'].get('numericValue', 0)
            metrics['speed_index'] = f"{si/1000:.2f}초"

        return metrics

    def _extract_issues(self, audits: Dict[str, Any], categories: Dict[str, Any]) -> list:
        """문제점 추출 및 분류"""

        issues = []

        # 성능 문제
        performance_audits = categories.get('performance', {}).get('auditRefs', [])
        for audit_ref in performance_audits:
            audit_id = audit_ref.get('id')
            if audit_id in audits:
                audit = audits[audit_id]
                if audit.get('score', 1) < 0.9:  # 90% 미만인 경우
                    issues.append(f"🚀 성능: {audit.get('title', audit_id)}")

        # SEO 문제
        seo_audits = categories.get('seo', {}).get('auditRefs', [])
        for audit_ref in seo_audits:
            audit_id = audit_ref.get('id')
            if audit_id in audits:
                audit = audits[audit_id]
                if audit.get('score', 1) < 1.0:  # 100% 미만인 경우
                    issues.append(f"🔍 SEO: {audit.get('title', audit_id)}")

        # 접근성 문제
        accessibility_audits = categories.get('accessibility', {}).get('auditRefs', [])
        for audit_ref in accessibility_audits:
            audit_id = audit_ref.get('id')
            if audit_id in audits:
                audit = audits[audit_id]
                if audit.get('score', 1) < 0.9:  # 90% 미만인 경우
                    issues.append(f"♿ 접근성: {audit.get('title', audit_id)}")

        return issues[:8]  # 최대 8개까지만 표시

    def _estimate_recovery_time(self, score: int, issue_count: int) -> int:
        """회복 시간 예측 - 설정 기반"""
        return seo_config.calculate_recovery_time(score, issue_count)

    def _determine_emergency_level(self, score: int) -> str:
        """응급 레벨 결정 - 설정 기반"""
        emergency_info = seo_config.determine_emergency_level(score)
        return f"{emergency_info['emoji']} {emergency_info['label']}"

    def __del__(self):
        """Clean up temporary directory on object destruction."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass


# 전역 인스턴스
lighthouse_analyzer = PlaywrightLighthouseAnalyzer()


async def analyze_website_with_lighthouse(url: str, strategy: str = "mobile") -> Dict[str, Any]:
    """외부에서 사용할 수 있는 간단한 인터페이스"""
    return await lighthouse_analyzer.analyze_website(url, strategy)
