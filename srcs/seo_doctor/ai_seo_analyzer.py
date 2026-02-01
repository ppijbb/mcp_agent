#!/usr/bin/env python3
"""
AI SEO Analyzer
Gemini 2.5 Flash를 활용한 SEO 데이터 분석 및 추천 생성
"""

import logging
from typing import Dict, List, Any
import google.generativeai as genai
from .config_loader import seo_config

logger = logging.getLogger(__name__)


class SEOAIAnalyzer:
    """SEO 데이터 AI 분석기 - Gemini 2.5 Flash 활용"""

    def __init__(self):
        self.model_name = seo_config.get_ai_model_config()
        self.api_key = seo_config.get_ai_api_key()
        self.prompts = seo_config.get_analysis_prompts()

        # Gemini 설정
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        logger.info(f"SEO AI 분석기 초기화 완료 - 모델: {self.model_name}")

    async def analyze_lighthouse_data(self, lighthouse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Lighthouse 데이터 분석 및 SEO 진단 생성"""
        try:
            if not lighthouse_data or lighthouse_data.get('error'):
                return {
                    "analysis": "Lighthouse 데이터를 가져올 수 없습니다.",
                    "recommendations": [],
                    "critical_issues": [],
                    "quick_fixes": []
                }

            # 분석용 데이터 준비
            analysis_input = self._prepare_lighthouse_analysis_input(lighthouse_data)

            # Gemini를 통한 분석
            prompt = self._build_seo_analysis_prompt(analysis_input)
            response = await self._generate_analysis(prompt)

            # 결과 파싱 및 구조화
            analysis_result = self._parse_seo_analysis(response, lighthouse_data)

            logger.info(f"Lighthouse 데이터 분석 완료 - 점수: {lighthouse_data.get('overall_score', 0)}")
            return analysis_result

        except Exception as e:
            logger.error(f"Lighthouse 데이터 분석 오류: {e}")
            return {
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "recommendations": [],
                "critical_issues": [],
                "quick_fixes": []
            }

    async def analyze_competitors(self, competitors_data: List[Dict[str, Any]],
                                 target_url: str) -> List[Dict[str, Any]]:
        """경쟁사 데이터 분석 및 전략적 인사이트 생성"""
        try:
            if not competitors_data:
                return []

            competitor_analyses = []

            for competitor in competitors_data:
                # 경쟁사별 분석
                prompt = self._build_competitor_analysis_prompt(competitor, target_url)
                response = await self._generate_analysis(prompt)

                # 결과 파싱
                analysis = self._parse_competitor_analysis(response, competitor)
                competitor_analyses.append(analysis)

            logger.info(f"경쟁사 분석 완료 - {len(competitor_analyses)}개 경쟁사")
            return competitor_analyses

        except Exception as e:
            logger.error(f"경쟁사 분석 오류: {e}")
            return []

    async def generate_seo_prescription(self,
                                       lighthouse_analysis: Dict[str, Any],
                                       competitor_analyses: List[Dict[str, Any]],
                                       target_url: str) -> Dict[str, Any]:
        """통합 SEO 처방전 생성"""
        try:
            # 통합 분석 데이터 준비
            combined_data = {
                "target_url": target_url,
                "lighthouse_analysis": lighthouse_analysis,
                "competitor_analyses": competitor_analyses
            }

            # Gemini를 통한 처방전 생성
            prompt = self._build_prescription_prompt(combined_data)
            response = await self._generate_analysis(prompt)

            # 결과 파싱
            prescription = self._parse_prescription(response)

            logger.info("SEO 처방전 생성 완료")
            return prescription

        except Exception as e:
            logger.error(f"처방전 생성 오류: {e}")
            return {
                "emergency_treatment": [],
                "weekly_medicine": [],
                "monthly_checkup": [],
                "competitive_moves": [],
                "expected_results": "처방전 생성 중 오류 발생"
            }

    def _prepare_lighthouse_analysis_input(self, lighthouse_data: Dict[str, Any]) -> str:
        """Lighthouse 데이터를 분석 가능한 형식으로 준비"""
        scores = lighthouse_data.get('scores', {})
        metrics = lighthouse_data.get('metrics', {})
        issues = lighthouse_data.get('issues', [])

        input_text = f"""
URL: {lighthouse_data.get('url', 'N/A')}
전체 점수: {lighthouse_data.get('overall_score', 0)}/100

카테고리별 점수:
- 성능: {scores.get('performance', 0)}/100
- SEO: {scores.get('seo', 0)}/100
- 접근성: {scores.get('accessibility', 0)}/100
- 모범 사례: {scores.get('best_practices', 0)}/100

주요 메트릭:
- LCP (최대 콘텐츠 표시 시간): {metrics.get('lcp', 'N/A')}
- FCP (첫 콘텐츠 표시 시간): {metrics.get('fcp', 'N/A')}
- CLS (누적 레이아웃 이동): {metrics.get('cls', 'N/A')}
- TBT (총 차단 시간): {metrics.get('tbt', 'N/A')}
- Speed Index: {metrics.get('speed_index', 'N/A')}

발견된 문제점 ({len(issues)}개):
""" + "\n".join([f"- {issue}" for issue in issues[:10]])

        return input_text

    def _build_seo_analysis_prompt(self, analysis_input: str) -> str:
        """SEO 분석 프롬프트 생성"""
        base_prompt = self.prompts.get('seo_analysis', '')

        return f"""
{base_prompt}

{analysis_input}

다음 형식으로 분석 결과를 제공해주세요:
1. 현재 SEO 상태 평가 (한 문장 요약)
2. 주요 문제점 3-5개 (우선순위 순)
3. 즉시 실행 가능한 빠른 수정사항 3-5개
4. 단계별 개선 방안
5. 예상 회복 시간 및 전략
"""

    def _build_competitor_analysis_prompt(self, competitor: Dict[str, Any],
                                         target_url: str) -> str:
        """경쟁사 분석 프롬프트 생성"""
        base_prompt = self.prompts.get('competitor_analysis', '')

        return f"""
{base_prompt}

분석 대상 사이트: {target_url}
경쟁사: {competitor.get('url', 'N/A')}

경쟁사 데이터:
{competitor}

다음 형식으로 분석 결과를 제공해주세요:
1. 경쟁사 SEO 강점 분석
2. 경쟁사 약점 및 취약점
3. 훔칠 만한 전술 3-5개
4. 콘텐츠 갭 분석
5. 차별화 전략 제안
"""

    def _build_prescription_prompt(self, combined_data: Dict[str, Any]) -> str:
        """처방전 생성 프롬프트"""
        lighthouse_analysis = combined_data['lighthouse_analysis']
        competitor_analyses = combined_data['competitor_analyses']

        competitor_summary = "\n".join([
            f"- {comp.get('competitor_url', 'N/A')}: {comp.get('threat_level', 'N/A')}"
            for comp in competitor_analyses[:5]
        ])

        return f"""
다음 SEO 진단 결과를 바탕으로 실행 가능한 처방전을 작성해주세요.

사이트: {combined_data['target_url']}

Lighthouse 분석 결과:
{lighthouse_analysis.get('analysis', 'N/A')}

경쟁사 분석 요약:
{competitor_summary}

다음 형식으로 처방전을 제공해주세요:
1. 응급 처치 (즉시 실행해야 할 3-5개 항목)
2. 주간 처방 (매주 실행할 3-5개 항목)
3. 월간 체크업 (매월 점검할 3-5개 항목)
4. 경쟁 우위 확보 전략 (3-5개 항목)
5. 예상 결과 (구체적 수치 포함)
"""

    async def _generate_analysis(self, prompt: str) -> str:
        """Gemini를 통한 분석 생성"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini 분석 생성 오류: {e}")
            raise

    def _parse_seo_analysis(self, response: str, lighthouse_data: Dict[str, Any]) -> Dict[str, Any]:
        """SEO 분석 결과 파싱"""
        return {
            "analysis": response,
            "recommendations": self._extract_recommendations(response),
            "critical_issues": lighthouse_data.get('issues', [])[:5],
            "quick_fixes": self._extract_quick_fixes(response)
        }

    def _parse_competitor_analysis(self, response: str, competitor: Dict[str, Any]) -> Dict[str, Any]:
        """경쟁사 분석 결과 파싱"""
        return {
            "competitor_url": competitor.get('url', 'N/A'),
            "analysis": response,
            "threat_level": self._assess_threat_level(response),
            "steal_worthy_tactics": self._extract_tactics(response),
            "vulnerabilities": self._extract_vulnerabilities(response),
            "content_gaps": self._extract_content_gaps(response)
        }

    def _parse_prescription(self, response: str) -> Dict[str, Any]:
        """처방전 파싱"""
        return {
            "emergency_treatment": self._extract_list_items(response, "응급 처치"),
            "weekly_medicine": self._extract_list_items(response, "주간 처방"),
            "monthly_checkup": self._extract_list_items(response, "월간 체크업"),
            "competitive_moves": self._extract_list_items(response, "경쟁 우위"),
            "expected_results": self._extract_expected_results(response)
        }

    def _extract_recommendations(self, text: str) -> List[str]:
        """추천사항 추출"""
        lines = text.split('\n')
        recommendations = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['추천', 'recommend', '개선', '방안']):
                clean_line = line.strip().lstrip('- ').lstrip('* ').lstrip('1234567890. ')
                if len(clean_line) > 10:
                    recommendations.append(clean_line)
        return recommendations[:5]

    def _extract_quick_fixes(self, text: str) -> List[str]:
        """빠른 수정사항 추출"""
        lines = text.split('\n')
        fixes = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['빠른', 'quick', '즉시', '수정']):
                clean_line = line.strip().lstrip('- ').lstrip('* ').lstrip('1234567890. ')
                if len(clean_line) > 10:
                    fixes.append(clean_line)
        return fixes[:5]

    def _assess_threat_level(self, text: str) -> str:
        """위협 레벨 평가"""
        if any(keyword in text.lower() for keyword in ['지배', 'dominating', '강력']):
            return "👑 지배중"
        elif any(keyword in text.lower() for keyword in ['상승', 'rising', '급증']):
            return "📈 급상승"
        elif any(keyword in text.lower() for keyword in ['하락', 'declining', '약화']):
            return "📉 하락"
        elif any(keyword in text.lower() for keyword in ['약함', 'weak', '취약']):
            return "😴 약함"
        else:
            return "➡️ 안정"

    def _extract_tactics(self, text: str) -> List[str]:
        """전술 추출"""
        return self._extract_list_items(text, "전술")

    def _extract_vulnerabilities(self, text: str) -> List[str]:
        """약점 추출"""
        return self._extract_list_items(text, "약점")

    def _extract_content_gaps(self, text: str) -> List[str]:
        """콘텐츠 갭 추출"""
        return self._extract_list_items(text, "갭")

    def _extract_list_items(self, text: str, keyword: str) -> List[str]:
        """특정 키워드 섹션의 리스트 항목 추출"""
        lines = text.split('\n')
        items = []
        in_section = False

        for line in lines:
            if keyword.lower() in line.lower():
                in_section = True
                continue

            if in_section:
                if line.strip().startswith(('-', '*', '•')) or line.strip()[0:2].replace('.', '').isdigit():
                    clean_line = line.strip().lstrip('- ').lstrip('* ').lstrip('• ').lstrip('1234567890. ')
                    if len(clean_line) > 5:
                        items.append(clean_line)
                elif line.strip() == '':
                    continue
                elif not line.strip().startswith((' ', '\t')):
                    in_section = False

        return items[:5]

    def _extract_expected_results(self, text: str) -> str:
        """예상 결과 추출"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if '예상 결과' in line or 'expected results' in line.lower():
                # 다음 몇 줄을 결합
                result_lines = lines[i:i+5]
                return ' '.join([l.strip() for l in result_lines if l.strip()])
        return "SEO 개선을 통한 트래픽 증가 예상"
