#!/usr/bin/env python3
"""
AI Analyzer for Travel Scout
Gemini 2.5 Flash를 활용한 여행 데이터 분석 및 추천 생성
"""

import logging
from typing import Dict, List, Any
import google.generativeai as genai
from .config_loader import config

logger = logging.getLogger(__name__)


class TravelAIAnalyzer:
    """여행 데이터 AI 분석기 - Gemini 2.5 Flash 활용"""

    def __init__(self):
        self.model_name = config.get_ai_model_config()
        self.api_key = config.get_ai_api_key()
        self.prompts = config.get_analysis_prompts()

        # Gemini 설정
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        logger.info(f"AI 분석기 초기화 완료 - 모델: {self.model_name}")

    async def analyze_hotel_data(self, hotels_data: List[Dict[str, Any]],
                                search_params: Dict[str, Any]) -> Dict[str, Any]:
        """호텔 데이터 분석 및 추천 생성"""
        try:
            if not hotels_data:
                return {
                    "analysis": "호텔 데이터가 없습니다.",
                    "recommendations": [],
                    "price_analysis": {},
                    "quality_ranking": []
                }

            # 분석용 데이터 준비
            analysis_data = {
                "hotels": hotels_data,
                "search_params": search_params,
                "total_hotels": len(hotels_data)
            }

            # Gemini를 통한 분석
            prompt = self._build_hotel_analysis_prompt(analysis_data)
            response = await self._generate_analysis(prompt)

            # 결과 파싱 및 구조화
            analysis_result = self._parse_hotel_analysis(response, hotels_data)

            logger.info(f"호텔 데이터 분석 완료 - {len(hotels_data)}개 호텔")
            return analysis_result

        except Exception as e:
            logger.error(f"호텔 데이터 분석 오류: {e}")
            return {
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "recommendations": [],
                "price_analysis": {},
                "quality_ranking": []
            }

    async def analyze_flight_data(self, flights_data: List[Dict[str, Any]],
                                 search_params: Dict[str, Any]) -> Dict[str, Any]:
        """항공편 데이터 분석 및 추천 생성"""
        try:
            if not flights_data:
                return {
                    "analysis": "항공편 데이터가 없습니다.",
                    "recommendations": [],
                    "price_analysis": {},
                    "airline_ranking": []
                }

            # 분석용 데이터 준비
            analysis_data = {
                "flights": flights_data,
                "search_params": search_params,
                "total_flights": len(flights_data)
            }

            # Gemini를 통한 분석
            prompt = self._build_flight_analysis_prompt(analysis_data)
            response = await self._generate_analysis(prompt)

            # 결과 파싱 및 구조화
            analysis_result = self._parse_flight_analysis(response, flights_data)

            logger.info(f"항공편 데이터 분석 완료 - {len(flights_data)}개 항공편")
            return analysis_result

        except Exception as e:
            logger.error(f"항공편 데이터 분석 오류: {e}")
            return {
                "analysis": f"분석 중 오류 발생: {str(e)}",
                "recommendations": [],
                "price_analysis": {},
                "airline_ranking": []
            }

    async def generate_travel_recommendations(self, hotel_analysis: Dict[str, Any],
                                            flight_analysis: Dict[str, Any],
                                            search_params: Dict[str, Any]) -> Dict[str, Any]:
        """통합 여행 추천 생성"""
        try:
            # 통합 분석 데이터 준비
            combined_data = {
                "hotel_analysis": hotel_analysis,
                "flight_analysis": flight_analysis,
                "search_params": search_params
            }

            # Gemini를 통한 통합 분석
            prompt = self._build_combined_analysis_prompt(combined_data)
            response = await self._generate_analysis(prompt)

            # 결과 파싱
            recommendations = self._parse_combined_analysis(response)

            logger.info("통합 여행 추천 생성 완료")
            return recommendations

        except Exception as e:
            logger.error(f"통합 추천 생성 오류: {e}")
            return {
                "summary": f"추천 생성 중 오류 발생: {str(e)}",
                "best_combinations": [],
                "budget_breakdown": {},
                "travel_tips": []
            }

    def _build_hotel_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """호텔 분석 프롬프트 생성"""
        base_prompt = self.prompts.get('hotel_analysis', '')

        hotels_text = "\n".join([
            f"- {hotel.get('name', 'N/A')} | 가격: {hotel.get('price', 'N/A')} | 평점: {hotel.get('rating', 'N/A')}"
            for hotel in data['hotels']
        ])

        return f"""
{base_prompt}

검색 조건:
- 목적지: {data['search_params'].get('destination', 'N/A')}
- 체크인: {data['search_params'].get('check_in', 'N/A')}
- 체크아웃: {data['search_params'].get('check_out', 'N/A')}
- 투숙객: {data['search_params'].get('guests', 'N/A')}명

호텔 데이터 ({data['total_hotels']}개):
{hotels_text}

다음 형식으로 분석 결과를 제공해주세요:
1. 전체 분석 요약
2. 가격대별 추천 (예산별)
3. 품질 순위 (평점 기준)
4. 최적 선택 3개
5. 주의사항 및 팁
"""

    def _build_flight_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """항공편 분석 프롬프트 생성"""
        base_prompt = self.prompts.get('flight_analysis', '')

        flights_text = "\n".join([
            f"- {flight.get('airline', 'N/A')} | 가격: {flight.get('price', 'N/A')} | 소요시간: {flight.get('duration', 'N/A')}"
            for flight in data['flights']
        ])

        return f"""
{base_prompt}

검색 조건:
- 출발지: {data['search_params'].get('origin', 'N/A')}
- 목적지: {data['search_params'].get('destination', 'N/A')}
- 출발일: {data['search_params'].get('departure_date', 'N/A')}
- 귀국일: {data['search_params'].get('return_date', 'N/A')}

항공편 데이터 ({data['total_flights']}개):
{flights_text}

다음 형식으로 분석 결과를 제공해주세요:
1. 전체 분석 요약
2. 가격대별 추천 (예산별)
3. 항공사별 신뢰도 순위
4. 최적 선택 3개
5. 주의사항 및 팁
"""

    def _build_combined_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """통합 분석 프롬프트 생성"""
        return f"""
다음 호텔과 항공편 분석 결과를 바탕으로 최적의 여행 패키지를 추천해주세요.

호텔 분석:
{data['hotel_analysis'].get('analysis', 'N/A')}

항공편 분석:
{data['flight_analysis'].get('analysis', 'N/A')}

검색 조건:
- 목적지: {data['search_params'].get('destination', 'N/A')}
- 기간: {data['search_params'].get('check_in', 'N/A')} ~ {data['search_params'].get('check_out', 'N/A')}

다음 형식으로 통합 추천을 제공해주세요:
1. 최적 조합 3가지 (호텔 + 항공편)
2. 예산별 총 비용 분석
3. 여행 팁 및 주의사항
4. 예약 전략
5. 대안 옵션
"""

    async def _generate_analysis(self, prompt: str) -> str:
        """Gemini를 통한 분석 생성"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini 분석 생성 오류: {e}")
            raise

    def _parse_hotel_analysis(self, response: str, hotels_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """호텔 분석 결과 파싱"""
        return {
            "analysis": response,
            "recommendations": self._extract_recommendations(response),
            "price_analysis": self._extract_price_analysis(response),
            "quality_ranking": self._extract_quality_ranking(response, hotels_data)
        }

    def _parse_flight_analysis(self, response: str, flights_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """항공편 분석 결과 파싱"""
        return {
            "analysis": response,
            "recommendations": self._extract_recommendations(response),
            "price_analysis": self._extract_price_analysis(response),
            "airline_ranking": self._extract_airline_ranking(response, flights_data)
        }

    def _parse_combined_analysis(self, response: str) -> Dict[str, Any]:
        """통합 분석 결과 파싱"""
        return {
            "summary": response,
            "best_combinations": self._extract_combinations(response),
            "budget_breakdown": self._extract_budget_breakdown(response),
            "travel_tips": self._extract_travel_tips(response)
        }

    def _extract_recommendations(self, text: str) -> List[str]:
        """추천사항 추출"""
        # 간단한 추출 로직 - 실제로는 더 정교한 파싱 필요
        lines = text.split('\n')
        recommendations = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['추천', 'recommend', 'best', 'optimal']):
                recommendations.append(line.strip())
        return recommendations[:5]  # 최대 5개

    def _extract_price_analysis(self, text: str) -> Dict[str, str]:
        """가격 분석 추출"""
        # 간단한 추출 로직
        return {
            "budget_range": "분석 중...",
            "value_analysis": "분석 중...",
            "price_trends": "분석 중..."
        }

    def _extract_quality_ranking(self, text: str, hotels_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """품질 순위 추출"""
        # 호텔 데이터를 평점 기준으로 정렬
        sorted_hotels = sorted(hotels_data,
                             key=lambda x: self._extract_rating_numeric(x.get('rating', '0')),
                             reverse=True)
        return sorted_hotels[:3]  # 상위 3개

    def _extract_airline_ranking(self, text: str, flights_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """항공사 순위 추출"""
        # 항공편 데이터를 가격 기준으로 정렬
        sorted_flights = sorted(flights_data,
                              key=lambda x: self._extract_price_numeric(x.get('price', '0')))
        return sorted_flights[:3]  # 상위 3개

    def _extract_combinations(self, text: str) -> List[Dict[str, str]]:
        """최적 조합 추출"""
        return [
            {"hotel": "추천 호텔 1", "flight": "추천 항공편 1", "total_cost": "예상 비용"},
            {"hotel": "추천 호텔 2", "flight": "추천 항공편 2", "total_cost": "예상 비용"},
            {"hotel": "추천 호텔 3", "flight": "추천 항공편 3", "total_cost": "예상 비용"}
        ]

    def _extract_budget_breakdown(self, text: str) -> Dict[str, str]:
        """예산 분석 추출"""
        return {
            "hotel_cost": "호텔 비용",
            "flight_cost": "항공편 비용",
            "total_estimated": "총 예상 비용",
            "savings_tips": "절약 팁"
        }

    def _extract_travel_tips(self, text: str) -> List[str]:
        """여행 팁 추출"""
        return [
            "여행 팁 1",
            "여행 팁 2",
            "여행 팁 3"
        ]

    def _extract_rating_numeric(self, rating_text: str) -> float:
        """평점 텍스트에서 숫자 추출"""
        try:
            import re
            match = re.search(r'(\d+\.?\d*)', str(rating_text))
            return float(match.group(1)) if match else 0.0
        except:
            return 0.0

    def _extract_price_numeric(self, price_text: str) -> float:
        """가격 텍스트에서 숫자 추출"""
        try:
            import re
            match = re.search(r'(\d+)', str(price_text).replace(',', ''))
            return float(match.group(1)) if match else float('inf')
        except:
            return float('inf')
