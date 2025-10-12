"""
AI Text Analyzer - Gemini 2.5 Flash Lite 전용

Uses Google Gemini 2.5 Flash Lite to analyze text for:
1. Resource query intent classification (offer/request)
2. Interest categorization for social connections
3. Context understanding without hard-coded keywords
"""

import json
import os
from typing import Dict, List, Tuple, Optional
import asyncio
from datetime import datetime
import google.generativeai as genai
from .config import get_llm_config
from .exceptions import ExternalDataUnavailableError


class AITextAnalyzer:
    """AI-powered text analysis using Google Gemini 2.5 Flash Lite."""
    
    def __init__(self):
        """Initialize the AI text analyzer with Gemini configuration."""
        self.config = get_llm_config()
        self._setup_gemini_client()
    
    def _setup_gemini_client(self):
        """Setup Gemini client with configuration."""
        if not self.config.api_key:
            raise ExternalDataUnavailableError(
                "GOOGLE_API_KEY environment variable is required for AI text analysis"
            )
        
        genai.configure(api_key=self.config.api_key)
        self.model = genai.GenerativeModel(self.config.model)
    
    async def analyze_resource_intent(self, query: str) -> Tuple[str, Dict]:
        """
        Analyze user query to understand if they're offering or requesting resources.
        
        Returns:
            Tuple[str, Dict]: (intent_type, analysis_result)
            - intent_type: "offer", "request", or "general"
            - analysis_result: detailed analysis including extracted items and context
        """
        try:
            prompt = f"""
사용자의 텍스트를 분석하여 의도를 파악해주세요.

텍스트: "{query}"

다음 중 하나로 분류하고 JSON 형태로 응답해주세요:

{{
    "intent": "offer|request|general",
    "confidence": 0.0-1.0,
    "extracted_items": ["아이템1", "아이템2"],
    "context": "상황 설명",
    "reasoning": "분류 이유"
}}

분류 기준:
- offer: 뭔가를 나누거나 제공하려는 의도 (예: 나눔, 드림, 공유)
- request: 뭔가를 필요로 하거나 구하려는 의도 (예: 필요, 구함, 급함)
- general: 단순 문의나 정보 요청

한국어와 영어 모두 이해하여 분석해주세요.
"""
            
            response = await self._call_gemini(prompt)
            result = json.loads(response)
            
            return result.get("intent", "general"), {
                "query": query,
                "confidence": result.get("confidence", 0.5),
                "extracted_items": result.get("extracted_items", []),
                "context": result.get("context", ""),
                "reasoning": result.get("reasoning", ""),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ExternalDataUnavailableError(f"Resource intent analysis failed: {e}") from e
    
    async def analyze_interests(self, interests_text: str) -> List[str]:
        """
        Analyze user interests and categorize them intelligently.
        
        Returns:
            List[str]: List of interest categories
        """
        try:
            prompt = f"""
사용자의 관심사 텍스트를 분석하여 적절한 카테고리로 분류해주세요.

텍스트: "{interests_text}"

다음과 같은 JSON 형태로 응답해주세요:

{{
    "categories": ["카테고리1", "카테고리2", "카테고리3"],
    "confidence": 0.0-1.0,
    "social_isolation_risk": "low|medium|high",
    "reasoning": "분류 이유"
}}

가능한 카테고리들 (이것에 국한되지 않음):
- fitness (운동, 헬스, 요가, 달리기 등)
- cooking (요리, 베이킹, 음식 등)
- reading (독서, 책, 문학 등)
- games (게임, 체스, 보드게임 등)
- music (음악, 악기, 노래 등)
- art (그림, 사진, 미술 등)
- technology (프로그래밍, IT, 컴퓨터 등)
- nature (하이킹, 등산, 자연 등)
- social (사교, 모임, 커뮤니티 등)
- learning (공부, 언어, 교육 등)
- volunteer (봉사, 도움, 기부 등)

사회적 고립 위험도도 함께 평가해주세요:
- high: 외로움, 새로 이사, 친구 없음 등의 표현
- medium: 소극적이지만 참여 의지
- low: 활발한 활동 의지

한국어와 영어 모두 분석 가능합니다.
"""
            
            response = await self._call_gemini(prompt)
            result = json.loads(response)
            return result.get("categories", ["social"])
            
        except Exception as e:
            raise ExternalDataUnavailableError(f"Interests analysis failed: {e}") from e
    
    async def assess_isolation_risk(self, user_profile: Dict) -> Tuple[str, str]:
        """
        Assess social isolation risk using AI analysis.
        
        Returns:
            Tuple[str, str]: (risk_level, reasoning)
        """
        try:
            interests = user_profile.get('interests', '')
            name = user_profile.get('name', 'Unknown')
            
            prompt = f"""
사용자 프로필을 분석하여 사회적 고립 위험도를 평가해주세요.

사용자 정보:
- 이름: {name}
- 관심사/상황: {interests}

다음 JSON 형태로 응답해주세요:

{{
    "risk_level": "low|medium|high",
    "confidence": 0.0-1.0,
    "risk_indicators": ["지표1", "지표2"],
    "protective_factors": ["보호요인1", "보호요인2"],
    "recommendations": ["추천사항1", "추천사항2"],
    "reasoning": "평가 이유"
}}

위험도 기준:
- high: 명시적 외로움 표현, 새로 이사, 친구 부족, 사회적 불안
- medium: 소극적 성향, 제한적 사회활동, 특정 상황으로 인한 일시적 고립
- low: 활발한 사회 참여 의지, 다양한 관심사, 긍정적 태도

한국어와 영어 모두 분석 가능합니다.
"""
            
            response = await self._call_gemini(prompt)
            result = json.loads(response)
            return result.get("risk_level", "low"), result.get("reasoning", "AI 분석 완료")
            
        except Exception as e:
            raise ExternalDataUnavailableError(f"Isolation risk assessment failed: {e}") from e
    
    async def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            return response.text
        except Exception as e:
            raise ExternalDataUnavailableError(f"Gemini API error: {e}") from e


# Global instance
ai_text_analyzer = AITextAnalyzer()