"""
AI Text Analyzer

Uses LLM to analyze text for:
1. Resource query intent classification (offer/request)
2. Interest categorization for social connections
3. Context understanding without hard-coded keywords
"""

import json
import os
from typing import Dict, List, Tuple, Optional
import asyncio
from datetime import datetime


class AITextAnalyzer:
    """AI-powered text analysis for Urban Hive agents."""
    
    def __init__(self, llm_provider="anthropic"):
        """Initialize the AI text analyzer with specified LLM provider."""
        self.llm_provider = llm_provider
        self._setup_llm_client()
    
    def _setup_llm_client(self):
        """Setup LLM client based on provider preference."""
        try:
            if self.llm_provider == "anthropic":
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    self.model = "claude-3-haiku-20240307"
                except ImportError:
                    print("Anthropic not available, falling back to OpenAI")
                    self._setup_openai_client()
            else:
                self._setup_openai_client()
        except Exception as e:
            print(f"Error setting up LLM client: {e}")
            self.client = None
    
    def _setup_openai_client(self):
        """Setup OpenAI client as fallback."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gemini-2.5-flash-lite-preview-06-07"
            self.llm_provider = "openai"
        except ImportError:
            print("No LLM client available")
            self.client = None
    
    async def analyze_resource_intent(self, query: str) -> Tuple[str, Dict]:
        """
        Analyze user query to understand if they're offering or requesting resources.
        
        Returns:
            Tuple[str, Dict]: (intent_type, analysis_result)
            - intent_type: "offer", "request", or "general"
            - analysis_result: detailed analysis including extracted items and context
        """
        if not self.client:
            raise RuntimeError("AI client not configured. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        
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
            
            if self.llm_provider == "anthropic":
                response = await self._call_anthropic(prompt)
            else:
                response = await self._call_openai(prompt)
            
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
            print(f"Error in AI resource intent analysis: {e}")
            # No fallback - raise the actual error for proper handling
            raise RuntimeError(f"Resource intent analysis failed: {e}") from e
    
    async def analyze_interests(self, interests_text: str) -> List[str]:
        """
        Analyze user interests and categorize them intelligently.
        
        Returns:
            List[str]: List of interest categories
        """
        if not self.client:
            raise RuntimeError("AI client not configured. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        
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
            
            if self.llm_provider == "anthropic":
                response = await self._call_anthropic(prompt)
            else:
                response = await self._call_openai(prompt)
            
            result = json.loads(response)
            return result.get("categories", ["social"])
            
        except Exception as e:
            print(f"Error in AI interests analysis: {e}")
            # No fallback - raise the actual error for proper handling
            raise RuntimeError(f"Interests analysis failed: {e}") from e
    
    async def assess_isolation_risk(self, user_profile: Dict) -> Tuple[str, str]:
        """
        Assess social isolation risk using AI analysis.
        
        Returns:
            Tuple[str, str]: (risk_level, reasoning)
        """
        if not self.client:
            raise RuntimeError("AI client not configured. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        
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
            
            if self.llm_provider == "anthropic":
                response = await self._call_anthropic(prompt)
            else:
                response = await self._call_openai(prompt)
            
            result = json.loads(response)
            return result.get("risk_level", "low"), result.get("reasoning", "AI 분석 완료")
            
        except Exception as e:
            print(f"Error in AI isolation risk assessment: {e}")
            # No fallback - raise the actual error for proper handling
            raise RuntimeError(f"Isolation risk assessment failed: {e}") from e
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return message.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
# All fallback methods removed - Urban Hive now uses real AI analysis only


# Global instance
ai_text_analyzer = AITextAnalyzer() 