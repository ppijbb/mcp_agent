"""
LLM Client - Gemini 2.5 Flash Lite Preview 클라이언트

이 클래스는 Agent의 추론 능력을 담당하는 핵심 구성요소입니다.
계획서에 명시된 대로 Gemini 2.5 Flash Lite Preview를 사용하여
진짜 Agent의 지능적 추론을 구현합니다.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class GeminiLLMClient:
    """
    Gemini 2.5 Flash Lite Preview 클라이언트
    
    Agent의 추론 능력을 제공하는 핵심 클래스입니다.
    단순한 텍스트 생성이 아닌 Agent의 지능적 사고를 담당합니다.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Google API 키 (환경변수 GOOGLE_API_KEY에서 자동 로드)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Gemini 설정
        genai.configure(api_key=self.api_key)
        
        # 모델 설정 - 계획서에 명시된 모델 사용
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite-preview-06-17",
            generation_config={
                "temperature": 0.7,  # Agent의 창의성 조절
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        # 통계 및 모니터링
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "created_at": datetime.now().isoformat()
        }
    
    async def complete(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        LLM 추론 실행 - Agent의 핵심 사고 과정
        
        이 메서드는 Agent의 지능적 추론을 담당합니다:
        - 복잡한 상황 분석
        - 전략적 계획 수립
        - 불확실성 하에서의 의사결정
        
        Args:
            prompt: 추론할 내용
            context: 추가 컨텍스트 정보
            
        Returns:
            LLM의 추론 결과
        """
        try:
            self.usage_stats["total_requests"] += 1
            
            # 컨텍스트가 있으면 프롬프트에 포함
            if context:
                enhanced_prompt = self._enhance_prompt_with_context(prompt, context)
            else:
                enhanced_prompt = prompt
            
            # Gemini 호출
            response = await self._call_gemini_async(enhanced_prompt)
            
            self.usage_stats["successful_requests"] += 1
            
            return response.text
            
        except Exception as e:
            self.usage_stats["failed_requests"] += 1
            raise LLMClientError(f"LLM 추론 실패: {str(e)}")
    
    async def reason_with_structure(self, prompt: str, expected_structure: Dict[str, str]) -> Dict[str, Any]:
        """
        구조화된 추론 실행
        
        Agent가 일관된 형태로 추론 결과를 반환하도록 합니다.
        
        Args:
            prompt: 추론할 내용
            expected_structure: 기대하는 응답 구조
            
        Returns:
            구조화된 추론 결과
        """
        structured_prompt = f"""
        {prompt}
        
        다음 JSON 형태로 응답해주세요:
        {json.dumps(expected_structure, ensure_ascii=False, indent=2)}
        
        각 필드의 의미:
        {self._explain_structure(expected_structure)}
        
        반드시 유효한 JSON 형태로만 응답하세요.
        """
        
        try:
            response = await self.complete(structured_prompt)
            
            # JSON 파싱 시도
            try:
                structured_result = json.loads(response)
                return structured_result
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 재시도
                return await self._retry_structured_response(structured_prompt)
                
        except Exception as e:
            raise LLMClientError(f"구조화된 추론 실패: {str(e)}")
    
    async def analyze_game_situation(self, game_state: Dict[str, Any], agent_persona: Dict[str, Any]) -> Dict[str, Any]:
        """
        게임 상황 분석 - 게임 Agent 전용 추론
        
        Args:
            game_state: 현재 게임 상태
            agent_persona: Agent의 페르소나 정보
            
        Returns:
            게임 상황 분석 결과
        """
        analysis_prompt = f"""
        당신은 테이블게임을 플레이하는 AI 플레이어입니다.
        
        당신의 성격:
        - 이름: {agent_persona.get('name', '알 수 없음')}
        - 성격 유형: {agent_persona.get('archetype', '알 수 없음')}
        - 특성: {agent_persona.get('traits', {})}
        
        현재 게임 상황:
        {json.dumps(game_state, ensure_ascii=False, indent=2)}
        
        이 상황을 분석하고 다음 정보를 제공해주세요:
        1. 현재 상황에 대한 이해
        2. 가능한 행동 옵션들
        3. 각 옵션의 장단점
        4. 당신의 성격에 맞는 최적 전략
        5. 행동에 대한 확신도 (0.0-1.0)
        """
        
        expected_structure = {
            "situation_analysis": "현재 상황 분석",
            "available_actions": ["가능한 행동들"],
            "action_evaluation": {"행동": "평가"},
            "recommended_action": "추천 행동",
            "reasoning": "추론 과정",
            "confidence": 0.8
        }
        
        return await self.reason_with_structure(analysis_prompt, expected_structure)
    
    async def _call_gemini_async(self, prompt: str):
        """Gemini API 비동기 호출"""
        # Gemini는 기본적으로 동기이므로 비동기로 래핑
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.model.generate_content, prompt)
    
    def _enhance_prompt_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """컨텍스트를 포함한 프롬프트 생성"""
        context_str = json.dumps(context, ensure_ascii=False, indent=2)
        
        return f"""
        컨텍스트 정보:
        {context_str}
        
        요청:
        {prompt}
        
        위 컨텍스트를 참고하여 답변해주세요.
        """
    
    def _explain_structure(self, structure: Dict[str, str]) -> str:
        """구조 설명 생성"""
        explanations = []
        for key, description in structure.items():
            explanations.append(f"- {key}: {description}")
        return "\n".join(explanations)
    
    async def _retry_structured_response(self, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """구조화된 응답 재시도"""
        for attempt in range(max_retries):
            try:
                retry_prompt = f"""
                {prompt}
                
                이전 응답이 유효한 JSON 형태가 아니었습니다.
                반드시 유효한 JSON만 응답하세요. 추가 설명은 하지 마세요.
                """
                
                response = await self.complete(retry_prompt)
                return json.loads(response)
                
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries - 1:
                    # 최종 실패 시 기본 구조 반환
                    return {
                        "error": "구조화된 응답 생성 실패",
                        "raw_response": response if 'response' in locals() else "",
                        "confidence": 0.1
                    }
                continue
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """사용 통계 반환"""
        success_rate = 0.0
        if self.usage_stats["total_requests"] > 0:
            success_rate = self.usage_stats["successful_requests"] / self.usage_stats["total_requests"]
        
        return {
            **self.usage_stats,
            "success_rate": success_rate,
            "current_time": datetime.now().isoformat()
        }
    
    async def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            test_response = await self.complete("안녕하세요. 연결 테스트입니다. '연결 성공'이라고 답변해주세요.")
            return "연결 성공" in test_response
        except Exception:
            return False


class LLMClientError(Exception):
    """LLM 클라이언트 에러"""
    pass


# 싱글톤 패턴으로 전역 LLM 클라이언트 제공
_global_llm_client: Optional[GeminiLLMClient] = None


def get_llm_client(api_key: Optional[str] = None) -> GeminiLLMClient:
    """전역 LLM 클라이언트 반환"""
    global _global_llm_client
    
    if _global_llm_client is None:
        _global_llm_client = GeminiLLMClient(api_key)
    
    return _global_llm_client


def reset_llm_client():
    """LLM 클라이언트 초기화 (테스트용)"""
    global _global_llm_client
    _global_llm_client = None 