"""
Game UI Analyzer Agent

보드게임 설명과 규칙을 분석하여 UI 명세서를 생성하는 LangGraph 에이전트
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.llm.fallback_llm import _try_fallback_llm, DirectHTTPLLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

logger = logging.getLogger(__name__)


class FallbackLangChainLLM:
    """LangChain 호환 Fallback LLM Wrapper"""
    
    def __init__(self, primary_model: str = "gemini-2.5-flash-lite"):
        """초기화"""
        self.primary_model = primary_model
        self.llm = None
        self.fallback_llm = None
        self._use_fallback_only = False
        self._init_llm()
    
    def _init_llm(self):
        """LLM 초기화 (event loop 체크 후 결정)"""
        try:
            # Event loop가 실행 중이고 닫히지 않았는지 확인
            try:
                loop = asyncio.get_running_loop()
                if loop.is_closed():
                    logger.warning("Event loop is closed, using fallback LLM only")
                    self._use_fallback_only = True
                    self._init_fallback_llm()
                    return
            except RuntimeError:
                # Event loop가 없는 경우도 fallback 사용
                logger.warning("No event loop available, using fallback LLM only")
                self._use_fallback_only = True
                self._init_fallback_llm()
                return
            
            # Event loop가 정상인 경우 기본 LLM 시도
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.primary_model,
                temperature=0.7,
                google_api_key=api_key
            )
            logger.info(f"Primary LLM ({self.primary_model}) 초기화 완료")
        except Exception as e:
            logger.warning(f"Primary LLM 초기화 실패: {e}, fallback 시도")
            self._use_fallback_only = True
            self._init_fallback_llm()
    
    def _init_fallback_llm(self):
        """Fallback LLM 초기화"""
        try:
            fallback_llm = _try_fallback_llm(self.primary_model, logger)
            if fallback_llm:
                self.fallback_llm = fallback_llm
                logger.info("Fallback LLM 초기화 완료")
            else:
                raise ValueError("Fallback LLM 초기화 실패")
        except Exception as e:
            logger.error(f"Fallback LLM 초기화 실패: {e}")
            raise
    
    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """비동기 LLM 호출 (fallback 지원)"""
        # Fallback만 사용하도록 설정된 경우
        if self._use_fallback_only:
            if self.fallback_llm:
                return await self._invoke_fallback(messages, **kwargs)
            else:
                raise RuntimeError("Fallback LLM이 초기화되지 않았습니다.")
        
        # 기본 LLM 시도
        if self.llm:
            try:
                # Event loop 체크
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    # Event loop가 없거나 닫혀있는 경우 fallback으로 전환
                    logger.warning("Event loop is closed or not available, using fallback LLM")
                    if not self.fallback_llm:
                        self._init_fallback_llm()
                    if self.fallback_llm:
                        self._use_fallback_only = True  # 이후부터는 fallback만 사용
                        return await self._invoke_fallback(messages, **kwargs)
                    raise RuntimeError("Event loop is closed and fallback LLM is not available")
                
                response = await self.llm.ainvoke(messages, **kwargs)
                return response
            except RuntimeError as e:
                error_str = str(e).lower()
                # Event loop 관련 에러인 경우 즉시 fallback으로 전환
                if "event loop is closed" in error_str or "event loop" in error_str:
                    logger.warning(f"Event loop 에러 발생, fallback으로 전환: {e}")
                    if not self.fallback_llm:
                        self._init_fallback_llm()
                    if self.fallback_llm:
                        self._use_fallback_only = True  # 이후부터는 fallback만 사용
                        return await self._invoke_fallback(messages, **kwargs)
                raise
            except Exception as e:
                error_str = str(e).lower()
                # 429, 503, quota, resource_exhausted 에러인 경우 fallback 시도
                if ("429" in error_str or "503" in error_str or "quota" in error_str or 
                    "resource_exhausted" in error_str or "overloaded" in error_str):
                    logger.warning(f"Primary LLM 오류 발생, fallback 시도: {e}")
                    if not self.fallback_llm:
                        self._init_fallback_llm()
                    if self.fallback_llm:
                        return await self._invoke_fallback(messages, **kwargs)
                raise
        
        # Fallback LLM 사용
        if self.fallback_llm:
            return await self._invoke_fallback(messages, **kwargs)
        
        raise RuntimeError("LLM이 초기화되지 않았습니다.")
    
    async def _invoke_fallback(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Fallback LLM 호출"""
        # LangChain 메시지를 텍스트로 변환
        prompt_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt_text += f"{msg.content}\n"
            elif isinstance(msg, AIMessage):
                prompt_text += f"Assistant: {msg.content}\n"
        
        # Fallback LLM 호출
        if isinstance(self.fallback_llm, DirectHTTPLLM):
            # DirectHTTPLLM은 generate_str 메서드 사용 (OpenAI 클라이언트 사용 안 함)
            result = await self.fallback_llm.generate_str(
                message=prompt_text.strip(),
                request_params=None
            )
            return AIMessage(content=result)
        elif isinstance(self.fallback_llm, GoogleAugmentedLLM):
            # GoogleAugmentedLLM은 augmented_generate 메서드 사용
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_messages.append({"role": "assistant", "content": msg.content})
            
            result = await self.fallback_llm.augmented_generate(
                RequestParams(
                    messages=formatted_messages,
                    tools_choice="none"
                )
            )
            return AIMessage(content=result.content)
        else:
            raise RuntimeError(f"지원하지 않는 fallback LLM 타입: {type(self.fallback_llm)}")


class GameUIAnalysisState(BaseModel):
    """게임 UI 분석 상태"""
    game_description: str = ""
    detailed_rules: str = ""
    messages: list = Field(default_factory=list)
    ui_spec: Dict[str, Any] = Field(default_factory=dict)
    analysis_result: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 0.0
    error_message: Optional[str] = None


class GameUIAnalyzer:
    """게임 UI 분석기"""
    
    def __init__(self):
        """초기화"""
        # Fallback LLM wrapper 사용
        self.llm = FallbackLangChainLLM(primary_model="gemini-2.5-flash-lite")
        
        # 그래프 구성
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 상태 그래프 구성"""
        workflow = StateGraph(GameUIAnalysisState)
        
        # 노드 추가
        workflow.add_node("analyze_game", self._analyze_game)
        workflow.add_node("generate_ui_spec", self._generate_ui_spec)
        workflow.add_node("validate_spec", self._validate_spec)
        
        # 엣지 추가
        workflow.set_entry_point("analyze_game")
        workflow.add_edge("analyze_game", "generate_ui_spec")
        workflow.add_edge("generate_ui_spec", "validate_spec")
        workflow.add_edge("validate_spec", END)
        
        return workflow
    
    async def _analyze_game(self, state: GameUIAnalysisState) -> GameUIAnalysisState:
        """게임 분석"""
        try:
            prompt = f"""다음 보드게임 정보를 분석해주세요:

게임 설명:
{state.game_description}

상세 규칙:
{state.detailed_rules}

다음 항목을 분석해주세요:
1. 게임의 핵심 메커니즘
2. 필요한 게임 요소 (보드, 카드, 토큰 등)
3. 플레이어 수와 게임 시간
4. 게임의 복잡도
5. UI에 필요한 주요 컴포넌트

JSON 형식으로 분석 결과를 반환해주세요:
{{
    "core_mechanisms": ["메커니즘1", "메커니즘2"],
    "game_elements": ["요소1", "요소2"],
    "player_count": {{"min": 2, "max": 4}},
    "game_duration": "30-60분",
    "complexity": "low|medium|high",
    "ui_components": ["컴포넌트1", "컴포넌트2"]
}}"""

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            analysis_text = response.content
            
            # JSON 추출 시도
            try:
                # JSON 코드 블록에서 추출
                if "```json" in analysis_text:
                    json_start = analysis_text.find("```json") + 7
                    json_end = analysis_text.find("```", json_start)
                    analysis_text = analysis_text[json_start:json_end].strip()
                elif "```" in analysis_text:
                    json_start = analysis_text.find("```") + 3
                    json_end = analysis_text.find("```", json_start)
                    analysis_text = analysis_text[json_start:json_end].strip()
                
                state.analysis_result = json.loads(analysis_text)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트로 저장
                state.analysis_result = {"raw_analysis": analysis_text}
            
            state.messages.append(f"게임 분석 완료")
            logger.info("게임 분석 완료")
            
        except Exception as e:
            logger.error(f"게임 분석 오류: {e}")
            state.error_message = f"게임 분석 실패: {str(e)}"
        
        return state
    
    async def _generate_ui_spec(self, state: GameUIAnalysisState) -> GameUIAnalysisState:
        """UI 명세서 생성"""
        try:
            analysis = state.analysis_result
            
            prompt = f"""다음 게임 분석 결과를 바탕으로 UI 명세서를 생성해주세요:

게임 설명:
{state.game_description}

분석 결과:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

다음 형식의 UI 명세서를 JSON으로 생성해주세요. 실제로 Streamlit에서 렌더링 가능한 UI 컴포넌트와 플레이어 인터페이스를 포함해야 합니다:
{{
    "game_name": "게임 이름",
    "board_type": "grid|card|tile|other",
    "components": [
        {{
            "type": "board|card|token|dice|other",
            "name": "컴포넌트 이름",
            "description": "설명",
            "properties": {{}},
            "ui_component": "st.board|st.card|st.container|st.columns",
            "player_interface": {{
                "visible_to": "all|self|others",
                "interactive": true,
                "actions": ["click", "drag", "select"]
            }}
        }}
    ],
    "layout": {{
        "type": "grid|stack|hand|other",
        "description": "레이아웃 설명",
        "streamlit_layout": {{
            "columns": 3,
            "rows": 2,
            "spacing": "medium"
        }}
    }},
    "interactions": [
        {{
            "type": "click|drag|select|other",
            "description": "상호작용 설명",
            "streamlit_component": "st.button|st.selectbox|st.slider"
        }}
    ],
    "player_interface": {{
        "hand_display": {{
            "component": "st.container",
            "layout": "horizontal|vertical",
            "cards_per_row": 5
        }},
        "action_buttons": [
            {{
                "label": "행동 이름",
                "component": "st.button",
                "enabled": true
            }}
        ],
        "status_display": {{
            "component": "st.metric|st.info",
            "show": ["score", "resources", "status"]
        }}
    }}
}}"""

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            spec_text = response.content
            
            # JSON 추출
            try:
                if "```json" in spec_text:
                    json_start = spec_text.find("```json") + 7
                    json_end = spec_text.find("```", json_start)
                    spec_text = spec_text[json_start:json_end].strip()
                elif "```" in spec_text:
                    json_start = spec_text.find("```") + 3
                    json_end = spec_text.find("```", json_start)
                    spec_text = spec_text[json_start:json_end].strip()
                
                state.ui_spec = json.loads(spec_text)
            except json.JSONDecodeError:
                state.ui_spec = {"raw_spec": spec_text, "error": "JSON 파싱 실패"}
            
            state.messages.append("UI 명세서 생성 완료")
            logger.info("UI 명세서 생성 완료")
            
        except Exception as e:
            logger.error(f"UI 명세서 생성 오류: {e}")
            state.error_message = f"UI 명세서 생성 실패: {str(e)}"
        
        return state
    
    async def _validate_spec(self, state: GameUIAnalysisState) -> GameUIAnalysisState:
        """명세서 검증 및 신뢰도 계산"""
        try:
            # 기본 검증
            has_name = bool(state.ui_spec.get("game_name"))
            has_components = bool(state.ui_spec.get("components"))
            has_layout = bool(state.ui_spec.get("layout"))
            
            # 신뢰도 계산 (0.0 ~ 1.0)
            confidence = 0.0
            if has_name:
                confidence += 0.3
            if has_components:
                confidence += 0.4
            if has_layout:
                confidence += 0.3
            
            state.confidence_score = confidence
            
            if confidence < 0.5:
                state.error_message = "UI 명세서가 불완전합니다. 신뢰도가 낮습니다."
            
            state.messages.append(f"검증 완료 (신뢰도: {confidence:.1%})")
            logger.info(f"명세서 검증 완료 (신뢰도: {confidence:.1%})")
            
        except Exception as e:
            logger.error(f"명세서 검증 오류: {e}")
            state.error_message = f"명세서 검증 실패: {str(e)}"
            state.confidence_score = 0.0
        
        return state


# 싱글톤 인스턴스
_ui_analyzer_instance: Optional[GameUIAnalyzer] = None


def get_game_ui_analyzer() -> GameUIAnalyzer:
    """게임 UI 분석기 인스턴스 반환 (싱글톤)"""
    global _ui_analyzer_instance
    if _ui_analyzer_instance is None:
        _ui_analyzer_instance = GameUIAnalyzer()
    return _ui_analyzer_instance


# A2A를 위한 모듈 레벨 app 변수
# standard_agent_runner가 이 변수를 찾을 수 있도록
_analyzer = get_game_ui_analyzer()
app = _analyzer.app

