"""
Game UI Analyzer Agent

보드게임 설명과 규칙을 분석하여 UI 명세서를 생성하는 LangGraph 에이전트
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os

logger = logging.getLogger(__name__)


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
        # LLM 초기화
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.7,
            google_api_key=api_key
        )
        
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

다음 형식의 UI 명세서를 JSON으로 생성해주세요:
{{
    "game_name": "게임 이름",
    "board_type": "grid|card|tile|other",
    "components": [
        {{
            "type": "board|card|token|dice|other",
            "name": "컴포넌트 이름",
            "description": "설명",
            "properties": {{}}
        }}
    ],
    "layout": {{
        "type": "grid|stack|hand|other",
        "description": "레이아웃 설명"
    }},
    "interactions": [
        {{
            "type": "click|drag|select|other",
            "description": "상호작용 설명"
        }}
    ]
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

