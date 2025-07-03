#!/usr/bin/env python3
"""
게임 UI 분석 에이전트 - LangGraph 기반
게임 설명을 분석하여 적절한 UI 구조를 결정하는 실제 AI 에이전트
"""

import json
import os
from typing import Dict, List, Any, TypedDict, Annotated
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class GameAnalysisState(TypedDict):
    """게임 분석 상태"""
    messages: Annotated[list, add_messages]
    game_description: str
    detailed_rules: str
    analysis_result: Dict[str, Any]
    ui_spec: Dict[str, Any]
    confidence_score: float
    error_message: str

class GameUIAnalyzerAgent:
    """게임 UI 분석 에이전트"""
    
    def __init__(self, model_name: str = None):
        # 환경변수나 기본값 사용 - Gemini 모델 설정
        if model_name is None:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        
        # Gemini API Key 확인
        if not os.getenv("GEMINI_API_KEY"):
            print("경고: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.1,
            convert_system_message_to_human=True,  # Gemini용 설정
            google_api_key=os.getenv("GEMINI_API_KEY")  # 직접 API 키 전달
        )
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _create_workflow(self) -> StateGraph:
        """워크플로우 생성"""
        
        workflow = StateGraph(GameAnalysisState)
        
        # 노드 추가
        workflow.add_node("analyze_game", self._analyze_game)
        workflow.add_node("determine_ui_type", self._determine_ui_type)
        workflow.add_node("generate_ui_spec", self._generate_ui_spec)
        workflow.add_node("validate_spec", self._validate_spec)
        
        # 엣지 추가
        workflow.set_entry_point("analyze_game")
        workflow.add_edge("analyze_game", "determine_ui_type")
        workflow.add_edge("determine_ui_type", "generate_ui_spec")
        workflow.add_edge("generate_ui_spec", "validate_spec")
        workflow.add_edge("validate_spec", END)
        
        return workflow
    
    def _analyze_game(self, state: GameAnalysisState) -> Dict[str, Any]:
        """게임 설명 분석"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 보드게임 전문가입니다. 주된 정보 소스인 '게임 설명' (BGG 기반)과 보조 정보 소스인 '추가 규칙' (웹 검색 기반)을 모두 참고하여 게임을 분석해주세요.

두 정보 간에 차이가 있을 경우, 공식 룰북일 가능성이 높은 '추가 규칙'을 더 신뢰해야 합니다.

두 정보를 종합하여 다음 정보를 추출해주세요:

1. 게임명 (추정)
2. 게임 카테고리 (추상전략, 카드게임, 소셜추론, 협력게임, 타일배치, 기타)
3. 보드 타입 (grid, card_layout, text_based, map, free_form, hybrid)
4. 플레이어 수 범위
5. 게임 복잡도 (simple, medium, complex)
6. 핵심 메커니즘 리스트
7. 특수 규칙이나 기능

JSON 형식으로 응답해주세요. '추가 규칙' 내용이 비어있다면 '게임 설명'만으로 분석해주세요.
            """),
            ("human", "게임 설명: {description}\n\n추가 규칙: {rules}")
        ])
        
        try:
            response = self.llm.invoke(
                analysis_prompt.format_messages(
                    description=state["game_description"],
                    rules=state.get("detailed_rules", "")
                )
            )
            
            # JSON 파싱 시도
            analysis_text = response.content
            
            # JSON 부분만 추출 (```json으로 감싸진 경우 처리)
            if "```json" in analysis_text:
                json_start = analysis_text.find("```json") + 7
                json_end = analysis_text.find("```", json_start)
                analysis_text = analysis_text[json_start:json_end]
            elif "```" in analysis_text:
                json_start = analysis_text.find("```") + 3
                json_end = analysis_text.find("```", json_start)
                analysis_text = analysis_text[json_start:json_end]
            
            analysis_result = json.loads(analysis_text.strip())
            
            return {
                **state,
                "analysis_result": analysis_result,
                "messages": state["messages"] + [
                    HumanMessage(content=f"게임 분석 완료: {state['game_description']}"),
                    response
                ]
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"게임 분석 실패: {str(e)}",
                "analysis_result": {"error": f"Failed to analyze game: {e}"}
            }
    
    def _determine_ui_type(self, state: GameAnalysisState) -> Dict[str, Any]:
        """UI 타입 결정"""
        
        # 만약 이전 단계에서 에러가 있었다면, UI 타입 결정을 건너뜀
        if state.get("error_message"):
            return {**state, "ui_spec": {"error": "Cannot determine UI type due to previous error."}}

        analysis = state.get("analysis_result", {})
        if not analysis or "error" in analysis:
            return {
                **state,
                "error_message": "UI 타입을 결정할 수 없습니다. 게임 분석 결과가 유효하지 않습니다.",
                "ui_spec": {"error": "Analysis result is invalid."}
            }
            
        board_type = analysis.get("보드_타입", "grid")
        category = analysis.get("게임_카테고리", "기타")
        mechanisms = analysis.get("핵심_메커니즘", [])
        
        # 보드 타입 매핑
        board_type_mapping = {
            "grid": "grid",
            "card_layout": "card_layout", 
            "text_based": "text_based",
            "map": "map",
            "free_form": "free_form",
            "hybrid": "hybrid"
        }
        
        # 카테고리별 기본 컴포넌트
        category_components = {
            "추상전략": ["turn_indicator", "action_buttons", "score_board"],
            "카드게임": ["player_hand", "action_buttons", "resource_tracker"],
            "소셜추론": ["chat", "player_list", "voting", "turn_indicator"],
            "협력게임": ["resource_tracker", "turn_indicator", "action_buttons"],
            "타일배치": ["resource_tracker", "action_buttons", "score_board"]
        }
        
        # 메커니즘별 추가 컴포넌트
        mechanism_components = {
            "투표": ["voting"],
            "베팅": ["resource_tracker"],
            "채팅": ["chat"],
            "실시간": ["timer"],
            "협력": ["shared_resources"]
        }
        
        # 최종 컴포넌트 결정
        required_components = set(category_components.get(category, ["action_buttons"]))
        
        for mechanism in mechanisms:
            if mechanism in mechanism_components:
                required_components.update(mechanism_components[mechanism])
        
        # 복잡도 계산
        complexity_score = len(mechanisms) + len(required_components)
        if complexity_score <= 3:
            complexity = "simple"
        elif complexity_score <= 6:
            complexity = "medium"
        else:
            complexity = "complex"
        
        ui_determination = {
            "board_type": board_type_mapping.get(board_type, "grid"),
            "required_components": list(required_components),
            "complexity": complexity,
            "estimated_confidence": 0.8 if board_type in board_type_mapping else 0.6
        }
        
        return {
            **state,
            "ui_spec": {**state.get("ui_spec", {}), **ui_determination}
        }
    
    def _generate_ui_spec(self, state: GameAnalysisState) -> Dict[str, Any]:
        """상세 UI 명세 생성"""
        
        spec_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 UI/UX 전문가입니다. 게임 분석 결과를 바탕으로 상세한 UI 명세를 생성해주세요.

다음 형식의 JSON으로 응답해주세요:
{{
    "layout_structure": {{
        "main_area": {{"rows": 8, "cols": 8, "특수기능들..."}},
        "sidebar": {{"controls": true, "player_info": true}},
        "bottom_panel": {{"hand_display": true, "chat": true}}
    }},
    "interaction_patterns": ["click_select", "drag_drop", "text_input"],
    "special_features": {{
        "coordinate_system": true,
        "animation": true,
        "real_time_updates": false
    }},
    "responsive_behavior": {{
        "mobile_friendly": true,
        "scalable_components": true
    }}
}}
            """),
            ("human", """
게임 분석: {analysis}
UI 타입: {ui_type}
필요 컴포넌트: {components}
복잡도: {complexity}

이 정보를 바탕으로 최적의 UI 명세를 생성해주세요.
            """)
        ])
        
        try:
            response = self.llm.invoke(
                spec_prompt.format_messages(
                    analysis=json.dumps(state["analysis_result"], ensure_ascii=False),
                    ui_type=state["ui_spec"]["board_type"],
                    components=state["ui_spec"]["required_components"],
                    complexity=state["ui_spec"]["complexity"]
                )
            )
            
            # JSON 파싱
            spec_text = response.content
            if "```json" in spec_text:
                json_start = spec_text.find("```json") + 7
                json_end = spec_text.find("```", json_start)
                spec_text = spec_text[json_start:json_end]
            elif "```" in spec_text:
                json_start = spec_text.find("```") + 3
                json_end = spec_text.find("```", json_start)
                spec_text = spec_text[json_start:json_end]
            
            detailed_spec = json.loads(spec_text.strip())
            
            return {
                **state,
                "ui_spec": {**state["ui_spec"], **detailed_spec},
                "messages": state["messages"] + [response]
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"UI 명세 생성 실패: {str(e)}",
                "ui_spec": {**state.get("ui_spec", {}), "error": f"Failed to generate UI spec: {e}"}
            }
    
    def _validate_spec(self, state: GameAnalysisState) -> Dict[str, Any]:
        """UI 명세 검증 및 신뢰도 계산"""
        
        ui_spec = state["ui_spec"]
        analysis = state["analysis_result"]
        
        # 이전 단계 에러 확인
        if "error" in ui_spec or "error" in analysis:
            return {
                **state,
                "confidence_score": 0.0,
                "error_message": state.get("error_message", "검증 실패: 이전 단계에서 오류 발생")
            }
        
        # 신뢰도 계산
        confidence_factors = []
        
        # 1. 보드 타입 일치성 검사
        board_type = ui_spec.get("board_type", "grid")
        if board_type in ["grid", "card_layout", "text_based"]:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)
        
        # 2. 컴포넌트 적절성 검사
        components = ui_spec.get("required_components", [])
        if len(components) >= 2:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # 3. 레이아웃 구조 완성도 검사
        layout = ui_spec.get("layout_structure", {})
        if "main_area" in layout and len(layout) >= 2:
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.65)
        
        # 4. 에러 여부 확인
        if state.get("error_message"):
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(1.0)
        
        # 최종 신뢰도 계산
        final_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return {
            **state,
            "confidence_score": final_confidence,
            "ui_spec": {
                **ui_spec,
                "game_name": analysis.get("게임명", "Unknown Game"),
                "generated_at": datetime.now().isoformat(),
                "validation_passed": final_confidence > 0.7
            }
        }
    
    async def analyze_game_for_ui(self, game_description: str, detailed_rules: str = "") -> Dict[str, Any]:
        """게임 분석 및 UI 명세 생성 (메인 인터페이스)"""
        
        initial_state = GameAnalysisState(
            messages=[],
            game_description=game_description,
            detailed_rules=detailed_rules,
            analysis_result={},
            ui_spec={},
            confidence_score=0.0,
            error_message=""
        )
        
        try:
            # 워크플로우 실행
            result = await self.app.ainvoke(initial_state)
            
            # 최종 결과에 에러가 있는지 확인
            success = not result.get("error_message")

            return {
                "success": success,
                "game_name": result.get("ui_spec", {}).get("game_name", "Unknown Game"),
                "board_type": result.get("ui_spec", {}).get("board_type", "grid"),
                "required_components": result.get("ui_spec", {}).get("required_components", []),
                "layout_structure": result.get("ui_spec", {}).get("layout_structure", {}),
                "interaction_patterns": result.get("ui_spec", {}).get("interaction_patterns", []),
                "special_features": result.get("ui_spec", {}).get("special_features", {}),
                "complexity": result.get("ui_spec", {}).get("complexity", "medium"),
                "confidence_score": result.get("confidence_score", 0.0),
                "analysis_result": result.get("analysis_result", {}),
                "error_message": result.get("error_message", ""),
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"워크플로우 실행 실패: {str(e)}",
                "game_name": "Error Game",
                "board_type": "grid",
                "required_components": ["action_buttons"],
                "layout_structure": {"main_area": {"rows": 3, "cols": 3}},
                "interaction_patterns": ["click_select"],
                "special_features": {},
                "complexity": "simple",
                "confidence_score": 0.3,
                "generated_at": datetime.now()
            }

# 싱글톤 에이전트 인스턴스
_game_ui_analyzer = None

def get_game_ui_analyzer() -> GameUIAnalyzerAgent:
    """게임 UI 분석 에이전트 싱글톤 인스턴스 반환"""
    global _game_ui_analyzer
    if _game_ui_analyzer is None:
        _game_ui_analyzer = GameUIAnalyzerAgent()
    return _game_ui_analyzer 