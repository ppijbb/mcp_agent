"""
게임 팩토리 - 게임별 특화 설정 및 인스턴스 생성

이 팩토리는 다양한 보드게임에 대한 특화된 설정과 Agent 행동을 제공합니다.
Agent들이 게임의 특성에 맞게 동작할 수 있도록 게임별 매개변수를 조정합니다.
"""

from typing import Dict, List, Any, Optional, Type
from enum import Enum
from dataclasses import dataclass
import json
import os
import uuid

from ..core import GameState, GamePhase
# ActionType은 현재 사용되지 않으므로 제거


class GameType(Enum):
    """게임 타입"""
    STRATEGY = "strategy"
    CARD = "card"
    DICE = "dice"
    SOCIAL = "social"
    PARTY = "party"
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"


class GameComplexity(Enum):
    """게임 복잡도 레벨"""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class GameTemplate:
    """게임 템플릿 - 게임별 기본 설정"""
    name: str
    game_type: GameType
    complexity: GameComplexity
    min_players: int
    max_players: int
    estimated_duration: int  # 분
    
    # AI 행동 매개변수
    ai_decision_time: float = 2.0  # 초
    ai_creativity_factor: float = 0.7  # 0-1
    ai_risk_tolerance: float = 0.5  # 0-1
    
    # 게임 특화 설정
    turn_based: bool = True
    simultaneous_actions: bool = False
    hidden_information: bool = False
    player_elimination: bool = False
    
    # 추천 페르소나 타입 (현재 사용되지 않음)
    recommended_personas: List[str] = None
    
    # 특수 규칙 힌트
    special_mechanics: List[str] = None
    victory_conditions: List[str] = None
    
    def __post_init__(self):
        if self.recommended_personas is None:
            self.recommended_personas = ["social", "strategic"]
        if self.special_mechanics is None:
            self.special_mechanics = []
        if self.victory_conditions is None:
            self.victory_conditions = ["점수 최대화"]


class GameFactory:
    """
    게임 팩토리 - 게임별 특화 설정 제공
    
    이 클래스는 Agent들이 다양한 보드게임에 적응할 수 있도록
    게임별 특화된 설정과 행동 매개변수를 제공합니다.
    """
    
    def __init__(self):
        """게임 템플릿들을 초기화"""
        self.templates: Dict[str, GameTemplate] = {}
        self.game_aliases: Dict[str, str] = {}  # 게임 별명 매핑
        self._load_default_templates()
        self._load_custom_templates()
    
    def _load_default_templates(self):
        """기본 게임 템플릿들 로드"""
        
        # 클래식 게임들
        self.templates["chess"] = GameTemplate(
            name="Chess",
            game_type=GameType.STRATEGY,
            complexity=GameComplexity.COMPLEX,
            min_players=2,
            max_players=2,
            estimated_duration=60,
            ai_creativity_factor=0.8,
            ai_risk_tolerance=0.4,
            hidden_information=False,
            player_elimination=True,
            recommended_personas=[PersonaArchetype.STRATEGIC, PersonaArchetype.ANALYTICAL],
            special_mechanics=["체크", "캐슬링", "앙파상"],
            victory_conditions=["체크메이트", "시간 승부"]
        )
        
        self.templates["틱택토"] = GameTemplate(
            name="틱택토",
            game_type=GameType.STRATEGY,
            complexity=GameComplexity.SIMPLE,
            min_players=2,
            max_players=2,
            estimated_duration=5,
            ai_decision_time=1.0,
            ai_creativity_factor=0.3,
            recommended_personas=[PersonaArchetype.SOCIAL],
            victory_conditions=["3개 연속 배치"]
        )
        
        # 파티 게임들
        self.templates["마피아"] = GameTemplate(
            name="마피아",
            game_type=GameType.SOCIAL_DEDUCTION,
            complexity=GameComplexity.MODERATE,
            min_players=6,
            max_players=12,
            estimated_duration=30,
            ai_creativity_factor=0.9,
            ai_risk_tolerance=0.6,
            hidden_information=True,
            player_elimination=True,
            recommended_personas=[PersonaArchetype.SOCIAL, PersonaArchetype.DECEPTIVE, PersonaArchetype.ANALYTICAL],
            special_mechanics=["투표", "역할 공개", "밤 페이즈"],
            victory_conditions=["마피아 전멸", "마피아가 시민과 동수"]
        )
        
        self.templates["뱅"] = GameTemplate(
            name="뱅",
            game_type=GameType.SOCIAL_DEDUCTION,
            complexity=GameComplexity.COMPLEX,
            min_players=4,
            max_players=7,
            estimated_duration=45,
            ai_creativity_factor=0.8,
            ai_risk_tolerance=0.7,
            hidden_information=True,
            player_elimination=True,
            recommended_personas=[PersonaArchetype.AGGRESSIVE, PersonaArchetype.STRATEGIC, PersonaArchetype.SOCIAL],
            special_mechanics=["거리 제한", "역할 카드", "장비 카드"],
            victory_conditions=["역할별 승리 조건"]
        )
        
        # 전략 게임들
        self.templates["카탄"] = GameTemplate(
            name="카탄",
            game_type=GameType.STRATEGY,
            complexity=GameComplexity.MODERATE,
            min_players=3,
            max_players=4,
            estimated_duration=90,
            ai_creativity_factor=0.6,
            ai_risk_tolerance=0.5,
            recommended_personas=[PersonaArchetype.STRATEGIC, PersonaArchetype.DIPLOMATIC],
            special_mechanics=["자원 수집", "교역", "도적"],
            victory_conditions=["10점 달성"]
        )
        
        # 별명 매핑
        self.game_aliases.update({
            "chess": "Chess",
            "tic-tac-toe": "틱택토",
            "tictactoe": "틱택토",
            "mafia": "마피아",
            "werewolf": "마피아",
            "bang": "뱅",
            "catan": "카탄",
            "settlers": "카탄"
        })
    
    def _load_custom_templates(self):
        """커스텀 게임 템플릿 로드 (JSON 파일에서)"""
        try:
            custom_file = os.path.join(os.path.dirname(__file__), "custom_games.json")
            if os.path.exists(custom_file):
                with open(custom_file, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                    
                for game_name, config in custom_data.items():
                    self.templates[game_name] = GameTemplate(**config)
                    
        except Exception as e:
            print(f"⚠️ 커스텀 게임 템플릿 로드 실패: {e}")
    
    def get_template(self, game_name: str) -> Optional[GameTemplate]:
        """
        게임 템플릿 반환
        
        Args:
            game_name: 게임 이름 (별명 포함)
            
        Returns:
            게임 템플릿 또는 None
        """
        # 직접 매칭
        if game_name in self.templates:
            return self.templates[game_name]
        
        # 별명 매칭
        normalized_name = game_name.lower().strip()
        if normalized_name in self.game_aliases:
            return self.templates[self.game_aliases[normalized_name]]
        
        # 부분 매칭
        for template_name in self.templates.keys():
            if normalized_name in template_name.lower():
                return self.templates[template_name]
        
        return None
    
    def create_game_metadata(self, game_name: str, **overrides) -> Dict[str, Any]:
        """
        게임 메타데이터 생성
        
        Args:
            game_name: 게임 이름
            **overrides: 덮어쓸 설정들
            
        Returns:
            GameMetadata 인스턴스
        """
        template = self.get_template(game_name)
        
        if template:
            metadata = GameMetadata(
                name=template.name,
                min_players=template.min_players,
                max_players=template.max_players,
                estimated_duration=template.estimated_duration,
                complexity=self._complexity_to_float(template.complexity),
                game_type=template.game_type,
                description=f"{template.name} - {template.complexity.value} 복잡도"
            )
        else:
            # 기본 메타데이터
            metadata = GameMetadata(
                name=game_name,
                min_players=2,
                max_players=4,
                estimated_duration=60,
                complexity=3.0,
                game_type=GameType.STRATEGY,
                description=f"{game_name} - 알려지지 않은 게임"
            )
        
        # 오버라이드 적용
        for key, value in overrides.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        return metadata
    
    def get_ai_config(self, game_name: str) -> Dict[str, Any]:
        """
        게임별 AI 설정 반환
        
        Args:
            game_name: 게임 이름
            
        Returns:
            AI 설정 딕셔너리
        """
        template = self.get_template(game_name)
        
        if template:
            return {
                "decision_time": template.ai_decision_time,
                "creativity_factor": template.ai_creativity_factor,
                "risk_tolerance": template.ai_risk_tolerance,
                "recommended_personas": [p.value for p in template.recommended_personas],
                "special_mechanics": template.special_mechanics,
                "victory_conditions": template.victory_conditions,
                "turn_based": template.turn_based,
                "simultaneous_actions": template.simultaneous_actions,
                "hidden_information": template.hidden_information,
                "player_elimination": template.player_elimination
            }
        else:
            # 기본 AI 설정
            return {
                "decision_time": 2.0,
                "creativity_factor": 0.5,
                "risk_tolerance": 0.5,
                "recommended_personas": ["casual", "strategic"],
                "special_mechanics": [],
                "victory_conditions": ["점수 최대화"],
                "turn_based": True,
                "simultaneous_actions": False,
                "hidden_information": False,
                "player_elimination": False
            }
    
    def _complexity_to_float(self, complexity: GameComplexity) -> float:
        """복잡도 enum을 float로 변환"""
        mapping = {
            GameComplexity.SIMPLE: 1.5,
            GameComplexity.MODERATE: 3.0,
            GameComplexity.COMPLEX: 4.0,
            GameComplexity.EXPERT: 5.0
        }
        return mapping.get(complexity, 3.0)
    
    def list_available_games(self) -> List[str]:
        """사용 가능한 게임 목록 반환"""
        return list(self.templates.keys())
    
    def get_games_by_player_count(self, player_count: int) -> List[str]:
        """플레이어 수에 맞는 게임 목록 반환"""
        suitable_games = []
        
        for name, template in self.templates.items():
            if template.min_players <= player_count <= template.max_players:
                suitable_games.append(name)
        
        return suitable_games
    
    def get_games_by_complexity(self, complexity: GameComplexity) -> List[str]:
        """복잡도별 게임 목록 반환"""
        return [
            name for name, template in self.templates.items()
            if template.complexity == complexity
        ]
    
    def add_custom_game(self, template: GameTemplate):
        """커스텀 게임 추가"""
        self.templates[template.name] = template
        print(f"✅ 커스텀 게임 추가: {template.name}")
    
    def save_custom_templates(self):
        """커스텀 템플릿들을 JSON 파일로 저장"""
        try:
            custom_file = os.path.join(os.path.dirname(__file__), "custom_games.json")
            
            # 기본 템플릿 제외하고 저장
            default_names = {"Chess", "TicTacToe", "Mafia", "Bang", "Catan"}
            custom_templates = {
                name: template for name, template in self.templates.items()
                if name not in default_names
            }
            
            if custom_templates:
                # dataclass를 dict로 변환
                serializable_data = {}
                for name, template in custom_templates.items():
                    template_dict = {
                        "name": template.name,
                        "game_type": template.game_type.value,
                        "complexity": template.complexity.value,
                        "min_players": template.min_players,
                        "max_players": template.max_players,
                        "estimated_duration": template.estimated_duration,
                        "ai_decision_time": template.ai_decision_time,
                        "ai_creativity_factor": template.ai_creativity_factor,
                        "ai_risk_tolerance": template.ai_risk_tolerance,
                        "turn_based": template.turn_based,
                        "simultaneous_actions": template.simultaneous_actions,
                        "hidden_information": template.hidden_information,
                        "player_elimination": template.player_elimination,
                        "recommended_personas": [p.value for p in template.recommended_personas],
                        "special_mechanics": template.special_mechanics,
                        "victory_conditions": template.victory_conditions
                    }
                    serializable_data[name] = template_dict
                
                with open(custom_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 커스텀 게임 템플릿 저장: {len(custom_templates)}개")
                
        except Exception as e:
            print(f"⚠️ 커스텀 템플릿 저장 실패: {e}")

    @classmethod
    def create_game_from_info(cls, game_info: 'GameInfo') -> GameState:
        # Implementation of create_game_from_info method
        pass

    @classmethod
    def create_avalon(cls) -> GameState:
        return cls.create_game_from_info(
            GameInfo(
                game_id="avalon",
                name="아발론",
                description="아서왕의 기사들과 모드레드의 하수인 간의 숨막히는 심리전",
                player_count=10,
                avg_play_time=45,
                game_mechanics=["Social Deduction", "Hidden Roles", "Team-Based"],
                recommended_personas=[PersonaArchetype.AGGRESSIVE, PersonaArchetype.STRATEGIC, PersonaArchetype.SOCIAL],
            )
        )

    @classmethod
    def create_uno(cls) -> GameState:
        return cls.create_game_from_info(
            GameInfo(
                game_id="uno",
                name="우노",
                description="같은 색이나 숫자의 카드를 내는 간단한 카드 게임",
                player_count=4,
                avg_play_time=20,
                game_mechanics=["Hand Management", "Matching"],
                recommended_personas=[PersonaArchetype.SOCIAL],
            )
        )
    
    @classmethod
    def create_custom_game(cls, name: str, description: str, player_count: int = 4) -> GameState:
        # Implementation of create_custom_game method
        pass


# 전역 팩토리 인스턴스
_global_game_factory: Optional[GameFactory] = None


def get_game_factory() -> GameFactory:
    """전역 게임 팩토리 반환"""
    global _global_game_factory
    
    if _global_game_factory is None:
        _global_game_factory = GameFactory()
    
    return _global_game_factory 