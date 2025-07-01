"""
AI 플레이어 페르소나 시스템
게임별 특화된 AI 성격과 전략 정의
"""

from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from dataclasses import dataclass
import random

class PersonaArchetype(Enum):
    """기본 페르소나 원형"""
    AGGRESSIVE = "aggressive"        # 공격적
    DEFENSIVE = "defensive"          # 수비적  
    ANALYTICAL = "analytical"        # 분석적
    INTUITIVE = "intuitive"          # 직관적
    COOPERATIVE = "cooperative"      # 협력적
    COMPETITIVE = "competitive"      # 경쟁적
    RISK_TAKER = "risk_taker"       # 위험 감수자
    CONSERVATIVE = "conservative"    # 보수적
    SOCIAL = "social"               # 사교적
    STRATEGIC = "strategic"         # 전략적

class CommunicationStyle(Enum):
    """의사소통 스타일"""
    VERBOSE = "verbose"             # 말이 많음
    CONCISE = "concise"            # 간결함
    FRIENDLY = "friendly"          # 친근함
    FORMAL = "formal"              # 격식을 차림
    HUMOROUS = "humorous"          # 유머러스
    SERIOUS = "serious"            # 진중함
    SARCASTIC = "sarcastic"        # 비꼬는 듯
    ENCOURAGING = "encouraging"    # 격려하는

@dataclass
class PersonaTraits:
    """페르소나 특성"""
    # 기본 성향 (0.0 - 1.0)
    aggression: float = 0.5        # 공격성
    risk_tolerance: float = 0.5    # 위험 감수성
    cooperation: float = 0.5       # 협력성
    patience: float = 0.5          # 인내심
    creativity: float = 0.5        # 창의성
    logic: float = 0.5            # 논리성
    
    # 게임 플레이 스타일
    planning_depth: float = 0.5    # 계획 깊이
    adaptability: float = 0.5      # 적응력
    bluffing_skill: float = 0.5    # 허풍/속임수 능력
    observation: float = 0.5       # 관찰력
    
    def randomize_within_archetype(self, archetype: PersonaArchetype) -> None:
        """원형에 맞게 특성을 랜덤화"""
        base_traits = self._get_archetype_base_traits(archetype)
        
        for trait, base_value in base_traits.items():
            # 기본값 ±0.2 범위에서 랜덤화
            variation = random.uniform(-0.2, 0.2)
            new_value = max(0.0, min(1.0, base_value + variation))
            setattr(self, trait, new_value)
    
    def _get_archetype_base_traits(self, archetype: PersonaArchetype) -> Dict[str, float]:
        """원형별 기본 특성값"""
        trait_map = {
            PersonaArchetype.AGGRESSIVE: {
                "aggression": 0.8, "risk_tolerance": 0.7, "cooperation": 0.3,
                "patience": 0.2, "planning_depth": 0.4
            },
            PersonaArchetype.DEFENSIVE: {
                "aggression": 0.2, "risk_tolerance": 0.3, "cooperation": 0.6,
                "patience": 0.8, "planning_depth": 0.7
            },
            PersonaArchetype.ANALYTICAL: {
                "logic": 0.9, "planning_depth": 0.8, "observation": 0.8,
                "creativity": 0.3, "patience": 0.7
            },
            PersonaArchetype.INTUITIVE: {
                "logic": 0.3, "creativity": 0.8, "adaptability": 0.8,
                "planning_depth": 0.3, "risk_tolerance": 0.6
            },
            PersonaArchetype.COOPERATIVE: {
                "cooperation": 0.9, "aggression": 0.2, "risk_tolerance": 0.4,
                "patience": 0.7
            },
            PersonaArchetype.STRATEGIC: {
                "planning_depth": 0.9, "logic": 0.8, "patience": 0.8,
                "observation": 0.7, "creativity": 0.6
            }
        }
        return trait_map.get(archetype, {})

@dataclass
class GameSpecificBehavior:
    """게임별 특화 행동 패턴"""
    preferred_actions: List[str]      # 선호하는 액션들
    avoided_actions: List[str]        # 피하는 액션들
    decision_factors: Dict[str, float] # 의사결정 요소별 가중치
    interaction_preferences: Dict[str, str] # 다른 플레이어와의 상호작용 선호도

class PersonaProfile(TypedDict):
    """완전한 페르소나 프로필"""
    # 기본 정보
    name: str
    archetype: PersonaArchetype
    communication_style: CommunicationStyle
    
    # 성격 특성
    traits: PersonaTraits
    
    # 배경 설정
    background_story: str
    catchphrases: List[str]          # 자주 쓰는 말
    
    # 게임별 행동
    game_behaviors: Dict[str, GameSpecificBehavior]
    
    # 메타 정보
    created_for_game: str
    difficulty_level: str            # easy, medium, hard

class PersonaGenerator:
    """페르소나 생성기"""
    
    @staticmethod
    def generate_for_game(
        game_name: str,
        game_type: str,
        count: int,
        difficulty: str = "medium"
    ) -> List[PersonaProfile]:
        """게임에 특화된 페르소나들을 생성"""
        
        # 게임 타입별 적합한 원형들 선택
        suitable_archetypes = PersonaGenerator._get_suitable_archetypes(game_type)
        
        personas = []
        for i in range(count):
            archetype = random.choice(suitable_archetypes)
            persona = PersonaGenerator._create_persona(
                game_name, archetype, difficulty, i
            )
            personas.append(persona)
        
        # 다양성 보장 - 너무 비슷한 페르소나들 조정
        PersonaGenerator._ensure_diversity(personas)
        
        return personas
    
    @staticmethod
    def _get_suitable_archetypes(game_type: str) -> List[PersonaArchetype]:
        """게임 타입별 적합한 페르소나 원형들"""
        archetype_map = {
            "strategy": [
                PersonaArchetype.ANALYTICAL, PersonaArchetype.STRATEGIC,
                PersonaArchetype.CONSERVATIVE, PersonaArchetype.COMPETITIVE
            ],
            "social": [
                PersonaArchetype.SOCIAL, PersonaArchetype.INTUITIVE,
                PersonaArchetype.AGGRESSIVE, PersonaArchetype.ANALYTICAL
            ],
            "negotiation": [
                PersonaArchetype.COOPERATIVE, PersonaArchetype.COMPETITIVE,
                PersonaArchetype.SOCIAL, PersonaArchetype.STRATEGIC
            ],
            "card": [
                PersonaArchetype.RISK_TAKER, PersonaArchetype.CONSERVATIVE,
                PersonaArchetype.ANALYTICAL, PersonaArchetype.INTUITIVE
            ],
            "board": [
                PersonaArchetype.STRATEGIC, PersonaArchetype.AGGRESSIVE,
                PersonaArchetype.DEFENSIVE, PersonaArchetype.ANALYTICAL
            ]
        }
        return archetype_map.get(game_type, list(PersonaArchetype))
    
    @staticmethod
    def _create_persona(
        game_name: str,
        archetype: PersonaArchetype,
        difficulty: str,
        index: int
    ) -> PersonaProfile:
        """개별 페르소나 생성"""
        
        traits = PersonaTraits()
        traits.randomize_within_archetype(archetype)
        
        # 난이도별 조정
        if difficulty == "easy":
            traits.logic *= 0.7
            traits.planning_depth *= 0.7
        elif difficulty == "hard":
            traits.logic *= 1.3
            traits.planning_depth *= 1.3
            traits.observation *= 1.2
        
        return PersonaProfile(
            name=PersonaGenerator._generate_name(archetype, index),
            archetype=archetype,
            communication_style=random.choice(list(CommunicationStyle)),
            traits=traits,
            background_story=PersonaGenerator._generate_background(archetype),
            catchphrases=PersonaGenerator._generate_catchphrases(archetype),
            game_behaviors={},  # 게임별로 나중에 채움
            created_for_game=game_name,
            difficulty_level=difficulty
        )
    
    @staticmethod
    def _generate_name(archetype: PersonaArchetype, index: int) -> str:
        """원형에 맞는 이름 생성"""
        name_pools = {
            PersonaArchetype.AGGRESSIVE: ["블레이즈", "스톰", "렉스", "볼트"],
            PersonaArchetype.ANALYTICAL: ["로직", "캘큘러스", "데이터", "알고"],
            PersonaArchetype.COOPERATIVE: ["하모니", "유니티", "앨리", "프렌드"],
            PersonaArchetype.STRATEGIC: ["체스", "택틱", "플랜", "마스터"],
        }
        
        names = name_pools.get(archetype, ["플레이어", "게이머", "에이스", "프로"])
        base_name = names[index % len(names)]
        return f"{base_name}_{index + 1}"
    
    @staticmethod
    def _generate_background(archetype: PersonaArchetype) -> str:
        """배경 스토리 생성"""
        stories = {
            PersonaArchetype.AGGRESSIVE: "승부욕이 강한 전직 스포츠 선수. 항상 이기려고 합니다.",
            PersonaArchetype.ANALYTICAL: "수학과 출신으로 모든 것을 계산하고 분석합니다.",
            PersonaArchetype.COOPERATIVE: "팀워크를 중시하는 사회복지사. 모두가 즐거웠으면 합니다.",
        }
        return stories.get(archetype, "평범한 게임 애호가입니다.")
    
    @staticmethod
    def _generate_catchphrases(archetype: PersonaArchetype) -> List[str]:
        """자주 쓰는 말 생성"""
        phrases = {
            PersonaArchetype.AGGRESSIVE: ["승부다!", "한 번 더!", "이번엔 내가 이긴다!"],
            PersonaArchetype.ANALYTICAL: ["계산해보니...", "확률상으로는...", "데이터가 말해주네요"],
            PersonaArchetype.COOPERATIVE: ["같이 해요!", "모두 화이팅!", "윈윈하죠!"],
        }
        return phrases.get(archetype, ["좋은 게임이네요!", "재미있어요!", "열심히 하겠습니다!"])
    
    @staticmethod
    def _ensure_diversity(personas: List[PersonaProfile]) -> None:
        """페르소나 다양성 보장"""
        # 같은 원형이 너무 많으면 일부를 다른 원형으로 변경
        archetype_counts = {}
        for persona in personas:
            archetype = persona["archetype"]
            archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        
        # 50% 이상이 같은 원형이면 조정
        max_same_archetype = len(personas) // 2
        for archetype, count in archetype_counts.items():
            if count > max_same_archetype:
                # 초과분을 다른 원형으로 변경
                excess = count - max_same_archetype
                other_archetypes = [a for a in PersonaArchetype if a != archetype]
                
                changed = 0
                for persona in personas:
                    if persona["archetype"] == archetype and changed < excess:
                        new_archetype = random.choice(other_archetypes)
                        persona["archetype"] = new_archetype
                        persona["traits"].randomize_within_archetype(new_archetype)
                        changed += 1 