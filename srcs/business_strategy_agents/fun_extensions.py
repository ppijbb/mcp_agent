"""
Fun Extensions for Most Hooking Business Strategy Agent

This module adds sparkle and entertainment features to the agent system.
"""

import random
import time
from datetime import datetime
from typing import Dict, List, Any, Generator
import json

class AgentPersonality:
    """에이전트 개성화 시스템"""
    
    PERSONAS = {
        "DATA_SCOUT": {
            "name": "스카우트 짱 🔍",
            "personality": "호기심 많고 정보에 목마른 탐정",
            "emojis": ["🔍", "🕵️", "📊", "💡"],
            "excited_phrases": [
                "와! 이런 흥미로운 데이터를 발견했어요!",
                "대박! 이 정보 완전 핵심이네요!",
                "오oh~ 숨겨진 보석 같은 데이터예요!"
            ],
            "working_phrases": [
                "열심히 데이터를 수집하고 있어요... 🔍",
                "품질 높은 정보만 골라내는 중이에요~",
                "시장의 숨겨진 신호들을 찾고 있어요!"
            ]
        },
        "TREND_ANALYZER": {
            "name": "트렌디 센세 📈",
            "personality": "세련되고 분석적인 패션 구루",
            "emojis": ["📈", "✨", "🌟", "💫"],
            "excited_phrases": [
                "이건 완전 핫트렌드예요! ✨",
                "감각적인 패턴을 발견했어요!",
                "이 트렌드, 완전 대세 될 것 같아요!"
            ],
            "working_phrases": [
                "트렌드의 깊은 의미를 분석하고 있어요...",
                "패턴 속에서 미래를 읽고 있어요 🔮",
                "시장의 맥박을 느끼고 있어요..."
            ]
        },
        "HOOKING_DETECTOR": {
            "name": "후킹 마스터 🎯",
            "personality": "날카롭고 직관적인 기회의 신",
            "emojis": ["🎯", "🔥", "⚡", "💥"],
            "excited_phrases": [
                "이거다!! 완전 대박 포인트예요! 🔥",
                "후킹 레이더가 미친듯이 반응해요!",
                "이건 놓치면 안 될 기회예요!"
            ],
            "working_phrases": [
                "후킹 포인트를 스캔하고 있어요... 🎯",
                "기회의 냄새를 맡고 있어요...",
                "시장의 빈틈을 찾아내는 중이에요!"
            ]
        },
        "STRATEGY_PLANNER": {
            "name": "전략 마에스트로 🎼",
            "personality": "체계적이고 창의적인 전략 천재",
            "emojis": ["🎼", "🧠", "⚖️", "🏆"],
            "excited_phrases": [
                "완벽한 전략을 설계했어요! 🏆",
                "이 계획, 정말 예술 작품이에요!",
                "ROI가 환상적일 것 같아요!"
            ],
            "working_phrases": [
                "전략적 마스터플랜을 그리고 있어요...",
                "리스크와 기회의 균형을 맞추는 중이에요 ⚖️",
                "실행 가능한 로드맵을 만들고 있어요..."
            ]
        }
    }
    
    @classmethod
    def get_reaction(cls, agent_role: str, emotion: str, context: str = "") -> str:
        """에이전트 반응 생성"""
        persona = cls.PERSONAS.get(agent_role.upper(), cls.PERSONAS["DATA_SCOUT"])
        
        if emotion == "excited":
            reaction = random.choice(persona["excited_phrases"])
        elif emotion == "working":
            reaction = random.choice(persona["working_phrases"])
        else:
            reaction = f"{persona['name']} 가 열심히 작업 중이에요!"
        
        emoji = random.choice(persona["emojis"])
        return f"{emoji} {reaction}"


class TrendBattleSystem:
    """트렌드 예측 배틀 시스템"""
    
    def __init__(self):
        self.leaderboard = []
        self.active_challenges = []
    
    def create_prediction_challenge(self) -> Dict[str, Any]:
        """예측 챌린지 생성"""
        topics = [
            "AI 스타트업", "메타버스 패션", "친환경 기술", 
            "게임 스트리밍", "디지털 헬스케어", "NFT 아트"
        ]
        
        challenge = {
            "id": f"challenge_{int(time.time())}",
            "topic": random.choice(topics),
            "description": f"다음 주 '{random.choice(topics)}' 분야의 바이럴 가능성을 예측하세요!",
            "deadline": time.time() + 604800,  # 1주일 후
            "participants": [],
            "ai_prediction": random.uniform(0.6, 0.9),  # AI의 예측
            "reward_points": random.randint(100, 500)
        }
        
        self.active_challenges.append(challenge)
        return challenge
    
    def submit_prediction(self, user_id: str, challenge_id: str, prediction: float) -> Dict[str, Any]:
        """사용자 예측 제출"""
        for challenge in self.active_challenges:
            if challenge["id"] == challenge_id:
                challenge["participants"].append({
                    "user_id": user_id,
                    "prediction": prediction,
                    "timestamp": time.time()
                })
                
                return {
                    "status": "success",
                    "message": f"예측이 제출되었습니다! (예측값: {prediction:.2f})",
                    "ai_hint": f"AI는 {challenge['ai_prediction']:.2f}로 예측했어요 🤖"
                }
        
        return {"status": "error", "message": "챌린지를 찾을 수 없습니다."}


class AchievementSystem:
    """성취 시스템"""
    
    ACHIEVEMENTS = {
        "first_analysis": {
            "name": "🔰 첫 분석 완료",
            "description": "첫 번째 비즈니스 분석을 완료했습니다!",
            "points": 100
        },
        "high_hooker": {
            "name": "🎯 후킹 마스터", 
            "description": "후킹 점수 0.8 이상을 달성했습니다!",
            "points": 250
        },
        "trend_explorer": {
            "name": "📊 트렌드 탐험가",
            "description": "10회 이상 분석을 완료했습니다!",
            "points": 300
        },
        "global_citizen": {
            "name": "🌍 글로벌 시민",
            "description": "모든 지역에서 분석을 완료했습니다!",
            "points": 500
        },
        "strategy_genius": {
            "name": "🧠 전략 천재",
            "description": "5개 이상의 전략을 생성했습니다!",
            "points": 400
        },
        "speed_demon": {
            "name": "⚡ 스피드 데몬",
            "description": "10초 이내에 분석을 완료했습니다!",
            "points": 200
        }
    }
    
    def __init__(self):
        self.user_achievements = {}
        self.user_stats = {}
    
    def check_achievements(self, user_id: str, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """새로운 성취 확인"""
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                "total_analyses": 0,
                "max_hooking_score": 0,
                "regions_analyzed": set(),
                "strategies_generated": 0,
                "fastest_analysis": float('inf')
            }
        
        if user_id not in self.user_achievements:
            self.user_achievements[user_id] = set()
        
        stats = self.user_stats[user_id]
        new_achievements = []
        
        # 통계 업데이트
        stats["total_analyses"] += 1
        
        if analysis_result.get("enhanced_insights"):
            max_score = max([insight.hooking_score for insight in analysis_result["enhanced_insights"]], default=0)
            stats["max_hooking_score"] = max(stats["max_hooking_score"], max_score)
        
        if analysis_result.get("execution_time"):
            stats["fastest_analysis"] = min(stats["fastest_analysis"], analysis_result["execution_time"])
        
        if analysis_result.get("strategies"):
            stats["strategies_generated"] += len(analysis_result["strategies"])
        
        # 성취 확인
        achievements_to_check = [
            ("first_analysis", stats["total_analyses"] >= 1),
            ("high_hooker", stats["max_hooking_score"] >= 0.8),
            ("trend_explorer", stats["total_analyses"] >= 10),
            ("strategy_genius", stats["strategies_generated"] >= 5),
            ("speed_demon", stats["fastest_analysis"] <= 10)
        ]
        
        for achievement_id, condition in achievements_to_check:
            if condition and achievement_id not in self.user_achievements[user_id]:
                self.user_achievements[user_id].add(achievement_id)
                new_achievements.append(self.ACHIEVEMENTS[achievement_id])
        
        return new_achievements


class TrendMusicGenerator:
    """트렌드 데이터를 음악으로 변환"""
    
    MUSICAL_MAPPINGS = {
        "hooking_score": {
            0.0: "🎵 조용한 피아노",
            0.3: "🎶 부드러운 기타", 
            0.6: "🎸 신나는 록",
            0.8: "🔥 강렬한 EDM",
            0.9: "💥 폭발적인 드럼"
        },
        "sentiment": {
            "positive": "💫 밝은 멜로디",
            "neutral": "🎼 안정적인 하모니", 
            "negative": "🌧️ 우울한 블루스"
        },
        "trend_direction": {
            "rising": "📈 상승하는 아르페지오",
            "stable": "➡️ 일정한 비트",
            "falling": "📉 하강하는 음계"
        }
    }
    
    def generate_soundtrack_description(self, analysis_result: Dict[str, Any]) -> str:
        """분석 결과의 사운드트랙 설명 생성"""
        if not analysis_result.get("enhanced_insights"):
            return "🎵 조용한 배경음악이 흐르고 있어요..."
        
        max_hooking = max([insight.hooking_score for insight in analysis_result["enhanced_insights"]], default=0)
        
        # 후킹 점수에 따른 음악 스타일
        music_style = "🎵 조용한 피아노"
        for score, style in self.MUSICAL_MAPPINGS["hooking_score"].items():
            if max_hooking >= score:
                music_style = style
        
        # 추가 효과
        insights_count = len(analysis_result["enhanced_insights"])
        if insights_count > 5:
            music_style += " + 🎺 트럼펫 팡파레"
        
        strategies_count = len(analysis_result.get("strategies", []))
        if strategies_count > 3:
            music_style += " + 🎻 승리의 바이올린"
        
        return f"🎧 현재 재생 중: {music_style}"


class FunAnalyticsRenderer:
    """재미있는 분석 결과 렌더링"""
    
    @staticmethod
    def create_circus_performance(insight) -> Dict[str, str]:
        """인사이트를 서커스 퍼포먼스로 변환"""
        score = insight.hooking_score
        
        if score >= 0.9:
            return {
                "performance": "🔥 불타는 고리 점프",
                "description": "완벽한 착지! 관중들이 열광합니다!",
                "effect": "✨💥🎆"
            }
        elif score >= 0.7:
            return {
                "performance": "🎪 공중 그네",
                "description": "우아한 공중 연기가 펼쳐집니다!",
                "effect": "🌟⭐✨"
            }
        elif score >= 0.5:
            return {
                "performance": "🤹 저글링 쇼",
                "description": "능숙한 공 던지기로 관중을 매혹시킵니다!",
                "effect": "🎭🎨🎪"
            }
        else:
            return {
                "performance": "🎭 마임 공연",
                "description": "조용하지만 의미 있는 표현을 보여줍니다.",
                "effect": "🤫👻🎪"
            }
    
    @staticmethod
    def generate_trend_weather(insights: List) -> str:
        """트렌드를 날씨로 표현"""
        if not insights:
            return "☁️ 흐린 날씨 - 트렌드가 명확하지 않아요"
        
        avg_score = sum(insight.hooking_score for insight in insights) / len(insights)
        
        if avg_score >= 0.8:
            return "☀️ 맑고 화창한 날씨 - 최고의 비즈니스 기회들이 가득해요!"
        elif avg_score >= 0.6:
            return "⛅ 구름 조금 - 좋은 기회들이 보이기 시작해요!"
        elif avg_score >= 0.4:
            return "🌥️ 흐린 날씨 - 기회가 있지만 주의 깊은 분석이 필요해요"
        else:
            return "🌧️ 비오는 날씨 - 아직은 기다리는 것이 좋을 것 같아요"


class InteractiveStoryTeller:
    """인터랙티브 스토리텔링"""
    
    STORY_TEMPLATES = {
        "hero_journey": [
            "🏰 옛날 옛적, {region}의 시장에서...",
            "🗡️ 용감한 기업가가 {keyword} 분야의 모험을 떠났습니다!",
            "🐉 하지만 {risk_factor}라는 무서운 용이 길을 막고 있었어요...",
            "✨ 다행히 {opportunity}라는 마법의 검을 발견했습니다!",
            "🏆 결국 {strategy}로 용을 물리치고 성공을 거두었답니다!"
        ],
        "detective": [
            "🔍 {region}에서 수상한 트렌드 사건이 발생했습니다...",
            "🕵️ 명탐정이 {keyword} 단서를 따라 수사를 시작했어요!",
            "🔎 증거를 분석한 결과, {insight}라는 중요한 발견이!",
            "💡 범인은 바로... {hooking_point}였습니다!",
            "⚖️ 사건 해결! {strategy}로 정의가 승리했어요!"
        ]
    }
    
    def create_analysis_story(self, analysis_result: Dict[str, Any], story_type: str = "hero_journey") -> List[str]:
        """분석 결과를 스토리로 변환"""
        if not analysis_result.get("enhanced_insights"):
            return ["📖 아직 스토리가 시작되지 않았어요..."]
        
        template = self.STORY_TEMPLATES.get(story_type, self.STORY_TEMPLATES["hero_journey"])
        insights = analysis_result["enhanced_insights"]
        top_insight = max(insights, key=lambda x: x.hooking_score)
        
        story_vars = {
            "region": top_insight.region.value,
            "keyword": ", ".join(top_insight.key_topics[:2]),
            "risk_factor": "경쟁이 치열한 시장",
            "opportunity": f"후킹 점수 {top_insight.hooking_score:.2f}의 기회",
            "strategy": "혁신적인 비즈니스 전략",
            "insight": f"{top_insight.trend_direction} 트렌드",
            "hooking_point": "숨겨진 시장 니즈"
        }
        
        story = []
        for line in template:
            try:
                formatted_line = line.format(**story_vars)
                story.append(formatted_line)
            except KeyError:
                story.append(line)
        
        return story


# 전역 인스턴스들
personality_system = AgentPersonality()
battle_system = TrendBattleSystem()
achievement_system = AchievementSystem()
music_generator = TrendMusicGenerator()
analytics_renderer = FunAnalyticsRenderer()
story_teller = InteractiveStoryTeller()


def get_fun_extensions():
    """재미있는 확장 기능들 반환"""
    return {
        "personality": personality_system,
        "battle": battle_system,
        "achievements": achievement_system,
        "music": music_generator,
        "analytics": analytics_renderer,
        "story": story_teller
    } 