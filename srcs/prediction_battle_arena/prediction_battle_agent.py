"""
Prediction Battle Arena 메인 에이전트

실시간 예측 배틀, 가상 화폐 베팅, 글로벌 리더보드를 통한
강렬한 도파민을 제공하는 에이전트 시스템
"""

import asyncio
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from srcs.core.agent.base import BaseAgent
from mcp_agent.agents.agent import Agent as MCP_Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from srcs.common.llm import (
    create_fallback_llm_for_agents,
    create_fallback_orchestrator_llm_factory,
    try_fallback_orchestrator_execution
)

from .models.battle import Battle, BattleStatus, BattleType
from .models.prediction import Prediction, PredictionResult
from .models.user import User, UserStats

from .tools.prediction_tools import PredictionTools
from .tools.betting_tools import BettingTools
from .tools.reward_tools import RewardTools
from .tools.leaderboard_tools import LeaderboardTools

from .services.realtime_service import RealtimeService
from .services.redis_service import RedisService
from .services.reward_service import RewardService

from .agents.prediction_agent import create_prediction_agent
from .agents.battle_manager_agent import create_battle_manager_agent
from .agents.reward_calculator_agent import create_reward_calculator_agent
from .agents.leaderboard_agent import create_leaderboard_agent
from .agents.social_feed_agent import create_social_feed_agent


class PredictionBattleAgent(BaseAgent):
    """
    Prediction Battle Arena 메인 에이전트
    
    실시간 예측 배틀, 가상 화폐 베팅, 글로벌 리더보드 관리
    """
    
    def __init__(self):
        super().__init__(
            name="PredictionBattleAgent",
            instruction="실시간 예측 배틀을 관리하고 사용자에게 강렬한 도파민을 제공하는 에이전트",
            server_names=["g-search", "fetch", "filesystem"]
        )
        
        # 데이터 디렉토리
        self.data_dir = Path("prediction_battle_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 도구 초기화
        self.prediction_tools = PredictionTools(str(self.data_dir))
        self.betting_tools = BettingTools(str(self.data_dir))
        self.reward_tools = RewardTools(str(self.data_dir))
        self.leaderboard_tools = LeaderboardTools(str(self.data_dir))
        
        # 서비스 초기화
        self.realtime_service = RealtimeService()
        self.redis_service = RedisService(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0"))
        )
        self.reward_service = RewardService()
        
        # 배틀 저장소 (메모리, 실제로는 Redis 사용)
        self.active_battles: Dict[str, Battle] = {}
        
        self.logger.info("PredictionBattleAgent initialized")
    
    async def run_workflow(
        self,
        user_id: str,
        battle_type: str = "quick",  # quick/standard/extended
        prediction_topic: Optional[str] = None,
        action: str = "join"  # create/join
    ) -> Dict[str, Any]:
        """
        예측 배틀 워크플로우 실행
        
        Args:
            user_id: 사용자 ID
            battle_type: 배틀 유형
            prediction_topic: 예측 주제 (선택)
            action: 액션 (create/join)
        Returns:
            배틀 결과
        """
        async with self.app.run() as app_context:
            context = app_context.context
            
            try:
                # Redis 연결
                await self.redis_service.connect()
                
                # LLM 팩토리 생성
                llm_factory_for_agents = create_fallback_llm_for_agents(
                    primary_model="gemini-2.5-flash",
                    logger_instance=self.logger
                )
                
                orchestrator_llm_factory = create_fallback_orchestrator_llm_factory(
                    primary_model="gemini-2.5-flash",
                    logger_instance=self.logger
                )
                
                # 특화 에이전트 생성
                agents = self._create_specialized_agents(
                    llm_factory_for_agents
                )
                
                # Orchestrator 생성
                orchestrator = Orchestrator(
                    llm_factory=orchestrator_llm_factory,
                    available_agents=agents,
                    plan_type="full",
                    max_loops=30
                )
                
                # 배틀 생성 또는 참가
                if action == "create":
                    battle = await self._create_battle(battle_type, prediction_topic)
                else:
                    battle = await self._join_battle(user_id, battle_type)
                
                if not battle:
                    return {"error": "배틀을 생성하거나 참가할 수 없습니다."}
                
                # 워크플로우 태스크 생성
                task = self._create_battle_task(
                    user_id,
                    battle,
                    prediction_topic
                )
                
                # Orchestrator 실행
                result = await try_fallback_orchestrator_execution(
                    orchestrator=orchestrator,
                    agents=agents,
                    task=task,
                    primary_model="gemini-2.5-flash",
                    logger_instance=self.logger,
                    max_loops=30
                )
                
                # 배틀 상태 저장
                await self.redis_service.set_battle_state(
                    battle.battle_id,
                    battle.to_dict(),
                    ttl=battle.get_duration_seconds() + 3600  # 배틀 시간 + 1시간
                )
                
                # 실시간 업데이트 시작
                asyncio.create_task(
                    self._start_realtime_updates(battle.battle_id)
                )
                
                return {
                    "battle_id": battle.battle_id,
                    "battle_type": battle.battle_type.value,
                    "status": battle.status.value,
                    "participants": list(battle.participants),
                    "result": result
                }
                
            except Exception as e:
                self.logger.error(f"Battle workflow failed: {e}", exc_info=True)
                raise
            finally:
                await self.redis_service.close()
    
    def _create_specialized_agents(self, llm_factory) -> List[MCP_Agent]:
        """특화 에이전트 생성"""
        agents = [
            create_prediction_agent(llm_factory, self.prediction_tools),
            create_battle_manager_agent(llm_factory),
            create_reward_calculator_agent(
                llm_factory,
                self.reward_tools,
                self.reward_service
            ),
            create_leaderboard_agent(
                llm_factory,
                self.leaderboard_tools,
                self.redis_service
            ),
            create_social_feed_agent(llm_factory)
        ]
        
        return agents
    
    async def _create_battle(
        self,
        battle_type: str,
        topic: Optional[str] = None
    ) -> Optional[Battle]:
        """배틀 생성"""
        battle_type_enum = BattleType[battle_type.upper()] if battle_type.upper() in BattleType.__members__ else BattleType.QUICK
        
        battle = Battle(
            battle_type=battle_type_enum,
            topic=topic,
            status=BattleStatus.WAITING
        )
        
        self.active_battles[battle.battle_id] = battle
        self.logger.info(f"Created battle: {battle.battle_id}")
        
        return battle
    
    async def _join_battle(
        self,
        user_id: str,
        battle_type: str
    ) -> Optional[Battle]:
        """배틀 참가"""
        battle_type_enum = BattleType[battle_type.upper()] if battle_type.upper() in BattleType.__members__ else BattleType.QUICK
        
        # 대기 중인 배틀 찾기
        for battle in self.active_battles.values():
            if (
                battle.battle_type == battle_type_enum and
                battle.status == BattleStatus.WAITING and
                battle.can_join()
            ):
                battle.participants.add(user_id)
                self.logger.info(f"User {user_id} joined battle {battle.battle_id}")
                return battle
        
        # 없으면 새로 생성
        return await self._create_battle(battle_type)
    
    def _create_battle_task(
        self,
        user_id: str,
        battle: Battle,
        topic: Optional[str] = None
    ) -> str:
        """배틀 태스크 생성"""
        task = f"""
        Manage a prediction battle for user {user_id}.
        
        Battle ID: {battle.battle_id}
        Battle Type: {battle.battle_type.value}
        Topic: {topic or battle.topic or "Generate a relevant prediction topic"}
        
        Steps:
        1. If topic is not provided, generate an interesting prediction topic using g-search
        2. Create a prediction for user {user_id} using prediction_agent
        3. Process betting if user places a bet
        4. Wait for battle to complete
        5. Calculate results and rewards using reward_calculator_agent
        6. Update leaderboard using leaderboard_agent
        7. Generate social feed content using social_feed_agent
        
        Provide real-time updates throughout the battle process.
        """
        
        return task
    
    async def _start_realtime_updates(self, battle_id: str):
        """실시간 업데이트 시작"""
        battle = self.active_battles.get(battle_id)
        if not battle:
            return
        
        # 배틀 진행 상황 업데이트
        while battle.status != BattleStatus.FINISHED:
            await asyncio.sleep(5)  # 5초마다 업데이트
            
            remaining = battle.get_remaining_seconds()
            if remaining is not None and remaining <= 0:
                battle.status = BattleStatus.FINISHED
                battle.ended_at = datetime.now()
            
            # 실시간 업데이트 브로드캐스트
            await self.realtime_service.broadcast_battle_update(
                battle_id,
                battle.to_dict()
            )
        
        self.logger.info(f"Realtime updates completed for battle {battle_id}")

