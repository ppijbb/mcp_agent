"""
Table Game Mate - 메인 진입점

LangGraph 패턴을 따르는 멀티 에이전트 보드게임 플랫폼
"""

import asyncio
import sys
from typing import Dict, List, Any
from datetime import datetime

from agents import GameAgent, AnalysisAgent, MonitoringAgent
from core import GameConfig, Player, SystemState, ErrorHandler, ErrorSeverity, ErrorCategory


class TableGameMate:
    """Table Game Mate 메인 시스템"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.system_state = SystemState()
        
        # 에이전트 초기화
        self.game_agent = GameAgent()
        self.analysis_agent = AnalysisAgent()
        self.monitoring_agent = MonitoringAgent()
        
        # 에이전트 등록
        self.agents = {
            "game_agent": self.game_agent,
            "analysis_agent": self.analysis_agent,
            "monitoring_agent": self.monitoring_agent
        }
        
        print("🎮 Table Game Mate 시스템이 초기화되었습니다")
    
    async def start_system(self) -> bool:
        """시스템 시작"""
        try:
            print("🚀 시스템 시작 중...")
            
            # 시스템 상태 업데이트
            self.system_state.status = "running"
            self.system_state.updated_at = datetime.now()
            
            # 모니터링 에이전트 시작
            monitoring_result = await self.monitoring_agent.monitor_system()
            if not monitoring_result["success"]:
                raise Exception(f"모니터링 에이전트 시작 실패: {monitoring_result['error']}")
            
            print("✅ 시스템이 성공적으로 시작되었습니다")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(e, "main_system", ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM_ERROR)
            print(f"❌ 시스템 시작 실패: {str(e)}")
            return False
    
    async def play_game(self, game_name: str, player_names: List[str]) -> Dict[str, Any]:
        """게임 실행"""
        try:
            print(f"🎯 게임 '{game_name}' 시작 - 플레이어: {', '.join(player_names)}")
            
            # 게임 설정 생성
            game_config = GameConfig(
                name=game_name,
                type="chess",  # 기본값
                min_players=2,
                max_players=4,
                estimated_duration=60
            )
            
            # 플레이어 생성
            players = []
            for i, name in enumerate(player_names):
                player = Player(
                    id=f"player_{i+1}",
                    name=name,
                    type="human" if i == 0 else "ai"
                )
                players.append(player)
            
            # 게임 실행
            game_result = await self.game_agent.play_game(game_config.model_dump(), [p.model_dump() for p in players])
            
            if game_result["success"]:
                print("🎉 게임이 성공적으로 완료되었습니다")
                
                # 게임 분석
                analysis_result = await self.analysis_agent.analyze_game(game_result["final_state"])
                if analysis_result["success"]:
                    print("📊 게임 분석이 완료되었습니다")
                else:
                    print(f"⚠️ 게임 분석 실패: {analysis_result['error']}")
                
                return {
                    "success": True,
                    "game_result": game_result,
                    "analysis_result": analysis_result
                }
            else:
                raise Exception(f"게임 실행 실패: {game_result['error']}")
                
        except Exception as e:
            await self.error_handler.handle_error(e, "main_system", ErrorSeverity.HIGH, ErrorCategory.GAME_ERROR)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            # 에러 요약
            error_summary = self.error_handler.get_error_summary()
            
            # 시스템 메트릭
            monitoring_result = await self.monitoring_agent.monitor_system()
            
            return {
                "system_status": self.system_state.status,
                "active_games": len(self.system_state.active_games),
                "error_summary": error_summary,
                "monitoring_status": monitoring_result.get("success", False),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            await self.error_handler.handle_error(e, "main_system", ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM_ERROR)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def shutdown_system(self) -> bool:
        """시스템 종료"""
        try:
            print("🛑 시스템 종료 중...")
            
            # 시스템 상태 업데이트
            self.system_state.status = "maintenance"
            self.system_state.updated_at = datetime.now()
            
            # 활성 게임 정리
            for game_id in self.system_state.active_games:
                print(f"게임 {game_id} 정리 중...")
            
            self.system_state.active_games.clear()
            
            print("✅ 시스템이 안전하게 종료되었습니다")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(e, "main_system", ErrorSeverity.HIGH, ErrorCategory.SYSTEM_ERROR)
            print(f"❌ 시스템 종료 중 오류: {str(e)}")
            return False


async def main():
    """메인 함수"""
    try:
        # Table Game Mate 시스템 생성
        system = TableGameMate()
        
        # 시스템 시작
        if not await system.start_system():
            print("시스템 시작에 실패했습니다")
            return
        
        # 시스템 상태 확인
        status = await system.get_system_status()
        print(f"시스템 상태: {status}")
        
        # 데모 게임 실행
        print("\n🎮 데모 게임을 시작합니다...")
        game_result = await system.play_game("체스", ["Alice", "Bob"])
        
        if game_result["success"]:
            print("✅ 데모 게임이 성공적으로 완료되었습니다")
        else:
            print(f"❌ 데모 게임 실패: {game_result['error']}")
        
        # 시스템 종료
        await system.shutdown_system()
        
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 프로그램이 종료되었습니다")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())
