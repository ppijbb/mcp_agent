"""
리더보드 관련 MCP 도구
"""

import logging
import json
from typing import List, Optional
from pathlib import Path
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UpdateLeaderboardInput(BaseModel):
    """리더보드 업데이트 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    score: float = Field(description="점수")
    category: Optional[str] = Field(default="global", description="카테고리")


class LeaderboardTools:
    """
    리더보드 관련 도구 모음

    리더보드 업데이트, 조회, 순위 계산 기능 제공
    """

    def __init__(self, data_dir: str = "prediction_battle_data"):
        """
        LeaderboardTools 초기화

        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard_file = self.data_dir / "leaderboard.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()

    def _load_data(self):
        """데이터 로드"""
        if self.leaderboard_file.exists():
            with open(self.leaderboard_file, 'r', encoding='utf-8') as f:
                self.leaderboard = json.load(f)
        else:
            self.leaderboard = {
                "global": {},
                "weekly": {},
                "monthly": {}
            }

    def _save_data(self):
        """데이터 저장"""
        with open(self.leaderboard_file, 'w', encoding='utf-8') as f:
            json.dump(self.leaderboard, f, indent=2, ensure_ascii=False)

    def _initialize_tools(self):
        """리더보드 도구 초기화"""
        self.tools.append(self._update_leaderboard_tool())
        self.tools.append(self._get_leaderboard_tool())
        self.tools.append(self._get_user_rank_tool())
        logger.info(f"Initialized {len(self.tools)} leaderboard tools")

    def _update_leaderboard_tool(self) -> BaseTool:
        @tool("leaderboard_update", args_schema=UpdateLeaderboardInput)
        def update_leaderboard(
            user_id: str,
            score: float,
            category: Optional[str] = "global"
        ) -> str:
            """
            리더보드를 업데이트합니다.

            Args:
                user_id: 사용자 ID
                score: 점수
                category: 카테고리 (global/weekly/monthly)
            Returns:
                업데이트 결과 (JSON 문자열)
            """
            logger.info(f"Updating leaderboard: user {user_id}, score {score}, category {category}")

            if category not in self.leaderboard:
                self.leaderboard[category] = {}

            # 기존 점수와 비교하여 더 높은 점수로 업데이트
            current_score = self.leaderboard[category].get(user_id, 0.0)
            if score > current_score:
                self.leaderboard[category][user_id] = score
                self._save_data()

                result = {
                    "user_id": user_id,
                    "old_score": current_score,
                    "new_score": score,
                    "category": category,
                    "updated": True
                }
            else:
                result = {
                    "user_id": user_id,
                    "current_score": current_score,
                    "attempted_score": score,
                    "category": category,
                    "updated": False
                }

            return json.dumps(result, ensure_ascii=False, indent=2)
        return update_leaderboard

    def _get_leaderboard_tool(self) -> BaseTool:
        @tool("leaderboard_get")
        def get_leaderboard(
            category: str = "global",
            limit: int = 100
        ) -> str:
            """
            리더보드를 조회합니다.

            Args:
                category: 카테고리 (global/weekly/monthly)
                limit: 조회할 상위 순위 수
            Returns:
                리더보드 데이터 (JSON 문자열)
            """
            logger.info(f"Getting leaderboard: category {category}, limit {limit}")

            if category not in self.leaderboard:
                return json.dumps({"error": "카테고리를 찾을 수 없습니다."}, ensure_ascii=False)

            # 점수 순으로 정렬
            rankings = sorted(
                self.leaderboard[category].items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

            result = {
                "category": category,
                "rankings": [
                    {
                        "rank": idx + 1,
                        "user_id": user_id,
                        "score": score
                    }
                    for idx, (user_id, score) in enumerate(rankings)
                ],
                "total_players": len(self.leaderboard[category])
            }

            return json.dumps(result, ensure_ascii=False, indent=2)
        return get_leaderboard

    def _get_user_rank_tool(self) -> BaseTool:
        @tool("leaderboard_get_user_rank")
        def get_user_rank(
            user_id: str,
            category: str = "global"
        ) -> str:
            """
            사용자의 순위를 조회합니다.

            Args:
                user_id: 사용자 ID
                category: 카테고리 (global/weekly/monthly)
            Returns:
                사용자 순위 정보 (JSON 문자열)
            """
            logger.info(f"Getting rank for user {user_id}, category {category}")

            if category not in self.leaderboard:
                return json.dumps({"error": "카테고리를 찾을 수 없습니다."}, ensure_ascii=False)

            user_score = self.leaderboard[category].get(user_id, 0.0)

            # 전체 순위 계산
            rankings = sorted(
                self.leaderboard[category].items(),
                key=lambda x: x[1],
                reverse=True
            )

            rank = None
            for idx, (uid, score) in enumerate(rankings):
                if uid == user_id:
                    rank = idx + 1
                    break

            result = {
                "user_id": user_id,
                "category": category,
                "rank": rank,
                "score": user_score,
                "total_players": len(self.leaderboard[category])
            }

            return json.dumps(result, ensure_ascii=False, indent=2)
        return get_user_rank

    def get_tools(self) -> List[BaseTool]:
        """모든 리더보드 도구 반환"""
        return self.tools

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 리더보드 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None
