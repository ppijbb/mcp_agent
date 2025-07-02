#!/usr/bin/env python3
"""
SQLite MCP Server - 데이터베이스 관리

게임 세션, 플레이어 통계, 게임 분석 등
SQLite 기반 영구 데이터 저장을 위한 MCP 서버
"""

import asyncio
import json
import sqlite3
import aiosqlite
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# MCP 관련 임포트
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️ MCP 패키지가 설치되지 않음. 시뮬레이션 모드로 실행")
    MCP_AVAILABLE = False


class SQLiteMCPServer:
    """SQLite 데이터베이스 관리를 위한 MCP 서버"""
    
    def __init__(self, db_path: str = "./game_data/table_game_mate.db"):
        self.db_path = Path(db_path)
        self.server = Server("sqlite-server") if MCP_AVAILABLE else None
        
        # 데이터베이스 디렉토리 생성
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 연결 풀 설정
        self.connection_pool_size = 10
        
        if MCP_AVAILABLE and self.server:
            self._register_tools()
    
    async def initialize_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        
        async with aiosqlite.connect(self.db_path) as db:
            # 게임 세션 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS game_sessions (
                    session_id TEXT PRIMARY KEY,
                    game_name TEXT NOT NULL,
                    player_count INTEGER NOT NULL,
                    game_config TEXT,  -- JSON
                    session_data TEXT, -- JSON
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'active',  -- active, completed, error
                    winner_ids TEXT,  -- JSON array
                    final_scores TEXT,  -- JSON
                    turn_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 플레이어 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    player_name TEXT NOT NULL,
                    player_type TEXT DEFAULT 'ai',  -- ai, human
                    persona_type TEXT,
                    final_score INTEGER DEFAULT 0,
                    turn_order INTEGER DEFAULT 0,
                    is_winner BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES game_sessions(session_id)
                )
            """)
            
            # 게임 액션 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS game_actions (
                    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    action_data TEXT,  -- JSON
                    is_valid BOOLEAN DEFAULT TRUE,
                    validation_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES game_sessions(session_id),
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                )
            """)
            
            # 게임 통계 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS game_statistics (
                    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_name TEXT NOT NULL,
                    total_sessions INTEGER DEFAULT 0,
                    total_players INTEGER DEFAULT 0,
                    avg_duration_minutes REAL DEFAULT 0,
                    avg_turns REAL DEFAULT 0,
                    most_common_winner_persona TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 플레이어 성과 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS player_performance (
                    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_type TEXT NOT NULL,
                    game_name TEXT NOT NULL,
                    total_games INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0,
                    avg_turn_time REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(persona_type, game_name)
                )
            """)
            
            # 인덱스 생성
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_game_name ON game_sessions(game_name)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON game_sessions(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_players_session ON players(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_actions_session ON game_actions(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_actions_player ON game_actions(player_id)")
            
            await db.commit()
        
        print("✅ SQLite 데이터베이스 초기화 완료")
    
    def _register_tools(self):
        """MCP 도구들 등록"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="save_game_session",
                    description="게임 세션 저장",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"},
                            "game_name": {"type": "string", "description": "게임 이름"},
                            "player_count": {"type": "number", "description": "플레이어 수"},
                            "game_config": {"type": "object", "description": "게임 설정"},
                            "session_data": {"type": "object", "description": "세션 데이터"}
                        },
                        "required": ["session_id", "game_name", "player_count", "session_data"]
                    }
                ),
                Tool(
                    name="update_game_session",
                    description="게임 세션 업데이트",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"},
                            "status": {"type": "string", "description": "게임 상태"},
                            "winner_ids": {"type": "array", "description": "승자 ID 목록"},
                            "final_scores": {"type": "object", "description": "최종 점수"},
                            "turn_count": {"type": "number", "description": "총 턴 수"}
                        },
                        "required": ["session_id"]
                    }
                ),
                Tool(
                    name="save_players",
                    description="플레이어 정보 저장",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"},
                            "players": {"type": "array", "description": "플레이어 목록"}
                        },
                        "required": ["session_id", "players"]
                    }
                ),
                Tool(
                    name="save_game_action",
                    description="게임 액션 저장",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"},
                            "player_id": {"type": "string", "description": "플레이어 ID"},
                            "turn_number": {"type": "number", "description": "턴 번호"},
                            "action_type": {"type": "string", "description": "액션 타입"},
                            "action_data": {"type": "object", "description": "액션 데이터"},
                            "is_valid": {"type": "boolean", "description": "유효성"}
                        },
                        "required": ["session_id", "player_id", "turn_number", "action_type"]
                    }
                ),
                Tool(
                    name="get_game_sessions",
                    description="게임 세션 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "game_name": {"type": "string", "description": "게임 이름 필터"},
                            "status": {"type": "string", "description": "상태 필터"},
                            "limit": {"type": "number", "description": "조회 제한 (기본값: 20)"},
                            "offset": {"type": "number", "description": "오프셋 (기본값: 0)"}
                        }
                    }
                ),
                Tool(
                    name="get_session_details",
                    description="세션 상세 정보 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"}
                        },
                        "required": ["session_id"]
                    }
                ),
                Tool(
                    name="get_game_statistics",
                    description="게임 통계 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "game_name": {"type": "string", "description": "게임 이름"},
                            "days": {"type": "number", "description": "조회 기간 (일, 기본값: 30)"}
                        }
                    }
                ),
                Tool(
                    name="get_player_performance",
                    description="플레이어 성과 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "persona_type": {"type": "string", "description": "페르소나 타입"},
                            "game_name": {"type": "string", "description": "게임 이름"}
                        }
                    }
                ),
                Tool(
                    name="update_statistics",
                    description="통계 업데이트",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "game_name": {"type": "string", "description": "게임 이름"}
                        }
                    }
                ),
                Tool(
                    name="execute_query",
                    description="직접 SQL 쿼리 실행 (읽기 전용)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL 쿼리"},
                            "params": {"type": "array", "description": "쿼리 매개변수"}
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """도구 호출 처리"""
            
            try:
                if name == "save_game_session":
                    result = await self.save_game_session(**arguments)
                elif name == "update_game_session":
                    result = await self.update_game_session(**arguments)
                elif name == "save_players":
                    result = await self.save_players(**arguments)
                elif name == "save_game_action":
                    result = await self.save_game_action(**arguments)
                elif name == "get_game_sessions":
                    result = await self.get_game_sessions(**arguments)
                elif name == "get_session_details":
                    result = await self.get_session_details(**arguments)
                elif name == "get_game_statistics":
                    result = await self.get_game_statistics(**arguments)
                elif name == "get_player_performance":
                    result = await self.get_player_performance(**arguments)
                elif name == "update_statistics":
                    result = await self.update_statistics(**arguments)
                elif name == "execute_query":
                    result = await self.execute_query(**arguments)
                else:
                    result = {"error": f"알 수 없는 도구: {name}"}
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
                
            except Exception as e:
                error_result = {
                    "error": f"도구 실행 실패: {str(e)}",
                    "tool": name,
                    "arguments": arguments
                }
                return [TextContent(
                    type="text",
                    text=json.dumps(error_result, ensure_ascii=False, indent=2)
                )]
    
    async def save_game_session(
        self,
        session_id: str,
        game_name: str,
        player_count: int,
        session_data: Dict[str, Any],
        game_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """게임 세션 저장"""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO game_sessions 
                    (session_id, game_name, player_count, game_config, session_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    session_id,
                    game_name,
                    player_count,
                    json.dumps(game_config) if game_config else None,
                    json.dumps(session_data)
                ))
                
                await db.commit()
            
            return {
                "success": True,
                "session_id": session_id,
                "message": "게임 세션 저장 완료"
            }
            
        except Exception as e:
            return {"error": f"세션 저장 실패: {str(e)}"}
    
    async def update_game_session(
        self,
        session_id: str,
        status: Optional[str] = None,
        winner_ids: Optional[List[str]] = None,
        final_scores: Optional[Dict[str, Any]] = None,
        turn_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """게임 세션 업데이트"""
        
        try:
            updates = []
            params = []
            
            if status:
                updates.append("status = ?")
                params.append(status)
                
                if status == "completed":
                    updates.append("end_time = CURRENT_TIMESTAMP")
            
            if winner_ids is not None:
                updates.append("winner_ids = ?")
                params.append(json.dumps(winner_ids))
            
            if final_scores is not None:
                updates.append("final_scores = ?")
                params.append(json.dumps(final_scores))
            
            if turn_count is not None:
                updates.append("turn_count = ?")
                params.append(turn_count)
            
            if not updates:
                return {"error": "업데이트할 데이터가 없음"}
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(session_id)
            
            async with aiosqlite.connect(self.db_path) as db:
                query = f"UPDATE game_sessions SET {', '.join(updates)} WHERE session_id = ?"
                await db.execute(query, params)
                await db.commit()
            
            return {
                "success": True,
                "session_id": session_id,
                "message": "게임 세션 업데이트 완료"
            }
            
        except Exception as e:
            return {"error": f"세션 업데이트 실패: {str(e)}"}
    
    async def save_players(self, session_id: str, players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """플레이어 정보 저장"""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 기존 플레이어 삭제
                await db.execute("DELETE FROM players WHERE session_id = ?", (session_id,))
                
                # 새 플레이어 저장
                for player in players:
                    await db.execute("""
                        INSERT INTO players 
                        (player_id, session_id, player_name, player_type, persona_type, 
                         final_score, turn_order, is_winner)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player.get("id", ""),
                        session_id,
                        player.get("name", ""),
                        player.get("type", "ai"),
                        player.get("persona_type"),
                        player.get("score", 0),
                        player.get("turn_order", 0),
                        player.get("is_winner", False)
                    ))
                
                await db.commit()
            
            return {
                "success": True,
                "session_id": session_id,
                "player_count": len(players),
                "message": "플레이어 정보 저장 완료"
            }
            
        except Exception as e:
            return {"error": f"플레이어 저장 실패: {str(e)}"}
    
    async def save_game_action(
        self,
        session_id: str,
        player_id: str,
        turn_number: int,
        action_type: str,
        action_data: Optional[Dict[str, Any]] = None,
        is_valid: bool = True
    ) -> Dict[str, Any]:
        """게임 액션 저장"""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO game_actions 
                    (session_id, player_id, turn_number, action_type, action_data, is_valid)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    player_id,
                    turn_number,
                    action_type,
                    json.dumps(action_data) if action_data else None,
                    is_valid
                ))
                
                await db.commit()
            
            return {
                "success": True,
                "message": "게임 액션 저장 완료"
            }
            
        except Exception as e:
            return {"error": f"액션 저장 실패: {str(e)}"}
    
    async def get_game_sessions(
        self,
        game_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """게임 세션 조회"""
        
        try:
            conditions = []
            params = []
            
            if game_name:
                conditions.append("game_name = ?")
                params.append(game_name)
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            params.extend([limit, offset])
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute(f"""
                    SELECT session_id, game_name, player_count, status, 
                           start_time, end_time, turn_count, winner_ids, final_scores
                    FROM game_sessions 
                    {where_clause}
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, params)
                
                sessions = []
                async for row in cursor:
                    session_dict = dict(row)
                    
                    # JSON 파싱
                    if session_dict["winner_ids"]:
                        session_dict["winner_ids"] = json.loads(session_dict["winner_ids"])
                    if session_dict["final_scores"]:
                        session_dict["final_scores"] = json.loads(session_dict["final_scores"])
                    
                    sessions.append(session_dict)
            
            return {
                "success": True,
                "sessions": sessions,
                "count": len(sessions),
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            return {"error": f"세션 조회 실패: {str(e)}"}
    
    async def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """세션 상세 정보 조회"""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # 세션 정보
                cursor = await db.execute("""
                    SELECT * FROM game_sessions WHERE session_id = ?
                """, (session_id,))
                
                session_row = await cursor.fetchone()
                if not session_row:
                    return {"error": f"세션을 찾을 수 없음: {session_id}"}
                
                session = dict(session_row)
                
                # JSON 파싱
                if session["game_config"]:
                    session["game_config"] = json.loads(session["game_config"])
                if session["session_data"]:
                    session["session_data"] = json.loads(session["session_data"])
                if session["winner_ids"]:
                    session["winner_ids"] = json.loads(session["winner_ids"])
                if session["final_scores"]:
                    session["final_scores"] = json.loads(session["final_scores"])
                
                # 플레이어 정보
                cursor = await db.execute("""
                    SELECT * FROM players WHERE session_id = ? ORDER BY turn_order
                """, (session_id,))
                
                players = []
                async for row in cursor:
                    players.append(dict(row))
                
                # 게임 액션 (최근 20개)
                cursor = await db.execute("""
                    SELECT * FROM game_actions 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 20
                """, (session_id,))
                
                actions = []
                async for row in cursor:
                    action = dict(row)
                    if action["action_data"]:
                        action["action_data"] = json.loads(action["action_data"])
                    actions.append(action)
            
            return {
                "success": True,
                "session": session,
                "players": players,
                "recent_actions": actions
            }
            
        except Exception as e:
            return {"error": f"세션 상세 조회 실패: {str(e)}"}
    
    async def get_game_statistics(
        self,
        game_name: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """게임 통계 조회"""
        
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                conditions = ["created_at >= ?"]
                params = [since_date.isoformat()]
                
                if game_name:
                    conditions.append("game_name = ?")
                    params.append(game_name)
                
                where_clause = "WHERE " + " AND ".join(conditions)
                
                # 기본 통계
                cursor = await db.execute(f"""
                    SELECT 
                        game_name,
                        COUNT(*) as total_sessions,
                        SUM(player_count) as total_players,
                        AVG(turn_count) as avg_turns,
                        AVG(CASE 
                            WHEN end_time IS NOT NULL 
                            THEN (julianday(end_time) - julianday(start_time)) * 24 * 60 
                            ELSE NULL 
                        END) as avg_duration_minutes
                    FROM game_sessions 
                    {where_clause}
                    GROUP BY game_name
                    ORDER BY total_sessions DESC
                """, params)
                
                stats = []
                async for row in cursor:
                    stats.append(dict(row))
                
                # 페르소나별 승률
                cursor = await db.execute(f"""
                    SELECT 
                        p.persona_type,
                        s.game_name,
                        COUNT(*) as total_games,
                        SUM(CASE WHEN p.is_winner THEN 1 ELSE 0 END) as wins,
                        ROUND(AVG(p.final_score), 2) as avg_score
                    FROM players p
                    JOIN game_sessions s ON p.session_id = s.session_id
                    {where_clause.replace('created_at', 's.created_at')}
                    GROUP BY p.persona_type, s.game_name
                    HAVING total_games >= 3
                    ORDER BY s.game_name, wins DESC
                """, params)
                
                persona_stats = []
                async for row in cursor:
                    row_dict = dict(row)
                    row_dict["win_rate"] = round(row_dict["wins"] / row_dict["total_games"], 3)
                    persona_stats.append(row_dict)
            
            return {
                "success": True,
                "period_days": days,
                "game_statistics": stats,
                "persona_performance": persona_stats,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"통계 조회 실패: {str(e)}"}
    
    async def get_player_performance(
        self,
        persona_type: Optional[str] = None,
        game_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """플레이어 성과 조회"""
        
        try:
            conditions = []
            params = []
            
            if persona_type:
                conditions.append("persona_type = ?")
                params.append(persona_type)
            
            if game_name:
                conditions.append("game_name = ?")
                params.append(game_name)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute(f"""
                    SELECT * FROM player_performance 
                    {where_clause}
                    ORDER BY win_rate DESC, total_games DESC
                """, params)
                
                performance = []
                async for row in cursor:
                    performance.append(dict(row))
            
            return {
                "success": True,
                "performance": performance,
                "count": len(performance)
            }
            
        except Exception as e:
            return {"error": f"성과 조회 실패: {str(e)}"}
    
    async def update_statistics(self, game_name: Optional[str] = None) -> Dict[str, Any]:
        """통계 업데이트"""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 게임별 통계 업데이트
                games_query = "SELECT DISTINCT game_name FROM game_sessions"
                params = []
                
                if game_name:
                    games_query += " WHERE game_name = ?"
                    params.append(game_name)
                
                cursor = await db.execute(games_query, params)
                games = [row[0] async for row in cursor]
                
                updated_games = []
                
                for game in games:
                    # 게임 통계 계산
                    cursor = await db.execute("""
                        SELECT 
                            COUNT(*) as total_sessions,
                            SUM(player_count) as total_players,
                            AVG(turn_count) as avg_turns,
                            AVG(CASE 
                                WHEN end_time IS NOT NULL 
                                THEN (julianday(end_time) - julianday(start_time)) * 24 * 60 
                                ELSE NULL 
                            END) as avg_duration
                        FROM game_sessions 
                        WHERE game_name = ? AND status = 'completed'
                    """, (game,))
                    
                    stats = await cursor.fetchone()
                    
                    # 가장 성공적인 페르소나 찾기
                    cursor = await db.execute("""
                        SELECT p.persona_type, COUNT(*) as wins
                        FROM players p
                        JOIN game_sessions s ON p.session_id = s.session_id
                        WHERE s.game_name = ? AND p.is_winner = 1
                        GROUP BY p.persona_type
                        ORDER BY wins DESC
                        LIMIT 1
                    """, (game,))
                    
                    top_persona_row = await cursor.fetchone()
                    top_persona = top_persona_row[0] if top_persona_row else None
                    
                    # 통계 업데이트
                    await db.execute("""
                        INSERT OR REPLACE INTO game_statistics 
                        (game_name, total_sessions, total_players, avg_duration_minutes, 
                         avg_turns, most_common_winner_persona, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        game,
                        stats[0] or 0,
                        stats[1] or 0,
                        stats[3] or 0,
                        stats[2] or 0,
                        top_persona
                    ))
                    
                    updated_games.append(game)
                
                # 플레이어 성과 통계 업데이트
                await db.execute("""
                    INSERT OR REPLACE INTO player_performance 
                    (persona_type, game_name, total_games, wins, avg_score, win_rate, last_updated)
                    SELECT 
                        p.persona_type,
                        s.game_name,
                        COUNT(*) as total_games,
                        SUM(CASE WHEN p.is_winner THEN 1 ELSE 0 END) as wins,
                        AVG(p.final_score) as avg_score,
                        CAST(SUM(CASE WHEN p.is_winner THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as win_rate,
                        CURRENT_TIMESTAMP
                    FROM players p
                    JOIN game_sessions s ON p.session_id = s.session_id
                    WHERE s.status = 'completed'
                    GROUP BY p.persona_type, s.game_name
                """)
                
                await db.commit()
            
            return {
                "success": True,
                "updated_games": updated_games,
                "message": "통계 업데이트 완료"
            }
            
        except Exception as e:
            return {"error": f"통계 업데이트 실패: {str(e)}"}
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> Dict[str, Any]:
        """직접 SQL 쿼리 실행 (읽기 전용)"""
        
        # 보안: 읽기 전용 쿼리만 허용
        query_upper = query.upper().strip()
        if not query_upper.startswith("SELECT"):
            return {"error": "읽기 전용 쿼리(SELECT)만 허용됩니다"}
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute(query, params or [])
                
                results = []
                async for row in cursor:
                    results.append(dict(row))
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "query": query
            }
            
        except Exception as e:
            return {"error": f"쿼리 실행 실패: {str(e)}"}


# MCP 서버 실행
async def main():
    """SQLite MCP 서버 실행"""
    
    if not MCP_AVAILABLE:
        print("❌ MCP 패키지가 필요합니다: pip install mcp")
        return
    
    sqlite_server = SQLiteMCPServer()
    await sqlite_server.initialize_database()
    
    async with stdio_server() as (read_stream, write_stream):
        await sqlite_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sqlite-server",
                server_version="1.0.0",
                capabilities={}
            )
        )


if __name__ == "__main__":
    asyncio.run(main())