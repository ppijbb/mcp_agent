#!/usr/bin/env python3
"""
Memory MCP Server - AI 메모리 관리

Agent 메모리, 게임 컨텍스트, 학습 데이터 등
AI 시스템의 기억과 학습을 위한 MCP 서버
"""

import asyncio
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid

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


class MemoryMCPServer:
    """AI 메모리 관리를 위한 MCP 서버"""
    
    def __init__(self, max_memory_items: int = 10000):
        self.server = Server("memory-server") if MCP_AVAILABLE else None
        
        # 메모리 저장소들
        self.agent_memories: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.game_contexts: Dict[str, Dict[str, Any]] = {}
        self.learning_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pattern_cache: Dict[str, Any] = {}
        self.session_memories: Dict[str, Dict[str, Any]] = {}
        
        # 설정
        self.max_memory_items = max_memory_items
        self.max_context_age = timedelta(hours=24)
        self.max_learning_items = 1000
        
        # 메모리 통계
        self.memory_stats = {
            "total_memories": 0,
            "total_contexts": 0,
            "total_patterns": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        if MCP_AVAILABLE and self.server:
            self._register_tools()
    
    def _register_tools(self):
        """MCP 도구들 등록"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="store_agent_memory",
                    description="에이전트 메모리 저장",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "에이전트 ID"},
                            "memory_type": {"type": "string", "description": "메모리 타입"},
                            "memory_data": {"type": "object", "description": "메모리 데이터"},
                            "importance": {"type": "number", "description": "중요도 (0-1, 기본값: 0.5)"},
                            "tags": {"type": "array", "description": "태그 목록"}
                        },
                        "required": ["agent_id", "memory_type", "memory_data"]
                    }
                ),
                Tool(
                    name="retrieve_agent_memory",
                    description="에이전트 메모리 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "에이전트 ID"},
                            "memory_type": {"type": "string", "description": "메모리 타입 필터"},
                            "tags": {"type": "array", "description": "태그 필터"},
                            "limit": {"type": "number", "description": "조회 제한 (기본값: 10)"},
                            "min_importance": {"type": "number", "description": "최소 중요도 (기본값: 0)"}
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="store_game_context",
                    description="게임 컨텍스트 저장",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"},
                            "context_type": {"type": "string", "description": "컨텍스트 타입"},
                            "context_data": {"type": "object", "description": "컨텍스트 데이터"},
                            "game_phase": {"type": "string", "description": "게임 단계"},
                            "turn_number": {"type": "number", "description": "턴 번호"}
                        },
                        "required": ["session_id", "context_type", "context_data"]
                    }
                ),
                Tool(
                    name="retrieve_game_context",
                    description="게임 컨텍스트 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"},
                            "context_type": {"type": "string", "description": "컨텍스트 타입 필터"},
                            "game_phase": {"type": "string", "description": "게임 단계 필터"},
                            "limit": {"type": "number", "description": "조회 제한 (기본값: 20)"}
                        },
                        "required": ["session_id"]
                    }
                ),
                Tool(
                    name="store_learning_data",
                    description="학습 데이터 저장",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "에이전트 ID"},
                            "learning_type": {"type": "string", "description": "학습 타입"},
                            "input_data": {"type": "object", "description": "입력 데이터"},
                            "output_data": {"type": "object", "description": "출력 데이터"},
                            "performance_score": {"type": "number", "description": "성능 점수"},
                            "feedback": {"type": "object", "description": "피드백 데이터"}
                        },
                        "required": ["agent_id", "learning_type", "input_data", "output_data"]
                    }
                ),
                Tool(
                    name="analyze_patterns",
                    description="패턴 분석",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "에이전트 ID"},
                            "pattern_type": {"type": "string", "description": "패턴 타입"},
                            "data_source": {"type": "string", "description": "데이터 소스 (memory, learning, context)"},
                            "min_occurrences": {"type": "number", "description": "최소 발생 횟수 (기본값: 3)"}
                        },
                        "required": ["agent_id", "pattern_type"]
                    }
                ),
                Tool(
                    name="get_memory_summary",
                    description="메모리 요약 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "에이전트 ID"},
                            "summary_type": {"type": "string", "description": "요약 타입 (recent, important, patterns)"}
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="clear_memory",
                    description="메모리 정리",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "에이전트 ID"},
                            "memory_type": {"type": "string", "description": "정리할 메모리 타입"},
                            "older_than_hours": {"type": "number", "description": "N시간 이전 데이터 정리"},
                            "keep_important": {"type": "boolean", "description": "중요 데이터 보존 (기본값: true)"}
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="export_memory",
                    description="메모리 내보내기",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "에이전트 ID"},
                            "export_format": {"type": "string", "description": "내보내기 형식 (json, pickle)"},
                            "include_learning": {"type": "boolean", "description": "학습 데이터 포함 (기본값: true)"}
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="get_memory_stats",
                    description="메모리 통계 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "detailed": {"type": "boolean", "description": "상세 통계 포함 (기본값: false)"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """도구 호출 처리"""
            
            try:
                if name == "store_agent_memory":
                    result = await self.store_agent_memory(**arguments)
                elif name == "retrieve_agent_memory":
                    result = await self.retrieve_agent_memory(**arguments)
                elif name == "store_game_context":
                    result = await self.store_game_context(**arguments)
                elif name == "retrieve_game_context":
                    result = await self.retrieve_game_context(**arguments)
                elif name == "store_learning_data":
                    result = await self.store_learning_data(**arguments)
                elif name == "analyze_patterns":
                    result = await self.analyze_patterns(**arguments)
                elif name == "get_memory_summary":
                    result = await self.get_memory_summary(**arguments)
                elif name == "clear_memory":
                    result = await self.clear_memory(**arguments)
                elif name == "export_memory":
                    result = await self.export_memory(**arguments)
                elif name == "get_memory_stats":
                    result = await self.get_memory_stats(**arguments)
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
    
    def _generate_memory_id(self) -> str:
        """메모리 ID 생성"""
        return str(uuid.uuid4())
    
    def _create_memory_item(
        self,
        memory_type: str,
        memory_data: Dict[str, Any],
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """메모리 아이템 생성"""
        
        return {
            "memory_id": self._generate_memory_id(),
            "memory_type": memory_type,
            "memory_data": memory_data,
            "importance": max(0.0, min(1.0, importance)),
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": datetime.now().isoformat()
        }
    
    async def store_agent_memory(
        self,
        agent_id: str,
        memory_type: str,
        memory_data: Dict[str, Any],
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """에이전트 메모리 저장"""
        
        try:
            memory_item = self._create_memory_item(memory_type, memory_data, importance, tags)
            
            # 에이전트별 메모리 저장
            if agent_id not in self.agent_memories:
                self.agent_memories[agent_id] = {}
            
            if memory_type not in self.agent_memories[agent_id]:
                self.agent_memories[agent_id][memory_type] = deque(maxlen=self.max_memory_items)
            
            self.agent_memories[agent_id][memory_type].append(memory_item)
            
            # 통계 업데이트
            self.memory_stats["total_memories"] += 1
            
            return {
                "success": True,
                "memory_id": memory_item["memory_id"],
                "agent_id": agent_id,
                "memory_type": memory_type,
                "importance": importance
            }
            
        except Exception as e:
            return {"error": f"메모리 저장 실패: {str(e)}"}
    
    async def retrieve_agent_memory(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> Dict[str, Any]:
        """에이전트 메모리 조회"""
        
        try:
            if agent_id not in self.agent_memories:
                return {
                    "success": True,
                    "memories": [],
                    "count": 0,
                    "agent_id": agent_id
                }
            
            memories = []
            agent_memory = self.agent_memories[agent_id]
            
            # 메모리 타입 필터링
            memory_types = [memory_type] if memory_type else agent_memory.keys()
            
            for mem_type in memory_types:
                if mem_type not in agent_memory:
                    continue
                
                for memory_item in agent_memory[mem_type]:
                    # 중요도 필터
                    if memory_item["importance"] < min_importance:
                        continue
                    
                    # 태그 필터
                    if tags and not any(tag in memory_item["tags"] for tag in tags):
                        continue
                    
                    # 접근 카운트 업데이트
                    memory_item["access_count"] += 1
                    memory_item["last_accessed"] = datetime.now().isoformat()
                    
                    memories.append(memory_item.copy())
            
            # 중요도와 최근 접근 시간으로 정렬
            memories.sort(
                key=lambda x: (x["importance"], x["last_accessed"]),
                reverse=True
            )
            
            # 제한 적용
            memories = memories[:limit]
            
            # 통계 업데이트
            if memories:
                self.memory_stats["cache_hits"] += 1
            else:
                self.memory_stats["cache_misses"] += 1
            
            return {
                "success": True,
                "memories": memories,
                "count": len(memories),
                "agent_id": agent_id,
                "filters": {
                    "memory_type": memory_type,
                    "tags": tags,
                    "min_importance": min_importance
                }
            }
            
        except Exception as e:
            return {"error": f"메모리 조회 실패: {str(e)}"}
    
    async def store_game_context(
        self,
        session_id: str,
        context_type: str,
        context_data: Dict[str, Any],
        game_phase: Optional[str] = None,
        turn_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """게임 컨텍스트 저장"""
        
        try:
            context_item = {
                "context_id": self._generate_memory_id(),
                "context_type": context_type,
                "context_data": context_data,
                "game_phase": game_phase,
                "turn_number": turn_number,
                "created_at": datetime.now().isoformat()
            }
            
            if session_id not in self.game_contexts:
                self.game_contexts[session_id] = {}
            
            if context_type not in self.game_contexts[session_id]:
                self.game_contexts[session_id][context_type] = deque(maxlen=1000)
            
            self.game_contexts[session_id][context_type].append(context_item)
            
            # 통계 업데이트
            self.memory_stats["total_contexts"] += 1
            
            return {
                "success": True,
                "context_id": context_item["context_id"],
                "session_id": session_id,
                "context_type": context_type
            }
            
        except Exception as e:
            return {"error": f"컨텍스트 저장 실패: {str(e)}"}
    
    async def retrieve_game_context(
        self,
        session_id: str,
        context_type: Optional[str] = None,
        game_phase: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """게임 컨텍스트 조회"""
        
        try:
            if session_id not in self.game_contexts:
                return {
                    "success": True,
                    "contexts": [],
                    "count": 0,
                    "session_id": session_id
                }
            
            contexts = []
            session_context = self.game_contexts[session_id]
            
            # 컨텍스트 타입 필터링
            context_types = [context_type] if context_type else session_context.keys()
            
            for ctx_type in context_types:
                if ctx_type not in session_context:
                    continue
                
                for context_item in session_context[ctx_type]:
                    # 게임 단계 필터
                    if game_phase and context_item.get("game_phase") != game_phase:
                        continue
                    
                    contexts.append(context_item.copy())
            
            # 생성 시간 순으로 정렬 (최신 먼저)
            contexts.sort(key=lambda x: x["created_at"], reverse=True)
            
            # 제한 적용
            contexts = contexts[:limit]
            
            return {
                "success": True,
                "contexts": contexts,
                "count": len(contexts),
                "session_id": session_id
            }
            
        except Exception as e:
            return {"error": f"컨텍스트 조회 실패: {str(e)}"}
    
    async def store_learning_data(
        self,
        agent_id: str,
        learning_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        performance_score: Optional[float] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """학습 데이터 저장"""
        
        try:
            learning_item = {
                "learning_id": self._generate_memory_id(),
                "learning_type": learning_type,
                "input_data": input_data,
                "output_data": output_data,
                "performance_score": performance_score,
                "feedback": feedback,
                "created_at": datetime.now().isoformat()
            }
            
            # 학습 데이터 저장 (제한된 크기)
            if len(self.learning_data[agent_id]) >= self.max_learning_items:
                self.learning_data[agent_id].pop(0)  # 가장 오래된 것 제거
            
            self.learning_data[agent_id].append(learning_item)
            
            return {
                "success": True,
                "learning_id": learning_item["learning_id"],
                "agent_id": agent_id,
                "learning_type": learning_type
            }
            
        except Exception as e:
            return {"error": f"학습 데이터 저장 실패: {str(e)}"}
    
    async def analyze_patterns(
        self,
        agent_id: str,
        pattern_type: str,
        data_source: str = "memory",
        min_occurrences: int = 3
    ) -> Dict[str, Any]:
        """패턴 분석"""
        
        try:
            # 캐시 키 생성
            cache_key = f"{agent_id}:{pattern_type}:{data_source}:{min_occurrences}"
            
            if cache_key in self.pattern_cache:
                self.memory_stats["cache_hits"] += 1
                return self.pattern_cache[cache_key]
            
            patterns = {}
            
            if data_source == "memory" and agent_id in self.agent_memories:
                patterns = await self._analyze_memory_patterns(
                    self.agent_memories[agent_id], pattern_type, min_occurrences
                )
            
            elif data_source == "learning" and agent_id in self.learning_data:
                patterns = await self._analyze_learning_patterns(
                    self.learning_data[agent_id], pattern_type, min_occurrences
                )
            
            elif data_source == "context":
                patterns = await self._analyze_context_patterns(
                    agent_id, pattern_type, min_occurrences
                )
            
            result = {
                "success": True,
                "agent_id": agent_id,
                "pattern_type": pattern_type,
                "data_source": data_source,
                "patterns": patterns,
                "pattern_count": len(patterns),
                "analyzed_at": datetime.now().isoformat()
            }
            
            # 캐시에 저장 (1시간 유효)
            self.pattern_cache[cache_key] = result
            self.memory_stats["cache_misses"] += 1
            self.memory_stats["total_patterns"] += len(patterns)
            
            return result
            
        except Exception as e:
            return {"error": f"패턴 분석 실패: {str(e)}"}
    
    async def _analyze_memory_patterns(
        self, agent_memory: Dict[str, Any], pattern_type: str, min_occurrences: int
    ) -> Dict[str, Any]:
        """메모리 패턴 분석"""
        
        patterns = defaultdict(int)
        
        for memory_type, memories in agent_memory.items():
            for memory_item in memories:
                data = memory_item["memory_data"]
                
                if pattern_type == "action_sequences":
                    # 액션 시퀀스 패턴
                    if "action" in data:
                        patterns[data["action"]] += 1
                
                elif pattern_type == "decision_contexts":
                    # 결정 컨텍스트 패턴
                    if "context" in data and "decision" in data:
                        context_key = str(sorted(data["context"].items()))
                        patterns[context_key] += 1
                
                elif pattern_type == "success_factors":
                    # 성공 요인 패턴
                    if "outcome" in data and data["outcome"] == "success":
                        for key, value in data.items():
                            if key != "outcome":
                                patterns[f"{key}:{value}"] += 1
        
        # 최소 발생 횟수 필터링
        return {k: v for k, v in patterns.items() if v >= min_occurrences}
    
    async def _analyze_learning_patterns(
        self, learning_data: List[Dict[str, Any]], pattern_type: str, min_occurrences: int
    ) -> Dict[str, Any]:
        """학습 데이터 패턴 분석"""
        
        patterns = defaultdict(int)
        
        for learning_item in learning_data:
            if pattern_type == "performance_trends":
                # 성능 트렌드 패턴
                score = learning_item.get("performance_score", 0)
                if score is not None:
                    score_range = f"{int(score * 10) * 10}-{int(score * 10) * 10 + 10}%"
                    patterns[score_range] += 1
            
            elif pattern_type == "error_patterns":
                # 에러 패턴
                feedback = learning_item.get("feedback", {})
                if "error" in feedback:
                    patterns[feedback["error"]] += 1
        
        return {k: v for k, v in patterns.items() if v >= min_occurrences}
    
    async def _analyze_context_patterns(
        self, agent_id: str, pattern_type: str, min_occurrences: int
    ) -> Dict[str, Any]:
        """컨텍스트 패턴 분석"""
        
        patterns = defaultdict(int)
        
        for session_id, contexts in self.game_contexts.items():
            for context_type, context_items in contexts.items():
                for context_item in context_items:
                    if pattern_type == "game_phase_transitions":
                        # 게임 단계 전환 패턴
                        if context_item.get("game_phase"):
                            patterns[context_item["game_phase"]] += 1
        
        return {k: v for k, v in patterns.items() if v >= min_occurrences}
    
    async def get_memory_summary(
        self, agent_id: str, summary_type: str = "recent"
    ) -> Dict[str, Any]:
        """메모리 요약 조회"""
        
        try:
            if agent_id not in self.agent_memories:
                return {
                    "success": True,
                    "summary": {},
                    "agent_id": agent_id,
                    "summary_type": summary_type
                }
            
            agent_memory = self.agent_memories[agent_id]
            summary = {}
            
            if summary_type == "recent":
                # 최근 메모리 요약
                for memory_type, memories in agent_memory.items():
                    recent_memories = list(memories)[-5:]  # 최근 5개
                    summary[memory_type] = {
                        "count": len(memories),
                        "recent_items": len(recent_memories),
                        "last_updated": recent_memories[-1]["created_at"] if recent_memories else None
                    }
            
            elif summary_type == "important":
                # 중요 메모리 요약
                for memory_type, memories in agent_memory.items():
                    important_memories = [m for m in memories if m["importance"] >= 0.7]
                    summary[memory_type] = {
                        "total_count": len(memories),
                        "important_count": len(important_memories),
                        "avg_importance": sum(m["importance"] for m in memories) / len(memories)
                    }
            
            elif summary_type == "patterns":
                # 패턴 요약
                pattern_result = await self.analyze_patterns(agent_id, "action_sequences")
                summary = {
                    "pattern_analysis": pattern_result.get("patterns", {}),
                    "total_patterns": pattern_result.get("pattern_count", 0)
                }
            
            return {
                "success": True,
                "summary": summary,
                "agent_id": agent_id,
                "summary_type": summary_type,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"메모리 요약 실패: {str(e)}"}
    
    async def clear_memory(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        older_than_hours: Optional[int] = None,
        keep_important: bool = True
    ) -> Dict[str, Any]:
        """메모리 정리"""
        
        try:
            if agent_id not in self.agent_memories:
                return {
                    "success": True,
                    "cleared_count": 0,
                    "agent_id": agent_id
                }
            
            cleared_count = 0
            cutoff_time = None
            
            if older_than_hours:
                cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            
            agent_memory = self.agent_memories[agent_id]
            memory_types = [memory_type] if memory_type else list(agent_memory.keys())
            
            for mem_type in memory_types:
                if mem_type not in agent_memory:
                    continue
                
                original_count = len(agent_memory[mem_type])
                
                if cutoff_time:
                    # 시간 기반 정리
                    filtered_memories = deque()
                    for memory_item in agent_memory[mem_type]:
                        created_at = datetime.fromisoformat(memory_item["created_at"])
                        
                        # 중요한 메모리는 보존
                        if keep_important and memory_item["importance"] >= 0.8:
                            filtered_memories.append(memory_item)
                        # 최근 메모리는 보존
                        elif created_at > cutoff_time:
                            filtered_memories.append(memory_item)
                    
                    agent_memory[mem_type] = filtered_memories
                else:
                    # 전체 정리
                    if keep_important:
                        important_memories = deque()
                        for memory_item in agent_memory[mem_type]:
                            if memory_item["importance"] >= 0.8:
                                important_memories.append(memory_item)
                        agent_memory[mem_type] = important_memories
                    else:
                        agent_memory[mem_type].clear()
                
                cleared_count += original_count - len(agent_memory[mem_type])
            
            return {
                "success": True,
                "cleared_count": cleared_count,
                "agent_id": agent_id,
                "memory_type": memory_type,
                "older_than_hours": older_than_hours,
                "keep_important": keep_important
            }
            
        except Exception as e:
            return {"error": f"메모리 정리 실패: {str(e)}"}
    
    async def export_memory(
        self,
        agent_id: str,
        export_format: str = "json",
        include_learning: bool = True
    ) -> Dict[str, Any]:
        """메모리 내보내기"""
        
        try:
            export_data = {
                "agent_id": agent_id,
                "export_timestamp": datetime.now().isoformat(),
                "memories": {},
                "learning_data": [] if include_learning else None
            }
            
            # 메모리 데이터
            if agent_id in self.agent_memories:
                for memory_type, memories in self.agent_memories[agent_id].items():
                    export_data["memories"][memory_type] = list(memories)
            
            # 학습 데이터
            if include_learning and agent_id in self.learning_data:
                export_data["learning_data"] = self.learning_data[agent_id]
            
            # 형식에 따른 직렬화
            if export_format == "json":
                serialized_data = json.dumps(export_data, ensure_ascii=False, indent=2)
                content_type = "application/json"
            elif export_format == "pickle":
                serialized_data = pickle.dumps(export_data).hex()
                content_type = "application/octet-stream"
            else:
                return {"error": f"지원되지 않는 형식: {export_format}"}
            
            return {
                "success": True,
                "agent_id": agent_id,
                "export_format": export_format,
                "content_type": content_type,
                "data_size": len(serialized_data),
                "serialized_data": serialized_data[:1000] + "..." if len(serialized_data) > 1000 else serialized_data,
                "full_data_available": True
            }
            
        except Exception as e:
            return {"error": f"메모리 내보내기 실패: {str(e)}"}
    
    async def get_memory_stats(self, detailed: bool = False) -> Dict[str, Any]:
        """메모리 통계 조회"""
        
        try:
            basic_stats = self.memory_stats.copy()
            basic_stats["active_agents"] = len(self.agent_memories)
            basic_stats["active_sessions"] = len(self.game_contexts)
            basic_stats["cache_hit_rate"] = (
                basic_stats["cache_hits"] / max(1, basic_stats["cache_hits"] + basic_stats["cache_misses"])
            )
            
            if not detailed:
                return {
                    "success": True,
                    "stats": basic_stats,
                    "generated_at": datetime.now().isoformat()
                }
            
            # 상세 통계
            detailed_stats = basic_stats.copy()
            
            # 에이전트별 메모리 통계
            agent_stats = {}
            for agent_id, memories in self.agent_memories.items():
                total_memories = sum(len(mem_list) for mem_list in memories.values())
                avg_importance = 0
                if total_memories > 0:
                    all_memories = []
                    for mem_list in memories.values():
                        all_memories.extend(mem_list)
                    avg_importance = sum(m["importance"] for m in all_memories) / len(all_memories)
                
                agent_stats[agent_id] = {
                    "memory_types": len(memories),
                    "total_memories": total_memories,
                    "avg_importance": round(avg_importance, 3),
                    "learning_items": len(self.learning_data.get(agent_id, []))
                }
            
            detailed_stats["agent_stats"] = agent_stats
            
            # 세션별 컨텍스트 통계
            session_stats = {}
            for session_id, contexts in self.game_contexts.items():
                total_contexts = sum(len(ctx_list) for ctx_list in contexts.values())
                session_stats[session_id] = {
                    "context_types": len(contexts),
                    "total_contexts": total_contexts
                }
            
            detailed_stats["session_stats"] = session_stats
            
            return {
                "success": True,
                "stats": detailed_stats,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"통계 조회 실패: {str(e)}"}


# MCP 서버 실행
async def main():
    """Memory MCP 서버 실행"""
    
    if not MCP_AVAILABLE:
        print("❌ MCP 패키지가 필요합니다: pip install mcp")
        return
    
    memory_server = MemoryMCPServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await memory_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="memory-server",
                server_version="1.0.0",
                capabilities={}
            )
        )


if __name__ == "__main__":
    asyncio.run(main())