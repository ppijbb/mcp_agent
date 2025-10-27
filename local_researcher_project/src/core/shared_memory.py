"""
Shared Memory System for Multi-Agent Orchestration

기존 local_researcher는 ChromaDB와 LangGraph를 사용 중이므로
이 두 가지를 기반으로 구현하여 추가 의존성 없이 통합
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class MemoryScope:
    """Memory scope enumeration."""
    GLOBAL = "global"
    SESSION = "session"
    AGENT = "agent"


class SharedMemory:
    """
    Multi-Agent Shared Memory System
    
    ChromaDB를 벡터 저장소로 사용하되, 
    간단한 파일 기반 메모리 시스템을 제공하여 추가 의존성 없이 작동
    """
    
    def __init__(self, storage_path: str = "./storage/memori", enable_chromadb: bool = True):
        """Initialize shared memory system."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.enable_chromadb = enable_chromadb
        
        # In-memory state
        self.memories: Dict[str, Any] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Try to initialize ChromaDB if enabled
        self.chroma_client = None
        if self.enable_chromadb:
            try:
                import chromadb
                self.chroma_client = chromadb.Client()
                logger.info("✅ ChromaDB initialized for vector search")
            except ImportError:
                logger.warning("⚠️ ChromaDB not available - using file-based storage only")
                logger.info("   Install with: pip install chromadb")
                self.enable_chromadb = False
        
        logger.info(f"SharedMemory initialized at {self.storage_path}")
    
    def write(self, 
              key: str, 
              value: Any, 
              scope: str = MemoryScope.GLOBAL,
              session_id: Optional[str] = None,
              agent_id: Optional[str] = None) -> bool:
        """
        Write to shared memory.
        
        Args:
            key: Memory key
            value: Memory value
            scope: Memory scope (global, session, agent)
            session_id: Session ID
            agent_id: Agent ID
            
        Returns:
            Success status
        """
        try:
            # Store in-memory
            memory_entry = {
                "key": key,
                "value": value,
                "scope": scope,
                "session_id": session_id,
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if scope == MemoryScope.GLOBAL:
                self.memories[key] = memory_entry
            elif scope == MemoryScope.SESSION:
                if session_id:
                    if session_id not in self.sessions:
                        self.sessions[session_id] = {}
                    self.sessions[session_id][key] = memory_entry
            elif scope == MemoryScope.AGENT:
                if agent_id and session_id:
                    if session_id not in self.sessions:
                        self.sessions[session_id] = {}
                    if "agents" not in self.sessions[session_id]:
                        self.sessions[session_id]["agents"] = {}
                    if agent_id not in self.sessions[session_id]["agents"]:
                        self.sessions[session_id]["agents"][agent_id] = {}
                    self.sessions[session_id]["agents"][agent_id][key] = memory_entry
            
            # Persist to file
            self._persist_memory(memory_entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write memory: {e}")
            return False
    
    def read(self, 
             key: str, 
             scope: str = MemoryScope.GLOBAL,
             session_id: Optional[str] = None,
             agent_id: Optional[str] = None) -> Optional[Any]:
        """
        Read from shared memory.
        
        Args:
            key: Memory key
            scope: Memory scope
            session_id: Session ID
            agent_id: Agent ID
            
        Returns:
            Memory value or None
        """
        try:
            if scope == MemoryScope.GLOBAL:
                if key in self.memories:
                    return self.memories[key]["value"]
            elif scope == MemoryScope.SESSION:
                if session_id and session_id in self.sessions:
                    if key in self.sessions[session_id]:
                        return self.sessions[session_id][key]["value"]
            elif scope == MemoryScope.AGENT:
                if agent_id and session_id:
                    if session_id in self.sessions:
                        if "agents" in self.sessions[session_id]:
                            if agent_id in self.sessions[session_id]["agents"]:
                                if key in self.sessions[session_id]["agents"][agent_id]:
                                    return self.sessions[session_id]["agents"][agent_id][key]["value"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read memory: {e}")
            return None
    
    def search(self, 
               query: str, 
               limit: int = 10,
               scope: str = MemoryScope.GLOBAL) -> List[Dict[str, Any]]:
        """
        Search memories by query.
        
        Args:
            query: Search query
            limit: Result limit
            scope: Memory scope
            
        Returns:
            List of matching memories
        """
        results = []
        
        try:
            # Simple keyword matching for now
            # If ChromaDB is available, use vector search
            if self.enable_chromadb and self.chroma_client:
                # ChromaDB vector search would be implemented here
                # For now, fall through to simple search
                pass
            
            # Simple keyword search
            query_lower = query.lower()
            search_spaces = []
            
            if scope == MemoryScope.GLOBAL:
                search_spaces.append(self.memories)
            elif scope == MemoryScope.SESSION:
                for session_data in self.sessions.values():
                    if isinstance(session_data, dict):
                        search_spaces.append(session_data)
            
            for space in search_spaces:
                for key, memory_entry in space.items():
                    if isinstance(memory_entry, dict):
                        key_str = key.lower()
                        value_str = str(memory_entry.get("value", "")).lower()
                        
                        if query_lower in key_str or query_lower in value_str:
                            results.append({
                                "key": key,
                                "value": memory_entry.get("value"),
                                "timestamp": memory_entry.get("timestamp")
                            })
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []
    
    def list_session_memories(self, session_id: str) -> Dict[str, Any]:
        """List all memories for a session."""
        return self.sessions.get(session_id, {})
    
    def clear_session(self, session_id: str) -> bool:
        """Clear all memories for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def _persist_memory(self, memory_entry: Dict[str, Any]) -> None:
        """Persist memory to disk."""
        try:
            memory_file = self.storage_path / "memories.jsonl"
            
            with open(memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(memory_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.warning(f"Failed to persist memory: {e}")


# Global shared memory instance
_shared_memory: Optional[SharedMemory] = None


def get_shared_memory() -> SharedMemory:
    """Get global shared memory instance."""
    global _shared_memory
    
    if _shared_memory is None:
        storage_path = os.getenv("MEMORI_STORAGE_PATH", "./storage/memori")
        enable_chromadb = os.getenv("ENABLE_CHROMADB", "true").lower() == "true"
        _shared_memory = SharedMemory(storage_path=storage_path, enable_chromadb=enable_chromadb)
    
    return _shared_memory


def init_shared_memory(storage_path: Optional[str] = None, enable_chromadb: Optional[bool] = None) -> SharedMemory:
    """Initialize shared memory system."""
    global _shared_memory
    
    if storage_path is None:
        storage_path = os.getenv("MEMORI_STORAGE_PATH", "./storage/memori")
    if enable_chromadb is None:
        enable_chromadb = os.getenv("ENABLE_CHROMADB", "true").lower() == "true"
    
    _shared_memory = SharedMemory(storage_path=storage_path, enable_chromadb=enable_chromadb)
    
    return _shared_memory

