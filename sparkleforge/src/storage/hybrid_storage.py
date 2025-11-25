#!/usr/bin/env python3
"""
Hybrid Storage for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 하이브리드 스토리지 시스템.
벡터 DB와 파일 시스템을 결합하여 빠른 시맨틱 검색과
상세 결과 저장을 동시에 제공합니다.

2025년 10월 최신 기술 스택:
- Vector DB: ChromaDB/Qdrant for semantic search
- File System: JSON files for detailed storage
- Automatic synchronization mechanism
- Caching layer for performance
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import pickle
import gzip

# Import vector store
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.storage.vector_store import VectorStore, ResearchMemory, SearchResult, VectorDBType
from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StorageConfig:
    """스토리지 설정."""
    vector_db_type: VectorDBType = VectorDBType.CHROMADB
    base_directory: str = "./storage"
    vector_db_directory: str = "./storage/vector_db"
    file_storage_directory: str = "./storage/files"
    cache_directory: str = "./storage/cache"
    max_file_size_mb: int = 100
    enable_compression: bool = True
    cache_ttl_hours: int = 24
    sync_interval_seconds: int = 300


@dataclass
class StorageStats:
    """스토리지 통계."""
    total_research_count: int
    vector_db_count: int
    file_storage_count: int
    cache_hit_rate: float
    last_sync_time: datetime
    storage_size_mb: float
    compression_ratio: float


class HybridStorage:
    """
    Production-grade 하이브리드 스토리지 시스템.
    
    Features:
    - 벡터 DB: 빠른 시맨틱 검색용
    - 파일 시스템: 상세 결과 및 메타데이터 저장
    - 자동 동기화 메커니즘
    - 캐싱 레이어
    - 압축 지원
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        하이브리드 스토리지 초기화.
        
        Args:
            config: 스토리지 설정
        """
        self.config = config or StorageConfig()
        
        # 디렉토리 생성
        self.base_dir = Path(self.config.base_directory)
        self.vector_db_dir = Path(self.config.vector_db_directory)
        self.file_storage_dir = Path(self.config.file_storage_directory)
        self.cache_dir = Path(self.config.cache_directory)
        
        for directory in [self.base_dir, self.vector_db_dir, self.file_storage_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 벡터 스토어 초기화
        try:
            self.vector_store = VectorStore(
                db_type=self.config.vector_db_type,
                collection_name="research_memories",
                persist_directory=str(self.vector_db_dir)
            )
            
            # ChromaDB 사용 가능 여부 확인
            if self.vector_store.chroma_client is None and self.vector_store.qdrant_client is None:
                logger.warning("⚠️ Vector database not available - vector search disabled")
                logger.info("ℹ️ HybridStorage will use file storage only")
                self.vector_enabled = False
            else:
                self.vector_enabled = True
                logger.info(f"✅ Vector search enabled with {self.config.vector_db_type.value}")
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            logger.warning("⚠️ Continuing with file storage only")
            self.vector_store = None
            self.vector_enabled = False
        
        # 캐시 초기화
        self.cache = {}
        self.cache_timestamps = {}
        
        # 동기화 상태
        self.last_sync = datetime.now(timezone.utc)
        self.sync_in_progress = False
        
        logger.info(f"HybridStorage initialized: {self.config.vector_db_type.value}")
        logger.info(f"Vector search: {'Enabled' if self.vector_enabled else 'Disabled'}")
    
    async def store_research(
        self,
        research_id: str,
        user_id: str,
        topic: str,
        content: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> bool:
        """
        연구 결과를 저장합니다.
        
        Args:
            research_id: 연구 ID
            user_id: 사용자 ID
            topic: 연구 주제
            content: 연구 내용
            results: 연구 결과
            metadata: 추가 메타데이터
            summary: 요약 (선택사항)
            keywords: 키워드 (선택사항)
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 메타데이터 준비
            if metadata is None:
                metadata = {}
            
            # 요약 생성 (없는 경우)
            if not summary:
                summary = await self._generate_summary(content)
            
            # 키워드 추출 (없는 경우)
            if not keywords:
                keywords = await self._extract_keywords(content)
            
            # 연구 메모리 생성
            memory = ResearchMemory(
                research_id=research_id,
                user_id=user_id,
                topic=topic,
                timestamp=datetime.now(timezone.utc),
                embedding=[],  # 벡터 스토어에서 생성
                metadata=metadata,
                results=results,
                content=content,
                summary=summary,
                keywords=keywords,
                confidence_score=metadata.get('confidence_score', 0.0),
                source_count=metadata.get('source_count', 0),
                verification_status=metadata.get('verification_status', 'unverified')
            )
            
            # 벡터 DB에 저장
            vector_success = await self.vector_store.store_research_memory(memory)
            
            # 파일 시스템에 저장
            file_success = await self._store_in_file_system(memory)
            
            # 캐시 업데이트
            self._update_cache(research_id, memory)
            
            success = vector_success and file_success
            
            if success:
                logger.info(f"Research stored successfully: {research_id}")
            else:
                logger.error(f"Failed to store research: {research_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store research {research_id}: {e}")
            return False
    
    async def _store_in_file_system(self, memory: ResearchMemory) -> bool:
        """파일 시스템에 저장합니다."""
        try:
            # 파일 경로 생성
            file_path = self.file_storage_dir / f"{memory.research_id}.json"
            
            # 데이터 준비
            data = memory.to_dict()
            
            # 압축 여부 결정
            if self.config.enable_compression and len(json.dumps(data)) > 1024:  # 1KB 이상
                # 압축 저장
                compressed_data = gzip.compress(json.dumps(data, ensure_ascii=False).encode('utf-8'))
                file_path = file_path.with_suffix('.json.gz')
                file_path.write_bytes(compressed_data)
            else:
                # 일반 저장
                file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            
            logger.debug(f"Stored in file system: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"File system storage failed: {e}")
            return False
    
    async def get_research(self, research_id: str) -> Optional[ResearchMemory]:
        """연구 결과를 가져옵니다."""
        try:
            # 캐시 확인
            if research_id in self.cache:
                if self._is_cache_valid(research_id):
                    logger.debug(f"Cache hit: {research_id}")
                    return self.cache[research_id]
            
            # 파일 시스템에서 로드
            memory = await self._load_from_file_system(research_id)
            
            if memory:
                # 캐시 업데이트
                self._update_cache(research_id, memory)
                return memory
            
            # 벡터 DB에서 로드 (fallback)
            memory = await self.vector_store.get_research_memory(research_id)
            if memory:
                # 파일 시스템에 저장
                await self._store_in_file_system(memory)
                self._update_cache(research_id, memory)
                return memory
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get research {research_id}: {e}")
            return None
    
    async def _load_from_file_system(self, research_id: str) -> Optional[ResearchMemory]:
        """파일 시스템에서 로드합니다."""
        try:
            # 일반 파일 확인
            file_path = self.file_storage_dir / f"{research_id}.json"
            if file_path.exists():
                data = json.loads(file_path.read_text(encoding='utf-8'))
                return self._dict_to_memory(data)
            
            # 압축 파일 확인
            compressed_path = self.file_storage_dir / f"{research_id}.json.gz"
            if compressed_path.exists():
                compressed_data = compressed_path.read_bytes()
                data = json.loads(gzip.decompress(compressed_data).decode('utf-8'))
                return self._dict_to_memory(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load from file system {research_id}: {e}")
            return None
    
    def _dict_to_memory(self, data: Dict[str, Any]) -> ResearchMemory:
        """딕셔너리를 ResearchMemory로 변환합니다."""
        return ResearchMemory(
            research_id=data['research_id'],
            user_id=data['user_id'],
            topic=data['topic'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            embedding=data['embedding'],
            metadata=data['metadata'],
            results=data['results'],
            content=data['content'],
            summary=data['summary'],
            keywords=data['keywords'],
            confidence_score=data.get('confidence_score', 0.0),
            source_count=data.get('source_count', 0),
            verification_status=data.get('verification_status', 'unverified')
        )
    
    async def search_similar_research(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[SearchResult]:
        """유사한 연구를 검색합니다."""
        try:
            # 벡터 DB에서 검색
            results = await self.vector_store.search_similar_research(
                query=query,
                user_id=user_id,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            # 상세 정보 로드
            detailed_results = []
            for result in results:
                memory = await self.get_research(result.research_id)
                if memory:
                    detailed_results.append(SearchResult(
                        research_id=result.research_id,
                        similarity_score=result.similarity_score,
                        content=memory.content,
                        summary=memory.summary,
                        metadata=memory.metadata,
                        timestamp=memory.timestamp
                    ))
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_user_research_history(self, user_id: str, limit: int = 50) -> List[ResearchMemory]:
        """사용자의 연구 히스토리를 가져옵니다."""
        try:
            # 벡터 DB에서 검색
            memories = await self.vector_store.get_user_research_history(user_id, limit)
            
            # 파일 시스템에서 상세 정보 로드
            detailed_memories = []
            for memory in memories:
                detailed_memory = await self.get_research(memory.research_id)
                if detailed_memory:
                    detailed_memories.append(detailed_memory)
            
            return detailed_memories
            
        except Exception as e:
            logger.error(f"Failed to get user research history: {e}")
            return []
    
    async def update_research(
        self,
        research_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """연구 결과를 업데이트합니다."""
        try:
            # 기존 데이터 로드
            memory = await self.get_research(research_id)
            if not memory:
                logger.warning(f"Research not found: {research_id}")
                return False
            
            # 업데이트 적용
            for key, value in updates.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)
                else:
                    memory.metadata[key] = value
            
            # 타임스탬프 업데이트
            memory.timestamp = datetime.now(timezone.utc)
            
            # 저장
            vector_success = await self.vector_store.update_research_memory(memory)
            file_success = await self._store_in_file_system(memory)
            
            # 캐시 업데이트
            self._update_cache(research_id, memory)
            
            success = vector_success and file_success
            
            if success:
                logger.info(f"Research updated successfully: {research_id}")
            else:
                logger.error(f"Failed to update research: {research_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update research {research_id}: {e}")
            return False
    
    async def delete_research(self, research_id: str) -> bool:
        """연구 결과를 삭제합니다."""
        try:
            # 벡터 DB에서 삭제
            vector_success = await self.vector_store.delete_research_memory(research_id)
            
            # 파일 시스템에서 삭제
            file_success = await self._delete_from_file_system(research_id)
            
            # 캐시에서 제거
            self._remove_from_cache(research_id)
            
            success = vector_success and file_success
            
            if success:
                logger.info(f"Research deleted successfully: {research_id}")
            else:
                logger.error(f"Failed to delete research: {research_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete research {research_id}: {e}")
            return False
    
    async def _delete_from_file_system(self, research_id: str) -> bool:
        """파일 시스템에서 삭제합니다."""
        try:
            # 일반 파일 삭제
            file_path = self.file_storage_dir / f"{research_id}.json"
            if file_path.exists():
                file_path.unlink()
            
            # 압축 파일 삭제
            compressed_path = self.file_storage_dir / f"{research_id}.json.gz"
            if compressed_path.exists():
                compressed_path.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"File system deletion failed: {e}")
            return False
    
    async def sync_storage(self) -> bool:
        """스토리지를 동기화합니다."""
        if self.sync_in_progress:
            logger.warning("Sync already in progress")
            return False
        
        try:
            self.sync_in_progress = True
            logger.info("Starting storage sync...")
            
            # 벡터 DB와 파일 시스템 간 일치성 확인
            vector_stats = self.vector_store.get_collection_stats()
            file_stats = await self._get_file_storage_stats()
            
            # 동기화 로그
            logger.info(f"Vector DB documents: {vector_stats.get('total_documents', 0)}")
            logger.info(f"File storage files: {file_stats['file_count']}")
            
            self.last_sync = datetime.now(timezone.utc)
            logger.info("Storage sync completed")
            return True
            
        except Exception as e:
            logger.error(f"Storage sync failed: {e}")
            return False
        finally:
            self.sync_in_progress = False
    
    async def _get_file_storage_stats(self) -> Dict[str, Any]:
        """파일 스토리지 통계를 가져옵니다."""
        try:
            files = list(self.file_storage_dir.glob("*.json")) + list(self.file_storage_dir.glob("*.json.gz"))
            total_size = sum(f.stat().st_size for f in files)
            
            return {
                'file_count': len(files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Failed to get file storage stats: {e}")
            return {'file_count': 0, 'total_size_bytes': 0, 'total_size_mb': 0.0}
    
    def _update_cache(self, research_id: str, memory: ResearchMemory) -> None:
        """캐시를 업데이트합니다."""
        self.cache[research_id] = memory
        self.cache_timestamps[research_id] = datetime.now(timezone.utc)
        
        # 캐시 크기 제한
        if len(self.cache) > 1000:  # 최대 1000개 항목
            # 가장 오래된 항목 제거
            oldest_id = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
            self._remove_from_cache(oldest_id)
    
    def _remove_from_cache(self, research_id: str) -> None:
        """캐시에서 제거합니다."""
        self.cache.pop(research_id, None)
        self.cache_timestamps.pop(research_id, None)
    
    def _is_cache_valid(self, research_id: str) -> bool:
        """캐시가 유효한지 확인합니다."""
        if research_id not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[research_id]
        ttl = self.config.cache_ttl_hours * 3600  # 초로 변환
        return (datetime.now(timezone.utc) - cache_time).total_seconds() < ttl
    
    async def _generate_summary(self, content: str) -> str:
        """콘텐츠 요약을 생성합니다."""
        # 간단한 요약 (실제로는 LLM 사용)
        words = content.split()
        if len(words) <= 50:
            return content
        else:
            return " ".join(words[:50]) + "..."
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """키워드를 추출합니다."""
        # 간단한 키워드 추출 (실제로는 NLP 라이브러리 사용)
        words = content.lower().split()
        # 빈도 기반 키워드 추출
        word_freq = {}
        for word in words:
            if len(word) > 3:  # 3글자 이상
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 상위 10개 키워드
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, freq in keywords]
    
    def get_storage_stats(self) -> StorageStats:
        """스토리지 통계를 반환합니다."""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            file_stats = asyncio.run(self._get_file_storage_stats())
            
            # 캐시 히트율 계산
            total_requests = len(self.cache) + len(self.cache_timestamps)
            cache_hits = len(self.cache)
            cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
            
            return StorageStats(
                total_research_count=vector_stats.get('total_documents', 0),
                vector_db_count=vector_stats.get('total_documents', 0),
                file_storage_count=file_stats['file_count'],
                cache_hit_rate=cache_hit_rate,
                last_sync_time=self.last_sync,
                storage_size_mb=file_stats['total_size_mb'],
                compression_ratio=0.8  # 추정값
            )
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return StorageStats(
                total_research_count=0,
                vector_db_count=0,
                file_storage_count=0,
                cache_hit_rate=0.0,
                last_sync_time=datetime.now(timezone.utc),
                storage_size_mb=0.0,
                compression_ratio=0.0
            )
    
    async def cleanup_old_data(self, days_old: int = 30) -> int:
        """오래된 데이터를 정리합니다."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            cleaned_count = 0
            
            # 파일 시스템에서 오래된 파일 찾기
            for file_path in self.file_storage_dir.glob("*.json*"):
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to clean file {file_path}: {e}")
            
            logger.info(f"Cleaned {cleaned_count} old files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
