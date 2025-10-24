#!/usr/bin/env python3
"""
Vector Store for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 벡터 데이터베이스 통합 시스템.
ChromaDB와 Qdrant를 지원하며, 연구 결과 임베딩 및 저장,
시맨틱 검색, 유사 연구 찾기 기능을 제공합니다.

2025년 10월 최신 기술 스택:
- ChromaDB 0.4.18+ for vector storage
- Qdrant for advanced vector operations
- sentence-transformers 2.2.2+ for embeddings
- Production-grade reliability patterns
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import os

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# MCP integration
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcp_integration import execute_tool
from src.core.reliability import execute_with_reliability
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class VectorDBType(Enum):
    """벡터 데이터베이스 타입."""
    CHROMADB = "chromadb"
    QDRANT = "qdrant"
    HYBRID = "hybrid"  # ChromaDB + Qdrant


@dataclass
class ResearchMemory:
    """연구 메모리 데이터 구조."""
    research_id: str
    user_id: str
    topic: str
    timestamp: datetime
    embedding: List[float]
    metadata: Dict[str, Any]
    results: Dict[str, Any]
    content: str
    summary: str
    keywords: List[str]
    confidence_score: float = 0.0
    source_count: int = 0
    verification_status: str = "unverified"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SearchResult:
    """검색 결과."""
    research_id: str
    similarity_score: float
    content: str
    summary: str
    metadata: Dict[str, Any]
    timestamp: datetime


class VectorStore:
    """
    Production-grade 벡터 데이터베이스 통합 시스템.
    
    Features:
    - ChromaDB 또는 Qdrant 클라이언트 초기화
    - 연구 결과 임베딩 및 저장
    - 시맨틱 검색 기능
    - 유사 연구 찾기
    - 하이브리드 스토리지 지원
    """
    
    def __init__(
        self,
        db_type: VectorDBType = VectorDBType.CHROMADB,
        collection_name: str = "research_memories",
        persist_directory: str = "./vector_db"
    ):
        """
        벡터 스토어 초기화.
        
        Args:
            db_type: 사용할 벡터 데이터베이스 타입
            collection_name: 컬렉션 이름
            persist_directory: 데이터 저장 디렉토리
        """
        self.db_type = db_type
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 임베딩 모델 초기화
        self.embedding_model = self._initialize_embedding_model()
        
        # 벡터 데이터베이스 클라이언트 초기화
        self.chroma_client = None
        self.qdrant_client = None
        self.collection = None
        
        if db_type in [VectorDBType.CHROMADB, VectorDBType.HYBRID]:
            self._initialize_chromadb()
        
        if db_type in [VectorDBType.QDRANT, VectorDBType.HYBRID]:
            self._initialize_qdrant()
        
        logger.info(f"VectorStore initialized with {db_type.value} database")
    
    def _initialize_embedding_model(self) -> Optional[SentenceTransformer]:
        """임베딩 모델을 초기화합니다."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, using MCP tools for embeddings")
            return None
        
        try:
            # 다국어 지원 모델 사용
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized: all-MiniLM-L6-v2")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            return None
    
    def _initialize_chromadb(self) -> None:
        """ChromaDB를 초기화합니다."""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available")
            return
        
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory / "chromadb"),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 생성 또는 가져오기
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Research memories and findings"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def _initialize_qdrant(self) -> None:
        """Qdrant를 초기화합니다."""
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant not available")
            return
        
        try:
            # 로컬 Qdrant 서버 사용
            self.qdrant_client = QdrantClient(
                path=str(self.persist_directory / "qdrant")
            )
            
            # 컬렉션 생성
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding size
                        distance=models.Distance.COSINE
                    )
                )
            except Exception:
                # 컬렉션이 이미 존재하는 경우
                pass
            
            logger.info(f"Qdrant initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self.qdrant_client = None
    
    async def generate_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩을 생성합니다."""
        try:
            if self.embedding_model:
                # 로컬 모델 사용
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            else:
                # MCP 도구 사용
                result = await execute_tool("embed_text", {
                    "text": text,
                    "model": "all-MiniLM-L6-v2"
                })
                
                if result.get('success', False):
                    return result.get('data', {}).get('embedding', [])
                else:
                    logger.warning("MCP embedding failed, using simple text hash")
                    # 간단한 해시 기반 임베딩 (fallback)
                    return self._simple_hash_embedding(text)
                    
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._simple_hash_embedding(text)
    
    def _simple_hash_embedding(self, text: str) -> List[float]:
        """간단한 해시 기반 임베딩 (fallback)."""
        import hashlib
        
        # 텍스트를 해시하고 384차원 벡터로 변환
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # 384차원 벡터 생성 (32바이트 해시를 반복하여 확장)
        embedding = []
        for i in range(384):
            byte_idx = i % 32
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        
        return embedding
    
    async def store_research_memory(self, memory: ResearchMemory) -> bool:
        """
        연구 메모리를 저장합니다.
        
        Args:
            memory: 저장할 연구 메모리
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 임베딩 생성
            if not memory.embedding:
                memory.embedding = await self.generate_embedding(memory.content)
            
            # ChromaDB에 저장
            if self.chroma_client and self.collection:
                await self._store_in_chromadb(memory)
            
            # Qdrant에 저장
            if self.qdrant_client:
                await self._store_in_qdrant(memory)
            
            logger.info(f"Research memory stored: {memory.research_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store research memory: {e}")
            return False
    
    async def _store_in_chromadb(self, memory: ResearchMemory) -> None:
        """ChromaDB에 저장합니다."""
        try:
            self.collection.add(
                ids=[memory.research_id],
                embeddings=[memory.embedding],
                documents=[memory.content],
                metadatas=[memory.metadata]
            )
            logger.debug(f"Stored in ChromaDB: {memory.research_id}")
        except Exception as e:
            logger.error(f"ChromaDB storage failed: {e}")
            raise
    
    async def _store_in_qdrant(self, memory: ResearchMemory) -> None:
        """Qdrant에 저장합니다."""
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=hash(memory.research_id) % (2**63),  # 64비트 정수로 변환
                        vector=memory.embedding,
                        payload={
                            "research_id": memory.research_id,
                            "user_id": memory.user_id,
                            "topic": memory.topic,
                            "timestamp": memory.timestamp.isoformat(),
                            "content": memory.content,
                            "summary": memory.summary,
                            "keywords": memory.keywords,
                            "confidence_score": memory.confidence_score,
                            "source_count": memory.source_count,
                            "verification_status": memory.verification_status,
                            **memory.metadata
                        }
                    )
                ]
            )
            logger.debug(f"Stored in Qdrant: {memory.research_id}")
        except Exception as e:
            logger.error(f"Qdrant storage failed: {e}")
            raise
    
    async def search_similar_research(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        유사한 연구를 검색합니다.
        
        Args:
            query: 검색 쿼리
            user_id: 사용자 ID (선택사항)
            limit: 결과 수 제한
            similarity_threshold: 유사도 임계값
            
        Returns:
            List[SearchResult]: 검색 결과
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.generate_embedding(query)
            
            results = []
            
            # ChromaDB에서 검색
            if self.chroma_client and self.collection:
                chroma_results = await self._search_in_chromadb(
                    query_embedding, user_id, limit, similarity_threshold
                )
                results.extend(chroma_results)
            
            # Qdrant에서 검색
            if self.qdrant_client:
                qdrant_results = await self._search_in_qdrant(
                    query_embedding, user_id, limit, similarity_threshold
                )
                results.extend(qdrant_results)
            
            # 중복 제거 및 정렬
            unique_results = self._deduplicate_results(results)
            sorted_results = sorted(
                unique_results,
                key=lambda x: x.similarity_score,
                reverse=True
            )
            
            logger.info(f"Found {len(sorted_results)} similar research results")
            return sorted_results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _search_in_chromadb(
        self,
        query_embedding: List[float],
        user_id: Optional[str],
        limit: int,
        similarity_threshold: float
    ) -> List[SearchResult]:
        """ChromaDB에서 검색합니다."""
        try:
            # 필터 조건 설정
            where_filter = {}
            if user_id:
                where_filter["user_id"] = user_id
            
            # 검색 실행
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter if where_filter else None
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, research_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity_score = 1 - distance  # 거리를 유사도로 변환
                    
                    if similarity_score >= similarity_threshold:
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        content = results['documents'][0][i] if results['documents'] else ""
                        
                        search_results.append(SearchResult(
                            research_id=research_id,
                            similarity_score=similarity_score,
                            content=content,
                            summary=metadata.get('summary', ''),
                            metadata=metadata,
                            timestamp=datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat()))
                        ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    async def _search_in_qdrant(
        self,
        query_embedding: List[float],
        user_id: Optional[str],
        limit: int,
        similarity_threshold: float
    ) -> List[SearchResult]:
        """Qdrant에서 검색합니다."""
        try:
            # 필터 조건 설정
            filter_conditions = None
            if user_id:
                filter_conditions = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                )
            
            # 검색 실행
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_conditions,
                score_threshold=similarity_threshold
            )
            
            search_results = []
            for result in results:
                payload = result.payload
                search_results.append(SearchResult(
                    research_id=payload.get('research_id', ''),
                    similarity_score=result.score,
                    content=payload.get('content', ''),
                    summary=payload.get('summary', ''),
                    metadata=payload,
                    timestamp=datetime.fromisoformat(payload.get('timestamp', datetime.now().isoformat()))
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """중복 결과를 제거합니다."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.research_id not in seen_ids:
                seen_ids.add(result.research_id)
                unique_results.append(result)
        
        return unique_results
    
    async def get_research_memory(self, research_id: str) -> Optional[ResearchMemory]:
        """특정 연구 메모리를 가져옵니다."""
        try:
            # ChromaDB에서 검색
            if self.chroma_client and self.collection:
                results = self.collection.get(ids=[research_id])
                if results['ids']:
                    metadata = results['metadatas'][0] if results['metadatas'] else {}
                    content = results['documents'][0] if results['documents'] else ""
                    embedding = results['embeddings'][0] if results['embeddings'] else []
                    
                    return ResearchMemory(
                        research_id=research_id,
                        user_id=metadata.get('user_id', ''),
                        topic=metadata.get('topic', ''),
                        timestamp=datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat())),
                        embedding=embedding,
                        metadata=metadata,
                        results=metadata.get('results', {}),
                        content=content,
                        summary=metadata.get('summary', ''),
                        keywords=metadata.get('keywords', []),
                        confidence_score=metadata.get('confidence_score', 0.0),
                        source_count=metadata.get('source_count', 0),
                        verification_status=metadata.get('verification_status', 'unverified')
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get research memory: {e}")
            return None
    
    async def update_research_memory(self, memory: ResearchMemory) -> bool:
        """연구 메모리를 업데이트합니다."""
        try:
            # 임베딩 재생성
            memory.embedding = await self.generate_embedding(memory.content)
            
            # 저장 (upsert)
            return await self.store_research_memory(memory)
            
        except Exception as e:
            logger.error(f"Failed to update research memory: {e}")
            return False
    
    async def delete_research_memory(self, research_id: str) -> bool:
        """연구 메모리를 삭제합니다."""
        try:
            # ChromaDB에서 삭제
            if self.chroma_client and self.collection:
                self.collection.delete(ids=[research_id])
            
            # Qdrant에서 삭제
            if self.qdrant_client:
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=[hash(research_id) % (2**63)]
                    )
                )
            
            logger.info(f"Research memory deleted: {research_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete research memory: {e}")
            return False
    
    async def get_user_research_history(self, user_id: str, limit: int = 50) -> List[ResearchMemory]:
        """사용자의 연구 히스토리를 가져옵니다."""
        try:
            # ChromaDB에서 사용자별 검색
            if self.chroma_client and self.collection:
                results = self.collection.query(
                    query_embeddings=[[0.0] * 384],  # 더미 임베딩
                    n_results=limit,
                    where={"user_id": user_id}
                )
                
                memories = []
                if results['ids'] and results['ids'][0]:
                    for i, research_id in enumerate(results['ids'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        content = results['documents'][0][i] if results['documents'] else ""
                        embedding = results['embeddings'][0][i] if results['embeddings'] else []
                        
                        memory = ResearchMemory(
                            research_id=research_id,
                            user_id=metadata.get('user_id', user_id),
                            topic=metadata.get('topic', ''),
                            timestamp=datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat())),
                            embedding=embedding,
                            metadata=metadata,
                            results=metadata.get('results', {}),
                            content=content,
                            summary=metadata.get('summary', ''),
                            keywords=metadata.get('keywords', []),
                            confidence_score=metadata.get('confidence_score', 0.0),
                            source_count=metadata.get('source_count', 0),
                            verification_status=metadata.get('verification_status', 'unverified')
                        )
                        memories.append(memory)
                
                return memories
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get user research history: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계를 반환합니다."""
        stats = {
            'db_type': self.db_type.value,
            'collection_name': self.collection_name,
            'chromadb_available': self.chroma_client is not None,
            'qdrant_available': self.qdrant_client is not None,
            'embedding_model_available': self.embedding_model is not None
        }
        
        try:
            if self.chroma_client and self.collection:
                count = self.collection.count()
                stats['total_documents'] = count
                stats['chromadb_status'] = 'active'
            else:
                stats['chromadb_status'] = 'inactive'
            
            if self.qdrant_client:
                info = self.qdrant_client.get_collection(self.collection_name)
                stats['qdrant_points'] = info.points_count
                stats['qdrant_status'] = 'active'
            else:
                stats['qdrant_status'] = 'inactive'
                
        except Exception as e:
            logger.warning(f"Failed to get collection stats: {e}")
            stats['error'] = str(e)
        
        return stats
