"""
Simple RAG Agent

기존 RAG Agent의 간단한 버전 (의존성 문제 해결)
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent_simple import BaseAgent, BaseAgentConfig


class SimpleRAGConfig(BaseAgentConfig):
    """간단한 RAG 설정"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_search_results = kwargs.get("max_search_results", 5)
        self.context_window_size = kwargs.get("context_window_size", 4000)
        self.similarity_threshold = kwargs.get("similarity_threshold", 0.7)


class SimpleRAGAgent(BaseAgent):
    """간단한 RAG 에이전트"""
    
    def __init__(self, config: SimpleRAGConfig):
        super().__init__(config)
        self.knowledge_base = {}
        
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """문서를 지식 베이스에 추가"""
        for doc in documents:
            doc_id = doc.get("id", f"doc_{len(self.knowledge_base)}")
            self.knowledge_base[doc_id] = doc
        self.logger.info(f"Added {len(documents)} documents to knowledge base")
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """문서 검색"""
        # 간단한 키워드 기반 검색
        query_words = query.lower().split()
        scored_docs = []
        
        for doc_id, doc in self.knowledge_base.items():
            content = doc.get("content", "").lower()
            score = sum(1 for word in query_words if word in content)
            
            if score > 0:
                scored_docs.append({
                    "doc_id": doc_id,
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": score
                })
        
        # 점수순으로 정렬하고 상위 k개 반환
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]
    
    async def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """컨텍스트를 기반으로 응답 생성"""
        context = "\n".join([doc["content"] for doc in context_docs])
        
        prompt = f"""
        다음 컨텍스트를 기반으로 질문에 답변하세요:
        
        컨텍스트:
        {context}
        
        질문: {query}
        
        답변:
        """
        
        response = await self.call_llm(prompt)
        return response.strip()
    
    async def process(self, query: str) -> Dict[str, Any]:
        """RAG 처리 메인 메서드"""
        self.logger.info(f"Processing RAG query: {query[:100]}...")
        
        # 1. 문서 검색
        search_results = await self.search_documents(query, self.config.max_search_results)
        
        # 2. 응답 생성
        response = await self.generate_response(query, search_results)
        
        # 3. 결과 구성
        result = {
            "query": query,
            "response": response,
            "source_documents": search_results,
            "num_sources": len(search_results),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return result
