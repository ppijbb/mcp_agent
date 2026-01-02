"""
벡터 데이터베이스 및 텍스트 분할기
효율적인 정보 검색 및 대용량 문서 처리
"""

import logging
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class HSPVectorStore:
    """HSP Agent 전용 벡터 데이터베이스"""
    
    def __init__(self, vector_store_type: str = "chroma", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.vector_store_type = vector_store_type
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        
        # 초기화
        self._initialize_embeddings()
        self._initialize_text_splitter()
        self._initialize_vector_store()
        
        logger.info(f"HSP Vector Store 초기화 완료: {vector_store_type}")
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"임베딩 모델 로드 완료: {self.embedding_model}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            self.embeddings = None
    
    def _initialize_text_splitter(self):
        """텍스트 분할기 초기화"""
        try:
            # 재귀적 문자 분할기 (한국어에 최적화)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len
            )
            
            # 토큰 기반 분할기 (백업용)
            self.token_splitter = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            
            logger.info("텍스트 분할기 초기화 완료")
        except Exception as e:
            logger.error(f"텍스트 분할기 초기화 실패: {e}")
    
    def _initialize_vector_store(self):
        """벡터 스토어 초기화"""
        try:
            if self.vector_store_type == "chroma":
                # Chroma 벡터 스토어
                persist_directory = "./chroma_db"
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                
            elif self.vector_store_type == "faiss":
                # FAISS 벡터 스토어
                self.vector_store = FAISS.from_texts(
                    ["초기화 텍스트"],
                    self.embeddings
                )
            
            logger.info(f"벡터 스토어 초기화 완료: {self.vector_store_type}")
            
        except Exception as e:
            logger.error(f"벡터 스토어 초기화 실패: {e}")
    
    def add_hobby_documents(self, hobby_data: List[Dict[str, Any]]) -> bool:
        """취미 관련 문서를 벡터 스토어에 추가"""
        try:
            if not self.vector_store or not self.text_splitter:
                logger.error("벡터 스토어 또는 텍스트 분할기가 초기화되지 않았습니다.")
                return False
            
            documents = []
            
            for hobby in hobby_data:
                # 취미 정보를 텍스트로 변환
                hobby_text = self._hobby_to_text(hobby)
                
                # 텍스트 분할
                chunks = self.text_splitter.split_text(hobby_text)
                
                # 청크를 Document 객체로 변환
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "hobby_id": hobby.get("id", ""),
                            "hobby_name": hobby.get("name", ""),
                            "category": hobby.get("category", ""),
                            "difficulty": hobby.get("difficulty", ""),
                            "chunk_index": i,
                            "source": "hobby_database",
                            "added_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            # 벡터 스토어에 문서 추가
            if self.vector_store_type == "chroma":
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
            elif self.vector_store_type == "faiss":
                self.vector_store.add_documents(documents)
            
            logger.info(f"{len(documents)}개의 취미 문서 청크를 벡터 스토어에 추가했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"취미 문서 추가 실패: {e}")
            return False
    
    def _hobby_to_text(self, hobby: Dict[str, Any]) -> str:
        """취미 데이터를 텍스트로 변환"""
        text_parts = []
        
        if hobby.get("name"):
            text_parts.append(f"취미 이름: {hobby['name']}")
        
        if hobby.get("description"):
            text_parts.append(f"설명: {hobby['description']}")
        
        if hobby.get("category"):
            text_parts.append(f"카테고리: {hobby['category']}")
        
        if hobby.get("difficulty"):
            text_parts.append(f"난이도: {hobby['difficulty']}")
        
        if hobby.get("benefits"):
            text_parts.append(f"장점: {', '.join(hobby['benefits']) if isinstance(hobby['benefits'], list) else hobby['benefits']}")
        
        if hobby.get("requirements"):
            text_parts.append(f"필요 사항: {', '.join(hobby['requirements']) if isinstance(hobby['requirements'], list) else hobby['requirements']}")
        
        if hobby.get("tips"):
            text_parts.append(f"팁: {hobby['tips']}")
        
        return "\n".join(text_parts)
    
    def search_similar_hobbies(self, query: str, user_profile: Dict[str, Any], 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """사용자 프로필 기반 유사 취미 검색"""
        try:
            if not self.vector_store:
                logger.error("벡터 스토어가 초기화되지 않았습니다.")
                return []
            
            # 사용자 프로필을 쿼리에 포함
            enhanced_query = self._enhance_query_with_profile(query, user_profile)
            
            # 벡터 검색 수행
            if self.vector_store_type == "chroma":
                results = self.vector_store.similarity_search_with_score(
                    enhanced_query, 
                    k=top_k
                )
            elif self.vector_store_type == "faiss":
                results = self.vector_store.similarity_search_with_score(
                    enhanced_query, 
                    k=top_k
                )
            
            # 결과 처리
            processed_results = []
            for doc, score in results:
                result = {
                    "hobby_name": doc.metadata.get("hobby_name", ""),
                    "category": doc.metadata.get("category", ""),
                    "difficulty": doc.metadata.get("difficulty", ""),
                    "content": doc.page_content,
                    "similarity_score": float(score),
                    "metadata": doc.metadata
                }
                processed_results.append(result)
            
            # 유사도 점수로 정렬
            processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"취미 검색 완료: {len(processed_results)}개 결과")
            return processed_results
            
        except Exception as e:
            logger.error(f"취미 검색 실패: {e}")
            return []
    
    def _enhance_query_with_profile(self, query: str, user_profile: Dict[str, Any]) -> str:
        """사용자 프로필을 쿼리에 포함하여 검색 품질 향상"""
        enhanced_parts = [query]
        
        if user_profile.get("interests"):
            interests = ", ".join(user_profile["interests"])
            enhanced_parts.append(f"관심사: {interests}")
        
        if user_profile.get("skill_level"):
            enhanced_parts.append(f"기술 수준: {user_profile['skill_level']}")
        
        if user_profile.get("time_availability"):
            enhanced_parts.append(f"가용 시간: {user_profile['time_availability']}")
        
        if user_profile.get("budget_range"):
            enhanced_parts.append(f"예산: {user_profile['budget_range']}")
        
        return " ".join(enhanced_parts)
    
    def add_community_documents(self, community_data: List[Dict[str, Any]]) -> bool:
        """커뮤니티 관련 문서를 벡터 스토어에 추가"""
        try:
            if not self.vector_store or not self.text_splitter:
                return False
            
            documents = []
            
            for community in community_data:
                # 커뮤니티 정보를 텍스트로 변환
                community_text = self._community_to_text(community)
                
                # 텍스트 분할
                chunks = self.text_splitter.split_text(community_text)
                
                # 청크를 Document 객체로 변환
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "community_id": community.get("id", ""),
                            "community_name": community.get("name", ""),
                            "type": community.get("type", ""),
                            "category": community.get("category", ""),
                            "members": community.get("members", 0),
                            "chunk_index": i,
                            "source": "community_database",
                            "added_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            # 벡터 스토어에 문서 추가
            if self.vector_store_type == "chroma":
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
            elif self.vector_store_type == "faiss":
                self.vector_store.add_documents(documents)
            
            logger.info(f"{len(documents)}개의 커뮤니티 문서 청크를 벡터 스토어에 추가했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"커뮤니티 문서 추가 실패: {e}")
            return False
    
    def _community_to_text(self, community: Dict[str, Any]) -> str:
        """커뮤니티 데이터를 텍스트로 변환"""
        text_parts = []
        
        if community.get("name"):
            text_parts.append(f"커뮤니티 이름: {community['name']}")
        
        if community.get("description"):
            text_parts.append(f"설명: {community['description']}")
        
        if community.get("type"):
            text_parts.append(f"유형: {community['type']}")
        
        if community.get("category"):
            text_parts.append(f"카테고리: {community['category']}")
        
        if community.get("activities"):
            text_parts.append(f"활동: {', '.join(community['activities']) if isinstance(community['activities'], list) else community['activities']}")
        
        if community.get("requirements"):
            text_parts.append(f"참여 요건: {community['requirements']}")
        
        return "\n".join(text_parts)
    
    def search_communities(self, query: str, user_profile: Dict[str, Any], 
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """사용자 프로필 기반 커뮤니티 검색"""
        try:
            if not self.vector_store:
                return []
            
            # 사용자 프로필을 쿼리에 포함
            enhanced_query = self._enhance_query_with_profile(query, user_profile)
            
            # 벡터 검색 수행
            results = self.vector_store.similarity_search_with_score(
                enhanced_query, 
                k=top_k,
                filter={"source": "community_database"}
            )
            
            # 결과 처리
            processed_results = []
            for doc, score in results:
                result = {
                    "community_name": doc.metadata.get("community_name", ""),
                    "type": doc.metadata.get("type", ""),
                    "category": doc.metadata.get("category", ""),
                    "members": doc.metadata.get("members", 0),
                    "content": doc.page_content,
                    "similarity_score": float(score),
                    "metadata": doc.metadata
                }
                processed_results.append(result)
            
            # 유사도 점수로 정렬
            processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"커뮤니티 검색 실패: {e}")
            return []
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """벡터 스토어 통계 정보"""
        try:
            if not self.vector_store:
                return {"error": "벡터 스토어가 초기화되지 않았습니다."}
            
            if self.vector_store_type == "chroma":
                collection = self.vector_store._collection
                count = collection.count() if collection else 0
                
                return {
                    "type": "chroma",
                    "document_count": count,
                    "embedding_dimension": self.embeddings.client.get_sentence_embedding_dimension() if self.embeddings else 0,
                    "persist_directory": "./chroma_db"
                }
            
            elif self.vector_store_type == "faiss":
                return {
                    "type": "faiss",
                    "document_count": len(self.vector_store.docstore._dict),
                    "embedding_dimension": self.vector_store.embedding_function.client.get_sentence_embedding_dimension() if hasattr(self.vector_store.embedding_function, 'client') else 0
                }
            
            return {"error": "알 수 없는 벡터 스토어 타입"}
            
        except Exception as e:
            logger.error(f"벡터 스토어 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    def clear_vector_store(self) -> bool:
        """벡터 스토어 초기화"""
        try:
            if self.vector_store_type == "chroma":
                # Chroma DB 파일 삭제
                import shutil
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                
                # 새로운 벡터 스토어 생성
                self._initialize_vector_store()
                
            elif self.vector_store_type == "faiss":
                # FAISS 인덱스 초기화
                self._initialize_vector_store()
            
            logger.info("벡터 스토어 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"벡터 스토어 초기화 실패: {e}")
            return False

