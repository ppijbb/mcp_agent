"""
LangChain 기반 대화 메모리 관리
ConversationBufferMemory를 활용한 대화 기록 관리
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class HSPMemoryManager:
    """HSP Agent 전용 메모리 관리자"""
    
    def __init__(self, max_token_limit: int = 4000):
        self.max_token_limit = max_token_limit
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=max_token_limit
        )
        
        # 장기 메모리 (요약 기반)
        self.summary_memory = ConversationSummaryMemory(
            llm=None,  # LLM은 나중에 설정
            memory_key="conversation_summary",
            return_messages=True
        )
        
        # 사용자별 메모리 저장소
        self.user_memories = {}
        
        # 취미 관련 컨텍스트 메모리
        self.hobby_context_memory = {}
        
        logger.info("HSP Memory Manager 초기화 완료")
    
    def add_user_message(self, user_id: str, message: str, context: Dict[str, Any] = None):
        """사용자 메시지 추가"""
        try:
            if user_id not in self.user_memories:
                self.user_memories[user_id] = ConversationBufferMemory(
                    memory_key="user_chat_history",
                    return_messages=True
                )
            
            # 메시지 생성
            human_message = HumanMessage(content=message)
            
            # 컨텍스트가 있으면 메시지에 포함
            if context:
                context_str = json.dumps(context, ensure_ascii=False)
                human_message.content = f"{message}\n\n컨텍스트: {context_str}"
            
            # 사용자별 메모리에 저장
            self.user_memories[user_id].chat_memory.add_user_message(human_message.content)
            
            # 전역 메모리에도 저장
            self.conversation_memory.chat_memory.add_user_message(human_message.content)
            
            logger.info(f"사용자 메시지 추가 완료: {user_id}")
            
        except Exception as e:
            logger.error(f"사용자 메시지 추가 실패: {e}")
    
    def add_ai_message(self, user_id: str, message: str, agent_type: str = "general"):
        """AI 응답 메시지 추가"""
        try:
            if user_id not in self.user_memories:
                self.user_memories[user_id] = ConversationBufferMemory(
                    memory_key="user_chat_history",
                    return_messages=True
                )
            
            # 에이전트 타입 정보를 포함한 메시지
            ai_message_content = f"[{agent_type}] {message}"
            
            # 사용자별 메모리에 저장
            self.user_memories[user_id].chat_memory.add_ai_message(ai_message_content)
            
            # 전역 메모리에도 저장
            self.conversation_memory.chat_memory.add_ai_message(ai_message_content)
            
            logger.info(f"AI 메시지 추가 완료: {user_id} - {agent_type}")
            
        except Exception as e:
            logger.error(f"AI 메시지 추가 실패: {e}")
    
    def get_user_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """사용자별 대화 기록 조회"""
        try:
            if user_id not in self.user_memories:
                return []
            
            memory = self.user_memories[user_id]
            messages = memory.chat_memory.messages
            
            # 최근 메시지들만 반환
            recent_messages = messages[-limit:] if len(messages) > limit else messages
            
            conversation_history = []
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    conversation_history.append({
                        "role": "user",
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat()
                    })
                elif isinstance(msg, AIMessage):
                    conversation_history.append({
                        "role": "assistant",
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return conversation_history
            
        except Exception as e:
            logger.error(f"대화 기록 조회 실패: {e}")
            return []
    
    def update_hobby_context(self, user_id: str, hobby_data: Dict[str, Any]):
        """취미 관련 컨텍스트 업데이트"""
        try:
            if user_id not in self.hobby_context_memory:
                self.hobby_context_memory[user_id] = {}
            
            # 기존 컨텍스트와 병합
            current_context = self.hobby_context_memory[user_id]
            current_context.update(hobby_data)
            
            # 타임스탬프 추가
            current_context["last_updated"] = datetime.now().isoformat()
            
            logger.info(f"취미 컨텍스트 업데이트 완료: {user_id}")
            
        except Exception as e:
            logger.error(f"취미 컨텍스트 업데이트 실패: {e}")
    
    def get_hobby_context(self, user_id: str) -> Dict[str, Any]:
        """취미 관련 컨텍스트 조회"""
        return self.hobby_context_memory.get(user_id, {})
    
    def create_memory_prompt(self, user_id: str, current_query: str) -> str:
        """메모리 기반 프롬프트 생성"""
        try:
            # 사용자별 대화 기록
            conversation_history = self.get_user_conversation_history(user_id, limit=5)
            
            # 취미 컨텍스트
            hobby_context = self.get_hobby_context(user_id)
            
            # 프롬프트 구성
            prompt_parts = [
                "=== 대화 기록 ===",
            ]
            
            for msg in conversation_history:
                role = "사용자" if msg["role"] == "user" else "AI"
                prompt_parts.append(f"{role}: {msg['content']}")
            
            if hobby_context:
                prompt_parts.append("\n=== 취미 컨텍스트 ===")
                prompt_parts.append(json.dumps(hobby_context, ensure_ascii=False, indent=2))
            
            prompt_parts.append(f"\n=== 현재 질문 ===")
            prompt_parts.append(current_query)
            
            prompt_parts.append("\n위의 대화 기록과 컨텍스트를 고려하여 답변해주세요.")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"메모리 프롬프트 생성 실패: {e}")
            return current_query
    
    def clear_user_memory(self, user_id: str):
        """사용자별 메모리 초기화"""
        try:
            if user_id in self.user_memories:
                del self.user_memories[user_id]
            
            if user_id in self.hobby_context_memory:
                del self.hobby_context_memory[user_id]
            
            logger.info(f"사용자 메모리 초기화 완료: {user_id}")
            
        except Exception as e:
            logger.error(f"사용자 메모리 초기화 실패: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 사용 통계"""
        try:
            total_users = len(self.user_memories)
            total_contexts = len(self.hobby_context_memory)
            
            # 전역 메모리 크기
            global_messages = len(self.conversation_memory.chat_memory.messages)
            
            return {
                "total_users": total_users,
                "total_contexts": total_contexts,
                "global_messages": global_messages,
                "max_token_limit": self.max_token_limit
            }
            
        except Exception as e:
            logger.error(f"메모리 통계 조회 실패: {e}")
            return {}
