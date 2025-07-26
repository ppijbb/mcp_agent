"""
LLM 응답 모델

LLM과의 상호작용을 위한 데이터 구조 정의
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import uuid


class ResponseStatus(Enum):
    """LLM 응답 상태"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_FORMAT = "invalid_format"
    PARTIAL = "partial"


class ResponseType(Enum):
    """LLM 응답 타입"""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class LLMResponse:
    """LLM 응답 기본 모델"""
    # 기본 정보
    response_id: str
    request_id: str
    model_name: str
    
    # 응답 내용
    content: str
    response_type: ResponseType = ResponseType.TEXT
    status: ResponseStatus = ResponseStatus.SUCCESS
    
    # 메타데이터
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    token_count: Optional[int] = None
    
    # 에러 정보
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # 추가 정보
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        """성공 여부 확인"""
        return self.status == ResponseStatus.SUCCESS
    
    def is_json(self) -> bool:
        """JSON 응답 여부 확인"""
        return self.response_type == ResponseType.JSON
    
    def get_json_content(self) -> Optional[Dict[str, Any]]:
        """JSON 내용 파싱"""
        if not self.is_json():
            return None
        
        try:
            return json.loads(self.content)
        except json.JSONDecodeError:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "response_id": self.response_id,
            "request_id": self.request_id,
            "model_name": self.model_name,
            "content": self.content,
            "response_type": self.response_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "token_count": self.token_count,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMResponse':
        """딕셔너리에서 생성"""
        return cls(
            response_id=data["response_id"],
            request_id=data["request_id"],
            model_name=data["model_name"],
            content=data["content"],
            response_type=ResponseType(data.get("response_type", "text")),
            status=ResponseStatus(data.get("status", "success")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            processing_time=data.get("processing_time"),
            token_count=data.get("token_count"),
            error_message=data.get("error_message"),
            error_code=data.get("error_code"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ParsedLLMResponse:
    """파싱된 LLM 응답"""
    # 원본 응답
    original_response: LLMResponse
    
    # 파싱된 데이터
    parsed_data: Dict[str, Any]
    parsing_success: bool = True
    parsing_errors: List[str] = field(default_factory=list)
    
    # 파싱 메타데이터
    parsing_time: Optional[float] = None
    parsing_method: Optional[str] = None
    
    def get_field(self, field_name: str, default: Any = None) -> Any:
        """특정 필드 값 조회"""
        return self.parsed_data.get(field_name, default)
    
    def has_field(self, field_name: str) -> bool:
        """필드 존재 여부 확인"""
        return field_name in self.parsed_data
    
    def get_all_fields(self) -> Dict[str, Any]:
        """모든 필드 반환"""
        return self.parsed_data.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "original_response": self.original_response.to_dict(),
            "parsed_data": self.parsed_data,
            "parsing_success": self.parsing_success,
            "parsing_errors": self.parsing_errors,
            "parsing_time": self.parsing_time,
            "parsing_method": self.parsing_method
        }


@dataclass
class LLMRequest:
    """LLM 요청 모델"""
    # 기본 정보
    request_id: str
    model_name: str
    
    # 요청 내용
    prompt: str
    system_message: Optional[str] = None
    
    # 요청 옵션
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    # 응답 형식
    response_format: Optional[str] = None
    expected_structure: Optional[Dict[str, Any]] = None
    
    # 메타데이터
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "system_message": self.system_message,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": self.response_format,
            "expected_structure": self.expected_structure,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMRequest':
        """딕셔너리에서 생성"""
        return cls(
            request_id=data["request_id"],
            model_name=data["model_name"],
            prompt=data["prompt"],
            system_message=data.get("system_message"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            frequency_penalty=data.get("frequency_penalty"),
            presence_penalty=data.get("presence_penalty"),
            response_format=data.get("response_format"),
            expected_structure=data.get("expected_structure"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            metadata=data.get("metadata", {})
        )


@dataclass
class LLMConversation:
    """LLM 대화 모델"""
    # 대화 정보
    conversation_id: str
    model_name: str
    
    # 메시지 히스토리
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # 대화 상태
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs):
        """메시지 추가"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def add_user_message(self, content: str, **kwargs):
        """사용자 메시지 추가"""
        self.add_message("user", content, **kwargs)
    
    def add_assistant_message(self, content: str, **kwargs):
        """어시스턴트 메시지 추가"""
        self.add_message("assistant", content, **kwargs)
    
    def add_system_message(self, content: str, **kwargs):
        """시스템 메시지 추가"""
        self.add_message("system", content, **kwargs)
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """메시지 목록 반환"""
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:]
    
    def clear_messages(self):
        """메시지 히스토리 초기화"""
        self.messages.clear()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "conversation_id": self.conversation_id,
            "model_name": self.model_name,
            "messages": self.messages,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConversation':
        """딕셔너리에서 생성"""
        return cls(
            conversation_id=data["conversation_id"],
            model_name=data["model_name"],
            messages=data.get("messages", []),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            metadata=data.get("metadata", {})
        )


# 편의 함수들
def create_llm_response(
    request_id: str,
    content: str,
    model_name: str = "unknown",
    response_type: ResponseType = ResponseType.TEXT,
    **kwargs
) -> LLMResponse:
    """LLM 응답 생성"""
    import uuid
    return LLMResponse(
        response_id=str(uuid.uuid4()),
        request_id=request_id,
        model_name=model_name,
        content=content,
        response_type=response_type,
        **kwargs
    )


def create_error_response(
    request_id: str,
    error_message: str,
    error_code: Optional[str] = None,
    model_name: str = "unknown"
) -> LLMResponse:
    """에러 응답 생성"""
    return LLMResponse(
        response_id=str(uuid.uuid4()),
        request_id=request_id,
        model_name=model_name,
        content="",
        response_type=ResponseType.ERROR,
        status=ResponseStatus.ERROR,
        error_message=error_message,
        error_code=error_code
    )


def parse_json_response(response: LLMResponse) -> ParsedLLMResponse:
    """JSON 응답 파싱"""
    import time
    start_time = time.time()
    
    try:
        if response.response_type != ResponseType.JSON:
            return ParsedLLMResponse(
                original_response=response,
                parsed_data={},
                parsing_success=False,
                parsing_errors=["응답이 JSON 형식이 아닙니다"]
            )
        
        parsed_data = json.loads(response.content)
        parsing_time = time.time() - start_time
        
        return ParsedLLMResponse(
            original_response=response,
            parsed_data=parsed_data,
            parsing_success=True,
            parsing_time=parsing_time,
            parsing_method="json.loads"
        )
        
    except json.JSONDecodeError as e:
        parsing_time = time.time() - start_time
        return ParsedLLMResponse(
            original_response=response,
            parsed_data={},
            parsing_success=False,
            parsing_errors=[f"JSON 파싱 실패: {str(e)}"],
            parsing_time=parsing_time,
            parsing_method="json.loads"
        )
    except Exception as e:
        parsing_time = time.time() - start_time
        return ParsedLLMResponse(
            original_response=response,
            parsed_data={},
            parsing_success=False,
            parsing_errors=[f"파싱 중 예외 발생: {str(e)}"],
            parsing_time=parsing_time,
            parsing_method="unknown"
        ) 