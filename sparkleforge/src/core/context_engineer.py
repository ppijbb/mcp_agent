"""
컨텍스트 엔지니어링 시스템

토큰 한계 내에서 입력·출력·도구 설명을 최적화하는 설계
컨텍스트 윈도우 내 토큰 수 제한 관리, 중요도 기반 컨텍스트 압축 및 요약
서브 에이전트 간 컨텍스트 교환 최적화
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """컨텍스트 우선순위."""
    CRITICAL = 5  # 반드시 보존해야 하는 정보
    HIGH = 4      # 중요한 정보
    MEDIUM = 3    # 보통 중요도
    LOW = 2       # 낮은 중요도
    OPTIONAL = 1  # 선택적 정보


class ContextType(Enum):
    """컨텍스트 유형."""
    SYSTEM_PROMPT = "system_prompt"
    USER_QUERY = "user_query"
    TOOL_RESULTS = "tool_results"
    AGENT_MEMORY = "agent_memory"
    CONVERSATION_HISTORY = "conversation_history"
    METADATA = "metadata"


@dataclass
class ContextChunk:
    """컨텍스트 청크."""
    content: str
    content_type: ContextType
    priority: ContextPriority
    timestamp: float
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)  # 의존하는 다른 청크들
    compressed: bool = False
    original_size: int = 0

    def __post_init__(self):
        self.original_size = len(self.content)
        self.token_count = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """토큰 수 추정 (대략적)."""
        # 간단한 추정: 영어 기준 단어 수의 1.3배
        word_count = len(self.content.split())
        return int(word_count * 1.3)

    def compress(self, max_tokens: int) -> 'ContextChunk':
        """청크 압축."""
        if self.token_count <= max_tokens or self.compressed:
            return self

        # 압축 로직
        compressed_content = self._compress_content(max_tokens)
        compressed_chunk = ContextChunk(
            content=compressed_content,
            content_type=self.content_type,
            priority=self.priority,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
            dependencies=self.dependencies.copy(),
            compressed=True,
            original_size=self.original_size
        )
        compressed_chunk.metadata['compression_ratio'] = len(compressed_content) / len(self.content)

        return compressed_chunk

    def _compress_content(self, max_tokens: int) -> str:
        """내용 압축."""
        if self.content_type == ContextType.TOOL_RESULTS:
            return self._compress_tool_results(max_tokens)
        elif self.content_type == ContextType.CONVERSATION_HISTORY:
            return self._compress_conversation(max_tokens)
        elif self.content_type == ContextType.AGENT_MEMORY:
            return self._compress_memory(max_tokens)
        else:
            # 기본 압축: 중요 부분 추출
            return self._extract_important_parts(max_tokens)

    def _compress_tool_results(self, max_tokens: int) -> str:
        """도구 결과 압축."""
        try:
            # JSON 파싱 시도
            data = json.loads(self.content)
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
                if isinstance(results, list):
                    # 상위 3개 결과만 유지
                    compressed_results = results[:3]
                    summary = f"총 {len(results)}개 결과 중 {len(compressed_results)}개 표시"
                    return json.dumps({
                        'results': compressed_results,
                        'summary': summary,
                        'total_count': len(results)
                    }, ensure_ascii=False)
        except:
            pass

        # 기본 압축
        return self._extract_important_parts(max_tokens)

    def _compress_conversation(self, max_tokens: int) -> str:
        """대화 히스토리 압축."""
        try:
            # JSON 파싱 시도
            messages = json.loads(self.content)
            if isinstance(messages, list):
                # 최근 5개 메시지만 유지
                recent_messages = messages[-5:] if len(messages) > 5 else messages
                summary = f"총 {len(messages)}개 메시지 중 최근 {len(recent_messages)}개 표시"
                return json.dumps({
                    'messages': recent_messages,
                    'summary': summary,
                    'total_count': len(messages)
                }, ensure_ascii=False)
        except:
            pass

        return self._extract_important_parts(max_tokens)

    def _compress_memory(self, max_tokens: int) -> str:
        """메모리 압축."""
        try:
            memory = json.loads(self.content)
            if isinstance(memory, dict):
                # 중요한 키만 유지
                important_keys = ['goals', 'findings', 'decisions', 'status']
                compressed_memory = {
                    k: v for k, v in memory.items()
                    if k in important_keys
                }
                return json.dumps(compressed_memory, ensure_ascii=False)
        except:
            pass

        return self._extract_important_parts(max_tokens)

    def _extract_important_parts(self, max_tokens: int) -> str:
        """중요 부분 추출."""
        content = self.content
        target_chars = max_tokens * 3  # 대략적인 문자 수

        if len(content) <= target_chars:
            return content

        # 처음과 끝 부분 유지 (중요한 정보가 양쪽에 있을 수 있음)
        start_chars = int(target_chars * 0.6)
        end_chars = target_chars - start_chars

        compressed = content[:start_chars] + "\n...\n" + content[-end_chars:]
        return compressed


@dataclass
class ContextWindowConfig:
    """컨텍스트 윈도우 설정."""
    max_tokens: int = 8000
    min_tokens: int = 1000
    system_prompt_tokens: int = 1000
    tool_descriptions_tokens: int = 1500
    conversation_tokens: int = 4000
    metadata_tokens: int = 500

    # 압축 설정
    enable_auto_compression: bool = True
    compression_threshold: float = 0.8  # 80% 사용 시 압축 시작
    importance_based_preservation: bool = True

    # 최적화 설정
    enable_token_estimation: bool = True
    adaptive_allocation: bool = True


class ContextEngineer:
    """
    컨텍스트 엔지니어링 시스템.

    토큰 한계를 효율적으로 관리하고, 중요도 기반으로 컨텍스트를 최적화.
    """

    def __init__(self, config: Optional[ContextWindowConfig] = None):
        """초기화."""
        self.config = config or ContextWindowConfig()
        self.context_chunks: List[ContextChunk] = []
        self.token_usage_history: List[Dict[str, Any]] = []
        self.compression_stats: Dict[str, int] = defaultdict(int)

        logger.info("ContextEngineer initialized with config: "
                   f"max_tokens={self.config.max_tokens}, "
                   f"auto_compression={self.config.enable_auto_compression}")

    async def add_context(
        self,
        content: str,
        content_type: ContextType,
        priority: ContextPriority = ContextPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Set[str]] = None
    ) -> str:
        """
        컨텍스트 추가.

        Args:
            content: 컨텍스트 내용
            content_type: 컨텍스트 유형
            priority: 우선순위
            metadata: 메타데이터
            dependencies: 의존 관계

        Returns:
            청크 ID
        """
        chunk_id = f"{content_type.value}_{int(time.time() * 1000)}_{len(self.context_chunks)}"

        chunk = ContextChunk(
            content=content,
            content_type=content_type,
            priority=priority,
            timestamp=time.time(),
            metadata=metadata or {},
            dependencies=dependencies or set()
        )

        self.context_chunks.append(chunk)
        logger.debug(f"Added context chunk: {chunk_id}, tokens={chunk.token_count}, type={content_type.value}")

        return chunk_id

    async def optimize_context(self, available_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        컨텍스트 최적화.

        Args:
            available_tokens: 사용 가능한 토큰 수 (None이면 설정값 사용)

        Returns:
            최적화 결과
        """
        max_tokens = available_tokens or self.config.max_tokens

        # 현재 토큰 사용량 계산
        total_tokens = sum(chunk.token_count for chunk in self.context_chunks)
        usage_ratio = total_tokens / max_tokens if max_tokens > 0 else 1.0

        logger.info(f"Context optimization: {total_tokens}/{max_tokens} tokens ({usage_ratio:.1%})")

        # 압축 필요 여부 확인
        if usage_ratio > self.config.compression_threshold and self.config.enable_auto_compression:
            await self._compress_context(max_tokens)

        # 최적화된 컨텍스트 생성
        optimized_context = await self._build_optimized_context(max_tokens)

        # 통계 기록
        stats = {
            'total_chunks': len(self.context_chunks),
            'total_tokens': total_tokens,
            'available_tokens': max_tokens,
            'usage_ratio': usage_ratio,
            'compressed_chunks': sum(1 for c in self.context_chunks if c.compressed),
            'optimization_applied': usage_ratio > self.config.compression_threshold
        }

        self.token_usage_history.append({
            'timestamp': time.time(),
            'stats': stats,
            'optimized_tokens': sum(len(c.content.split()) * 1.3 for c in optimized_context['chunks'])
        })

        logger.info(f"Context optimized: {stats['compressed_chunks']} chunks compressed, "
                   f"final ratio: {stats['usage_ratio']:.1%}")

        return {
            'chunks': optimized_context['chunks'],
            'stats': stats,
            'allocation': optimized_context['allocation']
        }

    async def _compress_context(self, max_tokens: int):
        """컨텍스트 압축."""
        logger.info("Starting context compression...")

        # 우선순위별 정렬 (낮은 우선순위부터 압축)
        sorted_chunks = sorted(
            self.context_chunks,
            key=lambda c: (c.priority.value, -c.timestamp)  # 우선순위 낮은 순, 최근 것 우선
        )

        target_tokens = int(max_tokens * 0.7)  # 70% 목표
        current_tokens = sum(c.token_count for c in self.context_chunks)

        for chunk in sorted_chunks:
            if current_tokens <= target_tokens:
                break

            if chunk.priority != ContextPriority.CRITICAL:  # CRITICAL은 압축하지 않음
                original_tokens = chunk.token_count
                compressed_chunk = chunk.compress(int(chunk.token_count * 0.5))  # 50% 압축

                if compressed_chunk.token_count < original_tokens:
                    # 청크 교체
                    idx = self.context_chunks.index(chunk)
                    self.context_chunks[idx] = compressed_chunk
                    current_tokens -= (original_tokens - compressed_chunk.token_count)

                    self.compression_stats['chunks_compressed'] += 1
                    self.compression_stats['tokens_saved'] += (original_tokens - compressed_chunk.token_count)

                    logger.debug(f"Compressed chunk: {original_tokens} -> {compressed_chunk.token_count} tokens")

        logger.info(f"Context compression completed: {self.compression_stats['chunks_compressed']} chunks compressed, "
                   f"{self.compression_stats['tokens_saved']} tokens saved")

    async def _build_optimized_context(self, max_tokens: int) -> Dict[str, Any]:
        """최적화된 컨텍스트 구축."""
        # 토큰 할당 계산
        allocation = self._calculate_token_allocation(max_tokens)

        # 유형별 청크 분류
        chunks_by_type = defaultdict(list)
        for chunk in self.context_chunks:
            chunks_by_type[chunk.content_type].append(chunk)

        # 각 유형별 최적화
        optimized_chunks = []

        for content_type, chunks in chunks_by_type.items():
            allocated_tokens = allocation.get(content_type.value, 0)
            if allocated_tokens > 0:
                selected_chunks = await self._select_chunks_for_type(chunks, allocated_tokens)
                optimized_chunks.extend(selected_chunks)

        # 우선순위별 정렬 (중요한 것 먼저)
        optimized_chunks.sort(key=lambda c: (-c.priority.value, -c.timestamp))

        return {
            'chunks': optimized_chunks,
            'allocation': allocation
        }

    def _calculate_token_allocation(self, max_tokens: int) -> Dict[str, int]:
        """토큰 할당 계산."""
        if not self.config.adaptive_allocation:
            # 고정 할당
            return {
                'system_prompt': self.config.system_prompt_tokens,
                'user_query': int(max_tokens * 0.05),
                'tool_results': int(max_tokens * 0.4),
                'agent_memory': int(max_tokens * 0.2),
                'conversation_history': int(max_tokens * 0.25),
                'metadata': self.config.metadata_tokens
            }

        # 적응형 할당
        allocation = {}

        # 시스템 프롬프트 고정
        allocation['system_prompt'] = min(self.config.system_prompt_tokens, int(max_tokens * 0.15))

        # 도구 설명 고정
        allocation['tool_descriptions'] = min(self.config.tool_descriptions_tokens, int(max_tokens * 0.2))

        # 남은 토큰 분배
        remaining_tokens = max_tokens - allocation['system_prompt'] - allocation['tool_descriptions']

        # 유형별 청크 수에 따라 동적 할당
        type_counts = defaultdict(int)
        for chunk in self.context_chunks:
            type_counts[chunk.content_type] += 1

        total_chunks = sum(type_counts.values())

        if total_chunks > 0:
            for content_type, count in type_counts.items():
                if content_type.value not in ['system_prompt', 'tool_descriptions']:
                    ratio = count / total_chunks
                    allocation[content_type.value] = int(remaining_tokens * ratio)

        return allocation

    async def _select_chunks_for_type(self, chunks: List[ContextChunk], allocated_tokens: int) -> List[ContextChunk]:
        """유형별 청크 선택."""
        if not chunks:
            return []

        # 우선순위별 정렬
        sorted_chunks = sorted(chunks, key=lambda c: (-c.priority.value, -c.timestamp))

        selected_chunks = []
        used_tokens = 0

        for chunk in sorted_chunks:
            if used_tokens + chunk.token_count <= allocated_tokens:
                selected_chunks.append(chunk)
                used_tokens += chunk.token_count
            elif chunk.priority == ContextPriority.CRITICAL:
                # CRITICAL은 강제 포함 (압축)
                compressed_chunk = chunk.compress(allocated_tokens - used_tokens)
                if compressed_chunk.token_count > 0:
                    selected_chunks.append(compressed_chunk)
                    used_tokens += compressed_chunk.token_count

        return selected_chunks

    async def exchange_context_with_sub_agent(
        self,
        sub_agent_id: str,
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        서브 에이전트와 컨텍스트 교환.

        Args:
            sub_agent_id: 서브 에이전트 ID
            shared_context: 공유할 컨텍스트

        Returns:
            교환 결과
        """
        # 공유 컨텍스트를 청크로 변환
        shared_chunks = []
        for key, value in shared_context.items():
            if isinstance(value, (str, int, float, bool)):
                content = str(value)
            elif isinstance(value, (list, dict)):
                content = json.dumps(value, ensure_ascii=False)
            else:
                continue

            chunk = ContextChunk(
                content=content,
                content_type=ContextType.AGENT_MEMORY,
                priority=ContextPriority.HIGH,
                timestamp=time.time(),
                metadata={'source': sub_agent_id, 'key': key}
            )
            shared_chunks.append(chunk)

        # 중복 제거 및 통합
        await self._merge_shared_chunks(shared_chunks)

        # 응답 컨텍스트 생성
        response_context = await self._generate_response_context()

        logger.info(f"Context exchanged with sub-agent {sub_agent_id}: "
                   f"{len(shared_chunks)} chunks received, "
                   f"{len(response_context.get('chunks', []))} chunks sent")

        return response_context

    async def _merge_shared_chunks(self, shared_chunks: List[ContextChunk]):
        """공유 청크 통합."""
        for shared_chunk in shared_chunks:
            # 중복 확인
            duplicate_found = False
            for existing_chunk in self.context_chunks:
                if (existing_chunk.content_type == shared_chunk.content_type and
                    existing_chunk.metadata.get('key') == shared_chunk.metadata.get('key') and
                    abs(existing_chunk.timestamp - shared_chunk.timestamp) < 60):  # 1분 내
                    # 최신 것으로 업데이트
                    if shared_chunk.timestamp > existing_chunk.timestamp:
                        idx = self.context_chunks.index(existing_chunk)
                        self.context_chunks[idx] = shared_chunk
                    duplicate_found = True
                    break

            if not duplicate_found:
                self.context_chunks.append(shared_chunk)

    async def _generate_response_context(self) -> Dict[str, Any]:
        """응답 컨텍스트 생성."""
        # 중요한 청크들만 선택
        important_chunks = [
            chunk for chunk in self.context_chunks
            if chunk.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]
        ]

        # 최근 청크 우선 (최대 10개)
        recent_chunks = sorted(important_chunks, key=lambda c: -c.timestamp)[:10]

        return {
            'chunks': recent_chunks,
            'summary': {
                'total_chunks': len(self.context_chunks),
                'shared_chunks': len(recent_chunks),
                'compression_applied': any(c.compressed for c in recent_chunks)
            }
        }

    def get_context_stats(self) -> Dict[str, Any]:
        """컨텍스트 통계."""
        type_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        total_tokens = 0
        compressed_count = 0

        for chunk in self.context_chunks:
            type_counts[chunk.content_type] += 1
            priority_counts[chunk.priority] += 1
            total_tokens += chunk.token_count
            if chunk.compressed:
                compressed_count += 1

        return {
            'total_chunks': len(self.context_chunks),
            'total_tokens': total_tokens,
            'compressed_chunks': compressed_count,
            'type_distribution': dict(type_counts),
            'priority_distribution': dict(priority_counts),
            'compression_stats': dict(self.compression_stats),
            'token_usage_history': self.token_usage_history[-10:]  # 최근 10개
        }

    async def cleanup_old_context(self, max_age_hours: int = 24):
        """오래된 컨텍스트 정리."""
        cutoff_time = time.time() - (max_age_hours * 3600)

        original_count = len(self.context_chunks)
        self.context_chunks = [
            chunk for chunk in self.context_chunks
            if chunk.timestamp > cutoff_time or chunk.priority == ContextPriority.CRITICAL
        ]

        removed_count = original_count - len(self.context_chunks)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old context chunks (>{max_age_hours}h)")

        return removed_count


# 전역 컨텍스트 엔지니어 인스턴스
_context_engineer = None

def get_context_engineer() -> ContextEngineer:
    """전역 컨텍스트 엔지니어 인스턴스 반환."""
    global _context_engineer
    if _context_engineer is None:
        _context_engineer = ContextEngineer()
    return _context_engineer

def set_context_engineer(engineer: ContextEngineer):
    """전역 컨텍스트 엔지니어 설정."""
    global _context_engineer
    _context_engineer = engineer
