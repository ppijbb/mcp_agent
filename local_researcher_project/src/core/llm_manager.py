"""
Multi-Model Orchestration (혁신 3)

역할 기반 모델 선택, 동적 모델 전환, 비용 최적화, Weighted Ensemble을 통한
다중 모델 오케스트레이션 시스템.
"""

import asyncio
import time
import logging
import requests
import os
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.researcher_config import get_llm_config, get_agent_config
from src.core.reliability import execute_with_reliability

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """작업 유형."""
    PLANNING = "planning"
    DEEP_REASONING = "deep_reasoning"
    VERIFICATION = "verification"
    GENERATION = "generation"
    COMPRESSION = "compression"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CREATIVE = "creative"


class Provider(Enum):
    """LLM 제공자."""
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """모델 설정."""
    name: str
    provider: str
    model_id: str
    temperature: float
    max_tokens: int
    cost_per_token: float
    speed_rating: float  # 1-10, 높을수록 빠름
    quality_rating: float  # 1-10, 높을수록 품질 좋음
    capabilities: List[TaskType]


@dataclass
class ModelResult:
    """모델 실행 결과."""
    content: str
    model_used: str
    execution_time: float
    confidence: float
    cost: float
    metadata: Dict[str, Any] = None


class ModelPerformanceTracker:
    """모델 성능 추적기."""
    
    def __init__(self):
        self.performance_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_execution(
        self,
        model_name: str,
        task_type: TaskType,
        execution_time: float,
        success: bool,
        quality_score: float = None
    ):
        """실행 기록."""
        if model_name not in self.performance_stats:
            self.performance_stats[model_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "avg_quality": 0.0,
                "task_performance": {}
            }
        
        stats = self.performance_stats[model_name]
        stats["total_executions"] += 1
        stats["total_time"] += execution_time
        
        if success:
            stats["successful_executions"] += 1
        
        if quality_score is not None:
            current_avg = stats["avg_quality"]
            total = stats["successful_executions"]
            stats["avg_quality"] = (current_avg * (total - 1) + quality_score) / total
        
        # 작업별 성능 추적
        task_key = task_type.value
        if task_key not in stats["task_performance"]:
            stats["task_performance"][task_key] = {
                "executions": 0,
                "successes": 0,
                "avg_time": 0.0,
                "avg_quality": 0.0
            }
        
        task_stats = stats["task_performance"][task_key]
        task_stats["executions"] += 1
        if success:
            task_stats["successes"] += 1
        
        # 평균 시간 업데이트
        current_avg_time = task_stats["avg_time"]
        task_stats["avg_time"] = (current_avg_time * (task_stats["executions"] - 1) + execution_time) / task_stats["executions"]
        
        # 평균 품질 업데이트
        if quality_score is not None and success:
            current_avg_quality = task_stats["avg_quality"]
            task_stats["avg_quality"] = (current_avg_quality * (task_stats["successes"] - 1) + quality_score) / task_stats["successes"]
    
    def get_model_score(self, model_name: str, task_type: TaskType = None) -> float:
        """모델 점수 반환."""
        if model_name not in self.performance_stats:
            return 0.0
        
        stats = self.performance_stats[model_name]
        
        if task_type:
            task_key = task_type.value
            if task_key not in stats["task_performance"]:
                return 0.0
            
            task_stats = stats["task_performance"][task_key]
            if task_stats["executions"] == 0:
                return 0.0
            
            success_rate = task_stats["successes"] / task_stats["executions"]
            avg_quality = task_stats["avg_quality"]
            speed_score = 1.0 / (1.0 + task_stats["avg_time"])  # 시간이 짧을수록 높은 점수
            
            return (success_rate * 0.4 + avg_quality * 0.4 + speed_score * 0.2)
        else:
            # 전체 성능 점수
            if stats["total_executions"] == 0:
                return 0.0
            
            success_rate = stats["successful_executions"] / stats["total_executions"]
            avg_quality = stats["avg_quality"]
            avg_time = stats["total_time"] / stats["total_executions"]
            speed_score = 1.0 / (1.0 + avg_time)
            
            return (success_rate * 0.4 + avg_quality * 0.4 + speed_score * 0.2)


class MultiModelOrchestrator:
    """다중 모델 오케스트레이터 (혁신 3)."""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        
        # Provider별 API 키 검증
        self._validate_provider_config()
        
        self.models: Dict[str, ModelConfig] = {}
        self.performance_tracker = ModelPerformanceTracker()
        self.model_clients: Dict[str, Any] = {}
        
        self._initialize_models()
        self._initialize_clients()
    
    def _validate_provider_config(self):
        """Provider별 API 키 검증."""
        if self.llm_config.provider == "openrouter":
            if not os.getenv("OPENROUTER_API_KEY"):
                raise ValueError("OPENROUTER_API_KEY is required for OpenRouter LLM provider")
        elif self.llm_config.provider == "google":
            if not self.llm_config.api_key:
                raise ValueError("GOOGLE_API_KEY is required for Google LLM provider")
    
    def _initialize_models(self):
        """모델 초기화."""
        # Gemini 2.5 Flash Lite (빠른 계획, 압축)
        self.models["gemini-flash-lite"] = ModelConfig(
            name="gemini-flash-lite",
            provider="google",
            model_id="gemini-2.5-flash-lite",
            temperature=0.1,
            max_tokens=4000,
            cost_per_token=0.0001,
            speed_rating=9.0,
            quality_rating=7.0,
            capabilities=[TaskType.PLANNING, TaskType.COMPRESSION, TaskType.RESEARCH]
        )
        
        # Gemini 2.5 Pro (복잡한 추론, 분석)
        self.models["gemini-pro"] = ModelConfig(
            name="gemini-pro",
            provider="google",
            model_id="gemini-2.5-pro",
            temperature=0.2,
            max_tokens=8000,
            cost_per_token=0.0005,
            speed_rating=6.0,
            quality_rating=9.0,
            capabilities=[TaskType.DEEP_REASONING, TaskType.ANALYSIS, TaskType.SYNTHESIS]
        )
        
        # Gemini 2.5 Flash (균형잡힌 성능)
        self.models["gemini-flash"] = ModelConfig(
            name="gemini-flash",
            provider="google",
            model_id="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=4000,
            cost_per_token=0.0002,
            speed_rating=8.0,
            quality_rating=8.0,
            capabilities=[TaskType.GENERATION, TaskType.VERIFICATION, TaskType.RESEARCH]
        )
        
        # OpenRouter 모델 로딩 비활성화 (Gemini만 사용)
        # OpenRouter API 키가 있으면 로드 시도, 없으면 무시
        if os.getenv("OPENROUTER_API_KEY"):
            try:
                self._load_openrouter_models()
            except Exception as e:
                logger.warning(f"OpenRouter models not loaded (Gemini will be used instead): {e}")
        else:
            logger.info("OpenRouter disabled - using Gemini only")
    
    def _load_openrouter_models(self):
        """OpenRouter API에서 무료 모델들을 동적으로 로드 (선택적)."""
        try:
            openrouter_models = self._fetch_openrouter_models()
            for model_data in openrouter_models:
                if self._is_free_model(model_data):
                    model_name = self._generate_model_name(model_data)
                    model_config = self._create_model_config(model_data, model_name)
                    self.models[model_name] = model_config
                    logger.info(f"Loaded OpenRouter model: {model_name} ({model_data['id']})")
        except Exception as e:
            logger.warning(f"Failed to load OpenRouter models: {e}")
            # 예외를 raise하지 않음 - Gemini만 사용하도록 함
    
    def _fetch_openrouter_models(self):
        """OpenRouter API에서 모델 목록을 가져옴."""
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
    
    def _is_free_model(self, model_data):
        """모델이 무료인지 확인."""
        pricing = model_data.get("pricing", {})
        prompt_price = pricing.get("prompt", "0")
        completion_price = pricing.get("completion", "0")
        
        # 무료 모델 조건: prompt와 completion 가격이 모두 0이거나 매우 낮음
        try:
            prompt_cost = float(prompt_price) if prompt_price else 0
            completion_cost = float(completion_price) if completion_price else 0
            return prompt_cost == 0 and completion_cost == 0
        except (ValueError, TypeError):
            return False
    
    def _generate_model_name(self, model_data):
        """모델 데이터에서 고유한 이름 생성."""
        model_id = model_data.get("id", "")
        # "provider/model-name:tag" 형식을 "provider-model-name-tag"로 변환
        name = model_id.replace("/", "-").replace(":", "-").replace("_", "-")
        return name
    
    def _create_model_config(self, model_data, model_name):
        """모델 데이터에서 ModelConfig 생성."""
        model_id = model_data.get("id", "")
        context_length = model_data.get("context_length", 4000)
        
        # 모델별 기본 설정
        capabilities = self._determine_capabilities(model_id)
        speed_rating = self._estimate_speed_rating(model_id)
        quality_rating = self._estimate_quality_rating(model_id)
        
        return ModelConfig(
            name=model_name,
            provider="openrouter",
            model_id=model_id,
            temperature=0.1,
            max_tokens=min(context_length, 4000),
            cost_per_token=0.0,  # 무료 모델
            speed_rating=speed_rating,
            quality_rating=quality_rating,
            capabilities=capabilities
        )
    
    def _determine_capabilities(self, model_id):
        """모델 ID를 기반으로 capabilities 결정."""
        capabilities = [TaskType.GENERATION]  # 기본
        
        # 모델명에 따른 capabilities 추가
        if any(keyword in model_id.lower() for keyword in ["reasoning", "reason", "think"]):
            capabilities.extend([TaskType.DEEP_REASONING, TaskType.ANALYSIS])
        
        if any(keyword in model_id.lower() for keyword in ["code", "coder", "programming"]):
            capabilities.extend([TaskType.RESEARCH, TaskType.COMPRESSION])
        
        if any(keyword in model_id.lower() for keyword in ["verify", "check", "validate"]):
            capabilities.extend([TaskType.VERIFICATION])
        
        if any(keyword in model_id.lower() for keyword in ["plan", "planning", "strategy"]):
            capabilities.append(TaskType.PLANNING)
        
        # 기본적으로 모든 작업 가능
        if not any(keyword in model_id.lower() for keyword in ["reasoning", "code", "verify", "plan"]):
            capabilities = [TaskType.PLANNING, TaskType.DEEP_REASONING, TaskType.VERIFICATION, 
                          TaskType.GENERATION, TaskType.COMPRESSION, TaskType.RESEARCH]
        
        return list(set(capabilities))  # 중복 제거
    
    def _estimate_speed_rating(self, model_id):
        """모델 ID를 기반으로 속도 등급 추정."""
        if any(keyword in model_id.lower() for keyword in ["flash", "fast", "lite", "small"]):
            return 8.0
        elif any(keyword in model_id.lower() for keyword in ["large", "big", "70b", "72b"]):
            return 6.0
        else:
            return 7.0
    
    def _estimate_quality_rating(self, model_id):
        """모델 ID를 기반으로 품질 등급 추정."""
        if any(keyword in model_id.lower() for keyword in ["70b", "72b", "large", "pro"]):
            return 9.0
        elif any(keyword in model_id.lower() for keyword in ["27b", "medium"]):
            return 7.5
        else:
            return 8.0
    
    
    def refresh_openrouter_models(self):
        """OpenRouter 모델 목록을 새로고침."""
        logger.info("Refreshing OpenRouter models...")
        # 기존 OpenRouter 모델들 제거
        openrouter_models = [name for name, config in self.models.items() 
                           if config.provider == "openrouter"]
        for model_name in openrouter_models:
            del self.models[model_name]
        
        # 새로 로드
        self._load_openrouter_models()
        logger.info(f"Refreshed OpenRouter models: {len([name for name, config in self.models.items() if config.provider == 'openrouter'])} models loaded")
    
    def _initialize_clients(self):
        """모델 클라이언트 초기화."""
        try:
            genai.configure(api_key=self.llm_config.api_key)
            
            for model_name, model_config in self.models.items():
                if model_config.provider == "google":
                    # Google Generative AI 클라이언트
                    self.model_clients[model_name] = genai.GenerativeModel(model_config.model_id)
                    
                    # LangChain 클라이언트 (선택적)
                    self.model_clients[f"{model_name}_langchain"] = ChatGoogleGenerativeAI(
                        model=model_config.model_id,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens,
                        google_api_key=self.llm_config.api_key
                    )
                
                elif model_config.provider == "openrouter":
                    # OpenRouter 클라이언트는 HTTP 요청으로 직접 처리
                    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                    if not openrouter_api_key:
                        raise ValueError(f"OpenRouter API key not found for {model_name}")
                    # OpenRouter는 HTTP 요청으로 직접 처리하므로 클라이언트 저장하지 않음
                    logger.info(f"OpenRouter model {model_name} configured for HTTP requests")
            
            logger.info("Model clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model clients: {e}")
            raise
    
    def select_model(
        self,
        task_type: TaskType,
        complexity: float = 5.0,
        budget: float = None
    ) -> str:
        """작업에 최적 모델 선택."""
        if budget is None:
            budget = self.llm_config.budget_limit
        
        # 작업 유형에 적합한 모델 필터링
        suitable_models = [
            name for name, config in self.models.items()
            if task_type in config.capabilities
        ]
        
        if not suitable_models:
            # 기본 모델 사용
            return "gemini-flash-lite"
        
        # OpenRouter 제외 - Gemini만 사용
        gemini_models = [
            name for name in suitable_models
            if self.models[name].provider == "google"
        ]
        
        # Gemini 모델 우선 사용
        if gemini_models:
            if complexity > 7.0 and "gemini-pro" in gemini_models:
                return "gemini-pro"
            elif complexity > 5.0 and "gemini-flash" in gemini_models:
                return "gemini-flash"
            else:
                # gemini-flash-lite 또는 첫 번째 Gemini 모델
                return "gemini-flash-lite" if "gemini-flash-lite" in gemini_models else gemini_models[0]
        
        # Gemini 모델이 없으면 기본 모델 사용
        return "gemini-flash-lite"
    
    async def execute_with_model(
        self,
        prompt: str,
        task_type: TaskType,
        model_name: str = None,
        system_message: str = None,
        **kwargs
    ) -> ModelResult:
        """모델로 실행."""
        if model_name is None:
            model_name = self.select_model(task_type)
        
        # 모델 클라이언트 확인
        model_name_clean = model_name.replace("_langchain", "")
        
        # OpenRouter 모델이 선택되었으면 Gemini로 폴백
        if model_name_clean in self.models and self.models[model_name_clean].provider == "openrouter":
            logger.warning(f"OpenRouter model {model_name} selected, falling back to Gemini")
            # Gemini 모델로 변경
            if "gemini-flash-lite" in self.model_clients:
                model_name = "gemini-flash-lite"
            elif "gemini-flash" in self.model_clients:
                model_name = "gemini-flash"
            elif "gemini-pro" in self.model_clients:
                model_name = "gemini-pro"
            else:
                # 사용 가능한 첫 번째 Gemini 모델
                gemini_models = [name for name in self.model_clients.keys() if "gemini" in name.lower()]
                if gemini_models:
                    model_name = gemini_models[0]
                else:
                    raise ValueError(f"No Gemini models available (OpenRouter model {model_name_clean} was requested)")
            model_name_clean = model_name
        
        if model_name not in self.model_clients:
            raise ValueError(f"Model {model_name} not available")
        
        start_time = time.time()
        
        try:
            # 모델 실행 (OpenRouter는 이미 폴백 처리됨)
            if model_name.endswith("_langchain"):
                # LangChain 클라이언트 사용
                result = await self._execute_langchain_model(
                    model_name, prompt, system_message, **kwargs
                )
            else:
                # Google Generative AI 클라이언트 사용 (OpenRouter는 폴백됨)
                result = await self._execute_gemini_model(
                    model_name, prompt, system_message, **kwargs
                )
            
            execution_time = time.time() - start_time
            
            # 비용 계산
            model_config = self.models[model_name.replace("_langchain", "")]
            cost = len(prompt.split()) * model_config.cost_per_token
            
            # 성능 기록
            self.performance_tracker.record_execution(
                model_name, task_type, execution_time, True, result.get("quality_score", 0.8)
            )
            
            return ModelResult(
                content=result["content"],
                model_used=model_name,
                execution_time=execution_time,
                confidence=result.get("confidence", 0.8),
                cost=cost,
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Model execution failed: {e}")
            
            # 실패 기록
            self.performance_tracker.record_execution(
                model_name, task_type, execution_time, False
            )
            
            raise
    
    async def _execute_gemini_model(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Gemini 모델 실행."""
        client = self.model_clients[model_name]
        model_config = self.models[model_name]
        
        # 프롬프트 구성
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        # 실행
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=model_config.temperature,
                    max_output_tokens=model_config.max_tokens
                )
            )
        )
        
        return {
            "content": response.text,
            "confidence": 0.8,  # 기본 신뢰도
            "quality_score": 0.8,
            "metadata": {
                "model": model_name,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens
            }
        }
    
    async def _execute_openrouter_model(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """OpenRouter 모델 실행."""
        model_config = self.models[model_name]
        
        # 메시지 구성
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # OpenRouter API 직접 호출
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mcp-agent.local",
            "X-Title": "MCP Agent Hub"
        }
        
        payload = {
            "model": model_config.model_id,
            "messages": messages,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            **kwargs
        }
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            return {
                "content": content,
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "provider": "openrouter",
                    "model_id": model_config.model_id,
                    "tokens_used": len(content.split()),
                    "usage": data.get("usage", {})
                }
            }
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise RuntimeError(f"OpenRouter model {model_name} failed: {e}")
    
    async def _execute_langchain_model(
        self,
        model_name: str,
        prompt: str,
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """LangChain 모델 실행."""
        client = self.model_clients[model_name]
        
        # 메시지 구성
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        # 실행
        response = await client.ainvoke(messages)
        
        return {
            "content": response.content,
            "confidence": 0.8,
            "quality_score": 0.8,
            "metadata": {
                "model": model_name,
                "response_type": type(response).__name__
            }
        }
    
    async def weighted_ensemble(
        self,
        prompt: str,
        task_type: TaskType,
        models: List[str] = None,
        weights: List[float] = None
    ) -> ModelResult:
        """Weighted Ensemble 실행."""
        if models is None:
            models = [self.select_model(task_type) for _ in range(3)]
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # 모든 모델로 실행
        results = []
        for model in models:
            try:
                result = await self.execute_with_model(prompt, task_type, model)
                results.append(result)
            except Exception as e:
                logger.warning(f"Model {model} failed in ensemble: {e}")
                continue
        
        if not results:
            raise RuntimeError("All models failed in ensemble")
        
        # 가중 평균으로 결과 통합
        total_weight = sum(weights[:len(results)])
        weighted_content = ""
        total_confidence = 0.0
        total_cost = 0.0
        total_time = 0.0
        
        for i, result in enumerate(results):
            weight = weights[i] / total_weight
            weighted_content += f"[{result.model_used}] {result.content}\n\n"
            total_confidence += result.confidence * weight
            total_cost += result.cost * weight
            total_time += result.execution_time * weight
        
        return ModelResult(
            content=weighted_content.strip(),
            model_used=f"ensemble({','.join([r.model_used for r in results])})",
            execution_time=total_time,
            confidence=total_confidence,
            cost=total_cost,
            metadata={
                "ensemble_models": [r.model_used for r in results],
                "weights": weights[:len(results)],
                "individual_results": [
                    {
                        "model": r.model_used,
                        "confidence": r.confidence,
                        "cost": r.cost
                    }
                    for r in results
                ]
            }
        )
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """모델 성능 통계 반환."""
        stats = {}
        for model_name in self.models:
            stats[model_name] = {
                "overall_score": self.performance_tracker.get_model_score(model_name),
                "task_scores": {
                    task_type.value: self.performance_tracker.get_model_score(model_name, task_type)
                    for task_type in TaskType
                }
            }
        return stats
    
    def get_best_model_for_task(self, task_type: TaskType) -> str:
        """작업에 최적 모델 반환."""
        best_model = None
        best_score = 0.0
        
        for model_name in self.models:
            score = self.performance_tracker.get_model_score(model_name, task_type)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model or "gemini-flash-lite"


# Global orchestrator instance (lazy initialization)
_llm_orchestrator = None

def get_llm_orchestrator() -> 'MultiModelOrchestrator':
    """Get or initialize global LLM orchestrator."""
    global _llm_orchestrator
    if _llm_orchestrator is None:
        _llm_orchestrator = MultiModelOrchestrator()
    return _llm_orchestrator


async def execute_llm_task(
    prompt: str,
    task_type: TaskType,
    model_name: str = None,
    system_message: str = None,
    use_ensemble: bool = False,
    **kwargs
) -> ModelResult:
    """LLM 작업 실행."""
    try:
        llm_orchestrator = get_llm_orchestrator()
        if use_ensemble:
            return await llm_orchestrator.weighted_ensemble(
                prompt, task_type, model_name, system_message, **kwargs
            )
        else:
            return await llm_orchestrator.execute_with_model(
                prompt, task_type, model_name, system_message, **kwargs
            )
    except Exception as e:
        logger.error(f"LLM task execution failed: {e}")
        raise


def get_best_model_for_task(task_type: TaskType) -> str:
    """작업에 최적 모델 반환."""
    llm_orchestrator = get_llm_orchestrator()
    return llm_orchestrator.get_best_model_for_task(task_type)


def get_model_performance_stats() -> Dict[str, Any]:
    """모델 성능 통계 반환."""
    llm_orchestrator = get_llm_orchestrator()
    return llm_orchestrator.get_model_performance_stats()
