"""
Multi-Model Orchestration (혁신 3)

역할 기반 모델 선택, 동적 모델 전환, 비용 최적화, Weighted Ensemble을 통한
다중 모델 오케스트레이션 시스템.
"""

import asyncio
import time
import logging
import os
import json
import requests
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI

from researcher_config import get_llm_config, get_agent_config
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


class Provider(Enum):
    """LLM 제공자."""
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
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
        
        self.models: Dict[str, ModelConfig] = {}
        self.performance_tracker = ModelPerformanceTracker()
        self.model_clients: Dict[str, Any] = {}
        
        self._initialize_models()
        self._initialize_clients()
    
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
        
        # OpenRouter 무료 모델들 (프로토타이핑용)
        self.models["qwen2.5-vl-72b"] = ModelConfig(
            name="qwen2.5-vl-72b",
            provider="openrouter",
            model_id="qwen/qwen2.5-vl-72b-instruct:free",
            temperature=0.1,
            max_tokens=4000,
            cost_per_token=0.0,  # 무료
            speed_rating=7.0,
            quality_rating=8.5,
            capabilities=[TaskType.DEEP_REASONING, TaskType.ANALYSIS, TaskType.SYNTHESIS]
        )
        
        self.models["deepseek-chat-v3"] = ModelConfig(
            name="deepseek-chat-v3",
            provider="openrouter",
            model_id="deepseek/deepseek-chat-v3-0324:free",
            temperature=0.1,
            max_tokens=4000,
            cost_per_token=0.0,  # 무료
            speed_rating=8.0,
            quality_rating=8.0,
            capabilities=[TaskType.PLANNING, TaskType.GENERATION, TaskType.RESEARCH]
        )
        
        self.models["llama-3.3-70b"] = ModelConfig(
            name="llama-3.3-70b",
            provider="openrouter",
            model_id="meta-llama/llama-3.3-70b-instruct:free",
            temperature=0.1,
            max_tokens=4000,
            cost_per_token=0.0,  # 무료
            speed_rating=6.0,
            quality_rating=9.0,
            capabilities=[TaskType.DEEP_REASONING, TaskType.VERIFICATION, TaskType.SYNTHESIS]
        )
        
        self.models["gemma-2-27b"] = ModelConfig(
            name="gemma-2-27b",
            provider="openrouter",
            model_id="google/gemma-2-27b-it:free",
            temperature=0.1,
            max_tokens=4000,
            cost_per_token=0.0,  # 무료
            speed_rating=7.5,
            quality_rating=7.5,
            capabilities=[TaskType.PLANNING, TaskType.COMPRESSION, TaskType.RESEARCH]
        )
    
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
                    # OpenRouter 클라이언트
                    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                    if openrouter_api_key:
                        self.model_clients[model_name] = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=openrouter_api_key
                        )
                    else:
                        logger.warning(f"OpenRouter API key not found for {model_name}")
            
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
        
        # OpenRouter 무료 모델 우선 선택 (프로토타이핑용)
        openrouter_models = [
            name for name in suitable_models
            if self.models[name].provider == "openrouter" and self.models[name].cost_per_token == 0.0
        ]
        
        if openrouter_models:
            # 복잡도에 따른 OpenRouter 모델 선택
            if complexity > 7.0 and "qwen2.5-vl-72b" in openrouter_models:
                return "qwen2.5-vl-72b"
            elif complexity > 6.0 and "llama-3.3-70b" in openrouter_models:
                return "llama-3.3-70b"
            elif complexity > 5.0 and "deepseek-chat-v3" in openrouter_models:
                return "deepseek-chat-v3"
            else:
                return "gemma-2-27b" if "gemma-2-27b" in openrouter_models else openrouter_models[0]
        
        # OpenRouter 모델이 없으면 Google 모델 사용
        if complexity > 7.0 and "gemini-pro" in suitable_models:
            return "gemini-pro"
        elif complexity > 5.0 and "gemini-flash" in suitable_models:
            return "gemini-flash"
        else:
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
        
        # OpenRouter 모델의 경우 API 키가 없으면 Google 모델로 폴백
        if model_name not in self.model_clients:
            if self.models[model_name].provider == "openrouter":
                logger.warning(f"OpenRouter model {model_name} not available, falling back to Google model")
                # Google 모델로 폴백
                fallback_models = [
                    name for name, config in self.models.items()
                    if config.provider == "google" and task_type in config.capabilities
                ]
                if fallback_models:
                    model_name = fallback_models[0]
                else:
                    model_name = "gemini-flash-lite"
            else:
                raise ValueError(f"Model {model_name} not available")
        
        start_time = time.time()
        
        try:
            # 모델 실행
            if model_name.endswith("_langchain"):
                # LangChain 클라이언트 사용
                result = await self._execute_langchain_model(
                    model_name, prompt, system_message, **kwargs
                )
            elif self.models[model_name.replace("_langchain", "")].provider == "openrouter":
                # OpenRouter 클라이언트 사용
                result = await self._execute_openrouter_model(
                    model_name, prompt, system_message, **kwargs
                )
            else:
                # Google Generative AI 클라이언트 사용
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
        client = self.model_clients[model_name]
        model_config = self.models[model_name]
        
        # 메시지 구성
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # OpenRouter API 호출
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model_config.model_id,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    **kwargs
                )
            )
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {
                    "model": model_name,
                    "provider": "openrouter",
                    "model_id": model_config.model_id,
                    "tokens_used": len(content.split()),
                    "usage": response.usage.__dict__ if hasattr(response, 'usage') else {}
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


# Global orchestrator instance
llm_orchestrator = MultiModelOrchestrator()


async def execute_llm_task(
    prompt: str,
    task_type: TaskType,
    model_name: str = None,
    system_message: str = None,
    use_ensemble: bool = False,
    **kwargs
) -> ModelResult:
    """LLM 작업 실행."""
    return await execute_with_reliability(
        llm_orchestrator.execute_with_model if not use_ensemble else llm_orchestrator.weighted_ensemble,
        prompt,
        task_type,
        model_name,
        system_message,
        **kwargs,
        component_name="llm_execution",
        save_state=True
    )


def get_best_model_for_task(task_type: TaskType) -> str:
    """작업에 최적 모델 반환."""
    return llm_orchestrator.get_best_model_for_task(task_type)


def get_model_performance_stats() -> Dict[str, Any]:
    """모델 성능 통계 반환."""
    return llm_orchestrator.get_model_performance_stats()
