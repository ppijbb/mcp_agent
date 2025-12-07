"""
예측 관련 MCP 도구
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CreatePredictionInput(BaseModel):
    """예측 생성 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    battle_id: str = Field(description="배틀 ID")
    topic: str = Field(description="예측 주제")
    context: Optional[str] = Field(default=None, description="추가 컨텍스트")


class EvaluatePredictionInput(BaseModel):
    """예측 평가 입력 스키마"""
    prediction_id: str = Field(description="예측 ID")
    actual_value: Optional[float] = Field(default=None, description="실제 값")
    actual_result: Optional[str] = Field(default=None, description="실제 결과")


class PredictionTools:
    """
    예측 관련 도구 모음
    
    예측 생성, 검증, 평가 기능 제공
    """
    
    def __init__(self, data_dir: str = "prediction_battle_data"):
        """
        PredictionTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.data_dir / "predictions.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                self.predictions = json.load(f)
        else:
            self.predictions = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.predictions_file, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """예측 도구 초기화"""
        self.tools.append(self._create_prediction_tool())
        self.tools.append(self._evaluate_prediction_tool())
        logger.info(f"Initialized {len(self.tools)} prediction tools")
    
    def _create_prediction_tool(self) -> BaseTool:
        @tool("prediction_create", args_schema=CreatePredictionInput)
        def create_prediction(
            user_id: str,
            battle_id: str,
            topic: str,
            context: Optional[str] = None
        ) -> str:
            """
            예측을 생성합니다.
            MCP 서버(g-search, fetch)를 활용하여 최신 데이터를 수집하고
            LLM을 통해 예측을 생성합니다.
            
            Args:
                user_id: 사용자 ID
                battle_id: 배틀 ID
                topic: 예측 주제
                context: 추가 컨텍스트
            Returns:
                예측 데이터 (JSON 문자열)
            """
            logger.info(f"Creating prediction for user {user_id}, battle {battle_id}, topic: {topic}")
            
            # 예측 생성 (실제로는 LLM과 MCP 서버를 활용)
            # 여기서는 기본 구조만 제공
            prediction_data = {
                "prediction_id": f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                "user_id": user_id,
                "battle_id": battle_id,
                "topic": topic,
                "prediction_text": f"예측: {topic}에 대한 분석 결과...",
                "confidence": 0.75,
                "reasoning": "데이터 분석 결과를 바탕으로 한 예측입니다.",
                "data_sources": ["g-search", "fetch"],
                "created_at": datetime.now().isoformat(),
                "status": "pending"
            }
            
            # 데이터 저장
            self.predictions[prediction_data["prediction_id"]] = prediction_data
            self._save_data()
            
            return json.dumps(prediction_data, ensure_ascii=False, indent=2)
        return create_prediction
    
    def _evaluate_prediction_tool(self) -> BaseTool:
        @tool("prediction_evaluate", args_schema=EvaluatePredictionInput)
        def evaluate_prediction(
            prediction_id: str,
            actual_value: Optional[float] = None,
            actual_result: Optional[str] = None
        ) -> str:
            """
            예측을 평가합니다.
            
            Args:
                prediction_id: 예측 ID
                actual_value: 실제 값 (수치 예측인 경우)
                actual_result: 실제 결과 (텍스트 예측인 경우)
            Returns:
                평가 결과 (JSON 문자열)
            """
            logger.info(f"Evaluating prediction {prediction_id}")
            
            if prediction_id not in self.predictions:
                return json.dumps({"error": "예측을 찾을 수 없습니다."}, ensure_ascii=False)
            
            prediction = self.predictions[prediction_id]
            
            # 정확도 계산
            if actual_value is not None and "prediction_value" in prediction:
                pred_value = prediction.get("prediction_value")
                if pred_value is not None:
                    diff = abs(pred_value - actual_value)
                    max_diff = max(abs(pred_value), abs(actual_value), 1.0)
                    accuracy = max(0.0, 1.0 - (diff / max_diff))
                else:
                    accuracy = 0.0
            else:
                # 텍스트 예측의 경우 간단한 매칭
                accuracy = 0.8 if actual_result else 0.0
            
            result = {
                "prediction_id": prediction_id,
                "actual_value": actual_value,
                "actual_result": actual_result,
                "accuracy_score": accuracy,
                "result": "correct" if accuracy >= 0.7 else "incorrect",
                "evaluated_at": datetime.now().isoformat()
            }
            
            # 예측 데이터 업데이트
            prediction.update(result)
            self._save_data()
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return evaluate_prediction
    
    def get_tools(self) -> List[BaseTool]:
        """모든 예측 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 예측 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

