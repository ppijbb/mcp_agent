"""
예측 데이터 모델
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from uuid import uuid4


class PredictionResult(Enum):
    """예측 결과"""
    PENDING = "pending"  # 대기 중
    CORRECT = "correct"  # 정확
    INCORRECT = "incorrect"  # 부정확
    PARTIAL = "partial"  # 부분 정확


@dataclass
class Prediction:
    """예측 데이터 모델"""
    
    prediction_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    battle_id: str = ""
    topic: str = ""
    
    # 예측 내용
    prediction_text: str = ""
    prediction_value: Optional[float] = None  # 수치 예측인 경우
    confidence: float = 0.0  # 0.0 ~ 1.0
    
    # 예측 근거
    reasoning: str = ""
    data_sources: List[str] = field(default_factory=list)
    
    # 결과
    actual_value: Optional[float] = None
    result: PredictionResult = PredictionResult.PENDING
    accuracy_score: float = 0.0  # 0.0 ~ 1.0
    
    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    evaluated_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    def calculate_accuracy(self) -> float:
        """정확도 계산"""
        if self.result == PredictionResult.PENDING:
            return 0.0
        
        if self.result == PredictionResult.CORRECT:
            return 1.0
        elif self.result == PredictionResult.INCORRECT:
            return 0.0
        elif self.result == PredictionResult.PARTIAL:
            # 부분 정확도 계산 (예: 수치 예측의 경우)
            if self.prediction_value is not None and self.actual_value is not None:
                diff = abs(self.prediction_value - self.actual_value)
                max_diff = max(abs(self.prediction_value), abs(self.actual_value), 1.0)
                return max(0.0, 1.0 - (diff / max_diff))
            return 0.5  # 기본값
        
        return 0.0
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "prediction_id": self.prediction_id,
            "user_id": self.user_id,
            "battle_id": self.battle_id,
            "topic": self.topic,
            "prediction_text": self.prediction_text,
            "prediction_value": self.prediction_value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "data_sources": self.data_sources,
            "actual_value": self.actual_value,
            "result": self.result.value,
            "accuracy_score": self.accuracy_score,
            "created_at": self.created_at.isoformat(),
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "metadata": self.metadata,
        }

