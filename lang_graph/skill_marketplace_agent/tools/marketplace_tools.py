"""
Marketplace 도구

강사 검색, 매칭, 결제, 수수료 계산 등 양면 시장 기능
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchInstructorsInput(BaseModel):
    """강사 검색 입력 스키마"""
    skill: str = Field(description="검색할 스킬")
    min_rating: Optional[float] = Field(default=0.0, ge=0.0, le=5.0, description="최소 평점")
    max_price: Optional[float] = Field(default=None, description="최대 수업료 (시간당)")
    location: Optional[str] = Field(default=None, description="위치 (온라인/오프라인)")


class MatchInstructorInput(BaseModel):
    """강사 매칭 입력 스키마"""
    learner_id: str = Field(description="학습자 ID")
    skill: str = Field(description="학습할 스킬")
    learner_level: str = Field(description="학습자 현재 수준 (beginner/intermediate/advanced)")
    budget: Optional[float] = Field(default=None, description="예산 (시간당)")
    preferred_time: Optional[str] = Field(default=None, description="선호 시간대")


class CalculateCommissionInput(BaseModel):
    """수수료 계산 입력 스키마"""
    transaction_amount: float = Field(description="거래 금액")
    commission_rate: Optional[float] = Field(default=0.15, ge=0.0, le=1.0, description="수수료율 (기본 15%)")


class ProcessPaymentInput(BaseModel):
    """결제 처리 입력 스키마"""
    learner_id: str = Field(description="학습자 ID")
    instructor_id: str = Field(description="강사 ID")
    amount: float = Field(description="결제 금액")
    session_id: str = Field(description="학습 세션 ID")


class CreateLearningSessionInput(BaseModel):
    """학습 세션 생성 입력 스키마"""
    learner_id: str = Field(description="학습자 ID")
    instructor_id: str = Field(description="강사 ID")
    skill: str = Field(description="학습할 스킬")
    scheduled_time: str = Field(description="예정 시간 (ISO format)")
    duration_hours: float = Field(description="수업 시간 (시간)")


class MarketplaceTools:
    """
    Marketplace 도구 모음
    
    강사 검색, 매칭, 결제, 수수료 계산 등 양면 시장 기능
    """
    
    def __init__(self, data_dir: str = "marketplace_data"):
        """
        MarketplaceTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.instructors_file = self.data_dir / "instructors.json"
        self.sessions_file = self.data_dir / "sessions.json"
        self.transactions_file = self.data_dir / "transactions.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.instructors_file.exists():
            with open(self.instructors_file, 'r', encoding='utf-8') as f:
                self.instructors = json.load(f)
        else:
            self.instructors = {}
        
        if self.sessions_file.exists():
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                self.sessions = json.load(f)
        else:
            self.sessions = {}
        
        if self.transactions_file.exists():
            with open(self.transactions_file, 'r', encoding='utf-8') as f:
                self.transactions = json.load(f)
        else:
            self.transactions = {}
    
    def _save_instructors(self):
        """강사 데이터 저장"""
        with open(self.instructors_file, 'w', encoding='utf-8') as f:
            json.dump(self.instructors, f, indent=2, ensure_ascii=False)
    
    def _save_sessions(self):
        """세션 데이터 저장"""
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(self.sessions, f, indent=2, ensure_ascii=False)
    
    def _save_transactions(self):
        """거래 데이터 저장"""
        with open(self.transactions_file, 'w', encoding='utf-8') as f:
            json.dump(self.transactions, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Marketplace 도구 초기화"""
        self.tools.append(self._create_search_instructors_tool())
        self.tools.append(self._create_match_instructor_tool())
        self.tools.append(self._create_calculate_commission_tool())
        self.tools.append(self._create_process_payment_tool())
        self.tools.append(self._create_create_learning_session_tool())
        logger.info(f"Initialized {len(self.tools)} marketplace tools")
    
    def _create_search_instructors_tool(self) -> BaseTool:
        @tool("marketplace_search_instructors", args_schema=SearchInstructorsInput)
        def search_instructors(
            skill: str,
            min_rating: Optional[float] = 0.0,
            max_price: Optional[float] = None,
            location: Optional[str] = None
        ) -> str:
            """
            강사를 검색합니다.
            Args:
                skill: 검색할 스킬
                min_rating: 최소 평점
                max_price: 최대 수업료
                location: 위치
            Returns:
                검색된 강사 목록 (JSON 문자열)
            """
            logger.info(f"Searching instructors for skill: {skill}, min_rating: {min_rating}, max_price: {max_price}")
            
            # 필터링된 강사 검색
            matched_instructors = []
            for instructor_id, instructor in self.instructors.items():
                if skill.lower() not in instructor.get("skills", []):
                    continue
                if instructor.get("rating", 0) < min_rating:
                    continue
                if max_price and instructor.get("hourly_rate", 0) > max_price:
                    continue
                if location and instructor.get("location") != location:
                    continue
                matched_instructors.append(instructor)
            
            # 평점 순으로 정렬
            matched_instructors.sort(key=lambda x: x.get("rating", 0), reverse=True)
            
            return json.dumps(matched_instructors, ensure_ascii=False, indent=2)
        return search_instructors
    
    def _create_match_instructor_tool(self) -> BaseTool:
        @tool("marketplace_match_instructor", args_schema=MatchInstructorInput)
        def match_instructor(
            learner_id: str,
            skill: str,
            learner_level: str,
            budget: Optional[float] = None,
            preferred_time: Optional[str] = None
        ) -> str:
            """
            학습자에게 최적의 강사를 매칭합니다.
            Args:
                learner_id: 학습자 ID
                skill: 학습할 스킬
                learner_level: 학습자 현재 수준
                budget: 예산
                preferred_time: 선호 시간대
            Returns:
                매칭된 강사 목록 및 매칭 점수 (JSON 문자열)
            """
            logger.info(f"Matching instructor for learner {learner_id}, skill: {skill}, level: {learner_level}")
            
            # 강사 검색
            candidates = []
            for instructor_id, instructor in self.instructors.items():
                if skill.lower() not in instructor.get("skills", []):
                    continue
                if budget and instructor.get("hourly_rate", 0) > budget:
                    continue
                
                # 매칭 점수 계산
                score = 0.0
                # 평점 점수 (40%)
                score += instructor.get("rating", 0) * 0.4
                # 수준 매칭 점수 (30%)
                if learner_level in instructor.get("teaches_levels", []):
                    score += 1.0 * 0.3
                # 가격 적합성 (20%)
                if budget:
                    price_ratio = instructor.get("hourly_rate", 0) / budget
                    if price_ratio <= 1.0:
                        score += (1.0 - price_ratio) * 0.2
                # 가용성 (10%)
                if preferred_time and preferred_time in instructor.get("available_times", []):
                    score += 0.1
                
                candidates.append({
                    "instructor_id": instructor_id,
                    "instructor": instructor,
                    "match_score": score
                })
            
            # 매칭 점수 순으로 정렬
            candidates.sort(key=lambda x: x["match_score"], reverse=True)
            
            return json.dumps(candidates[:5], ensure_ascii=False, indent=2)  # 상위 5명 반환
        return match_instructor
    
    def _create_calculate_commission_tool(self) -> BaseTool:
        @tool("marketplace_calculate_commission", args_schema=CalculateCommissionInput)
        def calculate_commission(transaction_amount: float, commission_rate: Optional[float] = 0.15) -> str:
            """
            거래 수수료를 계산합니다.
            Args:
                transaction_amount: 거래 금액
                commission_rate: 수수료율 (기본 15%)
            Returns:
                수수료 계산 결과 (JSON 문자열)
            """
            logger.info(f"Calculating commission: amount={transaction_amount}, rate={commission_rate}")
            commission = transaction_amount * commission_rate
            result = {
                "transaction_amount": transaction_amount,
                "commission_rate": commission_rate,
                "commission": commission,
                "instructor_payout": transaction_amount - commission,
                "platform_revenue": commission
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_commission
    
    def _create_process_payment_tool(self) -> BaseTool:
        @tool("marketplace_process_payment", args_schema=ProcessPaymentInput)
        def process_payment(
            learner_id: str,
            instructor_id: str,
            amount: float,
            session_id: str
        ) -> str:
            """
            결제를 처리합니다.
            Args:
                learner_id: 학습자 ID
                instructor_id: 강사 ID
                amount: 결제 금액
                session_id: 학습 세션 ID
            Returns:
                결제 처리 결과 (JSON 문자열)
            """
            logger.info(f"Processing payment: learner={learner_id}, instructor={instructor_id}, amount={amount}")
            
            # 수수료 계산
            commission_rate = 0.15
            commission = amount * commission_rate
            instructor_payout = amount - commission
            
            transaction = {
                "transaction_id": f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id}",
                "learner_id": learner_id,
                "instructor_id": instructor_id,
                "session_id": session_id,
                "amount": amount,
                "commission": commission,
                "instructor_payout": instructor_payout,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            self.transactions[transaction["transaction_id"]] = transaction
            self._save_transactions()
            
            return json.dumps(transaction, ensure_ascii=False, indent=2)
        return process_payment
    
    def _create_create_learning_session_tool(self) -> BaseTool:
        @tool("marketplace_create_learning_session", args_schema=CreateLearningSessionInput)
        def create_learning_session(
            learner_id: str,
            instructor_id: str,
            skill: str,
            scheduled_time: str,
            duration_hours: float
        ) -> str:
            """
            학습 세션을 생성합니다.
            Args:
                learner_id: 학습자 ID
                instructor_id: 강사 ID
                skill: 학습할 스킬
                scheduled_time: 예정 시간
                duration_hours: 수업 시간
            Returns:
                생성된 학습 세션 정보 (JSON 문자열)
            """
            logger.info(f"Creating learning session: learner={learner_id}, instructor={instructor_id}, skill={skill}")
            
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{learner_id}"
            
            # 강사 정보에서 시간당 요금 가져오기
            instructor = self.instructors.get(instructor_id, {})
            hourly_rate = instructor.get("hourly_rate", 0)
            total_amount = hourly_rate * duration_hours
            
            session = {
                "session_id": session_id,
                "learner_id": learner_id,
                "instructor_id": instructor_id,
                "skill": skill,
                "scheduled_time": scheduled_time,
                "duration_hours": duration_hours,
                "hourly_rate": hourly_rate,
                "total_amount": total_amount,
                "status": "scheduled",
                "created_at": datetime.now().isoformat()
            }
            
            self.sessions[session_id] = session
            self._save_sessions()
            
            return json.dumps(session, ensure_ascii=False, indent=2)
        return create_learning_session
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Marketplace 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Marketplace 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

