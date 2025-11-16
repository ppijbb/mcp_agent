"""
Learning 도구

학습 진행 추적, 스킬 향상 추적, 학습 리포트 생성
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RecordLearningProgressInput(BaseModel):
    """학습 진행 기록 입력 스키마"""
    learner_id: str = Field(description="학습자 ID")
    session_id: str = Field(description="학습 세션 ID")
    skill: str = Field(description="학습한 스킬")
    progress_percentage: float = Field(ge=0.0, le=100.0, description="진행률 (%)")
    notes: Optional[str] = Field(default=None, description="학습 노트")


class TrackSkillImprovementInput(BaseModel):
    """스킬 향상 추적 입력 스키마"""
    learner_id: str = Field(description="학습자 ID")
    skill: str = Field(description="추적할 스킬")
    period_days: Optional[int] = Field(default=30, description="추적 기간 (일)")


class GenerateLearningReportInput(BaseModel):
    """학습 리포트 생성 입력 스키마"""
    learner_id: str = Field(description="학습자 ID")
    period_days: Optional[int] = Field(default=30, description="리포트 기간 (일)")


class RecommendNextStepsInput(BaseModel):
    """다음 학습 단계 추천 입력 스키마"""
    learner_id: str = Field(description="학습자 ID")
    skill: str = Field(description="현재 학습 중인 스킬")
    current_level: str = Field(description="현재 수준")


class LearningTools:
    """
    Learning 도구 모음
    
    학습 진행 추적, 스킬 향상 추적, 학습 리포트 생성
    """
    
    def __init__(self, data_dir: str = "marketplace_data"):
        """
        LearningTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.data_dir / "learning_progress.json"
        self.skills_file = self.data_dir / "learner_skills.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
        else:
            self.progress = {}
        
        if self.skills_file.exists():
            with open(self.skills_file, 'r', encoding='utf-8') as f:
                self.learner_skills = json.load(f)
        else:
            self.learner_skills = {}
    
    def _save_progress(self):
        """진행 데이터 저장"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def _save_skills(self):
        """스킬 데이터 저장"""
        with open(self.skills_file, 'w', encoding='utf-8') as f:
            json.dump(self.learner_skills, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Learning 도구 초기화"""
        self.tools.append(self._create_record_learning_progress_tool())
        self.tools.append(self._create_track_skill_improvement_tool())
        self.tools.append(self._create_generate_learning_report_tool())
        self.tools.append(self._create_recommend_next_steps_tool())
        logger.info(f"Initialized {len(self.tools)} learning tools")
    
    def _create_record_learning_progress_tool(self) -> BaseTool:
        @tool("learning_record_progress", args_schema=RecordLearningProgressInput)
        def record_learning_progress(
            learner_id: str,
            session_id: str,
            skill: str,
            progress_percentage: float,
            notes: Optional[str] = None
        ) -> str:
            """
            학습 진행을 기록합니다.
            Args:
                learner_id: 학습자 ID
                session_id: 학습 세션 ID
                skill: 학습한 스킬
                progress_percentage: 진행률 (%)
                notes: 학습 노트
            Returns:
                기록 결과 메시지
            """
            logger.info(f"Recording learning progress for learner {learner_id}, session {session_id}")
            
            if learner_id not in self.progress:
                self.progress[learner_id] = []
            
            progress_record = {
                "learner_id": learner_id,
                "session_id": session_id,
                "skill": skill,
                "progress_percentage": progress_percentage,
                "notes": notes,
                "timestamp": datetime.now().isoformat()
            }
            self.progress[learner_id].append(progress_record)
            self._save_progress()
            
            # 학습자 스킬 레벨 업데이트
            if learner_id not in self.learner_skills:
                self.learner_skills[learner_id] = {}
            
            if skill not in self.learner_skills[learner_id]:
                self.learner_skills[learner_id][skill] = {
                    "level": "beginner",
                    "progress": 0.0,
                    "last_updated": datetime.now().isoformat()
                }
            
            self.learner_skills[learner_id][skill]["progress"] = progress_percentage
            self.learner_skills[learner_id][skill]["last_updated"] = datetime.now().isoformat()
            
            # 진행률에 따라 레벨 업데이트
            if progress_percentage >= 80:
                self.learner_skills[learner_id][skill]["level"] = "advanced"
            elif progress_percentage >= 50:
                self.learner_skills[learner_id][skill]["level"] = "intermediate"
            else:
                self.learner_skills[learner_id][skill]["level"] = "beginner"
            
            self._save_skills()
            
            return f"Learning progress recorded for learner {learner_id}: {skill} - {progress_percentage}%"
        return record_learning_progress
    
    def _create_track_skill_improvement_tool(self) -> BaseTool:
        @tool("learning_track_skill_improvement", args_schema=TrackSkillImprovementInput)
        def track_skill_improvement(
            learner_id: str,
            skill: str,
            period_days: Optional[int] = 30
        ) -> str:
            """
            스킬 향상을 추적합니다.
            Args:
                learner_id: 학습자 ID
                skill: 추적할 스킬
                period_days: 추적 기간 (일)
            Returns:
                스킬 향상 추적 결과 (JSON 문자열)
            """
            logger.info(f"Tracking skill improvement for learner {learner_id}, skill: {skill}")
            
            if learner_id not in self.progress:
                return json.dumps({"error": "No progress data found for this learner"}, ensure_ascii=False)
            
            cutoff_date = datetime.now() - timedelta(days=period_days)
            skill_progress = [
                record for record in self.progress[learner_id]
                if record["skill"] == skill and datetime.fromisoformat(record["timestamp"]) >= cutoff_date
            ]
            
            if not skill_progress:
                return json.dumps({"error": f"No progress data found for skill {skill} in the last {period_days} days"}, ensure_ascii=False)
            
            # 향상 분석
            initial_progress = skill_progress[0]["progress_percentage"]
            final_progress = skill_progress[-1]["progress_percentage"]
            improvement = final_progress - initial_progress
            
            result = {
                "learner_id": learner_id,
                "skill": skill,
                "period_days": period_days,
                "initial_progress": initial_progress,
                "final_progress": final_progress,
                "improvement": improvement,
                "improvement_rate": (improvement / initial_progress * 100) if initial_progress > 0 else 0,
                "sessions_count": len(skill_progress),
                "current_level": self.learner_skills.get(learner_id, {}).get(skill, {}).get("level", "unknown")
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return track_skill_improvement
    
    def _create_generate_learning_report_tool(self) -> BaseTool:
        @tool("learning_generate_report", args_schema=GenerateLearningReportInput)
        def generate_learning_report(
            learner_id: str,
            period_days: Optional[int] = 30
        ) -> str:
            """
            학습 리포트를 생성합니다.
            Args:
                learner_id: 학습자 ID
                period_days: 리포트 기간 (일)
            Returns:
                학습 리포트 (JSON 문자열)
            """
            logger.info(f"Generating learning report for learner {learner_id}")
            
            if learner_id not in self.progress:
                return json.dumps({"error": "No progress data found for this learner"}, ensure_ascii=False)
            
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_progress = [
                record for record in self.progress[learner_id]
                if datetime.fromisoformat(record["timestamp"]) >= cutoff_date
            ]
            
            if not recent_progress:
                return json.dumps({"error": f"No progress data found in the last {period_days} days"}, ensure_ascii=False)
            
            # 리포트 생성
            skills_learned = list(set([record["skill"] for record in recent_progress]))
            total_sessions = len(recent_progress)
            avg_progress = sum([record["progress_percentage"] for record in recent_progress]) / len(recent_progress)
            
            report = {
                "learner_id": learner_id,
                "period_days": period_days,
                "report_date": datetime.now().isoformat(),
                "total_sessions": total_sessions,
                "skills_learned": skills_learned,
                "average_progress": avg_progress,
                "skills_summary": {
                    skill: {
                        "sessions": len([r for r in recent_progress if r["skill"] == skill]),
                        "current_level": self.learner_skills.get(learner_id, {}).get(skill, {}).get("level", "unknown"),
                        "progress": self.learner_skills.get(learner_id, {}).get(skill, {}).get("progress", 0.0)
                    }
                    for skill in skills_learned
                }
            }
            
            return json.dumps(report, ensure_ascii=False, indent=2)
        return generate_learning_report
    
    def _create_recommend_next_steps_tool(self) -> BaseTool:
        @tool("learning_recommend_next_steps", args_schema=RecommendNextStepsInput)
        def recommend_next_steps(
            learner_id: str,
            skill: str,
            current_level: str
        ) -> str:
            """
            다음 학습 단계를 추천합니다.
            Args:
                learner_id: 학습자 ID
                skill: 현재 학습 중인 스킬
                current_level: 현재 수준
            Returns:
                다음 학습 단계 추천 (JSON 문자열)
            """
            logger.info(f"Recommending next steps for learner {learner_id}, skill: {skill}, level: {current_level}")
            
            # 현재 진행률 확인
            current_progress = self.learner_skills.get(learner_id, {}).get(skill, {}).get("progress", 0.0)
            
            # 다음 단계 추천
            next_steps = []
            if current_level == "beginner":
                next_steps = [
                    "기초 개념 복습",
                    "실습 프로젝트 시작",
                    "중급 강사와 세션 예약",
                    "관련 온라인 강의 수강"
                ]
            elif current_level == "intermediate":
                next_steps = [
                    "고급 개념 학습",
                    "복잡한 프로젝트 도전",
                    "고급 강사와 세션 예약",
                    "포트폴리오 구축"
                ]
            else:  # advanced
                next_steps = [
                    "전문가 수준 프로젝트",
                    "멘토링 제공 시작",
                    "관련 스킬 확장",
                    "인증 시험 준비"
                ]
            
            result = {
                "learner_id": learner_id,
                "skill": skill,
                "current_level": current_level,
                "current_progress": current_progress,
                "recommended_next_steps": next_steps,
                "target_level": "intermediate" if current_level == "beginner" else "advanced" if current_level == "intermediate" else "expert"
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return recommend_next_steps
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Learning 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Learning 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

