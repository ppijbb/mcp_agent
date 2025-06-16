"""
Roadmap Builder Module

요구사항 데이터를 기반으로 프로덕트 로드맵 구축
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

# 로컬 설정 imports - 절대 import로 수정
try:
    from config import ROADMAP_CONFIG
except ImportError:
    # Fallback 설정
    ROADMAP_CONFIG = {"phases": 4, "default_duration": 12}

logger = logging.getLogger(__name__)


class RoadmapBuilder:
    """
    프로덕트 로드맵 빌더
    
    요구사항 데이터를 분석하여 실현 가능한 개발 로드맵을 생성
    """
    
    def __init__(self):
        """RoadmapBuilder 초기화"""
        self.config = ROADMAP_CONFIG
        self.phases = self.config["phases"]
        self.estimation_factors = self.config["estimation_factors"]
        self.priority_levels = self.config["priority_levels"]
        self.sprint_length = self.config["default_sprint_length"]
        
        logger.info("RoadmapBuilder 초기화 완료")
    
    async def build_roadmap(self, requirements_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        요구사항 기반 로드맵 구축
        
        Args:
            requirements_data: 요구사항 데이터
            
        Returns:
            구축된 로드맵 데이터
        """
        try:
            logger.info("로드맵 구축 시작")
            
            # 1. 요구사항 분석 및 우선순위 설정
            analyzed_requirements = self._analyze_requirements(requirements_data)
            
            # 2. 개발 단계별 작업 분배
            phase_distribution = self._distribute_to_phases(analyzed_requirements)
            
            # 3. 타임라인 추정
            timeline = self._estimate_timeline(phase_distribution)
            
            # 4. 마일스톤 정의
            milestones = self._define_milestones(phase_distribution, timeline)
            
            # 5. 리스크 분석
            risks = self._analyze_risks(analyzed_requirements, timeline)
            
            # 6. 리소스 요구사항 계산
            resource_requirements = self._calculate_resources(phase_distribution)
            
            roadmap = {
                "project_name": requirements_data.get("product_name", "Product Roadmap"),
                "created_at": datetime.now().isoformat(),
                "phases": phase_distribution,
                "timeline": timeline,
                "milestones": milestones,
                "risks": risks,
                "resource_requirements": resource_requirements,
                "dependencies": self._identify_dependencies(analyzed_requirements),
                "success_metrics": requirements_data.get("success_metrics", []),
                "metadata": {
                    "total_requirements": len(analyzed_requirements),
                    "estimated_duration_weeks": timeline.get("total_weeks", 0),
                    "complexity_score": self._calculate_overall_complexity(analyzed_requirements),
                    "confidence_level": self._assess_confidence(analyzed_requirements)
                }
            }
            
            logger.info("로드맵 구축 완료")
            return roadmap
            
        except Exception as e:
            logger.error(f"로드맵 구축 실패: {str(e)}")
            raise
    
    def _analyze_requirements(self, requirements_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """요구사항 분석 및 구조화"""
        try:
            analyzed = []
            
            # User Stories 처리
            user_stories = requirements_data.get("user_stories", [])
            for i, story in enumerate(user_stories):
                analyzed.append({
                    "id": f"US-{i+1:03d}",
                    "type": "user_story",
                    "title": story if isinstance(story, str) else story.get("title", f"User Story {i+1}"),
                    "description": story if isinstance(story, str) else story.get("description", ""),
                    "priority": self._determine_priority(story),
                    "complexity": self._estimate_complexity(story),
                    "story_points": self._estimate_story_points(story),
                    "phase": self._assign_to_phase(story),
                    "dependencies": []
                })
            
            # Technical Requirements 처리
            tech_requirements = requirements_data.get("technical_requirements", [])
            for i, req in enumerate(tech_requirements):
                analyzed.append({
                    "id": f"TR-{i+1:03d}",
                    "type": "technical_requirement",
                    "title": req if isinstance(req, str) else req.get("title", f"Technical Requirement {i+1}"),
                    "description": req if isinstance(req, str) else req.get("description", ""),
                    "priority": self._determine_priority(req),
                    "complexity": self._estimate_complexity(req),
                    "story_points": self._estimate_story_points(req),
                    "phase": self._assign_to_phase(req),
                    "dependencies": []
                })
            
            return analyzed
            
        except Exception as e:
            logger.warning(f"요구사항 분석 중 오류: {str(e)}")
            return []
    
    def _determine_priority(self, requirement: Any) -> str:
        """요구사항 우선순위 결정"""
        if isinstance(requirement, dict):
            return requirement.get("priority", "Medium")
        
        # 키워드 기반 우선순위 추정
        req_text = str(requirement).lower()
        
        if any(keyword in req_text for keyword in ["critical", "essential", "must", "security", "login"]):
            return "Critical"
        elif any(keyword in req_text for keyword in ["important", "should", "core", "main"]):
            return "High"
        elif any(keyword in req_text for keyword in ["nice", "could", "optional", "enhancement"]):
            return "Low"
        else:
            return "Medium"
    
    def _estimate_complexity(self, requirement: Any) -> str:
        """요구사항 복잡도 추정"""
        if isinstance(requirement, dict):
            return requirement.get("complexity", "Medium")
        
        req_text = str(requirement).lower()
        
        if any(keyword in req_text for keyword in ["integration", "api", "database", "authentication", "complex"]):
            return "Complex"
        elif any(keyword in req_text for keyword in ["simple", "basic", "display", "show"]):
            return "Simple"
        else:
            return "Medium"
    
    def _estimate_story_points(self, requirement: Any) -> int:
        """스토리 포인트 추정"""
        if isinstance(requirement, dict):
            return requirement.get("story_points", 3)
        
        complexity = self._estimate_complexity(requirement)
        priority = self._determine_priority(requirement)
        
        base_points = {
            "Simple": 2,
            "Medium": 3,
            "Complex": 5
        }.get(complexity, 3)
        
        # 우선순위에 따른 조정
        if priority == "Critical":
            base_points += 1
        
        return min(base_points, 8)  # 최대 8포인트
    
    def _assign_to_phase(self, requirement: Any) -> str:
        """요구사항을 개발 단계에 할당"""
        req_text = str(requirement).lower()
        
        if any(keyword in req_text for keyword in ["research", "analysis", "discovery"]):
            return "discovery"
        elif any(keyword in req_text for keyword in ["design", "ui", "ux", "mockup"]):
            return "design"
        elif any(keyword in req_text for keyword in ["test", "qa", "validation"]):
            return "testing"
        elif any(keyword in req_text for keyword in ["deploy", "launch", "release"]):
            return "launch"
        else:
            return "development"
    
    def _distribute_to_phases(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """요구사항을 개발 단계별로 분배"""
        try:
            phase_distribution = {}
            
            for phase in self.phases:
                phase_requirements = [req for req in requirements if req["phase"] == phase]
                
                total_story_points = sum(req["story_points"] for req in phase_requirements)
                estimated_weeks = max(1, total_story_points // 10)  # 10 포인트 = 1주
                
                phase_distribution[phase] = {
                    "name": phase.title(),
                    "requirements": phase_requirements,
                    "total_story_points": total_story_points,
                    "estimated_weeks": estimated_weeks,
                    "priority_breakdown": self._get_priority_breakdown(phase_requirements),
                    "complexity_breakdown": self._get_complexity_breakdown(phase_requirements)
                }
            
            return phase_distribution
            
        except Exception as e:
            logger.warning(f"단계별 분배 중 오류: {str(e)}")
            return {}
    
    def _get_priority_breakdown(self, requirements: List[Dict[str, Any]]) -> Dict[str, int]:
        """우선순위별 요구사항 수 계산"""
        breakdown = {priority: 0 for priority in self.priority_levels}
        for req in requirements:
            priority = req.get("priority", "Medium")
            if priority in breakdown:
                breakdown[priority] += 1
        return breakdown
    
    def _get_complexity_breakdown(self, requirements: List[Dict[str, Any]]) -> Dict[str, int]:
        """복잡도별 요구사항 수 계산"""
        breakdown = {"Simple": 0, "Medium": 0, "Complex": 0}
        for req in requirements:
            complexity = req.get("complexity", "Medium")
            if complexity in breakdown:
                breakdown[complexity] += 1
        return breakdown
    
    def _estimate_timeline(self, phase_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """전체 타임라인 추정"""
        try:
            start_date = datetime.now()
            current_date = start_date
            total_weeks = 0
            
            phase_timelines = {}
            
            for phase_name, phase_data in phase_distribution.items():
                estimated_weeks = phase_data["estimated_weeks"]
                
                # 복잡도 및 리스크 버퍼 적용
                adjusted_weeks = int(estimated_weeks * self.estimation_factors["complexity_multiplier"])
                adjusted_weeks = int(adjusted_weeks * (1 + self.estimation_factors["risk_buffer"]))
                
                phase_start = current_date
                phase_end = current_date + timedelta(weeks=adjusted_weeks)
                
                phase_timelines[phase_name] = {
                    "start_date": phase_start.strftime("%Y-%m-%d"),
                    "end_date": phase_end.strftime("%Y-%m-%d"),
                    "duration_weeks": adjusted_weeks,
                    "sprint_count": max(1, adjusted_weeks // 2)  # 2주 스프린트
                }
                
                current_date = phase_end
                total_weeks += adjusted_weeks
            
            return {
                "project_start": start_date.strftime("%Y-%m-%d"),
                "project_end": current_date.strftime("%Y-%m-%d"),
                "total_weeks": total_weeks,
                "total_sprints": sum(pt["sprint_count"] for pt in phase_timelines.values()),
                "phases": phase_timelines
            }
            
        except Exception as e:
            logger.warning(f"타임라인 추정 중 오류: {str(e)}")
            return {}
    
    def _define_milestones(self, phase_distribution: Dict[str, Any], timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
        """주요 마일스톤 정의"""
        try:
            milestones = []
            
            phase_timelines = timeline.get("phases", {})
            
            for phase_name, phase_timeline in phase_timelines.items():
                milestone = {
                    "name": f"{phase_name.title()} Complete",
                    "date": phase_timeline["end_date"],
                    "phase": phase_name,
                    "description": f"{phase_name.title()} 단계 완료",
                    "deliverables": self._get_phase_deliverables(phase_name),
                    "success_criteria": self._get_phase_success_criteria(phase_name),
                    "dependencies": []
                }
                milestones.append(milestone)
            
            return milestones
            
        except Exception as e:
            logger.warning(f"마일스톤 정의 중 오류: {str(e)}")
            return []
    
    def _get_phase_deliverables(self, phase_name: str) -> List[str]:
        """단계별 주요 산출물"""
        deliverables_map = {
            "discovery": ["요구사항 문서", "사용자 리서치 결과", "기술 스택 결정"],
            "design": ["와이어프레임", "UI/UX 디자인", "프로토타입"],
            "development": ["핵심 기능 구현", "API 개발", "데이터베이스 설계"],
            "testing": ["테스트 케이스", "QA 리포트", "성능 테스트 결과"],
            "launch": ["배포 완료", "모니터링 설정", "사용자 가이드"]
        }
        return deliverables_map.get(phase_name, [f"{phase_name.title()} 산출물"])
    
    def _get_phase_success_criteria(self, phase_name: str) -> List[str]:
        """단계별 성공 기준"""
        criteria_map = {
            "discovery": ["모든 요구사항 문서화 완료", "기술적 타당성 검증"],
            "design": ["디자인 시스템 완성", "사용자 테스트 통과"],
            "development": ["핵심 기능 100% 구현", "코드 리뷰 완료"],
            "testing": ["모든 테스트 케이스 통과", "성능 기준 달성"],
            "launch": ["프로덕션 배포 완료", "모니터링 정상 작동"]
        }
        return criteria_map.get(phase_name, [f"{phase_name.title()} 목표 달성"])
    
    def _analyze_risks(self, requirements: List[Dict[str, Any]], timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
        """프로젝트 리스크 분석"""
        try:
            risks = []
            
            # 복잡도 기반 리스크
            complex_requirements = [req for req in requirements if req["complexity"] == "Complex"]
            if len(complex_requirements) > len(requirements) * 0.3:
                risks.append({
                    "type": "technical",
                    "level": "high",
                    "description": "복잡한 요구사항 비율이 높음 (30% 초과)",
                    "impact": "개발 일정 지연 가능성",
                    "mitigation": "복잡한 기능을 단계별로 분할하여 개발"
                })
            
            # 타임라인 기반 리스크
            total_weeks = timeline.get("total_weeks", 0)
            if total_weeks > 24:  # 6개월 초과
                risks.append({
                    "type": "schedule",
                    "level": "medium",
                    "description": "프로젝트 기간이 6개월을 초과함",
                    "impact": "요구사항 변경 및 범위 확대 위험",
                    "mitigation": "정기적인 스프린트 리뷰 및 우선순위 재조정"
                })
            
            # 의존성 기반 리스크
            dependencies = self._identify_dependencies(requirements)
            if len(dependencies) > 5:
                risks.append({
                    "type": "dependency",
                    "level": "medium",
                    "description": "외부 의존성이 많음",
                    "impact": "외부 요인으로 인한 지연 가능성",
                    "mitigation": "의존성 관리 계획 수립 및 대안 준비"
                })
            
            return risks
            
        except Exception as e:
            logger.warning(f"리스크 분석 중 오류: {str(e)}")
            return []
    
    def _identify_dependencies(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """요구사항 간 의존성 식별"""
        try:
            dependencies = []
            
            # 기본적인 의존성 패턴 식별
            auth_requirements = [req for req in requirements if "auth" in str(req).lower() or "login" in str(req).lower()]
            api_requirements = [req for req in requirements if "api" in str(req).lower()]
            db_requirements = [req for req in requirements if "database" in str(req).lower() or "data" in str(req).lower()]
            
            if auth_requirements and api_requirements:
                dependencies.append({
                    "type": "sequential",
                    "prerequisite": "Authentication System",
                    "dependent": "API Development",
                    "description": "API 개발 전 인증 시스템 구축 필요"
                })
            
            if db_requirements and api_requirements:
                dependencies.append({
                    "type": "sequential", 
                    "prerequisite": "Database Design",
                    "dependent": "API Development",
                    "description": "API 개발 전 데이터베이스 설계 완료 필요"
                })
            
            return dependencies
            
        except Exception as e:
            logger.warning(f"의존성 식별 중 오류: {str(e)}")
            return []
    
    def _calculate_resources(self, phase_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """리소스 요구사항 계산"""
        try:
            total_story_points = sum(phase["total_story_points"] for phase in phase_distribution.values())
            
            # 역할별 필요 인력 추정 (스토리 포인트 기반)
            developers_needed = max(2, total_story_points // 40)  # 40포인트당 개발자 1명
            designers_needed = max(1, total_story_points // 60)   # 60포인트당 디자이너 1명
            qa_needed = max(1, total_story_points // 80)          # 80포인트당 QA 1명
            
            return {
                "team_composition": {
                    "developers": developers_needed,
                    "designers": designers_needed,
                    "qa_engineers": qa_needed,
                    "product_manager": 1,
                    "total_team_size": developers_needed + designers_needed + qa_needed + 1
                },
                "skill_requirements": [
                    "Frontend Development",
                    "Backend Development", 
                    "UI/UX Design",
                    "Quality Assurance",
                    "Product Management"
                ],
                "estimated_cost": {
                    "development_hours": total_story_points * 8,  # 8시간/포인트
                    "total_person_months": total_story_points // 20  # 20포인트/월
                }
            }
            
        except Exception as e:
            logger.warning(f"리소스 계산 중 오류: {str(e)}")
            return {}
    
    def _calculate_overall_complexity(self, requirements: List[Dict[str, Any]]) -> float:
        """전체 프로젝트 복잡도 점수 계산"""
        if not requirements:
            return 0.0
        
        complexity_scores = {
            "Simple": 1,
            "Medium": 2,
            "Complex": 3
        }
        
        total_score = sum(complexity_scores.get(req["complexity"], 2) for req in requirements)
        max_possible = len(requirements) * 3
        
        return round(total_score / max_possible, 2)
    
    def _assess_confidence(self, requirements: List[Dict[str, Any]]) -> str:
        """로드맵 신뢰도 평가"""
        if not requirements:
            return "Low"
        
        # 복잡한 요구사항 비율
        complex_ratio = len([req for req in requirements if req["complexity"] == "Complex"]) / len(requirements)
        
        # 우선순위가 명확하지 않은 요구사항 비율
        unclear_priority = len([req for req in requirements if req["priority"] == "Medium"]) / len(requirements)
        
        if complex_ratio < 0.2 and unclear_priority < 0.4:
            return "High"
        elif complex_ratio < 0.4 and unclear_priority < 0.6:
            return "Medium"
        else:
            return "Low" 