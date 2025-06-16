"""
Requirement Generator Module

디자인 분석 결과를 바탕으로 프로덕트 요구사항 문서를 생성하는 모듈
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

# 로컬 설정 imports - 절대 import로 수정
try:
    from config import PRD_TEMPLATE_CONFIG, ANALYSIS_CONFIG
except ImportError:
    # Fallback 설정
    PRD_TEMPLATE_CONFIG = {"sections": ["overview", "requirements"]}
    ANALYSIS_CONFIG = {"requirement_depth": "detailed"}

logger = logging.getLogger(__name__)

class RequirementGenerator:
    """
    요구사항 생성기
    
    디자인 분석 결과를 바탕으로 체계적인 프로덕트 요구사항을 생성
    """
    
    def __init__(self):
        """RequirementGenerator 초기화"""
        self.prd_config = PRD_TEMPLATE_CONFIG
        self.analysis_config = ANALYSIS_CONFIG
        
        # 템플릿 및 패턴 라이브러리
        self._requirement_templates = self._load_requirement_templates()
        self._user_story_patterns = self._load_user_story_patterns()
        
        logger.info("RequirementGenerator 초기화 완료")
    
    async def generate_requirements(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        요구사항 생성 (메인 퍼블릭 메서드)
        
        Args:
            analysis_data: 디자인 분석 결과
            
        Returns:
            생성된 요구사항 딕셔너리
        """
        try:
            logger.info("요구사항 생성 시작")
            
            # 1. 기본 정보 추출
            basic_info = self._extract_basic_info(analysis_data)
            
            # 2. 기능 요구사항 생성
            functional_requirements = self._generate_functional_requirements(analysis_data)
            
            # 3. 비기능 요구사항 생성
            non_functional_requirements = self._generate_non_functional_requirements(analysis_data)
            
            # 4. 사용자 스토리 생성
            user_stories = self._generate_user_stories(analysis_data)
            
            # 5. 수용 기준 생성
            acceptance_criteria = self._generate_acceptance_criteria(functional_requirements)
            
            # 6. 기술적 제약사항 정의
            technical_constraints = self._define_technical_constraints(analysis_data)
            
            # 7. 성공 지표 정의
            success_metrics = self._define_success_metrics(analysis_data)
            
            # 8. 타임라인 및 마일스톤
            timeline = self._generate_timeline(functional_requirements)
            
            requirements = {
                "product_name": basic_info.get("product_name", "New Product"),
                "generated_at": datetime.now().isoformat(),
                "executive_summary": self._generate_executive_summary(basic_info, analysis_data),
                "problem_statement": self._generate_problem_statement(analysis_data),
                "solution_overview": self._generate_solution_overview(analysis_data),
                "target_audience": basic_info.get("target_audience", "End Users"),
                "key_features": self._extract_key_features(functional_requirements),
                "functional_requirements": functional_requirements,
                "non_functional_requirements": non_functional_requirements,
                "user_stories": user_stories,
                "acceptance_criteria": acceptance_criteria,
                "technical_constraints": technical_constraints,
                "success_metrics": success_metrics,
                "timeline_milestones": timeline,
                "priority_matrix": self._create_priority_matrix(functional_requirements),
                "risk_assessment": self._assess_risks(analysis_data),
                "resource_requirements": self._estimate_resources(analysis_data)
            }
            
            logger.info("요구사항 생성 완료")
            return requirements
            
        except Exception as e:
            logger.error(f"요구사항 생성 실패: {str(e)}")
            raise
    
    def _extract_basic_info(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """기본 정보 추출"""
        try:
            # 디자인 URL에서 프로젝트명 추출 시도
            url = analysis_data.get("url", "")
            product_name = "Product Planning Project"
            
            if "figma.com" in url:
                # Figma URL에서 프로젝트 힌트 추출
                if "login" in url.lower():
                    product_name = "Login System"
                elif "dashboard" in url.lower():
                    product_name = "Dashboard Application"
                elif "mobile" in url.lower():
                    product_name = "Mobile Application"
            
            return {
                "product_name": product_name,
                "target_audience": self._infer_target_audience(analysis_data),
                "platform": self._infer_platform(analysis_data),
                "complexity_level": analysis_data.get("overall_assessment", {}).get("development_complexity", "medium")
            }
        except Exception as e:
            logger.warning(f"기본 정보 추출 중 오류: {str(e)}")
            return {"product_name": "New Product", "target_audience": "General Users"}
    
    def _generate_functional_requirements(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """기능 요구사항 생성"""
        try:
            requirements = []
            
            # 컴포넌트 기반 요구사항
            component_analysis = analysis_data.get("component_analysis", {})
            components_detail = component_analysis.get("components_detail", [])
            
            for component in components_detail:
                comp_requirements = self._generate_component_requirements(component)
                requirements.extend(comp_requirements)
            
            # 인터랙션 기반 요구사항
            interaction_analysis = analysis_data.get("interaction_analysis", {})
            interaction_requirements = self._generate_interaction_requirements(interaction_analysis)
            requirements.extend(interaction_requirements)
            
            # 레이아웃 기반 요구사항
            layout_analysis = analysis_data.get("layout_analysis", {})
            layout_requirements = self._generate_layout_requirements(layout_analysis)
            requirements.extend(layout_requirements)
            
            # 중복 제거 및 우선순위 설정
            requirements = self._deduplicate_and_prioritize(requirements)
            
            return requirements
            
        except Exception as e:
            logger.warning(f"기능 요구사항 생성 중 오류: {str(e)}")
            return []
    
    def _generate_non_functional_requirements(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """비기능 요구사항 생성"""
        try:
            requirements = []
            
            # 성능 요구사항
            performance_analysis = analysis_data.get("performance_analysis", {})
            if performance_analysis.get("bundle_size_impact") == "high":
                requirements.append({
                    "id": "NFR-001",
                    "category": "performance",
                    "title": "번들 크기 최적화",
                    "description": "초기 로딩 시간 최적화를 위한 번들 크기 관리",
                    "acceptance_criteria": "초기 번들 크기 < 500KB, 지연 로딩 적용",
                    "priority": "high"
                })
            
            # 접근성 요구사항
            accessibility_analysis = analysis_data.get("accessibility_analysis", {})
            compliance_level = accessibility_analysis.get("compliance_level", "basic")
            
            if compliance_level in ["intermediate", "advanced"]:
                requirements.append({
                    "id": "NFR-002",
                    "category": "accessibility",
                    "title": "웹 접근성 준수",
                    "description": f"WCAG 2.1 {compliance_level.upper()} 수준 준수",
                    "acceptance_criteria": "스크린 리더 지원, 키보드 네비게이션, 색상 대비 4.5:1 이상",
                    "priority": "medium"
                })
            
            # 반응형 디자인 요구사항
            layout_analysis = analysis_data.get("layout_analysis", {})
            if layout_analysis.get("responsive_design", False):
                requirements.append({
                    "id": "NFR-003",
                    "category": "responsive",
                    "title": "반응형 디자인 구현",
                    "description": "다양한 디바이스에서 최적화된 사용자 경험 제공",
                    "acceptance_criteria": "모바일(375px), 태블릿(768px), 데스크톱(1200px) 지원",
                    "priority": "high"
                })
            
            # 보안 요구사항
            requirements.append({
                "id": "NFR-004",
                "category": "security",
                "title": "기본 보안 요구사항",
                "description": "사용자 데이터 보호 및 보안 정책 준수",
                "acceptance_criteria": "HTTPS 통신, 입력 데이터 검증, XSS/CSRF 방지",
                "priority": "high"
            })
            
            # 유지보수성 요구사항
            requirements.append({
                "id": "NFR-005",
                "category": "maintainability",
                "title": "코드 품질 및 유지보수성",
                "description": "장기적 유지보수를 위한 코드 품질 확보",
                "acceptance_criteria": "테스트 커버리지 80% 이상, 컴포넌트 재사용성 70% 이상",
                "priority": "medium"
            })
            
            return requirements
            
        except Exception as e:
            logger.warning(f"비기능 요구사항 생성 중 오류: {str(e)}")
            return []
    
    def _generate_user_stories(self, analysis_data: Dict[str, Any]) -> List[str]:
        """사용자 스토리 생성"""
        try:
            user_stories = []
            
            # 컴포넌트 기반 사용자 스토리
            component_analysis = analysis_data.get("component_analysis", {})
            components_detail = component_analysis.get("components_detail", [])
            
            for component in components_detail:
                comp_name = component.get("name", "")
                comp_type = component.get("type", "")
                
                if "button" in comp_name.lower():
                    user_stories.append(f"사용자로서, {comp_name}을 클릭하여 해당 액션을 수행할 수 있어야 한다.")
                
                elif "input" in comp_name.lower() or "field" in comp_name.lower():
                    user_stories.append(f"사용자로서, {comp_name}에 정보를 입력하고 유효성 검사를 받을 수 있어야 한다.")
                
                elif "card" in comp_name.lower():
                    user_stories.append(f"사용자로서, {comp_name}을 통해 관련 정보를 한눈에 볼 수 있어야 한다.")
                
                elif "modal" in comp_name.lower():
                    user_stories.append(f"사용자로서, {comp_name}을 통해 추가 정보를 확인하거나 액션을 수행할 수 있어야 한다.")
            
            # 인터랙션 기반 사용자 스토리
            interaction_analysis = analysis_data.get("interaction_analysis", {})
            interaction_types = interaction_analysis.get("interaction_types", [])
            
            if "form_input" in interaction_types:
                user_stories.append("사용자로서, 폼을 작성하고 제출할 때 실시간 검증 피드백을 받을 수 있어야 한다.")
            
            if "modal_interaction" in interaction_types:
                user_stories.append("사용자로서, 모달 창을 열고 닫을 때 직관적인 방법으로 조작할 수 있어야 한다.")
            
            # 접근성 기반 사용자 스토리
            accessibility_analysis = analysis_data.get("accessibility_analysis", {})
            if accessibility_analysis.get("keyboard_navigation"):
                user_stories.append("키보드 사용자로서, 마우스 없이도 모든 기능에 접근할 수 있어야 한다.")
            
            if accessibility_analysis.get("screen_reader_support"):
                user_stories.append("스크린 리더 사용자로서, 모든 UI 요소에 대한 적절한 설명을 들을 수 있어야 한다.")
            
            # 반응형 기반 사용자 스토리
            layout_analysis = analysis_data.get("layout_analysis", {})
            if layout_analysis.get("responsive_design", False):
                user_stories.append("모바일 사용자로서, 작은 화면에서도 모든 기능을 편리하게 사용할 수 있어야 한다.")
            
            return list(set(user_stories))  # 중복 제거
            
        except Exception as e:
            logger.warning(f"사용자 스토리 생성 중 오류: {str(e)}")
            return []
    
    def _generate_acceptance_criteria(self, functional_requirements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """수용 기준 생성"""
        try:
            acceptance_criteria = {}
            
            for req in functional_requirements:
                req_id = req.get("id", "")
                title = req.get("title", "")
                category = req.get("category", "")
                
                criteria = []
                
                if category == "ui_component":
                    criteria.extend([
                        "컴포넌트가 디자인 명세에 따라 정확히 렌더링되어야 함",
                        "모든 상태(기본, 호버, 활성, 비활성)가 올바르게 표시되어야 함",
                        "반응형 디자인이 다양한 화면 크기에서 작동해야 함"
                    ])
                
                elif category == "interaction":
                    criteria.extend([
                        "사용자 액션에 대한 즉각적인 피드백이 제공되어야 함",
                        "로딩 상태가 적절히 표시되어야 함",
                        "에러 상황에서 명확한 메시지가 표시되어야 함"
                    ])
                
                elif category == "form":
                    criteria.extend([
                        "모든 필수 필드가 검증되어야 함",
                        "유효하지 않은 입력에 대한 에러 메시지가 표시되어야 함",
                        "성공적인 제출 시 확인 메시지가 표시되어야 함"
                    ])
                
                elif category == "navigation":
                    criteria.extend([
                        "모든 네비게이션 링크가 올바른 페이지로 이동해야 함",
                        "현재 페이지가 명확히 표시되어야 함",
                        "브라우저 뒤로가기 버튼이 올바르게 작동해야 함"
                    ])
                
                # 공통 기준 추가
                criteria.extend([
                    "모든 브라우저에서 일관된 동작을 보여야 함",
                    "접근성 표준(WCAG 2.1)을 준수해야 함",
                    "성능 요구사항을 만족해야 함"
                ])
                
                acceptance_criteria[req_id] = criteria
            
            return acceptance_criteria
            
        except Exception as e:
            logger.warning(f"수용 기준 생성 중 오류: {str(e)}")
            return {}
    
    def _define_technical_constraints(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """기술적 제약사항 정의"""
        try:
            constraints = []
            
            # 성능 제약사항
            performance_analysis = analysis_data.get("performance_analysis", {})
            if performance_analysis.get("bundle_size_impact") in ["medium", "high"]:
                constraints.append({
                    "category": "performance",
                    "constraint": "번들 크기 제한",
                    "description": "초기 로딩 성능을 위한 번들 크기 제한",
                    "limit": "초기 번들 < 500KB, 총 번들 < 2MB"
                })
            
            # 브라우저 호환성
            constraints.append({
                "category": "compatibility",
                "constraint": "브라우저 지원",
                "description": "주요 브라우저에서의 호환성 보장",
                "limit": "Chrome 90+, Firefox 88+, Safari 14+, Edge 90+"
            })
            
            # 접근성 제약사항
            accessibility_analysis = analysis_data.get("accessibility_analysis", {})
            if accessibility_analysis.get("compliance_level") != "basic":
                constraints.append({
                    "category": "accessibility",
                    "constraint": "접근성 준수",
                    "description": "웹 접근성 지침 준수 필요",
                    "limit": "WCAG 2.1 AA 수준 준수"
                })
            
            # 반응형 제약사항
            layout_analysis = analysis_data.get("layout_analysis", {})
            if layout_analysis.get("responsive_design", False):
                constraints.append({
                    "category": "responsive",
                    "constraint": "화면 크기 지원",
                    "description": "다양한 디바이스 화면 크기 지원",
                    "limit": "최소 320px ~ 최대 1920px"
                })
            
            # 보안 제약사항
            constraints.append({
                "category": "security",
                "constraint": "보안 정책",
                "description": "기본 웹 보안 정책 준수",
                "limit": "HTTPS 필수, CSP 헤더 적용, 입력 검증 필수"
            })
            
            return constraints
            
        except Exception as e:
            logger.warning(f"기술적 제약사항 정의 중 오류: {str(e)}")
            return []
    
    def _define_success_metrics(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """성공 지표 정의"""
        try:
            metrics = []
            
            # 성능 지표
            metrics.append({
                "category": "performance",
                "metric": "페이지 로딩 시간",
                "target": "< 3초 (3G 네트워크 기준)",
                "measurement": "Lighthouse Performance Score > 90"
            })
            
            # 사용성 지표
            metrics.append({
                "category": "usability",
                "metric": "사용자 만족도",
                "target": "평균 4.0/5.0 이상",
                "measurement": "사용자 설문 조사 및 피드백"
            })
            
            # 접근성 지표
            accessibility_analysis = analysis_data.get("accessibility_analysis", {})
            if accessibility_analysis.get("compliance_level") != "basic":
                metrics.append({
                    "category": "accessibility",
                    "metric": "접근성 점수",
                    "target": "Lighthouse Accessibility Score > 95",
                    "measurement": "자동화된 접근성 테스트"
                })
            
            # 개발 효율성 지표
            component_analysis = analysis_data.get("component_analysis", {})
            if component_analysis.get("reusability_score", 0) > 0.5:
                metrics.append({
                    "category": "development",
                    "metric": "컴포넌트 재사용률",
                    "target": "> 70%",
                    "measurement": "코드 분석 도구를 통한 측정"
                })
            
            # 품질 지표
            metrics.append({
                "category": "quality",
                "metric": "테스트 커버리지",
                "target": "> 80%",
                "measurement": "Jest/Cypress 테스트 커버리지 리포트"
            })
            
            # 유지보수성 지표
            metrics.append({
                "category": "maintainability",
                "metric": "코드 품질 점수",
                "target": "SonarQube Grade A",
                "measurement": "정적 코드 분석 도구"
            })
            
            return metrics
            
        except Exception as e:
            logger.warning(f"성공 지표 정의 중 오류: {str(e)}")
            return []
    
    def _generate_timeline(self, functional_requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """타임라인 및 마일스톤 생성"""
        try:
            timeline = []
            
            # 요구사항을 우선순위별로 그룹화
            high_priority = [req for req in functional_requirements if req.get("priority") == "high"]
            medium_priority = [req for req in functional_requirements if req.get("priority") == "medium"]
            low_priority = [req for req in functional_requirements if req.get("priority") == "low"]
            
            current_date = datetime.now()
            
            # Phase 1: 핵심 기능 (고우선순위)
            if high_priority:
                phase1_duration = max(2, len(high_priority) * 0.5)  # 주 단위
                timeline.append({
                    "phase": "Phase 1 - Core Features",
                    "start_date": current_date.strftime("%Y-%m-%d"),
                    "end_date": (current_date + timedelta(weeks=phase1_duration)).strftime("%Y-%m-%d"),
                    "duration_weeks": int(phase1_duration),
                    "objectives": [req["title"] for req in high_priority[:5]],
                    "deliverables": ["MVP 버전", "핵심 컴포넌트", "기본 기능"],
                    "resources": ["Frontend Developer", "UI/UX Designer"]
                })
                current_date += timedelta(weeks=phase1_duration)
            
            # Phase 2: 추가 기능 (중우선순위)
            if medium_priority:
                phase2_duration = max(1, len(medium_priority) * 0.3)
                timeline.append({
                    "phase": "Phase 2 - Enhanced Features",
                    "start_date": current_date.strftime("%Y-%m-%d"),
                    "end_date": (current_date + timedelta(weeks=phase2_duration)).strftime("%Y-%m-%d"),
                    "duration_weeks": int(phase2_duration),
                    "objectives": [req["title"] for req in medium_priority[:5]],
                    "deliverables": ["향상된 UI", "추가 기능", "성능 최적화"],
                    "resources": ["Frontend Developer", "QA Engineer"]
                })
                current_date += timedelta(weeks=phase2_duration)
            
            # Phase 3: 최적화 및 마무리 (저우선순위)
            phase3_duration = max(1, len(low_priority) * 0.2 + 1)
            timeline.append({
                "phase": "Phase 3 - Polish & Launch",
                "start_date": current_date.strftime("%Y-%m-%d"),
                "end_date": (current_date + timedelta(weeks=phase3_duration)).strftime("%Y-%m-%d"),
                "duration_weeks": int(phase3_duration),
                "objectives": ["성능 최적화", "접근성 개선", "버그 수정", "문서화"],
                "deliverables": ["프로덕션 준비", "테스트 완료", "문서 작성"],
                "resources": ["Full Team", "DevOps Engineer"]
            })
            
            return timeline
            
        except Exception as e:
            logger.warning(f"타임라인 생성 중 오류: {str(e)}")
            return []
    
    # Helper methods
    def _generate_component_requirements(self, component: Dict[str, Any]) -> List[Dict[str, Any]]:
        """컴포넌트별 요구사항 생성"""
        requirements = []
        comp_name = component.get("name", "")
        comp_type = component.get("type", "")
        complexity = component.get("complexity", "medium")
        
        # 기본 컴포넌트 요구사항
        requirements.append({
            "id": f"FR-{comp_name.replace(' ', '-').lower()}",
            "category": "ui_component",
            "title": f"{comp_name} 컴포넌트 구현",
            "description": f"{comp_name} 컴포넌트의 디자인 및 기능 구현",
            "priority": "high" if complexity == "complex" else "medium",
            "complexity": complexity,
            "estimated_hours": {"simple": 4, "medium": 8, "complex": 16}.get(complexity, 8),
            "dependencies": [],
            "acceptance_criteria": f"{comp_name}이 디자인 명세에 따라 올바르게 작동해야 함"
        })
        
        # 컴포넌트별 특수 요구사항
        if "button" in comp_name.lower():
            requirements.append({
                "id": f"FR-{comp_name.replace(' ', '-').lower()}-interaction",
                "category": "interaction",
                "title": f"{comp_name} 인터랙션 구현",
                "description": "버튼 클릭, 호버, 포커스 상태 처리",
                "priority": "high",
                "complexity": "simple",
                "estimated_hours": 2,
                "dependencies": [f"FR-{comp_name.replace(' ', '-').lower()}"]
            })
        
        return requirements
    
    def _generate_interaction_requirements(self, interaction_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """인터랙션별 요구사항 생성"""
        requirements = []
        interaction_types = interaction_analysis.get("interaction_types", [])
        
        for i, interaction_type in enumerate(interaction_types):
            requirements.append({
                "id": f"FR-interaction-{i+1}",
                "category": "interaction",
                "title": f"{interaction_type.replace('_', ' ').title()} 구현",
                "description": f"{interaction_type} 인터랙션 패턴 구현",
                "priority": "medium",
                "complexity": "medium",
                "estimated_hours": 6
            })
        
        return requirements
    
    def _generate_layout_requirements(self, layout_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """레이아웃별 요구사항 생성"""
        requirements = []
        
        if layout_analysis.get("responsive_design", False):
            requirements.append({
                "id": "FR-responsive-layout",
                "category": "layout",
                "title": "반응형 레이아웃 구현",
                "description": "다양한 화면 크기에 대응하는 반응형 레이아웃",
                "priority": "high",
                "complexity": "complex",
                "estimated_hours": 12
            })
        
        if layout_analysis.get("auto_layout", False):
            requirements.append({
                "id": "FR-auto-layout",
                "category": "layout",
                "title": "자동 레이아웃 시스템",
                "description": "동적 콘텐츠에 대응하는 자동 레이아웃",
                "priority": "medium",
                "complexity": "medium",
                "estimated_hours": 8
            })
        
        return requirements
    
    def _deduplicate_and_prioritize(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거 및 우선순위 설정"""
        # 제목 기준으로 중복 제거
        seen_titles = set()
        unique_requirements = []
        
        for req in requirements:
            title = req.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                unique_requirements.append(req)
        
        # 우선순위 기준으로 정렬
        priority_order = {"high": 1, "medium": 2, "low": 3}
        unique_requirements.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 2))
        
        return unique_requirements
    
    def _create_priority_matrix(self, functional_requirements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """우선순위 매트릭스 생성"""
        matrix = {"high": [], "medium": [], "low": []}
        
        for req in functional_requirements:
            priority = req.get("priority", "medium")
            title = req.get("title", "")
            matrix[priority].append(title)
        
        return matrix
    
    def _assess_risks(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """리스크 평가"""
        risks = []
        
        # 복잡도 기반 리스크
        overall_assessment = analysis_data.get("overall_assessment", {})
        if overall_assessment.get("development_complexity") == "high":
            risks.append({
                "risk": "개발 복잡도 높음",
                "impact": "high",
                "probability": "medium",
                "mitigation": "경험 있는 개발자 투입, 점진적 개발 접근"
            })
        
        # 성능 리스크
        performance_analysis = analysis_data.get("performance_analysis", {})
        if performance_analysis.get("bundle_size_impact") == "high":
            risks.append({
                "risk": "성능 저하 가능성",
                "impact": "medium",
                "probability": "high",
                "mitigation": "코드 스플리팅, 지연 로딩, 번들 분석"
            })
        
        return risks
    
    def _estimate_resources(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """리소스 요구사항 추정"""
        overall_assessment = analysis_data.get("overall_assessment", {})
        estimated_effort = overall_assessment.get("estimated_effort", {})
        
        return {
            "development_team": {
                "frontend_developers": 1 if estimated_effort.get("effort_level") == "low" else 2,
                "ui_ux_designer": 1,
                "qa_engineer": 1
            },
            "timeline": {
                "estimated_weeks": estimated_effort.get("developer_weeks", 4),
                "sprints": estimated_effort.get("estimated_sprints", 2)
            },
            "tools_and_infrastructure": [
                "개발 환경 설정",
                "CI/CD 파이프라인",
                "테스팅 도구",
                "디자인 시스템 도구"
            ]
        }
    
    # Template and Pattern loading methods
    def _load_requirement_templates(self) -> Dict[str, Any]:
        """요구사항 템플릿 로드"""
        return {
            "functional": {
                "ui_component": "사용자는 {component_name}을 통해 {functionality}할 수 있어야 한다.",
                "interaction": "사용자가 {action}을 수행할 때 {response}가 발생해야 한다.",
                "form": "사용자는 {form_name}을 통해 {data}를 입력하고 제출할 수 있어야 한다."
            },
            "non_functional": {
                "performance": "시스템은 {condition} 조건에서 {target_time} 이내에 응답해야 한다.",
                "accessibility": "모든 기능은 {accessibility_standard} 표준을 준수해야 한다.",
                "security": "시스템은 {security_measure}를 통해 보안을 보장해야 한다."
            }
        }
    
    def _load_user_story_patterns(self) -> Dict[str, str]:
        """사용자 스토리 패턴 로드"""
        return {
            "basic": "사용자로서, {action}을 통해 {benefit}을 얻고 싶다.",
            "role_based": "{role}로서, {action}을 통해 {benefit}을 얻고 싶다.",
            "conditional": "{condition}일 때, {role}로서 {action}을 통해 {benefit}을 얻고 싶다."
        }
    
    def _infer_target_audience(self, analysis_data: Dict[str, Any]) -> str:
        """대상 사용자 추론"""
        # 간단한 휴리스틱 기반 추론
        url = analysis_data.get("url", "").lower()
        
        if "admin" in url or "dashboard" in url:
            return "관리자 및 내부 사용자"
        elif "mobile" in url or "app" in url:
            return "모바일 앱 사용자"
        elif "enterprise" in url or "business" in url:
            return "기업 사용자"
        else:
            return "일반 사용자"
    
    def _infer_platform(self, analysis_data: Dict[str, Any]) -> str:
        """플랫폼 추론"""
        layout_analysis = analysis_data.get("layout_analysis", {})
        
        if layout_analysis.get("responsive_design", False):
            return "웹 (반응형)"
        else:
            return "웹 (데스크톱)"
    
    def _extract_key_features(self, functional_requirements: List[Dict[str, Any]]) -> List[str]:
        """핵심 기능 추출"""
        key_features = []
        
        # 고우선순위 요구사항에서 핵심 기능 추출
        high_priority_reqs = [req for req in functional_requirements if req.get("priority") == "high"]
        
        for req in high_priority_reqs[:5]:  # 상위 5개만
            title = req.get("title", "")
            if title:
                key_features.append(title)
        
        return key_features
    
    def _generate_executive_summary(self, basic_info: Dict[str, Any], analysis_data: Dict[str, Any]) -> str:
        """경영진 요약 생성"""
        product_name = basic_info.get("product_name", "프로덕트")
        complexity = basic_info.get("complexity_level", "medium")
        
        component_count = analysis_data.get("component_analysis", {}).get("total_components", 0)
        
        return f"""
{product_name} 개발을 위한 요구사항 정의서입니다. 
디자인 분석 결과 {component_count}개의 주요 컴포넌트가 식별되었으며, 
개발 복잡도는 {complexity} 수준으로 평가됩니다.

본 프로젝트는 사용자 경험 향상과 기술적 완성도를 모두 고려하여 
체계적이고 확장 가능한 솔루션 구축을 목표로 합니다.
        """.strip()
    
    def _generate_problem_statement(self, analysis_data: Dict[str, Any]) -> str:
        """문제 정의 생성"""
        return """
현재 디자인된 사용자 인터페이스를 실제 동작하는 애플리케이션으로 구현하여 
사용자가 원활하고 직관적인 경험을 할 수 있도록 하는 것이 주요 과제입니다.

디자인과 개발 간의 일관성을 유지하면서도 성능, 접근성, 유지보수성을 
모두 고려한 고품질 솔루션이 필요합니다.
        """.strip()
    
    def _generate_solution_overview(self, analysis_data: Dict[str, Any]) -> str:
        """솔루션 개요 생성"""
        layout_analysis = analysis_data.get("layout_analysis", {})
        responsive = layout_analysis.get("responsive_design", False)
        
        solution = "컴포넌트 기반 아키텍처를 통해 재사용 가능하고 확장 가능한 UI 시스템을 구축합니다."
        
        if responsive:
            solution += " 반응형 디자인을 적용하여 다양한 디바이스에서 최적화된 경험을 제공합니다."
        
        solution += " 접근성과 성능을 고려한 모던 웹 표준을 준수하여 개발합니다."
        
        return solution 