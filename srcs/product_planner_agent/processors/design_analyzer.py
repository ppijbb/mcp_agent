"""
Design Analyzer Module

Figma 디자인 데이터를 분석하여 요구사항으로 변환하는 엔진
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# 로컬 설정 imports - 절대 import로 수정
try:
    from config import FIGMA_CONFIG, ANALYSIS_CONFIG
except ImportError:
    # Fallback 설정
    FIGMA_CONFIG = {"analysis_depth": "comprehensive"}
    ANALYSIS_CONFIG = {"timeout": 30}

logger = logging.getLogger(__name__)

class DesignAnalyzer:
    """
    디자인 분석 엔진
    
    Figma 디자인 메타데이터를 분석하여 프로덕트 요구사항으로 변환하는 핵심 로직
    """
    
    def __init__(self):
        """DesignAnalyzer 초기화"""
        self.analysis_config = ANALYSIS_CONFIG
        self.figma_config = FIGMA_CONFIG
        
        # 분석 결과 캐시
        self._analysis_cache = {}
        
        logger.info("DesignAnalyzer 초기화 완료")
    
    async def analyze_design(self, figma_url: str) -> Dict[str, Any]:
        """
        디자인 종합 분석 (외부 호출용 메인 메서드)
        
        Args:
            figma_url: Figma 디자인 URL
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            logger.info(f"디자인 분석 시작: {figma_url}")
            
            # 실제로는 FigmaIntegration을 통해 메타데이터를 받아야 하지만,
            # 여기서는 모의 데이터로 분석 로직을 구현
            mock_metadata = self._create_mock_metadata(figma_url)
            
            analysis_result = await self._perform_analysis(mock_metadata)
            
            logger.info("디자인 분석 완료")
            return analysis_result
            
        except Exception as e:
            logger.error(f"디자인 분석 실패: {str(e)}")
            raise
    
    async def _perform_analysis(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        실제 분석 수행
        
        Args:
            metadata: Figma 메타데이터
            
        Returns:
            분석 결과
        """
        try:
            parsed_data = metadata.get("parsed_data", {})
            
            # 1. 컴포넌트 분석
            component_analysis = self._analyze_components(parsed_data.get("components", []))
            
            # 2. 레이아웃 분석
            layout_analysis = self._analyze_layout(parsed_data.get("layout_info", {}))
            
            # 3. 인터랙션 분석
            interaction_analysis = self._analyze_interactions(parsed_data)
            
            # 4. 접근성 분석
            accessibility_analysis = self._analyze_accessibility(parsed_data)
            
            # 5. 성능 영향 분석
            performance_analysis = self._analyze_performance_impact(parsed_data)
            
            # 6. 종합 평가
            overall_assessment = self._generate_overall_assessment(
                component_analysis, layout_analysis, interaction_analysis,
                accessibility_analysis, performance_analysis
            )
            
            return {
                "url": metadata.get("figma_ids", {}).get("url", ""),
                "analysis_timestamp": datetime.now().isoformat(),
                "component_analysis": component_analysis,
                "layout_analysis": layout_analysis,
                "interaction_analysis": interaction_analysis,
                "accessibility_analysis": accessibility_analysis,
                "performance_analysis": performance_analysis,
                "overall_assessment": overall_assessment,
                "confidence_score": self._calculate_confidence_score(parsed_data),
                "recommendations": self._generate_recommendations(parsed_data)
            }
            
        except Exception as e:
            logger.error(f"분석 수행 중 오류: {str(e)}")
            raise
    
    def _analyze_components(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """컴포넌트 분석"""
        try:
            analysis = {
                "total_components": len(components),
                "component_types": {},
                "reusability_score": 0,
                "complexity_levels": {"simple": 0, "medium": 0, "complex": 0},
                "design_system_compliance": 0,
                "components_detail": []
            }
            
            for component in components:
                comp_type = component.get("type", "UNKNOWN")
                comp_name = component.get("name", "Unknown")
                
                # 타입별 카운트
                analysis["component_types"][comp_type] = analysis["component_types"].get(comp_type, 0) + 1
                
                # 복잡도 평가
                complexity = self._assess_component_complexity(component)
                analysis["complexity_levels"][complexity] += 1
                
                # 상세 정보
                detail = {
                    "name": comp_name,
                    "type": comp_type,
                    "complexity": complexity,
                    "reusability": self._assess_reusability(component),
                    "design_patterns": self._identify_design_patterns(component),
                    "development_requirements": self._extract_dev_requirements(component)
                }
                
                analysis["components_detail"].append(detail)
            
            # 전체 재사용성 점수 계산
            if components:
                analysis["reusability_score"] = sum(
                    comp["reusability"] for comp in analysis["components_detail"]
                ) / len(components)
            
            # 디자인 시스템 준수도 평가
            analysis["design_system_compliance"] = self._assess_design_system_compliance(components)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"컴포넌트 분석 중 오류: {str(e)}")
            return {"error": "컴포넌트 분석 실패", "total_components": 0}
    
    def _analyze_layout(self, layout_info: Dict[str, Any]) -> Dict[str, Any]:
        """레이아웃 분석"""
        try:
            return {
                "layout_type": layout_info.get("direction", "unknown"),
                "responsive_design": layout_info.get("responsive", False),
                "auto_layout": layout_info.get("auto_layout", False),
                "spacing_consistency": self._check_spacing_consistency(layout_info),
                "grid_usage": layout_info.get("grid_system", "none"),
                "layout_complexity": self._assess_layout_complexity(layout_info),
                "mobile_considerations": self._analyze_mobile_layout(layout_info),
                "accessibility_features": self._check_layout_accessibility(layout_info)
            }
        except Exception as e:
            logger.warning(f"레이아웃 분석 중 오류: {str(e)}")
            return {"error": "레이아웃 분석 실패"}
    
    def _analyze_interactions(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """인터랙션 분석"""
        try:
            components = parsed_data.get("components", [])
            
            interactions = {
                "interaction_types": [],
                "user_flows": [],
                "state_management": {},
                "validation_needs": [],
                "feedback_mechanisms": []
            }
            
            for component in components:
                comp_type = component.get("type", "")
                comp_name = component.get("name", "")
                
                # 인터랙션 타입 식별
                if "button" in comp_name.lower() or comp_type == "BUTTON":
                    interactions["interaction_types"].append("click_action")
                    interactions["feedback_mechanisms"].append("button_states")
                
                if "input" in comp_name.lower() or "field" in comp_name.lower():
                    interactions["interaction_types"].append("form_input")
                    interactions["validation_needs"].append("input_validation")
                
                if "modal" in comp_name.lower() or "popup" in comp_name.lower():
                    interactions["interaction_types"].append("modal_interaction")
                    interactions["state_management"]["modal_state"] = True
            
            # 중복 제거
            interactions["interaction_types"] = list(set(interactions["interaction_types"]))
            interactions["validation_needs"] = list(set(interactions["validation_needs"]))
            interactions["feedback_mechanisms"] = list(set(interactions["feedback_mechanisms"]))
            
            return interactions
            
        except Exception as e:
            logger.warning(f"인터랙션 분석 중 오류: {str(e)}")
            return {"error": "인터랙션 분석 실패"}
    
    def _analyze_accessibility(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """접근성 분석"""
        try:
            text_nodes = parsed_data.get("text_nodes", [])
            components = parsed_data.get("components", [])
            
            accessibility = {
                "color_contrast_needs": [],
                "keyboard_navigation": [],
                "screen_reader_support": [],
                "focus_management": [],
                "semantic_markup": [],
                "compliance_level": "basic"
            }
            
            # 텍스트 대비 확인 필요
            if text_nodes:
                accessibility["color_contrast_needs"].append("텍스트-배경 대비 확인")
            
            # 컴포넌트별 접근성 요구사항
            for component in components:
                comp_name = component.get("name", "").lower()
                
                if "button" in comp_name:
                    accessibility["keyboard_navigation"].append("버튼 키보드 접근")
                    accessibility["screen_reader_support"].append("버튼 레이블")
                
                if "input" in comp_name or "field" in comp_name:
                    accessibility["screen_reader_support"].append("입력 필드 레이블")
                    accessibility["focus_management"].append("입력 필드 포커스")
                
                if "form" in comp_name:
                    accessibility["semantic_markup"].append("폼 시맨틱 구조")
            
            # 준수 레벨 평가
            total_requirements = sum(len(v) for v in accessibility.values() if isinstance(v, list))
            if total_requirements > 10:
                accessibility["compliance_level"] = "advanced"
            elif total_requirements > 5:
                accessibility["compliance_level"] = "intermediate"
            
            return accessibility
            
        except Exception as e:
            logger.warning(f"접근성 분석 중 오류: {str(e)}")
            return {"error": "접근성 분석 실패"}
    
    def _analyze_performance_impact(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """성능 영향 분석"""
        try:
            components_count = len(parsed_data.get("components", []))
            frames_count = len(parsed_data.get("frames", []))
            
            performance = {
                "bundle_size_impact": "low",
                "rendering_complexity": "medium",
                "optimization_needs": [],
                "lazy_loading_candidates": [],
                "critical_path_items": []
            }
            
            # 번들 크기 영향
            if components_count > 20:
                performance["bundle_size_impact"] = "high"
                performance["optimization_needs"].append("컴포넌트 코드 스플리팅")
            elif components_count > 10:
                performance["bundle_size_impact"] = "medium"
            
            # 렌더링 복잡도
            if frames_count > 5:
                performance["rendering_complexity"] = "high"
                performance["optimization_needs"].append("가상화 고려")
            
            # 지연 로딩 후보
            if components_count > 15:
                performance["lazy_loading_candidates"].append("비중요 컴포넌트")
            
            # 크리티컬 패스
            performance["critical_path_items"] = ["메인 레이아웃", "기본 스타일"]
            
            return performance
            
        except Exception as e:
            logger.warning(f"성능 영향 분석 중 오류: {str(e)}")
            return {"error": "성능 영향 분석 실패"}
    
    def _generate_overall_assessment(self, *analyses) -> Dict[str, Any]:
        """종합 평가 생성"""
        try:
            component_analysis, layout_analysis, interaction_analysis, accessibility_analysis, performance_analysis = analyses
            
            assessment = {
                "development_complexity": "medium",
                "estimated_effort": self._estimate_development_effort(component_analysis),
                "technical_risks": [],
                "priority_areas": [],
                "success_factors": []
            }
            
            # 개발 복잡도 평가
            complexity_factors = 0
            
            if component_analysis.get("total_components", 0) > 10:
                complexity_factors += 1
                assessment["technical_risks"].append("많은 컴포넌트 관리")
            
            if layout_analysis.get("responsive_design", False):
                complexity_factors += 1
                assessment["priority_areas"].append("반응형 구현")
            
            if len(interaction_analysis.get("interaction_types", [])) > 3:
                complexity_factors += 1
                assessment["technical_risks"].append("복잡한 인터랙션")
            
            # 복잡도 레벨 설정
            if complexity_factors > 2:
                assessment["development_complexity"] = "high"
            elif complexity_factors == 0:
                assessment["development_complexity"] = "low"
            
            # 성공 요인
            if component_analysis.get("reusability_score", 0) > 0.7:
                assessment["success_factors"].append("높은 컴포넌트 재사용성")
            
            if accessibility_analysis.get("compliance_level") == "advanced":
                assessment["success_factors"].append("우수한 접근성 고려")
            
            return assessment
            
        except Exception as e:
            logger.warning(f"종합 평가 생성 중 오류: {str(e)}")
            return {"error": "종합 평가 생성 실패"}
    
    def _calculate_confidence_score(self, parsed_data: Dict[str, Any]) -> float:
        """분석 신뢰도 점수 계산"""
        try:
            score = 0.0
            
            # 데이터 완성도 기반
            if parsed_data.get("components"):
                score += 0.3
            if parsed_data.get("frames"):
                score += 0.2
            if parsed_data.get("text_nodes"):
                score += 0.2
            if parsed_data.get("variables"):
                score += 0.15
            if parsed_data.get("styles"):
                score += 0.15
            
            return min(score, 1.0)
            
        except:
            return 0.5
    
    def _generate_recommendations(self, parsed_data: Dict[str, Any]) -> List[str]:
        """개발 권장사항 생성"""
        recommendations = []
        
        try:
            components_count = len(parsed_data.get("components", []))
            
            if components_count > 15:
                recommendations.append("컴포넌트 라이브러리 구축 고려")
                recommendations.append("Storybook 등 컴포넌트 문서화 도구 사용")
            
            if parsed_data.get("variables"):
                recommendations.append("디자인 토큰 시스템 구축")
            
            if parsed_data.get("layout_info", {}).get("responsive", False):
                recommendations.append("모바일 퍼스트 접근법 적용")
                recommendations.append("브레이크포인트 전략 수립")
            
            recommendations.append("디자인-개발 간 정기적 동기화 프로세스 구축")
            recommendations.append("자동화된 디자인 토큰 추출 도구 도입")
            
        except Exception as e:
            logger.warning(f"권장사항 생성 중 오류: {str(e)}")
        
        return recommendations
    
    # Helper methods
    def _assess_component_complexity(self, component: Dict[str, Any]) -> str:
        """컴포넌트 복잡도 평가"""
        # 간단한 로직 - 실제로는 더 정교한 분석 필요
        properties = component.get("properties", {})
        if len(properties) > 5:
            return "complex"
        elif len(properties) > 2:
            return "medium"
        else:
            return "simple"
    
    def _assess_reusability(self, component: Dict[str, Any]) -> float:
        """재사용성 점수 (0.0 ~ 1.0)"""
        # 기본 점수
        score = 0.5
        
        # 컴포넌트 타입에 따른 점수 조정
        comp_type = component.get("type", "")
        if comp_type == "COMPONENT":
            score += 0.3
        
        # 속성의 다양성
        properties = component.get("properties", {})
        if len(properties) > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _identify_design_patterns(self, component: Dict[str, Any]) -> List[str]:
        """디자인 패턴 식별"""
        patterns = []
        comp_name = component.get("name", "").lower()
        
        if "button" in comp_name:
            patterns.append("action_pattern")
        if "card" in comp_name:
            patterns.append("content_pattern")
        if "modal" in comp_name:
            patterns.append("overlay_pattern")
        
        return patterns
    
    def _extract_dev_requirements(self, component: Dict[str, Any]) -> List[str]:
        """개발 요구사항 추출"""
        requirements = []
        comp_name = component.get("name", "").lower()
        
        if "button" in comp_name:
            requirements.extend(["클릭 이벤트 처리", "상태 관리 (hover, active, disabled)"])
        if "input" in comp_name:
            requirements.extend(["입력 검증", "에러 메시지 표시"])
        if "form" in comp_name:
            requirements.extend(["폼 제출 처리", "데이터 검증"])
        
        return requirements
    
    def _assess_design_system_compliance(self, components: List[Dict[str, Any]]) -> float:
        """디자인 시스템 준수도 평가"""
        if not components:
            return 0.0
        
        # 간단한 휴리스틱: 컴포넌트 이름에 일관성이 있는지 확인
        consistent_naming = 0
        for component in components:
            name = component.get("name", "")
            if any(keyword in name for keyword in ["Button", "Input", "Card", "Modal"]):
                consistent_naming += 1
        
        return consistent_naming / len(components)
    
    def _check_spacing_consistency(self, layout_info: Dict[str, Any]) -> bool:
        """간격 일관성 확인"""
        return layout_info.get("spacing") is not None
    
    def _assess_layout_complexity(self, layout_info: Dict[str, Any]) -> str:
        """레이아웃 복잡도 평가"""
        complexity_score = 0
        
        if layout_info.get("responsive", False):
            complexity_score += 1
        if layout_info.get("auto_layout", False):
            complexity_score += 1
        if layout_info.get("grid_system") != "none":
            complexity_score += 1
        
        if complexity_score >= 2:
            return "high"
        elif complexity_score == 1:
            return "medium"
        else:
            return "low"
    
    def _analyze_mobile_layout(self, layout_info: Dict[str, Any]) -> Dict[str, Any]:
        """모바일 레이아웃 분석"""
        return {
            "responsive_support": layout_info.get("responsive", False),
            "breakpoints": layout_info.get("breakpoints", []),
            "mobile_optimized": "mobile" in layout_info.get("breakpoints", [])
        }
    
    def _check_layout_accessibility(self, layout_info: Dict[str, Any]) -> List[str]:
        """레이아웃 접근성 체크"""
        features = []
        
        if layout_info.get("auto_layout", False):
            features.append("논리적 순서 유지")
        
        return features
    
    def _estimate_development_effort(self, component_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """개발 노력 추정"""
        total_components = component_analysis.get("total_components", 0)
        complexity_levels = component_analysis.get("complexity_levels", {})
        
        # 스토리 포인트 계산
        story_points = (
            complexity_levels.get("simple", 0) * 1 +
            complexity_levels.get("medium", 0) * 3 +
            complexity_levels.get("complex", 0) * 8
        )
        
        return {
            "story_points": story_points,
            "estimated_sprints": max(1, story_points // 20),
            "developer_weeks": max(1, story_points // 10),
            "effort_level": "high" if story_points > 40 else "medium" if story_points > 20 else "low"
        }
    
    def _create_mock_metadata(self, figma_url: str) -> Dict[str, Any]:
        """모의 메타데이터 생성 (개발/테스트용)"""
        return {
            "figma_ids": {"url": figma_url},
            "parsed_data": {
                "components": [
                    {
                        "name": "Primary Button",
                        "type": "COMPONENT",
                        "properties": {"variant": "primary", "size": "medium", "state": "default"}
                    },
                    {
                        "name": "Text Input",
                        "type": "COMPONENT",
                        "properties": {"type": "text", "required": True, "placeholder": "Enter text"}
                    },
                    {
                        "name": "Card Component",
                        "type": "COMPONENT",
                        "properties": {"elevation": "medium", "padding": "16px"}
                    }
                ],
                "frames": [
                    {"name": "Login Screen", "width": 375, "height": 812}
                ],
                "text_nodes": [
                    {"text": "Welcome", "font_size": 24, "font_weight": "bold"},
                    {"text": "Please sign in", "font_size": 16, "font_weight": "regular"}
                ],
                "variables": {
                    "colors": {"primary": "#007AFF", "secondary": "#8E8E93"},
                    "spacing": {"sm": 8, "md": 16, "lg": 24}
                },
                "styles": {
                    "text_styles": ["heading", "body", "caption"],
                    "color_styles": ["primary", "secondary", "neutral"]
                },
                "layout_info": {
                    "responsive": True,
                    "auto_layout": True,
                    "direction": "vertical",
                    "spacing": 16,
                    "breakpoints": ["mobile", "tablet", "desktop"],
                    "grid_system": "12-column"
                }
            }
        } 