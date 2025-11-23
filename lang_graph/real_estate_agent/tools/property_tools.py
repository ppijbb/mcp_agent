"""
부동산 매물 매칭 및 추천 도구

사용자 프로필 기반 추천, 스마트 필터링, 비교 분석, 투자 적합도 점수
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MatchPropertyInput(BaseModel):
    """매물 매칭 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    budget: float = Field(description="예산 (만원)")
    investment_goal: str = Field(description="투자 목표 (capital_gain/rental_income/both)")
    risk_tolerance: str = Field(description="리스크 성향 (conservative/moderate/aggressive)")
    preferred_regions: Optional[List[str]] = Field(default=None, description="선호 지역 목록")
    property_types: Optional[List[str]] = Field(default=None, description="선호 부동산 유형")


class ComparePropertiesInput(BaseModel):
    """매물 비교 입력 스키마"""
    property_ids: List[str] = Field(description="비교할 매물 ID 목록")


class PropertyTools:
    """
    부동산 매물 매칭 및 추천 도구 모음
    
    사용자 프로필 기반 추천, 스마트 필터링, 비교 분석, 투자 적합도 점수
    """
    
    def __init__(self, data_dir: str = "real_estate_data"):
        """
        PropertyTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.property_data_file = self.data_dir / "property_data.json"
        self.user_profiles_file = self.data_dir / "user_profiles.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.property_data_file.exists():
            with open(self.property_data_file, 'r', encoding='utf-8') as f:
                self.property_data = json.load(f)
        else:
            self.property_data = {}
        
        if self.user_profiles_file.exists():
            with open(self.user_profiles_file, 'r', encoding='utf-8') as f:
                self.user_profiles = json.load(f)
        else:
            self.user_profiles = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.property_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.property_data, f, indent=2, ensure_ascii=False)
        
        with open(self.user_profiles_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_profiles, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Property 도구 초기화"""
        self.tools.append(self._create_match_property_tool())
        self.tools.append(self._create_compare_properties_tool())
        self.tools.append(self._create_calculate_suitability_score_tool())
        logger.info(f"Initialized {len(self.tools)} property tools")
    
    def _create_match_property_tool(self) -> BaseTool:
        @tool("property_match", args_schema=MatchPropertyInput)
        def match_property(
            user_id: str,
            budget: float,
            investment_goal: str,
            risk_tolerance: str,
            preferred_regions: Optional[List[str]] = None,
            property_types: Optional[List[str]] = None
        ) -> str:
            """
            사용자 프로필에 맞는 매물을 추천합니다.
            Args:
                user_id: 사용자 ID
                budget: 예산 (만원)
                investment_goal: 투자 목표
                risk_tolerance: 리스크 성향
                preferred_regions: 선호 지역 목록
                property_types: 선호 부동산 유형
            Returns:
                매칭된 매물 목록 (JSON 문자열)
            """
            logger.info(f"Matching properties for user {user_id}, budget: {budget}만원")
            
            # 사용자 프로필 저장
            user_profile = {
                "user_id": user_id,
                "budget": budget,
                "investment_goal": investment_goal,
                "risk_tolerance": risk_tolerance,
                "preferred_regions": preferred_regions or [],
                "property_types": property_types or [],
                "updated_at": datetime.now().isoformat()
            }
            self.user_profiles[user_id] = user_profile
            self._save_data()
            
            # 매물 필터링 및 점수 계산
            matched_properties = []
            for prop_id, prop_data in self.property_data.items():
                score = self._calculate_match_score(user_profile, prop_data)
                if score > 50:  # 50점 이상만 추천
                    prop_data["match_score"] = score
                    matched_properties.append(prop_data)
            
            # 점수 순으로 정렬
            matched_properties.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            
            result = {
                "user_id": user_id,
                "matched_count": len(matched_properties),
                "properties": matched_properties[:10]  # 상위 10개만 반환
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return match_property
    
    def _create_compare_properties_tool(self) -> BaseTool:
        @tool("property_compare", args_schema=ComparePropertiesInput)
        def compare_properties(property_ids: List[str]) -> str:
            """
            여러 매물을 비교 분석합니다.
            Args:
                property_ids: 비교할 매물 ID 목록
            Returns:
                매물 비교 분석 결과 (JSON 문자열)
            """
            logger.info(f"Comparing properties: {property_ids}")
            
            properties = []
            for prop_id in property_ids:
                if prop_id in self.property_data:
                    properties.append(self.property_data[prop_id])
            
            if not properties:
                return json.dumps({"error": "매물을 찾을 수 없습니다."}, ensure_ascii=False)
            
            # 비교 지표 계산
            comparison = {
                "property_count": len(properties),
                "properties": properties,
                "comparison_metrics": {
                    "price_range": {
                        "min": min(p.get("price", 0) for p in properties),
                        "max": max(p.get("price", 0) for p in properties),
                        "avg": sum(p.get("price", 0) for p in properties) / len(properties)
                    },
                    "roi_range": {
                        "min": min(p.get("expected_roi", 0) for p in properties),
                        "max": max(p.get("expected_roi", 0) for p in properties),
                        "avg": sum(p.get("expected_roi", 0) for p in properties) / len(properties)
                    },
                    "yield_range": {
                        "min": min(p.get("yield_rate", 0) for p in properties),
                        "max": max(p.get("yield_rate", 0) for p in properties),
                        "avg": sum(p.get("yield_rate", 0) for p in properties) / len(properties)
                    }
                },
                "recommendations": self._generate_comparison_recommendations(properties)
            }
            
            return json.dumps(comparison, ensure_ascii=False, indent=2)
        return compare_properties
    
    def _create_calculate_suitability_score_tool(self) -> BaseTool:
        @tool("property_calculate_suitability")
        def calculate_suitability_score(
            property_id: str,
            user_id: str
        ) -> str:
            """
            매물의 투자 적합도 점수를 계산합니다.
            Args:
                property_id: 매물 ID
                user_id: 사용자 ID
            Returns:
                투자 적합도 점수 (JSON 문자열)
            """
            logger.info(f"Calculating suitability score for property {property_id}, user {user_id}")
            
            if property_id not in self.property_data:
                return json.dumps({"error": "매물을 찾을 수 없습니다."}, ensure_ascii=False)
            
            if user_id not in self.user_profiles:
                return json.dumps({"error": "사용자 프로필을 찾을 수 없습니다."}, ensure_ascii=False)
            
            property_data = self.property_data[property_id]
            user_profile = self.user_profiles[user_id]
            
            score = self._calculate_match_score(user_profile, property_data)
            
            result = {
                "property_id": property_id,
                "user_id": user_id,
                "suitability_score": score,
                "score_breakdown": {
                    "budget_match": self._calculate_budget_match(user_profile, property_data),
                    "goal_match": self._calculate_goal_match(user_profile, property_data),
                    "risk_match": self._calculate_risk_match(user_profile, property_data),
                    "location_match": self._calculate_location_match(user_profile, property_data)
                },
                "recommendation": "highly_recommended" if score >= 80 else "recommended" if score >= 60 else "not_recommended"
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return calculate_suitability_score
    
    def _calculate_match_score(self, user_profile: Dict[str, Any], property_data: Dict[str, Any]) -> float:
        """매칭 점수 계산"""
        score = 0.0
        
        # 예산 매칭 (40점)
        score += self._calculate_budget_match(user_profile, property_data) * 0.4
        
        # 목표 매칭 (30점)
        score += self._calculate_goal_match(user_profile, property_data) * 0.3
        
        # 리스크 매칭 (20점)
        score += self._calculate_risk_match(user_profile, property_data) * 0.2
        
        # 지역 매칭 (10점)
        score += self._calculate_location_match(user_profile, property_data) * 0.1
        
        return score
    
    def _calculate_budget_match(self, user_profile: Dict[str, Any], property_data: Dict[str, Any]) -> float:
        """예산 매칭 점수"""
        budget = user_profile.get("budget", 0)
        price = property_data.get("price", 0)
        
        if price == 0:
            return 0.0
        
        if price <= budget:
            # 예산 내: 가격이 낮을수록 높은 점수
            return 100.0 - ((budget - price) / budget * 50.0)
        else:
            # 예산 초과: 초과 비율에 따라 감점
            over_ratio = (price - budget) / budget
            return max(0.0, 100.0 - (over_ratio * 200.0))
    
    def _calculate_goal_match(self, user_profile: Dict[str, Any], property_data: Dict[str, Any]) -> float:
        """투자 목표 매칭 점수"""
        goal = user_profile.get("investment_goal", "")
        roi = property_data.get("expected_roi", 0)
        yield_rate = property_data.get("yield_rate", 0)
        
        if goal == "capital_gain":
            # 자본이득 목표: ROI가 높을수록 좋음
            return min(100.0, roi * 10.0)
        elif goal == "rental_income":
            # 임대 수익 목표: Yield가 높을수록 좋음
            return min(100.0, yield_rate * 10.0)
        else:  # both
            # 둘 다: 평균
            return (min(100.0, roi * 10.0) + min(100.0, yield_rate * 10.0)) / 2.0
    
    def _calculate_risk_match(self, user_profile: Dict[str, Any], property_data: Dict[str, Any]) -> float:
        """리스크 매칭 점수"""
        risk_tolerance = user_profile.get("risk_tolerance", "moderate")
        property_risk = property_data.get("risk_level", "moderate")
        
        risk_map = {"conservative": 1, "moderate": 2, "aggressive": 3}
        user_risk = risk_map.get(risk_tolerance, 2)
        prop_risk = risk_map.get(property_risk, 2)
        
        # 리스크 차이가 작을수록 높은 점수
        risk_diff = abs(user_risk - prop_risk)
        return 100.0 - (risk_diff * 30.0)
    
    def _calculate_location_match(self, user_profile: Dict[str, Any], property_data: Dict[str, Any]) -> float:
        """지역 매칭 점수"""
        preferred_regions = user_profile.get("preferred_regions", [])
        property_region = property_data.get("region", "")
        
        if not preferred_regions:
            return 50.0  # 선호 지역이 없으면 중간 점수
        
        if property_region in preferred_regions:
            return 100.0
        else:
            return 30.0  # 선호 지역이 아니면 낮은 점수
    
    def _generate_comparison_recommendations(self, properties: List[Dict[str, Any]]) -> List[str]:
        """비교 분석 기반 권장사항 생성"""
        recommendations = []
        
        if not properties:
            return recommendations
        
        # 최고 ROI 매물
        best_roi = max(properties, key=lambda p: p.get("expected_roi", 0))
        recommendations.append(f"최고 ROI: {best_roi.get('property_id')} (ROI: {best_roi.get('expected_roi', 0):.2f}%)")
        
        # 최고 Yield 매물
        best_yield = max(properties, key=lambda p: p.get("yield_rate", 0))
        recommendations.append(f"최고 Yield: {best_yield.get('property_id')} (Yield: {best_yield.get('yield_rate', 0):.2f}%)")
        
        # 최저 가격 매물
        cheapest = min(properties, key=lambda p: p.get("price", float('inf')))
        recommendations.append(f"최저 가격: {cheapest.get('property_id')} ({cheapest.get('price', 0)}만원)")
        
        return recommendations
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Property 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Property 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

