"""
부동산 시장 분석 도구

지역별 시세 분석, 가격 예측, 시장 트렌드 분석, 정책 영향 분석
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalyzeMarketTrendInput(BaseModel):
    """시장 트렌드 분석 입력 스키마"""
    region: str = Field(description="지역 (예: 서울 강남구, 경기 성남시)")
    property_type: str = Field(description="부동산 유형 (apartment/office_tel/commercial)")
    period_months: Optional[int] = Field(default=12, description="분석 기간 (월)")


class PredictPriceInput(BaseModel):
    """가격 예측 입력 스키마"""
    region: str = Field(description="지역")
    property_type: str = Field(description="부동산 유형")
    current_price: float = Field(description="현재 가격 (만원)")
    prediction_months: Optional[int] = Field(default=6, description="예측 기간 (월)")


class AnalyzePolicyImpactInput(BaseModel):
    """정책 영향 분석 입력 스키마"""
    policy_type: str = Field(description="정책 유형 (housing_policy/interest_rate/tax_policy)")
    region: Optional[str] = Field(default=None, description="영향받는 지역")


class MarketTools:
    """
    부동산 시장 분석 도구 모음
    
    지역별 시세 분석, 가격 예측, 시장 트렌드 분석, 정책 영향 분석
    """
    
    def __init__(self, data_dir: str = "real_estate_data"):
        """
        MarketTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.market_data_file = self.data_dir / "market_data.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.market_data_file.exists():
            with open(self.market_data_file, 'r', encoding='utf-8') as f:
                self.market_data = json.load(f)
        else:
            self.market_data = {}
    
    def _save_data(self):
        """데이터 저장"""
        with open(self.market_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.market_data, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """Market 도구 초기화"""
        self.tools.append(self._create_analyze_market_trend_tool())
        self.tools.append(self._create_predict_price_tool())
        self.tools.append(self._create_analyze_policy_impact_tool())
        self.tools.append(self._create_analyze_regional_growth_tool())
        logger.info(f"Initialized {len(self.tools)} market tools")
    
    def _create_analyze_market_trend_tool(self) -> BaseTool:
        @tool("market_analyze_trend", args_schema=AnalyzeMarketTrendInput)
        def analyze_market_trend(
            region: str,
            property_type: str,
            period_months: Optional[int] = 12
        ) -> str:
            """
            시장 트렌드를 분석합니다.
            Args:
                region: 지역
                property_type: 부동산 유형
                period_months: 분석 기간 (월)
            Returns:
                시장 트렌드 분석 결과 (JSON 문자열)
            """
            logger.info(f"Analyzing market trend for {region}, type: {property_type}")
            
            # 시장 데이터 키 생성
            data_key = f"{region}_{property_type}"
            
            # 기존 데이터가 있으면 사용, 없으면 기본값
            if data_key in self.market_data:
                trend_data = self.market_data[data_key]
            else:
                # 기본 트렌드 데이터 (실제로는 API나 데이터베이스에서 가져와야 함)
                trend_data = {
                    "region": region,
                    "property_type": property_type,
                    "price_trend": "stable",  # rising/falling/stable
                    "volume_trend": "stable",
                    "average_price": 0.0,
                    "price_change_rate": 0.0,
                    "volume_change_rate": 0.0
                }
            
            result = {
                "region": region,
                "property_type": property_type,
                "period_months": period_months,
                "trend_analysis": {
                    "price_trend": trend_data.get("price_trend", "stable"),
                    "volume_trend": trend_data.get("volume_trend", "stable"),
                    "average_price": trend_data.get("average_price", 0.0),
                    "price_change_rate": trend_data.get("price_change_rate", 0.0),
                    "volume_change_rate": trend_data.get("volume_change_rate", 0.0),
                    "market_phase": self._determine_market_phase(trend_data)
                },
                "recommendations": self._generate_trend_recommendations(trend_data)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return analyze_market_trend
    
    def _create_predict_price_tool(self) -> BaseTool:
        @tool("market_predict_price", args_schema=PredictPriceInput)
        def predict_price(
            region: str,
            property_type: str,
            current_price: float,
            prediction_months: Optional[int] = 6
        ) -> str:
            """
            부동산 가격을 예측합니다.
            Args:
                region: 지역
                property_type: 부동산 유형
                current_price: 현재 가격 (만원)
                prediction_months: 예측 기간 (월)
            Returns:
                가격 예측 결과 (JSON 문자열)
            """
            logger.info(f"Predicting price for {region}, type: {property_type}, current: {current_price}만원")
            
            # 간단한 예측 모델 (실제로는 ML 모델 사용)
            data_key = f"{region}_{property_type}"
            trend_data = self.market_data.get(data_key, {})
            price_change_rate = trend_data.get("price_change_rate", 0.0) / 100.0
            
            # 월별 가격 변화율 계산
            monthly_rate = price_change_rate / 12.0
            
            # 예측 가격 계산
            predicted_price = current_price * (1 + monthly_rate * prediction_months)
            price_change = predicted_price - current_price
            price_change_percentage = (price_change / current_price) * 100 if current_price > 0 else 0.0
            
            # 신뢰도 계산 (데이터가 많을수록 높음)
            confidence = min(90.0, 50.0 + abs(price_change_rate) * 2)
            
            result = {
                "region": region,
                "property_type": property_type,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change": price_change,
                "price_change_percentage": price_change_percentage,
                "prediction_months": prediction_months,
                "confidence": confidence,
                "scenarios": {
                    "optimistic": current_price * (1 + monthly_rate * prediction_months * 1.5),
                    "realistic": predicted_price,
                    "pessimistic": current_price * (1 + monthly_rate * prediction_months * 0.5)
                }
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return predict_price
    
    def _create_analyze_policy_impact_tool(self) -> BaseTool:
        @tool("market_analyze_policy_impact", args_schema=AnalyzePolicyImpactInput)
        def analyze_policy_impact(
            policy_type: str,
            region: Optional[str] = None
        ) -> str:
            """
            정책이 부동산 시장에 미치는 영향을 분석합니다.
            Args:
                policy_type: 정책 유형
                region: 영향받는 지역
            Returns:
                정책 영향 분석 결과 (JSON 문자열)
            """
            logger.info(f"Analyzing policy impact: {policy_type}, region: {region}")
            
            # 정책 유형별 영향 분석
            policy_impacts = {
                "housing_policy": {
                    "impact_level": "high",
                    "price_impact": -5.0,  # 가격 하락률 (%)
                    "volume_impact": -10.0,  # 거래량 변화 (%)
                    "description": "주택 정책은 시장에 큰 영향을 미칩니다."
                },
                "interest_rate": {
                    "impact_level": "high",
                    "price_impact": -3.0,
                    "volume_impact": -15.0,
                    "description": "금리 인상은 부동산 가격과 거래량에 부정적 영향을 미칩니다."
                },
                "tax_policy": {
                    "impact_level": "medium",
                    "price_impact": -2.0,
                    "volume_impact": -5.0,
                    "description": "세금 정책은 장기적으로 시장에 영향을 미칩니다."
                }
            }
            
            impact = policy_impacts.get(policy_type, {
                "impact_level": "low",
                "price_impact": 0.0,
                "volume_impact": 0.0,
                "description": "정책 영향이 제한적입니다."
            })
            
            result = {
                "policy_type": policy_type,
                "region": region,
                "impact_analysis": impact,
                "recommendations": self._generate_policy_recommendations(policy_type, impact)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return analyze_policy_impact
    
    def _create_analyze_regional_growth_tool(self) -> BaseTool:
        @tool("market_analyze_regional_growth")
        def analyze_regional_growth(region: str) -> str:
            """
            지역별 성장 잠재력을 분석합니다.
            Args:
                region: 지역
            Returns:
                지역 성장 잠재력 분석 결과 (JSON 문자열)
            """
            logger.info(f"Analyzing regional growth potential for {region}")
            
            # 지역별 성장 지표 (실제로는 인프라, 인구, 산업 데이터 기반)
            growth_factors = {
                "인프라 개발": 0.8,
                "인구 유입": 0.7,
                "산업 발전": 0.6,
                "교통 접근성": 0.75,
                "교육 환경": 0.65
            }
            
            overall_score = sum(growth_factors.values()) / len(growth_factors) * 100
            
            result = {
                "region": region,
                "growth_potential_score": overall_score,
                "growth_factors": growth_factors,
                "growth_phase": "emerging" if overall_score < 70 else "mature" if overall_score > 85 else "developing",
                "recommendations": self._generate_growth_recommendations(overall_score)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return analyze_regional_growth
    
    def _determine_market_phase(self, trend_data: Dict[str, Any]) -> str:
        """시장 단계 판단"""
        price_change = trend_data.get("price_change_rate", 0.0)
        volume_change = trend_data.get("volume_change_rate", 0.0)
        
        if price_change > 5 and volume_change > 10:
            return "bull_market"
        elif price_change < -5 and volume_change < -10:
            return "bear_market"
        elif price_change > 0 and volume_change < 0:
            return "overvalued"
        else:
            return "stable"
    
    def _generate_trend_recommendations(self, trend_data: Dict[str, Any]) -> List[str]:
        """트렌드 기반 권장사항 생성"""
        recommendations = []
        market_phase = self._determine_market_phase(trend_data)
        
        if market_phase == "bull_market":
            recommendations.append("상승장 단계입니다. 신중한 투자를 권장합니다.")
        elif market_phase == "bear_market":
            recommendations.append("하락장 단계입니다. 매수 기회를 모니터링하세요.")
        elif market_phase == "overvalued":
            recommendations.append("과열 단계입니다. 매도 타이밍을 고려하세요.")
        else:
            recommendations.append("안정적 시장입니다. 장기 투자 관점에서 접근하세요.")
        
        return recommendations
    
    def _generate_policy_recommendations(self, policy_type: str, impact: Dict[str, Any]) -> List[str]:
        """정책 영향 기반 권장사항 생성"""
        recommendations = []
        
        if impact["impact_level"] == "high":
            if impact["price_impact"] < 0:
                recommendations.append("정책 영향이 큽니다. 투자 결정을 신중히 하세요.")
                recommendations.append("가격 하락 가능성을 고려하여 현금 보유를 늘리는 것을 권장합니다.")
            else:
                recommendations.append("정책이 시장에 긍정적 영향을 미칠 수 있습니다.")
        
        return recommendations
    
    def _generate_growth_recommendations(self, score: float) -> List[str]:
        """성장 잠재력 기반 권장사항 생성"""
        recommendations = []
        
        if score >= 80:
            recommendations.append("높은 성장 잠재력을 가진 지역입니다. 장기 투자 관점에서 유리합니다.")
        elif score >= 60:
            recommendations.append("중간 수준의 성장 잠재력입니다. 신중한 투자 검토가 필요합니다.")
        else:
            recommendations.append("성장 잠재력이 제한적입니다. 투자 전 철저한 검토가 필요합니다.")
        
        return recommendations
    
    def get_tools(self) -> List[BaseTool]:
        """모든 Market 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 Market 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

