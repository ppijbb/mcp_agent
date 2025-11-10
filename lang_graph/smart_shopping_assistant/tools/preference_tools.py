"""
선호도 분석 도구

구매 이력 분석, 선호도 추출, 구매 패턴 분석
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PurchaseHistoryInput(BaseModel):
    """구매 이력 분석 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    history_file: Optional[str] = Field(default=None, description="구매 이력 파일 경로")


class PreferenceExtractionInput(BaseModel):
    """선호도 추출 입력 스키마"""
    purchase_history: List[Dict[str, Any]] = Field(description="구매 이력 리스트")


class PurchasePatternInput(BaseModel):
    """구매 패턴 분석 입력 스키마"""
    user_id: str = Field(description="사용자 ID")
    time_period: Optional[str] = Field(default="all", description="분석 기간 (all, month, year)")


class PreferenceTools:
    """
    선호도 분석 도구 모음
    
    구매 이력 분석, 선호도 추출, 구매 패턴 분석
    """
    
    def __init__(self, data_dir: str = "shopping_data"):
        """
        PreferenceTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tools: List[BaseTool] = []
        self._initialize_tools()
    
    def _initialize_tools(self):
        """선호도 도구 초기화"""
        self.tools.append(self._create_purchase_history_analysis_tool())
        self.tools.append(self._create_preference_extraction_tool())
        self.tools.append(self._create_purchase_pattern_analysis_tool())
        
        logger.info(f"Initialized {len(self.tools)} preference tools")
    
    def _create_purchase_history_analysis_tool(self) -> BaseTool:
        """구매 이력 분석 도구 생성"""
        
        @tool("analyze_purchase_history", args_schema=PurchaseHistoryInput)
        def analyze_purchase_history(user_id: str, history_file: Optional[str] = None) -> str:
            """
            구매 이력 분석
            
            Args:
                user_id: 사용자 ID
                history_file: 구매 이력 파일 경로 (선택)
            
            Returns:
                구매 이력 분석 결과
            """
            try:
                logger.info(f"Analyzing purchase history for user: {user_id}")
                
                # 구매 이력 파일 경로 결정
                if not history_file:
                    history_file = str(self.data_dir / f"{user_id}_purchase_history.json")
                
                history_path = Path(history_file)
                
                # 구매 이력 로드
                if history_path.exists():
                    with open(history_path, 'r', encoding='utf-8') as f:
                        purchase_history = json.load(f)
                else:
                    purchase_history = []
                
                # 기본 분석 (실제로는 LLM을 사용하여 더 정교한 분석 수행)
                total_purchases = len(purchase_history)
                total_spent = sum(item.get('price', 0) for item in purchase_history)
                categories = {}
                
                for item in purchase_history:
                    category = item.get('category', 'Unknown')
                    categories[category] = categories.get(category, 0) + 1
                
                analysis = {
                    "user_id": user_id,
                    "total_purchases": total_purchases,
                    "total_spent": total_spent,
                    "category_distribution": categories,
                    "average_price": total_spent / total_purchases if total_purchases > 0 else 0
                }
                
                return json.dumps(analysis, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error_msg = f"Error analyzing purchase history for user {user_id}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return analyze_purchase_history
    
    def _create_preference_extraction_tool(self) -> BaseTool:
        """선호도 추출 도구 생성"""
        
        @tool("extract_preferences", args_schema=PreferenceExtractionInput)
        def extract_preferences(purchase_history: List[Dict[str, Any]]) -> str:
            """
            구매 이력에서 선호도 추출
            
            Args:
                purchase_history: 구매 이력 리스트
            
            Returns:
                추출된 선호도 정보
            """
            try:
                logger.info(f"Extracting preferences from {len(purchase_history)} purchases")
                
                # 기본 선호도 추출 (실제로는 LLM을 사용하여 더 정교한 추출 수행)
                preferences = {
                    "preferred_categories": [],
                    "price_range": {"min": 0, "max": 0},
                    "preferred_brands": [],
                    "purchase_frequency": "unknown"
                }
                
                if purchase_history:
                    categories = {}
                    brands = {}
                    prices = []
                    
                    for item in purchase_history:
                        category = item.get('category')
                        brand = item.get('brand')
                        price = item.get('price', 0)
                        
                        if category:
                            categories[category] = categories.get(category, 0) + 1
                        if brand:
                            brands[brand] = brands.get(brand, 0) + 1
                        if price > 0:
                            prices.append(price)
                    
                    preferences["preferred_categories"] = sorted(
                        categories.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    preferences["preferred_brands"] = sorted(
                        brands.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    
                    if prices:
                        preferences["price_range"] = {
                            "min": min(prices),
                            "max": max(prices),
                            "average": sum(prices) / len(prices)
                        }
                
                return json.dumps(preferences, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error_msg = f"Error extracting preferences: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return extract_preferences
    
    def _create_purchase_pattern_analysis_tool(self) -> BaseTool:
        """구매 패턴 분석 도구 생성"""
        
        @tool("analyze_purchase_patterns", args_schema=PurchasePatternInput)
        def analyze_purchase_patterns(user_id: str, time_period: Optional[str] = "all") -> str:
            """
            구매 패턴 분석
            
            Args:
                user_id: 사용자 ID
                time_period: 분석 기간 (all, month, year)
            
            Returns:
                구매 패턴 분석 결과
            """
            try:
                logger.info(f"Analyzing purchase patterns for user: {user_id} (period: {time_period})")
                
                history_file = str(self.data_dir / f"{user_id}_purchase_history.json")
                history_path = Path(history_file)
                
                if not history_path.exists():
                    return json.dumps({
                        "user_id": user_id,
                        "time_period": time_period,
                        "message": "No purchase history found"
                    }, indent=2)
                
                with open(history_path, 'r', encoding='utf-8') as f:
                    purchase_history = json.load(f)
                
                # 패턴 분석 (실제로는 LLM을 사용하여 더 정교한 분석 수행)
                patterns = {
                    "user_id": user_id,
                    "time_period": time_period,
                    "total_purchases": len(purchase_history),
                    "purchase_frequency": "unknown",
                    "seasonal_patterns": {},
                    "category_trends": {}
                }
                
                return json.dumps(patterns, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error_msg = f"Error analyzing purchase patterns for user {user_id}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return analyze_purchase_patterns
    
    def get_tools(self) -> List[BaseTool]:
        """모든 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 도구 찾기"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

