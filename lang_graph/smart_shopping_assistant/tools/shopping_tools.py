"""
쇼핑 관련 도구

가격 검색, 제품 정보, 할인 정보 수집 등 쇼핑 관련 도구
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PriceSearchInput(BaseModel):
    """가격 검색 입력 스키마"""
    product_name: str = Field(description="제품 이름")
    category: Optional[str] = Field(default=None, description="제품 카테고리")


class ProductInfoInput(BaseModel):
    """제품 정보 검색 입력 스키마"""
    product_name: str = Field(description="제품 이름")
    brand: Optional[str] = Field(default=None, description="브랜드 이름")


class DiscountSearchInput(BaseModel):
    """할인 정보 검색 입력 스키마"""
    product_name: str = Field(description="제품 이름")
    max_discount: Optional[float] = Field(default=None, description="최대 할인율")


class StoreComparisonInput(BaseModel):
    """쇼핑몰 비교 입력 스키마"""
    product_name: str = Field(description="제품 이름")
    stores: Optional[List[str]] = Field(default=None, description="비교할 쇼핑몰 목록")


class ShoppingTools:
    """
    쇼핑 관련 도구 모음
    
    가격 검색, 제품 정보, 할인 정보 수집, 쇼핑몰 비교
    """
    
    def __init__(self):
        """ShoppingTools 초기화"""
        self.tools: List[BaseTool] = []
        self._initialize_tools()
    
    def _initialize_tools(self):
        """쇼핑 도구 초기화"""
        self.tools.append(self._create_price_search_tool())
        self.tools.append(self._create_product_info_tool())
        self.tools.append(self._create_discount_search_tool())
        self.tools.append(self._create_store_comparison_tool())
        
        logger.info(f"Initialized {len(self.tools)} shopping tools")
    
    def _create_price_search_tool(self) -> BaseTool:
        """가격 검색 도구 생성"""
        
        @tool("search_product_price", args_schema=PriceSearchInput)
        def search_product_price(product_name: str, category: Optional[str] = None) -> str:
            """
            제품 가격 검색
            
            Args:
                product_name: 제품 이름
                category: 제품 카테고리 (선택)
            
            Returns:
                가격 검색 결과
            """
            try:
                logger.info(f"Searching price for product: {product_name}")
                
                # 실제 구현에서는 MCP g-search를 통해 가격 검색
                search_query = f"{product_name} price"
                if category:
                    search_query += f" {category}"
                
                return f"Price search results for '{product_name}':\n" \
                       f"- Use MCP g-search tool to query product prices\n" \
                       f"- Search query: {search_query}\n" \
                       f"- Category: {category or 'Not specified'}\n" \
                       f"- Please ensure MCP g-search server is configured for actual search."
            
            except Exception as e:
                error_msg = f"Error searching price for '{product_name}': {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return search_product_price
    
    def _create_product_info_tool(self) -> BaseTool:
        """제품 정보 검색 도구 생성"""
        
        @tool("search_product_info", args_schema=ProductInfoInput)
        def search_product_info(product_name: str, brand: Optional[str] = None) -> str:
            """
            제품 정보 검색
            
            Args:
                product_name: 제품 이름
                brand: 브랜드 이름 (선택)
            
            Returns:
                제품 정보 검색 결과
            """
            try:
                logger.info(f"Searching product info: {product_name}")
                
                search_query = f"{product_name}"
                if brand:
                    search_query += f" {brand}"
                
                return f"Product info search results for '{product_name}':\n" \
                       f"- Use MCP g-search tool to query product information\n" \
                       f"- Search query: {search_query}\n" \
                       f"- Brand: {brand or 'Not specified'}\n" \
                       f"- Please ensure MCP g-search server is configured for actual search."
            
            except Exception as e:
                error_msg = f"Error searching product info for '{product_name}': {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return search_product_info
    
    def _create_discount_search_tool(self) -> BaseTool:
        """할인 정보 검색 도구 생성"""
        
        @tool("search_discounts", args_schema=DiscountSearchInput)
        def search_discounts(product_name: str, max_discount: Optional[float] = None) -> str:
            """
            할인 정보 검색
            
            Args:
                product_name: 제품 이름
                max_discount: 최대 할인율 (선택)
            
            Returns:
                할인 정보 검색 결과
            """
            try:
                logger.info(f"Searching discounts for: {product_name}")
                
                search_query = f"{product_name} discount sale"
                
                return f"Discount search results for '{product_name}':\n" \
                       f"- Use MCP g-search tool to query discount information\n" \
                       f"- Search query: {search_query}\n" \
                       f"- Max discount: {max_discount or 'Not specified'}\n" \
                       f"- Please ensure MCP g-search server is configured for actual search."
            
            except Exception as e:
                error_msg = f"Error searching discounts for '{product_name}': {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return search_discounts
    
    def _create_store_comparison_tool(self) -> BaseTool:
        """쇼핑몰 비교 도구 생성"""
        
        @tool("compare_stores", args_schema=StoreComparisonInput)
        def compare_stores(product_name: str, stores: Optional[List[str]] = None) -> str:
            """
            여러 쇼핑몰에서 가격 비교
            
            Args:
                product_name: 제품 이름
                stores: 비교할 쇼핑몰 목록 (선택)
            
            Returns:
                쇼핑몰별 가격 비교 결과
            """
            try:
                logger.info(f"Comparing stores for: {product_name}")
                
                store_list = stores or ["Amazon", "eBay", "Walmart", "Target"]
                
                return f"Store comparison results for '{product_name}':\n" \
                       f"- Use MCP g-search tool to query prices from multiple stores\n" \
                       f"- Stores to compare: {', '.join(store_list)}\n" \
                       f"- Please ensure MCP g-search server is configured for actual search."
            
            except Exception as e:
                error_msg = f"Error comparing stores for '{product_name}': {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return compare_stores
    
    def get_tools(self) -> List[BaseTool]:
        """모든 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 도구 찾기"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

