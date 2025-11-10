"""
규제 관련 도구

FDA 510(k), CE 마킹, ISO 13485 등 규제 관련 검색 및 검증 도구
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FDA510kSearchInput(BaseModel):
    """FDA 510(k) 검색 입력 스키마"""
    device_name: str = Field(description="의료기기 이름")
    device_class: Optional[str] = Field(default=None, description="의료기기 클래스")


class CEMarkingSearchInput(BaseModel):
    """CE 마킹 검색 입력 스키마"""
    device_type: str = Field(description="의료기기 유형")
    directive: Optional[str] = Field(default=None, description="EU 지시사항 (예: MDR, IVDR)")


class ISO13485SearchInput(BaseModel):
    """ISO 13485 검색 입력 스키마"""
    requirement: str = Field(description="ISO 13485 요구사항 키워드")


class RegulatoryTools:
    """
    규제 관련 도구 모음
    
    FDA 510(k), CE 마킹, ISO 13485 등 규제 정보 검색 및 검증
    """
    
    def __init__(self):
        """RegulatoryTools 초기화"""
        self.tools: List[BaseTool] = []
        self._initialize_tools()
    
    def _initialize_tools(self):
        """규제 도구 초기화"""
        self.tools.append(self._create_fda_510k_search_tool())
        self.tools.append(self._create_ce_marking_search_tool())
        self.tools.append(self._create_iso13485_search_tool())
        self.tools.append(self._create_regulatory_framework_analyzer_tool())
        
        logger.info(f"Initialized {len(self.tools)} regulatory tools")
    
    def _create_fda_510k_search_tool(self) -> BaseTool:
        """FDA 510(k) 검색 도구 생성"""
        
        @tool("search_fda_510k", args_schema=FDA510kSearchInput)
        def search_fda_510k(device_name: str, device_class: Optional[str] = None) -> str:
            """
            FDA 510(k) 사전시장 승인 정보 검색
            
            Args:
                device_name: 의료기기 이름
                device_class: 의료기기 클래스 (선택)
            
            Returns:
                FDA 510(k) 검색 결과
            """
            try:
                logger.info(f"Searching FDA 510(k) for device: {device_name}")
                
                # 실제 구현에서는 MCP g-search를 통해 FDA 데이터베이스 검색
                # 또는 FDA API를 직접 호출
                search_query = f"FDA 510(k) {device_name}"
                if device_class:
                    search_query += f" class {device_class}"
                
                # MCP g-search 도구를 사용하여 검색 수행
                # 여기서는 기본 응답 반환 (실제 구현에서는 MCP 통합 필요)
                return f"FDA 510(k) search results for '{device_name}':\n" \
                       f"- Use MCP g-search tool to query FDA database\n" \
                       f"- Search query: {search_query}\n" \
                       f"- Device class: {device_class or 'Not specified'}\n" \
                       f"- Please ensure MCP g-search server is configured for actual search."
            
            except Exception as e:
                error_msg = f"Error searching FDA 510(k) for '{device_name}': {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return search_fda_510k
    
    def _create_ce_marking_search_tool(self) -> BaseTool:
        """CE 마킹 검색 도구 생성"""
        
        @tool("search_ce_marking", args_schema=CEMarkingSearchInput)
        def search_ce_marking(device_type: str, directive: Optional[str] = None) -> str:
            """
            CE 마킹 요구사항 검색
            
            Args:
                device_type: 의료기기 유형
                directive: EU 지시사항 (MDR, IVDR 등)
            
            Returns:
                CE 마킹 검색 결과
            """
            try:
                logger.info(f"Searching CE marking requirements for device type: {device_type}")
                
                # 실제 구현에서는 MCP g-search를 통해 EU 규제 정보 검색
                search_query = f"CE marking {device_type}"
                if directive:
                    search_query += f" {directive}"
                
                return f"CE marking search results for '{device_type}':\n" \
                       f"- Use MCP g-search tool to query EU regulatory database\n" \
                       f"- Search query: {search_query}\n" \
                       f"- Directive: {directive or 'MDR/IVDR'}\n" \
                       f"- Please ensure MCP g-search server is configured for actual search."
            
            except Exception as e:
                error_msg = f"Error searching CE marking for '{device_type}': {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return search_ce_marking
    
    def _create_iso13485_search_tool(self) -> BaseTool:
        """ISO 13485 검색 도구 생성"""
        
        @tool("search_iso13485", args_schema=ISO13485SearchInput)
        def search_iso13485(requirement: str) -> str:
            """
            ISO 13485 품질 관리 시스템 요구사항 검색
            
            Args:
                requirement: ISO 13485 요구사항 키워드
            
            Returns:
                ISO 13485 검색 결과
            """
            try:
                logger.info(f"Searching ISO 13485 requirements: {requirement}")
                
                # 실제 구현에서는 ISO 표준 문서 검색
                search_query = f"ISO 13485 {requirement}"
                
                return f"ISO 13485 search results for '{requirement}':\n" \
                       f"- Use MCP g-search tool to query ISO standards\n" \
                       f"- Search query: {search_query}\n" \
                       f"- Please ensure MCP g-search server is configured for actual search."
            
            except Exception as e:
                error_msg = f"Error searching ISO 13485 for '{requirement}': {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return search_iso13485
    
    def _create_regulatory_framework_analyzer_tool(self) -> BaseTool:
        """규제 프레임워크 분석 도구 생성"""
        
        @tool("analyze_regulatory_framework")
        def analyze_regulatory_framework(device_info: str) -> str:
            """
            의료기기 정보를 기반으로 적용 가능한 규제 프레임워크 분석
            
            Args:
                device_info: 의료기기 정보 (JSON 형식 또는 텍스트)
            
            Returns:
                적용 가능한 규제 프레임워크 목록 및 요구사항
            """
            try:
                logger.info(f"Analyzing regulatory framework for device: {device_info}")
                
                # 실제 구현에서는 LLM을 사용하여 규제 프레임워크 분석
                return f"Regulatory framework analysis for device:\n" \
                       f"- Device info: {device_info}\n" \
                       f"- Potential frameworks: FDA 510(k), CE Marking (MDR/IVDR), ISO 13485\n" \
                       f"- Use LLM-based analysis to determine specific requirements."
            
            except Exception as e:
                error_msg = f"Error analyzing regulatory framework: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return analyze_regulatory_framework
    
    def get_tools(self) -> List[BaseTool]:
        """모든 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 도구 찾기"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

