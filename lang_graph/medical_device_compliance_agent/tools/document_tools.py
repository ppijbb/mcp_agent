"""
문서 처리 도구

규제 문서, 테스트 리포트 등 문서 처리 관련 도구
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentParseInput(BaseModel):
    """문서 파싱 입력 스키마"""
    file_path: str = Field(description="파싱할 문서 경로")
    document_type: Optional[str] = Field(default=None, description="문서 유형 (PDF, DOCX 등)")


class DocumentGenerateInput(BaseModel):
    """문서 생성 입력 스키마"""
    template_path: Optional[str] = Field(default=None, description="템플릿 경로")
    output_path: str = Field(description="출력 파일 경로")
    content: Dict[str, Any] = Field(description="문서 내용 (딕셔너리)")


class DocumentTools:
    """
    문서 처리 도구 모음
    
    규제 문서 파싱, 테스트 리포트 생성 등
    """
    
    def __init__(self):
        """DocumentTools 초기화"""
        self.tools: List[BaseTool] = []
        self._initialize_tools()
    
    def _initialize_tools(self):
        """문서 도구 초기화"""
        self.tools.append(self._create_document_parse_tool())
        self.tools.append(self._create_document_generate_tool())
        self.tools.append(self._create_report_generate_tool())
        
        logger.info(f"Initialized {len(self.tools)} document tools")
    
    def _create_document_parse_tool(self) -> BaseTool:
        """문서 파싱 도구 생성"""
        
        @tool("parse_document", args_schema=DocumentParseInput)
        def parse_document(file_path: str, document_type: Optional[str] = None) -> str:
            """
            문서 파싱 (PDF, DOCX, TXT 등)
            
            Args:
                file_path: 파싱할 문서 경로
                document_type: 문서 유형 (선택, 자동 감지 가능)
            
            Returns:
                파싱된 문서 내용
            """
            try:
                logger.info(f"Parsing document: {file_path}")
                
                path = Path(file_path)
                if not path.exists():
                    return f"Error: File not found: {file_path}"
                
                # 기본 텍스트 파일 읽기
                if path.suffix.lower() == '.txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return content
                
                # PDF, DOCX 등은 추가 라이브러리 필요
                # 실제 구현에서는 PyPDF2, python-docx 등을 사용
                return f"Document parsing for {file_path}:\n" \
                       f"- Document type: {document_type or 'auto-detect'}\n" \
                       f"- For PDF/DOCX, additional libraries required (PyPDF2, python-docx)"
            
            except Exception as e:
                error_msg = f"Error parsing document {file_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return parse_document
    
    def _create_document_generate_tool(self) -> BaseTool:
        """문서 생성 도구 생성"""
        
        @tool("generate_document", args_schema=DocumentGenerateInput)
        def generate_document(
            output_path: str,
            content: Dict[str, Any],
            template_path: Optional[str] = None
        ) -> str:
            """
            문서 생성 (마크다운, 텍스트 등)
            
            Args:
                output_path: 출력 파일 경로
                content: 문서 내용 (딕셔너리)
                template_path: 템플릿 경로 (선택)
            
            Returns:
                성공 메시지
            """
            try:
                logger.info(f"Generating document: {output_path}")
                
                # 기본 마크다운 문서 생성
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # 딕셔너리를 마크다운으로 변환
                markdown_content = self._dict_to_markdown(content)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                logger.info(f"Generated document: {output_path}")
                return f"Successfully generated document: {output_path}"
            
            except Exception as e:
                error_msg = f"Error generating document {output_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return generate_document
    
    def _create_report_generate_tool(self) -> BaseTool:
        """리포트 생성 도구 생성"""
        
        @tool("generate_compliance_report")
        def generate_compliance_report(
            report_type: str,
            device_info: str,
            test_results: str,
            output_path: str
        ) -> str:
            """
            규제 컴플라이언스 리포트 생성
            
            Args:
                report_type: 리포트 유형 (FDA 510(k), CE Marking 등)
                device_info: 의료기기 정보
                test_results: 테스트 결과
                output_path: 출력 파일 경로
            
            Returns:
                성공 메시지
            """
            try:
                logger.info(f"Generating compliance report: {report_type}")
                
                # 리포트 내용 구성
                report_content = {
                    "report_type": report_type,
                    "device_info": device_info,
                    "test_results": test_results,
                    "generated_at": str(Path(output_path).stat().st_mtime if Path(output_path).exists() else "N/A")
                }
                
                # 문서 생성 도구 사용
                return self._create_document_generate_tool().invoke({
                    "output_path": output_path,
                    "content": report_content
                })
            
            except Exception as e:
                error_msg = f"Error generating compliance report: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return generate_compliance_report
    
    def _dict_to_markdown(self, data: Dict[str, Any], level: int = 1) -> str:
        """딕셔너리를 마크다운으로 변환"""
        lines = []
        for key, value in data.items():
            header = "#" * level
            if isinstance(value, dict):
                lines.append(f"{header} {key}")
                lines.append(self._dict_to_markdown(value, level + 1))
            elif isinstance(value, list):
                lines.append(f"{header} {key}")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._dict_to_markdown(item, level + 1))
                    else:
                        lines.append(f"- {item}")
            else:
                lines.append(f"{header} {key}")
                lines.append(f"{value}")
        return "\n".join(lines)
    
    def get_tools(self) -> List[BaseTool]:
        """모든 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 도구 찾기"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

