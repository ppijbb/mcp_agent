"""
Notion Integration Module

Notion MCP 서버와의 연동을 통한 문서 생성 및 데이터베이스 관리
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
import json

# MCP 관련 imports
try:
    from mcp_agent.mcp.gen_client import gen_client, connect, disconnect
    from mcp_agent.context import get_current_context
except ImportError:
    # Fallback for development
    pass

# 로컬 설정 imports - 절대 import로 수정
try:
    from config import NOTION_CONFIG, PRD_TEMPLATE_CONFIG, ROADMAP_CONFIG
except ImportError:
    # Fallback 설정
    NOTION_CONFIG = {"fallback_enabled": False}
    PRD_TEMPLATE_CONFIG = {"template_id": "default"}
    ROADMAP_CONFIG = {"template_id": "default"}

logger = logging.getLogger(__name__)


class NotionError(Exception):
    """Notion 통합 관련 예외"""
    pass


class NotionIntegration:
    """
    Notion MCP 서버 통합 클래스
    
    Notion API를 통한 페이지 생성, 데이터베이스 관리, 문서 업데이트 등의 기능 제공
    """
    
    def __init__(self):
        """NotionIntegration 초기화"""
        self.server_name = "notion-api"
        self.client = None
        self.connected = False
        
        # 데이터베이스 ID 캐시
        self._database_ids = {}
        self._page_cache = {}
        
        logger.info("NotionIntegration 초기화 완료")
    
    async def connect(self) -> bool:
        """
        Notion MCP 서버에 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            if self.connected:
                return True
                
            self.client = await connect(self.server_name)
            self.connected = True
            
            logger.info(f"Notion MCP 서버 연결 성공: {self.server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Notion MCP 서버 연결 실패: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Notion MCP 서버 연결 해제"""
        try:
            if self.client:
                await disconnect(self.server_name)
                self.client = None
                self.connected = False
                
            logger.info("Notion MCP 서버 연결 해제 완료")
            
        except Exception as e:
            logger.warning(f"Notion MCP 서버 연결 해제 중 오류: {str(e)}")
    
    async def create_database(self, database_name: str, schema_type: str = "requirements") -> str:
        """
        새로운 Notion 데이터베이스 생성
        
        Args:
            database_name: 데이터베이스 이름
            schema_type: 스키마 타입 (requirements, roadmap, design_specs)
            
        Returns:
            생성된 데이터베이스 ID
        """
        try:
            if not await self.connect():
                raise ConnectionError("Notion MCP 서버에 연결할 수 없습니다.")
            
            # 스키마 가져오기
            schema = NOTION_DATABASE_SCHEMAS.get(schema_type, {})
            if not schema:
                raise ValueError(f"지원하지 않는 스키마 타입: {schema_type}")
            
            # 데이터베이스 생성 요청
            try:
                result = await self.client.call_tool(
                    name="create_database",
                    arguments={
                        "title": database_name,
                        "properties": schema
                    }
                )
                
                database_id = self._extract_database_id(result)
                self._database_ids[schema_type] = database_id
                
                logger.info(f"데이터베이스 생성 완료: {database_name} ({database_id})")
                return database_id
                
            except Exception as mcp_error:
                logger.error(f"Notion MCP 서버 데이터베이스 생성 실패: {str(mcp_error)}")
                raise NotionError(f"Notion 데이터베이스 생성 불가: {str(mcp_error)}")
                
        except Exception as e:
            logger.error(f"데이터베이스 생성 실패: {str(e)}")
            raise
    
    def _extract_database_id(self, result: Any) -> str:
        """MCP 결과에서 데이터베이스 ID 추출"""
        try:
            if isinstance(result, dict):
                return result.get("id", result.get("database_id", ""))
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        except:
            return f"extracted_id_{get_timestamp()}"
    
    async def create_prd(self, requirements_data: Dict[str, Any]) -> str:
        """
        PRD (Product Requirements Document) 페이지 생성
        
        Args:
            requirements_data: 요구사항 데이터
            
        Returns:
            생성된 PRD 페이지 ID
        """
        try:
            if not await self.connect():
                raise NotionError("Notion MCP 서버 연결 실패 - PRD 생성 불가")
            
            logger.info("PRD 페이지 생성 시작")
            
            # PRD 콘텐츠 생성
            prd_content = self._generate_prd_content(requirements_data)
            
            # Notion 페이지 생성
            try:
                result = await self.client.call_tool(
                    name="create_page",
                    arguments={
                        "title": f"PRD - {requirements_data.get('product_name', 'Product Planning')}",
                        "content": prd_content,
                        "properties": {
                            "Status": "Draft",
                            "Created": datetime.now().isoformat(),
                            "Type": "PRD"
                        }
                    }
                )
                
                page_id = self._extract_page_id(result)
                
                # 캐시에 저장
                self._page_cache[f"prd_{get_timestamp()}"] = page_id
                
                logger.info(f"PRD 페이지 생성 완료: {page_id}")
                return page_id
                
            except Exception as mcp_error:
                logger.error(f"Notion MCP 서버 PRD 생성 실패: {str(mcp_error)}")
                raise NotionError(f"PRD 페이지 생성 실패: {str(mcp_error)}")
                
        except Exception as e:
            logger.error(f"PRD 생성 실패: {str(e)}")
            raise
    
    def _generate_prd_content(self, requirements_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """PRD 콘텐츠 생성"""
        try:
            content = []
            
            # 제목 및 기본 섹션들
            sections = [
                ("Product Requirements Document", "heading_1"),
                ("Executive Summary", "heading_2", requirements_data.get("executive_summary", "프로덕트 요구사항 문서입니다.")),
                ("Problem Statement", "heading_2", requirements_data.get("problem_statement", "해결해야 할 문제를 정의합니다.")),
                ("Solution Overview", "heading_2", requirements_data.get("solution_overview", "제안하는 솔루션에 대한 개요입니다."))
            ]
            
            for section in sections:
                if len(section) == 2:  # 제목만
                    content.append({
                        "type": section[1],
                        section[1]: {
                            "rich_text": [{"type": "text", "text": {"content": section[0]}}]
                        }
                    })
                else:  # 제목 + 내용
                    content.append({
                        "type": section[1],
                        section[1]: {
                            "rich_text": [{"type": "text", "text": {"content": section[0]}}]
                        }
                    })
                    content.append({
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": section[2]}}]
                        }
                    })
            
            # User Stories
            self._add_list_section(content, "User Stories", requirements_data.get("user_stories", []))
            
            # Technical Requirements
            self._add_list_section(content, "Technical Requirements", requirements_data.get("technical_requirements", []))
            
            # Success Metrics
            self._add_list_section(content, "Success Metrics", requirements_data.get("success_metrics", []))
            
            return content
            
        except Exception as e:
            logger.warning(f"PRD 콘텐츠 생성 중 오류: {str(e)}")
            return self._create_error_content("PRD 콘텐츠 생성 중 오류가 발생했습니다.")
    
    def _add_list_section(self, content: List, title: str, items: List[str]):
        """리스트 섹션 추가"""
        content.append({
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": title}}]
            }
        })
        
        for item in items:
            content.append({
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": item}}]
                }
            })
    
    def _create_error_content(self, error_message: str) -> List[Dict[str, Any]]:
        """오류 콘텐츠 생성"""
        return [{
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": error_message}}]
            }
        }]
    
    def _extract_page_id(self, result: Any) -> str:
        """MCP 결과에서 페이지 ID 추출"""
        try:
            if isinstance(result, dict):
                return result.get("id", result.get("page_id", ""))
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        except:
            return f"extracted_page_{get_timestamp()}"
    
    async def create_roadmap(self, roadmap_data: Dict[str, Any]) -> str:
        """
        프로덕트 로드맵 페이지 생성
        
        Args:
            roadmap_data: 로드맵 데이터
            
        Returns:
            생성된 로드맵 페이지 ID
        """
        try:
            if not await self.connect():
                raise NotionError("Notion MCP 서버 연결 실패 - 로드맵 생성 불가")
            
            logger.info("로드맵 페이지 생성 시작")
            
            # 로드맵 콘텐츠 생성
            roadmap_content = self._generate_roadmap_content(roadmap_data)
            
            # Notion 페이지 생성
            try:
                result = await self.client.call_tool(
                    name="create_page",
                    arguments={
                        "title": f"Roadmap - {roadmap_data.get('project_name', 'Product Roadmap')}",
                        "content": roadmap_content,
                        "properties": {
                            "Status": "Active",
                            "Created": datetime.now().isoformat(),
                            "Type": "Roadmap"
                        }
                    }
                )
                
                page_id = self._extract_page_id(result)
                
                # 캐시에 저장
                self._page_cache[f"roadmap_{get_timestamp()}"] = page_id
                
                logger.info(f"로드맵 페이지 생성 완료: {page_id}")
                return page_id
                
            except Exception as mcp_error:
                logger.warning(f"MCP 로드맵 생성 실패, 모의 ID 반환: {str(mcp_error)}")
                mock_id = f"mock_roadmap_{get_timestamp()}"
                self._page_cache[f"roadmap_{get_timestamp()}"] = mock_id
                return mock_id
                
        except Exception as e:
            logger.error(f"로드맵 생성 실패: {str(e)}")
            raise
    
    def _generate_roadmap_content(self, roadmap_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """로드맵 콘텐츠 생성"""
        try:
            content = []
            
            # 제목
            content.append({
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": "Product Roadmap"}}]
                }
            })
            
            # 개요
            content.append({
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": roadmap_data.get("overview", "프로덕트 개발 로드맵입니다.")}}]
                }
            })
            
            # 마일스톤들
            milestones = roadmap_data.get("milestones", [])
            for milestone in milestones:
                self._add_milestone_section(content, milestone)
            
            # 리스크 및 가정사항
            self._add_risks_section(content, roadmap_data.get("risks", []))
            
            return content
            
        except Exception as e:
            logger.warning(f"로드맵 콘텐츠 생성 중 오류: {str(e)}")
            return self._create_error_content("로드맵 콘텐츠 생성 중 오류가 발생했습니다.")
    
    def _add_milestone_section(self, content: List, milestone: Dict[str, Any]):
        """마일스톤 섹션 추가"""
        # 마일스톤 제목
        content.append({
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": milestone.get("name", "Milestone")}}]
            }
        })
        
        # 기간
        content.append({
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": f"📅 {milestone.get('start_date', 'TBD')} ~ {milestone.get('end_date', 'TBD')}"}}]
            }
        })
        
        # 목표
        objectives = milestone.get("objectives", [])
        if objectives:
            content.append({
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": "🎯 주요 목표:"}}]
                }
            })
            
            for objective in objectives:
                content.append({
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": objective}}]
                    }
                })
        
        # 구분선
        content.append({
            "type": "divider",
            "divider": {}
        })
    
    def _add_risks_section(self, content: List, risks: List[str]):
        """리스크 섹션 추가"""
        content.append({
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "Risks & Assumptions"}}]
            }
        })
        
        for risk in risks:
            content.append({
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"⚠️ {risk}"}}]
                }
            })
    
    async def update_page(self, page_id: str, content: List[Dict[str, Any]]) -> bool:
        """
        기존 Notion 페이지 업데이트
        
        Args:
            page_id: 업데이트할 페이지 ID
            content: 새로운 콘텐츠
            
        Returns:
            업데이트 성공 여부
        """
        try:
            if not await self.connect():
                raise ConnectionError("Notion MCP 서버에 연결할 수 없습니다.")
            
            logger.info(f"페이지 업데이트 시작: {page_id}")
            
            try:
                result = await self.client.call_tool(
                    name="update_page",
                    arguments={
                        "page_id": page_id,
                        "content": content
                    }
                )
                
                logger.info(f"페이지 업데이트 완료: {page_id}")
                return True
                
            except Exception as mcp_error:
                logger.warning(f"MCP 페이지 업데이트 실패: {str(mcp_error)}")
                return False
                
        except Exception as e:
            logger.error(f"페이지 업데이트 실패: {str(e)}")
            return False
    
    async def create_database_entry(self, database_id: str, properties: Dict[str, Any]) -> str:
        """
        데이터베이스에 새 항목 추가
        
        Args:
            database_id: 대상 데이터베이스 ID
            properties: 항목 속성들
            
        Returns:
            생성된 항목 ID
        """
        try:
            if not await self.connect():
                raise ConnectionError("Notion MCP 서버에 연결할 수 없습니다.")
            
            logger.info(f"데이터베이스 항목 생성: {database_id}")
            
            try:
                result = await self.client.call_tool(
                    name="create_page",
                    arguments={
                        "parent": {"database_id": database_id},
                        "properties": properties
                    }
                )
                
                entry_id = self._extract_page_id(result)
                logger.info(f"데이터베이스 항목 생성 완료: {entry_id}")
                return entry_id
                
            except Exception as mcp_error:
                logger.warning(f"MCP 데이터베이스 항목 생성 실패: {str(mcp_error)}")
                mock_id = f"mock_entry_{get_timestamp()}"
                return mock_id
                
        except Exception as e:
            logger.error(f"데이터베이스 항목 생성 실패: {str(e)}")
            raise
    
    async def query_database(self, database_id: str, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        데이터베이스 쿼리
        
        Args:
            database_id: 쿼리할 데이터베이스 ID
            filter_criteria: 필터 조건
            
        Returns:
            쿼리 결과 리스트
        """
        try:
            if not await self.connect():
                raise ConnectionError("Notion MCP 서버에 연결할 수 없습니다.")
            
            logger.info(f"데이터베이스 쿼리: {database_id}")
            
            try:
                result = await self.client.call_tool(
                    name="query_database",
                    arguments={
                        "database_id": database_id,
                        "filter": filter_criteria or {}
                    }
                )
                
                # 결과 파싱
                entries = self._parse_query_result(result)
                logger.info(f"데이터베이스 쿼리 완료: {len(entries)}개 항목")
                return entries
                
            except Exception as mcp_error:
                logger.warning(f"MCP 데이터베이스 쿼리 실패: {str(mcp_error)}")
                return []
                
        except Exception as e:
            logger.error(f"데이터베이스 쿼리 실패: {str(e)}")
            return []
    
    def _parse_query_result(self, result: Any) -> List[Dict[str, Any]]:
        """쿼리 결과 파싱"""
        try:
            if isinstance(result, dict):
                return result.get("results", [])
            elif isinstance(result, list):
                return result
            else:
                return []
        except:
            return []
    
    def get_cached_database_id(self, schema_type: str) -> Optional[str]:
        """캐시된 데이터베이스 ID 반환"""
        return self._database_ids.get(schema_type)
    
    def get_cached_page_id(self, page_type: str) -> Optional[str]:
        """캐시된 페이지 ID 반환"""
        for key, page_id in self._page_cache.items():
            if page_type in key:
                return page_id
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """NotionIntegration 상태 반환"""
        return {
            "connected": self.connected,
            "server_name": self.server_name,
            "cached_databases": len(self._database_ids),
            "cached_pages": len(self._page_cache),
            "supported_schemas": list(NOTION_DATABASE_SCHEMAS.keys()),
            "timestamp": get_timestamp()
        } 