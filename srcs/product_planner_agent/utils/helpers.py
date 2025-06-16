"""
Product Planner Agent Helpers

프로덕트 기획자 Agent 전용 헬퍼 함수들
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)

# 로컬 설정 imports - 절대 import로 수정
try:
    from config import ANALYSIS_CONFIG, PRD_TEMPLATE_CONFIG
except ImportError:
    # Fallback 설정
    ANALYSIS_CONFIG = {"complexity_weights": {"low": 1, "medium": 2, "high": 3}}
    PRD_TEMPLATE_CONFIG = {"sections": ["overview", "requirements", "timeline"]}


class ProductPlannerHelpers:
    """
    Product Planner Agent 헬퍼 클래스
    
    공통 유틸리티 함수들과 데이터 처리 로직 제공
    """
    
    @staticmethod
    def format_timestamp(timestamp: Optional[datetime] = None) -> str:
        """타임스탬프 포맷팅"""
        if timestamp is None:
            timestamp = datetime.now()
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def extract_figma_file_id(figma_url: str) -> Optional[str]:
        """Figma URL에서 파일 ID 추출"""
        try:
            # Figma URL 패턴: https://www.figma.com/file/{file_id}/{file_name}
            pattern = r'figma\.com/file/([a-zA-Z0-9]+)'
            match = re.search(pattern, figma_url)
            return match.group(1) if match else None
        except Exception as e:
            logger.warning(f"Figma 파일 ID 추출 실패: {str(e)}")
            return None
    
    @staticmethod
    def extract_figma_node_id(figma_url: str) -> Optional[str]:
        """Figma URL에서 노드 ID 추출"""
        try:
            # 노드 ID 패턴: ?node-id={node_id}
            pattern = r'node-id=([^&]+)'
            match = re.search(pattern, figma_url)
            return match.group(1) if match else None
        except Exception as e:
            logger.warning(f"Figma 노드 ID 추출 실패: {str(e)}")
            return None
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """파일명 안전화"""
        try:
            # 특수문자 제거 및 공백을 언더스코어로 변경
            sanitized = re.sub(r'[^\w\s-]', '', filename)
            sanitized = re.sub(r'[-\s]+', '_', sanitized)
            return sanitized.strip('_')
        except Exception as e:
            logger.warning(f"파일명 안전화 실패: {str(e)}")
            return "untitled"
    
    @staticmethod
    def calculate_complexity_score(components: List[Dict[str, Any]]) -> float:
        """컴포넌트 기반 복잡도 점수 계산"""
        try:
            if not components:
                return 0.0
            
            total_score = 0
            for component in components:
                # 기본 점수
                base_score = 1
                
                # 컴포넌트 타입별 가중치
                comp_type = component.get("type", "").upper()
                type_weights = {
                    "COMPONENT": 2,
                    "INSTANCE": 1.5,
                    "FRAME": 1,
                    "GROUP": 1.2,
                    "TEXT": 0.5
                }
                base_score *= type_weights.get(comp_type, 1)
                
                # 속성 개수에 따른 가중치
                properties = component.get("properties", {})
                if isinstance(properties, dict):
                    base_score *= (1 + len(properties) * 0.1)
                
                total_score += base_score
            
            # 정규화 (0-1 범위)
            max_possible = len(components) * 3  # 최대 가능 점수
            return min(total_score / max_possible, 1.0)
            
        except Exception as e:
            logger.warning(f"복잡도 점수 계산 실패: {str(e)}")
            return 0.5  # 기본값
    
    @staticmethod
    def estimate_development_time(story_points: int, team_size: int = 3) -> Dict[str, Any]:
        """스토리 포인트 기반 개발 시간 추정"""
        try:
            # 기본 추정: 1 스토리 포인트 = 1일 (8시간)
            base_hours = story_points * 8
            
            # 팀 크기에 따른 조정
            adjusted_hours = base_hours / max(team_size, 1)
            
            # 버퍼 추가 (20%)
            buffered_hours = adjusted_hours * 1.2
            
            days = buffered_hours / 8
            weeks = days / 5  # 주 5일 근무
            
            return {
                "story_points": story_points,
                "estimated_hours": round(buffered_hours, 1),
                "estimated_days": round(days, 1),
                "estimated_weeks": round(weeks, 1),
                "team_size": team_size
            }
            
        except Exception as e:
            logger.warning(f"개발 시간 추정 실패: {str(e)}")
            return {
                "story_points": story_points,
                "estimated_hours": story_points * 8,
                "estimated_days": story_points,
                "estimated_weeks": story_points / 5,
                "team_size": team_size
            }
    
    @staticmethod
    def prioritize_requirements(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """요구사항 우선순위 정렬"""
        try:
            priority_order = {
                "Critical": 4,
                "High": 3,
                "Medium": 2,
                "Low": 1
            }
            
            def get_priority_score(req):
                priority = req.get("priority", "Medium")
                return priority_order.get(priority, 2)
            
            return sorted(requirements, key=get_priority_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"요구사항 우선순위 정렬 실패: {str(e)}")
            return requirements
    
    @staticmethod
    def generate_requirement_id(req_type: str, index: int) -> str:
        """요구사항 ID 생성"""
        try:
            type_prefixes = {
                "user_story": "US",
                "technical_requirement": "TR",
                "business_requirement": "BR",
                "functional_requirement": "FR",
                "non_functional_requirement": "NFR"
            }
            
            prefix = type_prefixes.get(req_type, "REQ")
            return f"{prefix}-{index:03d}"
            
        except Exception as e:
            logger.warning(f"요구사항 ID 생성 실패: {str(e)}")
            return f"REQ-{index:03d}"
    
    @staticmethod
    def validate_prd_data(prd_data: Dict[str, Any]) -> Dict[str, Any]:
        """PRD 데이터 유효성 검증"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # 필수 필드 검증
            required_fields = [
                "product_name",
                "executive_summary",
                "problem_statement",
                "solution_overview"
            ]
            
            for field in required_fields:
                if not prd_data.get(field):
                    validation_result["errors"].append(f"필수 필드 누락: {field}")
                    validation_result["is_valid"] = False
            
            # 선택적 필드 검증
            optional_fields = [
                "user_stories",
                "technical_requirements",
                "success_metrics"
            ]
            
            for field in optional_fields:
                if field in prd_data and not prd_data[field]:
                    validation_result["warnings"].append(f"선택적 필드가 비어있음: {field}")
            
            # 데이터 타입 검증
            list_fields = ["user_stories", "technical_requirements", "success_metrics"]
            for field in list_fields:
                if field in prd_data and not isinstance(prd_data[field], list):
                    validation_result["errors"].append(f"잘못된 데이터 타입: {field} (리스트 필요)")
                    validation_result["is_valid"] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"PRD 데이터 검증 실패: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"검증 프로세스 오류: {str(e)}"],
                "warnings": []
            }
    
    @staticmethod
    def format_notion_content(content: str, content_type: str = "paragraph") -> Dict[str, Any]:
        """Notion 콘텐츠 포맷팅"""
        try:
            notion_block = {
                "type": content_type,
                content_type: {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content
                            }
                        }
                    ]
                }
            }
            
            return notion_block
            
        except Exception as e:
            logger.warning(f"Notion 콘텐츠 포맷팅 실패: {str(e)}")
            return {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": str(content)
                            }
                        }
                    ]
                }
            }
    
    @staticmethod
    def extract_key_insights(analysis_data: Dict[str, Any]) -> List[str]:
        """분석 데이터에서 핵심 인사이트 추출"""
        try:
            insights = []
            
            # 컴포넌트 인사이트
            components = analysis_data.get("parsed_data", {}).get("components", [])
            if components:
                insights.append(f"총 {len(components)}개의 디자인 컴포넌트 발견")
                
                component_types = {}
                for comp in components:
                    comp_type = comp.get("type", "Unknown")
                    component_types[comp_type] = component_types.get(comp_type, 0) + 1
                
                for comp_type, count in component_types.items():
                    insights.append(f"{comp_type} 타입 컴포넌트: {count}개")
            
            # 프레임 인사이트
            frames = analysis_data.get("parsed_data", {}).get("frames", [])
            if frames:
                insights.append(f"총 {len(frames)}개의 화면/프레임 구성")
            
            # 변수 인사이트
            variables = analysis_data.get("parsed_data", {}).get("variables", {})
            if variables:
                insights.append("디자인 시스템 변수 사용 확인됨")
            
            # 복잡도 인사이트
            metadata = analysis_data.get("metadata", {})
            if metadata.get("is_mock_data"):
                insights.append("⚠️ 모의 데이터 기반 분석 결과")
            
            return insights[:10]  # 최대 10개 인사이트
            
        except Exception as e:
            logger.warning(f"핵심 인사이트 추출 실패: {str(e)}")
            return ["분석 데이터에서 인사이트를 추출할 수 없습니다."]
    
    @staticmethod
    def create_status_summary(agent_status: Dict[str, Any]) -> str:
        """Agent 상태 요약 생성"""
        try:
            status = agent_status.get("status", "unknown")
            agent_name = agent_status.get("agent_name", "Product Planner Agent")
            
            summary_parts = [f"🤖 {agent_name} 상태: {status.upper()}"]
            
            if status == "ready":
                summary_parts.append("✅ 모든 시스템 정상 작동")
                
                # 서버 상태
                servers = agent_status.get("servers", [])
                if servers:
                    summary_parts.append(f"🔗 연결된 MCP 서버: {', '.join(servers)}")
                
                # 현재 작업 상태
                current_states = []
                if agent_status.get("current_analysis"):
                    current_states.append("분석 완료")
                if agent_status.get("current_requirements"):
                    current_states.append("요구사항 생성")
                if agent_status.get("current_roadmap"):
                    current_states.append("로드맵 구축")
                
                if current_states:
                    summary_parts.append(f"📋 현재 상태: {', '.join(current_states)}")
                
                # ReAct 패턴 상태
                if agent_status.get("react_pattern") == "implemented":
                    summary_parts.append("🧠 ReAct 패턴 활성화")
                
            elif status == "error":
                error = agent_status.get("error", "알 수 없는 오류")
                summary_parts.append(f"❌ 오류: {error}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"상태 요약 생성 실패: {str(e)}")
            return f"Agent 상태 요약을 생성할 수 없습니다: {str(e)}" 