"""
Product Planner Validators

프로덕트 기획자 Agent 전용 데이터 검증 로직
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import urlparse
import json

# 로컬 설정 imports - 절대 import로 수정
try:
    from config import FIGMA_CONFIG, NOTION_CONFIG
except ImportError:
    # Fallback 설정
    FIGMA_CONFIG = {"url_patterns": ["figma.com"]}
    NOTION_CONFIG = {"url_patterns": ["notion.so"]}

logger = logging.getLogger(__name__)

class ProductPlannerValidators:
    """
    프로덕트 기획자 Agent 검증기
    
    입력 데이터, URL, 설정 등의 유효성을 검증하는 헬퍼 클래스
    """
    
    def __init__(self):
        """ProductPlannerValidators 초기화"""
        logger.info("ProductPlannerValidators 초기화 완료")
    
    def validate_figma_url(self, url: str) -> bool:
        """
        Figma URL 유효성 검증
        
        Args:
            url: 검증할 Figma URL
            
        Returns:
            URL 유효성 여부
        """
        try:
            if not url or not isinstance(url, str):
                return False
            
            # 기본 URL 형식 검증
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Figma 도메인 검증
            if "figma.com" not in parsed.netloc.lower():
                return False
            
            # Figma 파일 URL 패턴 검증
            figma_patterns = [
                r'/file/[a-zA-Z0-9]+',  # 파일 URL
                r'/proto/[a-zA-Z0-9]+',  # 프로토타입 URL
                r'/design/[a-zA-Z0-9]+', # 디자인 URL
            ]
            
            path = parsed.path
            for pattern in figma_patterns:
                if re.search(pattern, path):
                    logger.debug(f"유효한 Figma URL: {url}")
                    return True
            
            logger.warning(f"유효하지 않은 Figma URL 패턴: {url}")
            return False
            
        except Exception as e:
            logger.error(f"Figma URL 검증 중 오류: {str(e)}")
            return False
    
    def validate_notion_page_id(self, page_id: str) -> bool:
        """
        Notion 페이지 ID 유효성 검증
        
        Args:
            page_id: 검증할 Notion 페이지 ID
            
        Returns:
            페이지 ID 유효성 여부
        """
        try:
            if not page_id or not isinstance(page_id, str):
                return False
            
            # Notion UUID 패턴 검증 (하이픈 포함/미포함 모두 허용)
            uuid_pattern = r'^[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12}$'
            
            # 하이픈 제거 후 검증
            clean_id = page_id.replace('-', '')
            if len(clean_id) == 32 and re.match(r'^[a-f0-9]{32}$', clean_id):
                logger.debug(f"유효한 Notion 페이지 ID: {page_id}")
                return True
            
            # mock ID 패턴도 허용 (개발/테스트용)
            if page_id.startswith('mock_'):
                logger.debug(f"유효한 Mock 페이지 ID: {page_id}")
                return True
            
            logger.warning(f"유효하지 않은 Notion 페이지 ID: {page_id}")
            return False
            
        except Exception as e:
            logger.error(f"Notion 페이지 ID 검증 중 오류: {str(e)}")
            return False
    
    def validate_requirements_data(self, requirements: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        요구사항 데이터 유효성 검증
        
        Args:
            requirements: 검증할 요구사항 딕셔너리
            
        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        errors = []
        
        try:
            if not isinstance(requirements, dict):
                errors.append("요구사항 데이터는 딕셔너리 형태여야 합니다.")
                return False, errors
            
            # 필수 필드 검증
            required_fields = [
                "product_name",
                "functional_requirements",
                "user_stories"
            ]
            
            for field in required_fields:
                if field not in requirements:
                    errors.append(f"필수 필드 누락: {field}")
                elif not requirements[field]:
                    errors.append(f"필수 필드가 비어있음: {field}")
            
            # 기능 요구사항 검증
            if "functional_requirements" in requirements:
                func_reqs = requirements["functional_requirements"]
                if not isinstance(func_reqs, list):
                    errors.append("functional_requirements는 리스트 형태여야 합니다.")
                else:
                    for i, req in enumerate(func_reqs):
                        req_errors = self._validate_requirement_item(req, f"functional_requirements[{i}]")
                        errors.extend(req_errors)
            
            # 사용자 스토리 검증
            if "user_stories" in requirements:
                user_stories = requirements["user_stories"]
                if not isinstance(user_stories, list):
                    errors.append("user_stories는 리스트 형태여야 합니다.")
                elif not user_stories:
                    errors.append("최소 1개의 사용자 스토리가 필요합니다.")
            
            # 성공 지표 검증
            if "success_metrics" in requirements:
                success_metrics = requirements["success_metrics"]
                if not isinstance(success_metrics, list):
                    errors.append("success_metrics는 리스트 형태여야 합니다.")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.debug("요구사항 데이터 검증 성공")
            else:
                logger.warning(f"요구사항 데이터 검증 실패: {len(errors)}개 오류")
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"요구사항 데이터 검증 중 오류: {str(e)}")
            errors.append(f"검증 중 예외 발생: {str(e)}")
            return False, errors
    
    def validate_roadmap_data(self, roadmap: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        로드맵 데이터 유효성 검증
        
        Args:
            roadmap: 검증할 로드맵 딕셔너리
            
        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        errors = []
        
        try:
            if not isinstance(roadmap, dict):
                errors.append("로드맵 데이터는 딕셔너리 형태여야 합니다.")
                return False, errors
            
            # 필수 필드 검증
            required_fields = ["project_name", "milestones"]
            
            for field in required_fields:
                if field not in roadmap:
                    errors.append(f"필수 필드 누락: {field}")
            
            # 마일스톤 검증
            if "milestones" in roadmap:
                milestones = roadmap["milestones"]
                if not isinstance(milestones, list):
                    errors.append("milestones는 리스트 형태여야 합니다.")
                elif len(milestones) == 0:
                    errors.append("최소 1개의 마일스톤이 필요합니다.")
                else:
                    for i, milestone in enumerate(milestones):
                        milestone_errors = self._validate_milestone_item(milestone, f"milestones[{i}]")
                        errors.extend(milestone_errors)
            
            # 리스크 검증
            if "risks" in roadmap:
                risks = roadmap["risks"]
                if not isinstance(risks, list):
                    errors.append("risks는 리스트 형태여야 합니다.")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.debug("로드맵 데이터 검증 성공")
            else:
                logger.warning(f"로드맵 데이터 검증 실패: {len(errors)}개 오류")
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"로드맵 데이터 검증 중 오류: {str(e)}")
            errors.append(f"검증 중 예외 발생: {str(e)}")
            return False, errors
    
    def validate_analysis_data(self, analysis: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        분석 데이터 유효성 검증
        
        Args:
            analysis: 검증할 분석 결과 딕셔너리
            
        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        errors = []
        
        try:
            if not isinstance(analysis, dict):
                errors.append("분석 데이터는 딕셔너리 형태여야 합니다.")
                return False, errors
            
            # 필수 섹션 검증
            required_sections = [
                "component_analysis",
                "layout_analysis",
                "interaction_analysis"
            ]
            
            for section in required_sections:
                if section not in analysis:
                    errors.append(f"필수 분석 섹션 누락: {section}")
            
            # 컴포넌트 분석 검증
            if "component_analysis" in analysis:
                comp_analysis = analysis["component_analysis"]
                if not isinstance(comp_analysis, dict):
                    errors.append("component_analysis는 딕셔너리 형태여야 합니다.")
                elif "total_components" not in comp_analysis:
                    errors.append("component_analysis에 total_components 필드 누락")
            
            # 신뢰도 점수 검증
            if "confidence_score" in analysis:
                score = analysis["confidence_score"]
                if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                    errors.append("confidence_score는 0과 1 사이의 숫자여야 합니다.")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.debug("분석 데이터 검증 성공")
            else:
                logger.warning(f"분석 데이터 검증 실패: {len(errors)}개 오류")
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"분석 데이터 검증 중 오류: {str(e)}")
            errors.append(f"검증 중 예외 발생: {str(e)}")
            return False, errors
    
    def validate_project_name(self, project_name: str) -> bool:
        """
        프로젝트명 유효성 검증
        
        Args:
            project_name: 검증할 프로젝트명
            
        Returns:
            프로젝트명 유효성 여부
        """
        try:
            if not project_name or not isinstance(project_name, str):
                return False
            
            # 길이 검증
            if len(project_name.strip()) < 2:
                logger.warning("프로젝트명이 너무 짧습니다.")
                return False
            
            if len(project_name) > 100:
                logger.warning("프로젝트명이 너무 깁니다.")
                return False
            
            # 특수문자 검증 (기본적인 문자, 숫자, 공백, 하이픈, 언더스코어만 허용)
            if not re.match(r'^[a-zA-Z0-9가-힣\s\-_]+$', project_name):
                logger.warning("프로젝트명에 허용되지 않는 문자가 포함되어 있습니다.")
                return False
            
            logger.debug(f"유효한 프로젝트명: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"프로젝트명 검증 중 오류: {str(e)}")
            return False
    
    def validate_email(self, email: str) -> bool:
        """
        이메일 주소 유효성 검증
        
        Args:
            email: 검증할 이메일 주소
            
        Returns:
            이메일 유효성 여부
        """
        try:
            if not email or not isinstance(email, str):
                return False
            
            # 기본 이메일 패턴 검증
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            if re.match(email_pattern, email):
                logger.debug(f"유효한 이메일: {email}")
                return True
            else:
                logger.warning(f"유효하지 않은 이메일 형식: {email}")
                return False
            
        except Exception as e:
            logger.error(f"이메일 검증 중 오류: {str(e)}")
            return False
    
    def validate_json_structure(self, data: str, expected_keys: List[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        JSON 구조 유효성 검증
        
        Args:
            data: JSON 문자열
            expected_keys: 예상되는 키 리스트 (선택사항)
            
        Returns:
            (유효성 여부, 파싱된 데이터 또는 None)
        """
        try:
            if not data or not isinstance(data, str):
                return False, None
            
            # JSON 파싱
            parsed_data = json.loads(data)
            
            # 예상 키 검증
            if expected_keys:
                for key in expected_keys:
                    if key not in parsed_data:
                        logger.warning(f"예상 키 누락: {key}")
                        return False, None
            
            logger.debug("JSON 구조 검증 성공")
            return True, parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {str(e)}")
            return False, None
        except Exception as e:
            logger.error(f"JSON 구조 검증 중 오류: {str(e)}")
            return False, None
    
    def validate_priority_level(self, priority: str) -> bool:
        """
        우선순위 레벨 유효성 검증
        
        Args:
            priority: 검증할 우선순위
            
        Returns:
            우선순위 유효성 여부
        """
        valid_priorities = ["critical", "high", "medium", "low"]
        
        if not priority or not isinstance(priority, str):
            return False
        
        is_valid = priority.lower() in valid_priorities
        
        if not is_valid:
            logger.warning(f"유효하지 않은 우선순위: {priority}")
        
        return is_valid
    
    def validate_complexity_level(self, complexity: str) -> bool:
        """
        복잡도 레벨 유효성 검증
        
        Args:
            complexity: 검증할 복잡도
            
        Returns:
            복잡도 유효성 여부
        """
        valid_complexities = ["simple", "medium", "complex", "low", "high"]
        
        if not complexity or not isinstance(complexity, str):
            return False
        
        is_valid = complexity.lower() in valid_complexities
        
        if not is_valid:
            logger.warning(f"유효하지 않은 복잡도: {complexity}")
        
        return is_valid
    
    def validate_date_format(self, date_str: str, format_pattern: str = "%Y-%m-%d") -> bool:
        """
        날짜 형식 유효성 검증
        
        Args:
            date_str: 검증할 날짜 문자열
            format_pattern: 예상 날짜 형식 (기본: YYYY-MM-DD)
            
        Returns:
            날짜 형식 유효성 여부
        """
        try:
            if not date_str or not isinstance(date_str, str):
                return False
            
            from datetime import datetime
            datetime.strptime(date_str, format_pattern)
            
            logger.debug(f"유효한 날짜 형식: {date_str}")
            return True
            
        except ValueError:
            logger.warning(f"유효하지 않은 날짜 형식: {date_str}")
            return False
        except Exception as e:
            logger.error(f"날짜 형식 검증 중 오류: {str(e)}")
            return False
    
    # Private helper methods
    def _validate_requirement_item(self, requirement: Dict[str, Any], context: str) -> List[str]:
        """개별 요구사항 항목 검증"""
        errors = []
        
        try:
            if not isinstance(requirement, dict):
                errors.append(f"{context}: 요구사항 항목은 딕셔너리 형태여야 합니다.")
                return errors
            
            # 필수 필드 검증
            required_fields = ["title", "category", "priority"]
            for field in required_fields:
                if field not in requirement:
                    errors.append(f"{context}: 필수 필드 누락 - {field}")
                elif not requirement[field]:
                    errors.append(f"{context}: 필수 필드가 비어있음 - {field}")
            
            # 우선순위 검증
            if "priority" in requirement:
                if not self.validate_priority_level(requirement["priority"]):
                    errors.append(f"{context}: 유효하지 않은 우선순위 - {requirement['priority']}")
            
            # 복잡도 검증
            if "complexity" in requirement:
                if not self.validate_complexity_level(requirement["complexity"]):
                    errors.append(f"{context}: 유효하지 않은 복잡도 - {requirement['complexity']}")
            
        except Exception as e:
            errors.append(f"{context}: 검증 중 오류 - {str(e)}")
        
        return errors
    
    def _validate_milestone_item(self, milestone: Dict[str, Any], context: str) -> List[str]:
        """개별 마일스톤 항목 검증"""
        errors = []
        
        try:
            if not isinstance(milestone, dict):
                errors.append(f"{context}: 마일스톤 항목은 딕셔너리 형태여야 합니다.")
                return errors
            
            # 필수 필드 검증
            required_fields = ["name", "start_date", "end_date"]
            for field in required_fields:
                if field not in milestone:
                    errors.append(f"{context}: 필수 필드 누락 - {field}")
            
            # 날짜 형식 검증
            if "start_date" in milestone:
                if not self.validate_date_format(milestone["start_date"]):
                    errors.append(f"{context}: 유효하지 않은 시작 날짜 형식 - {milestone['start_date']}")
            
            if "end_date" in milestone:
                if not self.validate_date_format(milestone["end_date"]):
                    errors.append(f"{context}: 유효하지 않은 종료 날짜 형식 - {milestone['end_date']}")
            
            # 목표 검증
            if "objectives" in milestone:
                objectives = milestone["objectives"]
                if not isinstance(objectives, list):
                    errors.append(f"{context}: objectives는 리스트 형태여야 합니다.")
                elif len(objectives) == 0:
                    errors.append(f"{context}: 최소 1개의 목표가 필요합니다.")
            
        except Exception as e:
            errors.append(f"{context}: 검증 중 오류 - {str(e)}")
        
        return errors
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """검증기 상태 및 지원 기능 요약"""
        return {
            "validator_name": "ProductPlannerValidators",
            "supported_validations": [
                "figma_url",
                "notion_page_id", 
                "requirements_data",
                "roadmap_data",
                "analysis_data",
                "project_name",
                "email",
                "json_structure",
                "priority_level",
                "complexity_level",
                "date_format"
            ],
            "valid_priorities": ["critical", "high", "medium", "low"],
            "valid_complexities": ["simple", "medium", "complex", "low", "high"],
            "default_date_format": "%Y-%m-%d"
        } 