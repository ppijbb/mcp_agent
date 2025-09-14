"""
Enhanced Graph Generator Agent

기존 GraphGeneratorAgent에 도메인 특화, 보안, 질의 최적화 기능을 통합한 버전
"""

import asyncio
import json
import hashlib
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from .base_agent_simple import BaseAgent, BaseAgentConfig


class DomainType(Enum):
    """지원하는 도메인 타입"""
    GENERAL = "general"
    BIOMEDICAL = "biomedical"
    FINANCE = "finance"
    TECHNICAL = "technical"


class DataClassification(Enum):
    """데이터 분류"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class QueryType(Enum):
    """질의 타입"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    RELATIONAL = "relational"


@dataclass
class DomainEntitySchema:
    """도메인별 엔티티 스키마"""
    entity_type: str
    attributes: List[str]
    relationships: List[str]
    validation_rules: Dict[str, Any]


@dataclass
class SecurityEvent:
    """보안 이벤트"""
    event_id: str
    timestamp: datetime
    threat_type: str
    severity: str
    description: str


class EnhancedGraphGeneratorConfig(BaseAgentConfig):
    """향상된 그래프 생성기 설정"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 도메인 특화 설정
        self.enable_domain_specialization = kwargs.get("enable_domain_specialization", False)
        self.domain_type = kwargs.get("domain_type", DomainType.GENERAL)
        
        # 보안 설정
        self.enable_security_privacy = kwargs.get("enable_security_privacy", True)
        self.default_data_classification = kwargs.get("default_data_classification", DataClassification.INTERNAL)
        
        # 질의 최적화 설정
        self.enable_query_optimization = kwargs.get("enable_query_optimization", True)
        
        # 그래프 생성 설정
        self.entity_extraction_threshold = kwargs.get("entity_extraction_threshold", 0.7)
        self.relationship_extraction_threshold = kwargs.get("relationship_extraction_threshold", 0.6)
        self.enable_quality_validation = kwargs.get("enable_quality_validation", True)


class EnhancedGraphGeneratorAgent(BaseAgent):
    """향상된 그래프 생성 에이전트 - 모든 기능 통합"""
    
    def __init__(self, config: EnhancedGraphGeneratorConfig):
        super().__init__(config)
        self.domain_type = config.domain_type
        self.entity_schemas = self._load_domain_schemas()
        self.privacy_rules = self._load_privacy_rules()
        self.security_events = []
        
    def _load_domain_schemas(self) -> Dict[str, DomainEntitySchema]:
        """도메인별 엔티티 스키마 로드"""
        schemas = {
            DomainType.BIOMEDICAL: {
                "gene": DomainEntitySchema(
                    entity_type="gene",
                    attributes=["name", "symbol", "chromosome", "function"],
                    relationships=["interacts_with", "regulates", "associated_with"],
                    validation_rules={"name": "required", "symbol": "required"}
                ),
                "protein": DomainEntitySchema(
                    entity_type="protein",
                    attributes=["name", "uniprot_id", "function", "location"],
                    relationships=["interacts_with", "encoded_by", "involved_in"],
                    validation_rules={"name": "required"}
                ),
                "disease": DomainEntitySchema(
                    entity_type="disease",
                    attributes=["name", "icd_code", "symptoms", "treatments"],
                    relationships=["caused_by", "treated_with", "associated_with"],
                    validation_rules={"name": "required"}
                )
            },
            DomainType.FINANCE: {
                "company": DomainEntitySchema(
                    entity_type="company",
                    attributes=["name", "ticker", "sector", "market_cap"],
                    relationships=["competes_with", "acquires", "invests_in"],
                    validation_rules={"name": "required", "ticker": "required"}
                ),
                "person": DomainEntitySchema(
                    entity_type="person",
                    attributes=["name", "title", "company", "expertise"],
                    relationships=["works_for", "reports_to", "collaborates_with"],
                    validation_rules={"name": "required"}
                )
            },
            DomainType.TECHNICAL: {
                "technology": DomainEntitySchema(
                    entity_type="technology",
                    attributes=["name", "category", "version", "description"],
                    relationships=["depends_on", "replaces", "compatible_with"],
                    validation_rules={"name": "required", "category": "required"}
                ),
                "api": DomainEntitySchema(
                    entity_type="api",
                    attributes=["name", "endpoint", "method", "parameters"],
                    relationships=["uses", "calls", "integrates_with"],
                    validation_rules={"name": "required", "endpoint": "required"}
                )
            }
        }
        
        return schemas.get(self.domain_type, {})
    
    def _load_privacy_rules(self) -> List[Dict[str, Any]]:
        """프라이버시 규칙 로드"""
        return [
            {
                "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "action": "anonymize",
                "sensitivity_score": 0.8,
                "description": "이메일 주소 익명화"
            },
            {
                "pattern": r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
                "action": "anonymize",
                "sensitivity_score": 0.9,
                "description": "전화번호 익명화"
            },
            {
                "pattern": r'\b\d{3}-\d{2}-\d{4}\b',
                "action": "encrypt",
                "sensitivity_score": 1.0,
                "description": "주민등록번호 암호화"
            }
        ]
    
    async def classify_data(self, data: Dict[str, Any]) -> DataClassification:
        """데이터 분류"""
        if not self.config.enable_security_privacy:
            return self.config.default_data_classification
        
        content = json.dumps(data, ensure_ascii=False)
        sensitivity_score = 0.0
        
        for rule in self.privacy_rules:
            if re.search(rule["pattern"], content, re.IGNORECASE):
                sensitivity_score += rule["sensitivity_score"]
        
        if sensitivity_score >= 0.9:
            return DataClassification.RESTRICTED
        elif sensitivity_score >= 0.7:
            return DataClassification.CONFIDENTIAL
        elif sensitivity_score >= 0.3:
            return DataClassification.INTERNAL
        else:
            return DataClassification.PUBLIC
    
    async def protect_privacy(self, data: Dict[str, Any], classification: DataClassification) -> Dict[str, Any]:
        """프라이버시 보호 적용"""
        if not self.config.enable_security_privacy:
            return data
        
        protected_data = data.copy()
        content = json.dumps(protected_data, ensure_ascii=False)
        
        for rule in self.privacy_rules:
            if rule["action"] == "anonymize":
                content = self._anonymize_data(content, rule)
            elif rule["action"] == "encrypt":
                content = self._encrypt_data(content, rule)
        
        try:
            protected_data = json.loads(content)
        except json.JSONDecodeError:
            protected_data = {"error": "Failed to process protected data"}
        
        return protected_data
    
    def _anonymize_data(self, content: str, rule: Dict[str, Any]) -> str:
        """데이터 익명화"""
        def anonymize_match(match):
            original = match.group(0)
            anonymized = hashlib.sha256(original.encode()).hexdigest()[:8]
            return f"[ANONYMIZED_{anonymized}]"
        
        return re.sub(rule["pattern"], anonymize_match, content, flags=re.IGNORECASE)
    
    def _encrypt_data(self, content: str, rule: Dict[str, Any]) -> str:
        """데이터 암호화"""
        def encrypt_match(match):
            original = match.group(0)
            encrypted = original.encode().hex()
            return f"[ENCRYPTED_{encrypted}]"
        
        return re.sub(rule["pattern"], encrypt_match, content, flags=re.IGNORECASE)
    
    async def extract_domain_entities(self, text: str) -> List[Dict[str, Any]]:
        """도메인별 엔티티 추출"""
        if not self.config.enable_domain_specialization:
            return []
        
        prompt = f"""
        다음 텍스트에서 {self.domain_type.value} 분야의 엔티티를 추출하세요:
        
        텍스트: {text}
        
        추출할 엔티티 타입: {list(self.entity_schemas.keys())}
        
        각 엔티티에 대해 다음 정보를 포함하세요:
        - entity_type: 엔티티 타입
        - name: 엔티티 이름
        - attributes: 관련 속성들
        - confidence: 추출 신뢰도 (0-1)
        """
        
        response = await self.call_llm(prompt)
        return self._parse_entities(response)
    
    async def extract_domain_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """도메인별 관계 추출"""
        if not self.config.enable_domain_specialization:
            return []
        
        prompt = f"""
        다음 텍스트와 엔티티들에서 {self.domain_type.value} 분야의 관계를 추출하세요:
        
        텍스트: {text}
        
        엔티티들: {json.dumps(entities, ensure_ascii=False, indent=2)}
        
        각 관계에 대해 다음 정보를 포함하세요:
        - relationship_type: 관계 타입
        - source_entity: 소스 엔티티
        - target_entity: 타겟 엔티티
        - properties: 관계 속성들
        - confidence: 추출 신뢰도 (0-1)
        """
        
        response = await self.call_llm(prompt)
        return self._parse_relationships(response)
    
    def validate_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """엔티티 도메인 검증"""
        if not self.config.enable_domain_specialization:
            return {"valid": True, "errors": []}
        
        entity_type = entity.get("entity_type")
        if entity_type not in self.entity_schemas:
            return {"valid": False, "errors": [f"Unknown entity type: {entity_type}"]}
        
        schema = self.entity_schemas[entity_type]
        errors = []
        
        for attr, rule in schema.validation_rules.items():
            if rule == "required" and attr not in entity:
                errors.append(f"Missing required attribute: {attr}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_relationship(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """관계 도메인 검증"""
        if not self.config.enable_domain_specialization:
            return {"valid": True, "errors": []}
        
        # 간단한 검증 로직
        required_fields = ["relationship_type", "source_entity", "target_entity"]
        errors = []
        
        for field in required_fields:
            if field not in relationship:
                errors.append(f"Missing required field: {field}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    async def optimize_query(self, query: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """질의 최적화"""
        if not self.config.enable_query_optimization:
            return {"original_query": query, "optimized_query": query}
        
        prompt = f"""
        다음 질의를 {self.domain_type.value} 분야에 최적화하세요:
        
        원본 질의: {query}
        
        최적화된 질의를 생성하세요:
        1. 키워드 확장
        2. 동의어 추가
        3. 관련 개념 포함
        4. 검색 전략 제안
        """
        
        response = await self.call_llm(prompt)
        
        return {
            "original_query": query,
            "optimized_query": response.strip(),
            "domain_type": self.domain_type.value,
            "optimization_timestamp": datetime.now().isoformat()
        }
    
    def _parse_entities(self, response: str) -> List[Dict[str, Any]]:
        """LLM 응답에서 엔티티 파싱"""
        try:
            entities = json.loads(response)
            if isinstance(entities, list):
                return entities
            else:
                return [entities]
        except json.JSONDecodeError:
            entities = []
            lines = response.split('\n')
            for line in lines:
                if 'entity_type' in line and 'name' in line:
                    try:
                        entity = json.loads(line)
                        entities.append(entity)
                    except:
                        continue
            return entities
    
    def _parse_relationships(self, response: str) -> List[Dict[str, Any]]:
        """LLM 응답에서 관계 파싱"""
        try:
            relationships = json.loads(response)
            if isinstance(relationships, list):
                return relationships
            else:
                return [relationships]
        except json.JSONDecodeError:
            relationships = []
            lines = response.split('\n')
            for line in lines:
                if 'relationship_type' in line and 'source_entity' in line:
                    try:
                        rel = json.loads(line)
                        relationships.append(rel)
                    except:
                        continue
            return relationships
    
    async def process(self, text_units: List[Dict[str, Any]]) -> Dict[str, Any]:
        """메인 처리 메서드 - 모든 기능 통합"""
        self.logger.info(f"Processing {len(text_units)} text units with enhanced features")
        
        # 1. 데이터 분류
        data_classification = await self.classify_data({"text_units": text_units})
        self.logger.info(f"Data classified as: {data_classification.value}")
        
        # 2. 프라이버시 보호
        protected_data = await self.protect_privacy({"text_units": text_units}, data_classification)
        
        # 3. 도메인별 엔티티 추출
        all_entities = []
        all_relationships = []
        
        for unit in protected_data.get("text_units", text_units):
            text = unit.get("text_unit", "")
            
            # 엔티티 추출
            entities = await self.extract_domain_entities(text)
            for entity in entities:
                entity["text_unit_id"] = unit.get("id")
                all_entities.append(entity)
            
            # 관계 추출
            relationships = await self.extract_domain_relationships(text, entities)
            for relationship in relationships:
                relationship["text_unit_id"] = unit.get("id")
                all_relationships.append(relationship)
        
        # 4. 검증
        if self.config.enable_quality_validation:
            validated_entities = []
            validated_relationships = []
            
            for entity in all_entities:
                validation = self.validate_entity(entity)
                if validation["valid"]:
                    validated_entities.append(entity)
                else:
                    self.logger.warning(f"Entity validation failed: {validation['errors']}")
            
            for relationship in all_relationships:
                validation = self.validate_relationship(relationship)
                if validation["valid"]:
                    validated_relationships.append(relationship)
                else:
                    self.logger.warning(f"Relationship validation failed: {validation['errors']}")
            
            all_entities = validated_entities
            all_relationships = validated_relationships
        
        # 5. 결과 구성
        result = {
            "status": "completed",
            "entities": all_entities,
            "relationships": all_relationships,
            "domain_type": self.domain_type.value,
            "data_classification": data_classification.value,
            "security_enabled": self.config.enable_security_privacy,
            "domain_specialization_enabled": self.config.enable_domain_specialization,
            "query_optimization_enabled": self.config.enable_query_optimization,
            "processing_timestamp": datetime.now().isoformat(),
            "stats": {
                "total_entities": len(all_entities),
                "total_relationships": len(all_relationships),
                "text_units_processed": len(text_units)
            }
        }
        
        return result
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """향상된 상태 정보"""
        base_status = self.get_metrics()
        
        return {
            **base_status,
            "domain_type": self.domain_type.value,
            "entity_schemas_count": len(self.entity_schemas),
            "privacy_rules_count": len(self.privacy_rules),
            "security_events_count": len(self.security_events),
            "features_enabled": {
                "domain_specialization": self.config.enable_domain_specialization,
                "security_privacy": self.config.enable_security_privacy,
                "query_optimization": self.config.enable_query_optimization,
                "quality_validation": self.config.enable_quality_validation
            }
        }
