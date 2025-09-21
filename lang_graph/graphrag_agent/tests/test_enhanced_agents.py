"""
Enhanced GraphRAG Agents Test Suite

새로 추가된 에이전트들의 통합 테스트
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from agents.domain_specialist_agent import (
    DomainSpecialistAgent, DomainSpecialistConfig, DomainType, 
    DomainEntitySchema, DomainRelationshipSchema
)
from agents.realtime_update_agent import (
    RealtimeUpdateAgent, RealtimeUpdateConfig, DataSource, DataSourceType,
    UpdateEvent, UpdateStrategy
)
from agents.query_optimizer_agent import (
    QueryOptimizerAgent, QueryOptimizerConfig, QueryType, SearchStrategy,
    QueryIntent, SearchResult
)
from agents.security_privacy_agent import (
    SecurityPrivacyAgent, SecurityPrivacyConfig, DataClassification, 
    PrivacyLevel, SecurityThreat, SecurityPolicy, PrivacyRule
)
from agents.prompt_engine_agent import (
    PromptEngineAgent, PromptEngineConfig, PromptType, PromptTemplate,
    PromptContext, PromptResult
)


class TestDomainSpecialistAgent:
    """Domain Specialist Agent 테스트"""
    
    @pytest.fixture
    def domain_config(self):
        return DomainSpecialistConfig(
            openai_api_key="test-key",
            model_name="test-model",
            domain_type=DomainType.BIOMEDICAL
        )
    
    @pytest.fixture
    def domain_agent(self, domain_config):
        return DomainSpecialistAgent(domain_config)
    
    def test_initialization(self, domain_agent):
        """에이전트 초기화 테스트"""
        assert domain_agent.domain_type == DomainType.BIOMEDICAL
        assert len(domain_agent.entity_schemas) > 0
        assert len(domain_agent.relationship_schemas) > 0
    
    def test_load_domain_schemas(self, domain_agent):
        """도메인 스키마 로드 테스트"""
        schemas = domain_agent._load_domain_schemas()
        assert "gene" in schemas
        assert "protein" in schemas
        assert "disease" in schemas
        
        gene_schema = schemas["gene"]
        assert gene_schema.entity_type == "gene"
        assert "name" in gene_schema.attributes
        assert "interacts_with" in gene_schema.relationships
    
    def test_validate_entity(self, domain_agent):
        """엔티티 검증 테스트"""
        valid_entity = {
            "entity_type": "gene",
            "name": "BRCA1",
            "symbol": "BRCA1"
        }
        
        result = domain_agent.validate_entity(valid_entity)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        invalid_entity = {
            "entity_type": "gene",
            "name": "BRCA1"
            # symbol 누락
        }
        
        result = domain_agent.validate_entity(invalid_entity)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_validate_relationship(self, domain_agent):
        """관계 검증 테스트"""
        valid_relationship = {
            "relationship_type": "interacts_with",
            "source_entity": {"entity_type": "gene", "name": "BRCA1"},
            "target_entity": {"entity_type": "protein", "name": "BRCA1_protein"},
            "confidence": 0.8
        }
        
        result = domain_agent.validate_relationship(valid_relationship)
        assert result["valid"] is True
        
        invalid_relationship = {
            "relationship_type": "interacts_with",
            "source_entity": {"entity_type": "disease", "name": "cancer"},
            "target_entity": {"entity_type": "protein", "name": "BRCA1_protein"},
            "confidence": 0.5  # 너무 낮음
        }
        
        result = domain_agent.validate_relationship(invalid_relationship)
        assert result["valid"] is False


class TestRealtimeUpdateAgent:
    """Real-time Update Agent 테스트"""
    
    @pytest.fixture
    def update_config(self):
        return RealtimeUpdateConfig(
            openai_api_key="test-key",
            model_name="test-model",
            update_strategy=UpdateStrategy.SMART_MERGE
        )
    
    @pytest.fixture
    def update_agent(self, update_config):
        return RealtimeUpdateAgent(update_config)
    
    @pytest.fixture
    def sample_data_source(self):
        return DataSource(
            source_id="test_api",
            source_type=DataSourceType.API,
            endpoint="https://api.example.com/data",
            update_frequency=300,
            credentials={"Authorization": "Bearer token"}
        )
    
    def test_initialization(self, update_agent):
        """에이전트 초기화 테스트"""
        assert update_agent.update_strategy == UpdateStrategy.SMART_MERGE
        assert update_agent.running is False
        assert len(update_agent.data_sources) == 0
    
    def test_add_data_source(self, update_agent, sample_data_source):
        """데이터 소스 추가 테스트"""
        update_agent.data_sources.append(sample_data_source)
        assert len(update_agent.data_sources) == 1
        assert update_agent.data_sources[0].source_id == "test_api"
    
    def test_should_update_source(self, update_agent, sample_data_source):
        """업데이트 필요 여부 확인 테스트"""
        # 처음에는 업데이트 필요
        assert update_agent._should_update_source(sample_data_source) is True
        
        # 최근 업데이트 후에는 불필요
        sample_data_source.last_update = datetime.now()
        assert update_agent._should_update_source(sample_data_source) is False
    
    def test_calculate_data_hash(self, update_agent):
        """데이터 해시 계산 테스트"""
        data1 = {"key": "value"}
        data2 = {"key": "value"}
        data3 = {"key": "different"}
        
        hash1 = update_agent._calculate_data_hash(data1)
        hash2 = update_agent._calculate_data_hash(data2)
        hash3 = update_agent._calculate_data_hash(data3)
        
        assert hash1 == hash2  # 같은 데이터
        assert hash1 != hash3  # 다른 데이터
    
    def test_generate_event_id(self, update_agent, sample_data_source):
        """이벤트 ID 생성 테스트"""
        data = {"test": "data"}
        event_id = update_agent._generate_event_id(sample_data_source, data)
        
        assert event_id.startswith("test_api_")
        assert len(event_id) > 20  # 충분한 길이


class TestQueryOptimizerAgent:
    """Query Optimizer Agent 테스트"""
    
    @pytest.fixture
    def query_config(self):
        return QueryOptimizerConfig(
            openai_api_key="test-key",
            model_name="test-model",
            enable_intent_analysis=True,
            enable_query_expansion=True
        )
    
    @pytest.fixture
    def query_agent(self, query_config):
        return QueryOptimizerAgent(query_config)
    
    def test_initialization(self, query_agent):
        """에이전트 초기화 테스트"""
        assert query_agent.config.enable_intent_analysis is True
        assert query_agent.config.enable_query_expansion is True
        assert len(query_agent.query_cache) == 0
    
    def test_select_search_strategy(self, query_agent):
        """검색 전략 선택 테스트"""
        # 관계 질의는 그래프 순회
        relational_intent = QueryIntent(
            query_type=QueryType.RELATIONAL,
            entities=["entity1", "entity2"],
            relationships=["relates_to"],
            confidence=0.8
        )
        
        strategy = query_agent._select_search_strategy(relational_intent, {})
        assert strategy == SearchStrategy.GRAPH_TRAVERSAL
        
        # 복합 질의는 하이브리드
        complex_intent = QueryIntent(
            query_type=QueryType.COMPLEX,
            entities=["entity1"],
            relationships=["relates_to"],
            confidence=0.8,
            complexity_score=0.9
        )
        
        strategy = query_agent._select_search_strategy(complex_intent, {})
        assert strategy == SearchStrategy.HYBRID
    
    def test_calculate_node_relevance(self, query_agent):
        """노드 관련도 계산 테스트"""
        query = "Apple company"
        node_data = {
            "name": "Apple Inc.",
            "description": "Technology company",
            "attributes": {"sector": "technology"}
        }
        
        relevance = query_agent._calculate_node_relevance(query, node_data)
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.5  # "Apple"이 이름에 포함되어 있음
    
    def test_calculate_similarity(self, query_agent):
        """유사도 계산 테스트"""
        embedding1 = [0.1, 0.2, 0.3, 0.4]
        embedding2 = [0.1, 0.2, 0.3, 0.4]  # 동일
        embedding3 = [0.9, 0.8, 0.7, 0.6]  # 반대
        
        sim1 = query_agent._calculate_similarity(embedding1, embedding2)
        sim2 = query_agent._calculate_similarity(embedding1, embedding3)
        
        assert sim1 == 1.0  # 완전 동일
        assert sim2 < 0.0   # 반대 방향


class TestSecurityPrivacyAgent:
    """Security & Privacy Agent 테스트"""
    
    @pytest.fixture
    def security_config(self):
        return SecurityPrivacyConfig(
            openai_api_key="test-key",
            model_name="test-model",
            enable_data_classification=True,
            enable_privacy_protection=True
        )
    
    @pytest.fixture
    def security_agent(self, security_config):
        return SecurityPrivacyAgent(security_config)
    
    def test_initialization(self, security_agent):
        """에이전트 초기화 테스트"""
        assert len(security_agent.security_policies) > 0
        assert len(security_agent.privacy_rules) > 0
        assert len(security_agent.security_events) == 0
    
    def test_classify_data(self, security_agent):
        """데이터 분류 테스트"""
        # 공개 데이터
        public_data = {"content": "일반적인 정보"}
        classification = await security_agent.classify_data(public_data)
        assert classification == DataClassification.PUBLIC
        
        # 민감한 데이터 (이메일 포함)
        sensitive_data = {"content": "연락처: test@example.com"}
        classification = await security_agent.classify_data(sensitive_data)
        assert classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]
    
    def test_anonymize_data(self, security_agent):
        """데이터 익명화 테스트"""
        rule = PrivacyRule(
            rule_id="email_test",
            name="Email Test",
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            action="anonymize",
            sensitivity_score=0.8,
            description="Email anonymization"
        )
        
        content = "연락처: test@example.com"
        anonymized = security_agent._anonymize_data(content, rule)
        
        assert "test@example.com" not in anonymized
        assert "[ANONYMIZED_" in anonymized
    
    def test_encrypt_data(self, security_agent):
        """데이터 암호화 테스트"""
        rule = PrivacyRule(
            rule_id="ssn_test",
            name="SSN Test",
            pattern=r'\b\d{3}-\d{2}-\d{4}\b',
            action="encrypt",
            sensitivity_score=1.0,
            description="SSN encryption"
        )
        
        content = "SSN: 123-45-6789"
        encrypted = security_agent._encrypt_data(content, rule)
        
        assert "123-45-6789" not in encrypted
        assert "[ENCRYPTED_" in encrypted
    
    def test_detect_threats(self, security_agent):
        """위협 탐지 테스트"""
        malicious_data = {
            "content": "<script>alert('xss')</script>"
        }
        
        threats = await security_agent.detect_threats(malicious_data, {})
        assert len(threats) > 0
        assert any(t.threat_type == SecurityThreat.INJECTION_ATTACK for t in threats)
    
    def test_get_security_status(self, security_agent):
        """보안 상태 조회 테스트"""
        status = security_agent.get_security_status()
        
        assert "policies_count" in status
        assert "privacy_rules_count" in status
        assert "security_events_count" in status
        assert status["policies_count"] > 0


class TestPromptEngineAgent:
    """Prompt Engine Agent 테스트"""
    
    @pytest.fixture
    def prompt_config(self):
        return PromptEngineConfig(
            openai_api_key="test-key",
            model_name="test-model",
            enable_dynamic_prompts=True,
            enable_context_awareness=True
        )
    
    @pytest.fixture
    def prompt_agent(self, prompt_config):
        return PromptEngineAgent(prompt_config)
    
    @pytest.fixture
    def sample_context(self):
        return PromptContext(
            domain="biomedical",
            task_type="entity_extraction",
            user_level="expert",
            language="ko",
            cultural_context="Korean medical research"
        )
    
    def test_initialization(self, prompt_agent):
        """에이전트 초기화 테스트"""
        assert len(prompt_agent.prompt_templates) > 0
        assert len(prompt_agent.few_shot_examples) > 0
        assert len(prompt_agent.prompt_history) == 0
    
    def test_load_prompt_templates(self, prompt_agent):
        """프롬프트 템플릿 로드 테스트"""
        templates = prompt_agent.prompt_templates
        
        assert PromptType.ENTITY_EXTRACTION.value in templates
        assert PromptType.RELATIONSHIP_EXTRACTION.value in templates
        assert PromptType.QUERY_OPTIMIZATION.value in templates
        
        # 각 타입별로 템플릿이 있는지 확인
        for prompt_type in templates:
            type_templates = templates[prompt_type]
            assert PromptTemplate.STRUCTURED.value in type_templates
            assert PromptTemplate.CONVERSATIONAL.value in type_templates
    
    def test_substitute_variables(self, prompt_agent):
        """변수 치환 테스트"""
        template = "안녕하세요 {name}님, {domain} 분야에서 {task}를 수행합니다."
        variables = {
            "name": "홍길동",
            "domain": "의학",
            "task": "진단"
        }
        
        result = prompt_agent._substitute_variables(template, variables)
        expected = "안녕하세요 홍길동님, 의학 분야에서 진단을 수행합니다."
        assert result == expected
    
    def test_calculate_complexity(self, prompt_agent):
        """복잡도 계산 테스트"""
        simple_text = "안녕하세요. 반갑습니다."
        complex_text = "이것은 매우 복잡한 문장입니다. 여러 개의 구문과 특수 문자 [{}]를 포함하고 있습니다."
        
        simple_complexity = prompt_agent._calculate_complexity(simple_text)
        complex_complexity = prompt_agent._calculate_complexity(complex_text)
        
        assert simple_complexity < complex_complexity
        assert 0 <= simple_complexity <= 10
        assert 0 <= complex_complexity <= 10
    
    def test_assess_response_quality(self, prompt_agent):
        """응답 품질 평가 테스트"""
        good_response = '{"entities": [{"name": "test", "confidence": 0.9}]}'
        bad_response = "error occurred"
        
        good_quality = prompt_agent._assess_response_quality(good_response)
        bad_quality = prompt_agent._assess_response_quality(bad_response)
        
        assert good_quality > bad_quality
        assert 0 <= good_quality <= 1
        assert 0 <= bad_quality <= 1
    
    def test_truncate_prompt(self, prompt_agent):
        """프롬프트 길이 제한 테스트"""
        long_prompt = "안녕하세요. " * 1000  # 매우 긴 프롬프트
        
        truncated = prompt_agent._truncate_prompt(long_prompt)
        assert len(truncated) <= prompt_agent.config.max_prompt_length
        assert truncated.endswith("...")
    
    def test_get_prompt_statistics(self, prompt_agent):
        """프롬프트 통계 조회 테스트"""
        # 빈 상태
        stats = prompt_agent.get_prompt_statistics()
        assert stats["total_prompts"] == 0
        
        # 가짜 히스토리 추가
        fake_result = PromptResult(
            prompt="test prompt",
            template_used="structured",
            variables={},
            confidence=0.9,
            metadata={
                "prompt_type": "entity_extraction",
                "length": 100,
                "timestamp": datetime.now().isoformat()
            }
        )
        prompt_agent.prompt_history.append(fake_result)
        
        stats = prompt_agent.get_prompt_statistics()
        assert stats["total_prompts"] == 1
        assert stats["average_length"] == 100
        assert "structured" in stats["template_usage"]


class TestIntegration:
    """통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_domain_specialist_with_biomedical_data(self):
        """생명과학 도메인 전문가 통합 테스트"""
        config = DomainSpecialistConfig(
            openai_api_key="test-key",
            model_name="test-model",
            domain_type=DomainType.BIOMEDICAL
        )
        agent = DomainSpecialistAgent(config)
        
        # 생명과학 텍스트 처리
        text = "BRCA1 유전자는 유방암과 난소암의 위험을 증가시킵니다."
        
        with patch.object(agent, 'call_llm', return_value='{"entities": [{"name": "BRCA1", "type": "gene"}]}'):
            entities = await agent.extract_domain_entities(text)
            assert len(entities) > 0
    
    @pytest.mark.asyncio
    async def test_security_privacy_with_sensitive_data(self):
        """보안 및 프라이버시 통합 테스트"""
        config = SecurityPrivacyConfig(
            openai_api_key="test-key",
            model_name="test-model"
        )
        agent = SecurityPrivacyAgent(config)
        
        # 민감한 데이터 처리
        sensitive_data = {
            "name": "홍길동",
            "email": "hong@example.com",
            "ssn": "123-45-6789"
        }
        
        # 데이터 분류
        classification = await agent.classify_data(sensitive_data)
        assert classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]
        
        # 프라이버시 보호
        protected_data = await agent.protect_privacy(sensitive_data, classification)
        assert "hong@example.com" not in str(protected_data)
        assert "123-45-6789" not in str(protected_data)
    
    @pytest.mark.asyncio
    async def test_query_optimizer_with_complex_query(self):
        """질의 최적화 통합 테스트"""
        config = QueryOptimizerConfig(
            openai_api_key="test-key",
            model_name="test-model"
        )
        agent = QueryOptimizerAgent(config)
        
        # 복잡한 질의 처리
        query = "Apple과 Microsoft의 경쟁 관계는?"
        graph_data = {
            "entities": [
                {"id": "apple", "name": "Apple Inc.", "type": "company"},
                {"id": "microsoft", "name": "Microsoft", "type": "company"}
            ],
            "relationships": [
                {"source": "apple", "target": "microsoft", "type": "competes_with"}
            ]
        }
        
        with patch.object(agent, 'call_llm') as mock_llm:
            mock_llm.return_value = '{"query_type": "comparative", "entities": ["Apple", "Microsoft"]}'
            
            result = await agent.optimize_query(query, graph_data)
            assert result["original_query"] == query
            assert "optimized_query" in result
            assert "search_strategy" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
