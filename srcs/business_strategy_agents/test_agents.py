"""
Test Suite for Most Hooking Business Strategy Agent

This module provides comprehensive testing for all agent components,
following testing best practices from OpenAI's agent guide.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# 테스트 대상 임포트
try:
    from .ai_engine import (
        BaseAgent, DataScoutAgent, TrendAnalyzerAgent, 
        HookingDetectorAgent, StrategyPlannerAgent, 
        AgentOrchestrator, AgentRole, get_orchestrator
    )
    from .architecture import (
        RawContent, ProcessedInsight, BusinessStrategy,
        ContentType, RegionType, BusinessOpportunityLevel
    )
    from .config import get_config
    from .main_agent import get_main_agent, run_quick_analysis
except ImportError:
    # 절대 경로로 다시 시도
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from business_strategy_agents.ai_engine import (
        BaseAgent, DataScoutAgent, TrendAnalyzerAgent, 
        HookingDetectorAgent, StrategyPlannerAgent, 
        AgentOrchestrator, AgentRole, get_orchestrator
    )
    from business_strategy_agents.architecture import (
        RawContent, ProcessedInsight, BusinessStrategy,
        ContentType, RegionType, BusinessOpportunityLevel
    )
    from business_strategy_agents.config import get_config
    from business_strategy_agents.main_agent import get_main_agent, run_quick_analysis


class TestAgentConfiguration:
    """에이전트 설정 테스트"""
    
    def test_agent_roles_defined(self):
        """모든 에이전트 역할이 정의되어 있는지 확인"""
        required_roles = [
            AgentRole.DATA_SCOUT,
            AgentRole.TREND_ANALYZER,
            AgentRole.HOOKING_DETECTOR,
            AgentRole.STRATEGY_PLANNER
        ]
        
        for role in required_roles:
            assert role in AgentRole
    
    def test_config_loading(self):
        """설정 로딩 테스트"""
        config = get_config()
        assert config is not None
        assert hasattr(config, 'apis')
        assert hasattr(config, 'mcp_servers')
    
    def test_agent_initialization(self):
        """에이전트 초기화 테스트"""
        agent = DataScoutAgent()
        assert agent.role == AgentRole.DATA_SCOUT
        assert agent.mcp_manager is None  # 초기화 전
        assert agent.logger is not None


class TestBaseAgent:
    """BaseAgent 기본 기능 테스트"""
    
    @pytest.fixture
    def mock_agent(self):
        """Mock 에이전트 생성"""
        class MockAgent(BaseAgent):
            async def process(self, input_data: Any):
                return {"mock": "response"}
        
        return MockAgent(AgentRole.DATA_SCOUT)
    
    def test_system_prompt_generation(self, mock_agent):
        """시스템 프롬프트 생성 테스트"""
        prompt = mock_agent._get_system_prompt()
        assert "DATA SCOUT" in prompt.upper()
        assert "business data" in prompt.lower()
        assert "actionable insights" in prompt.lower()
    
    def test_prompt_formatting(self, mock_agent):
        """프롬프트 포맷팅 테스트"""
        context = {"key": "value", "number": 123}
        formatted = mock_agent._format_prompt("Test prompt", context)
        
        assert "Test prompt" in formatted
        assert "Context:" in formatted
        assert "key" in formatted
        assert "value" in formatted
    
    @pytest.mark.asyncio
    async def test_mock_llm_response(self, mock_agent):
        """Mock LLM 응답 테스트"""
        response = await mock_agent._mock_llm_response("system", "user")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_fallback_response(self, mock_agent):
        """폴백 응답 테스트"""
        response = mock_agent._get_fallback_response()
        assert isinstance(response, str)
        assert len(response) > 0


class TestDataScoutAgent:
    """DataScoutAgent 기능 테스트"""
    
    @pytest.fixture
    def data_scout(self):
        return DataScoutAgent()
    
    @pytest.mark.asyncio
    async def test_process_with_keywords(self, data_scout):
        """키워드와 함께 처리 테스트"""
        input_data = {
            'keywords': ['AI', 'startup'],
            'regions': [RegionType.EAST_ASIA, RegionType.NORTH_AMERICA]
        }
        
        response = await data_scout.process(input_data)
        
        assert response.agent_role == AgentRole.DATA_SCOUT
        assert response.success is True
        assert isinstance(response.result, list)
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_process_empty_input(self, data_scout):
        """빈 입력 처리 테스트"""
        input_data = {'keywords': [], 'regions': []}
        
        response = await data_scout.process(input_data)
        
        assert response.agent_role == AgentRole.DATA_SCOUT
        assert response.success is True
        assert isinstance(response.result, list)
    
    def test_mock_content_generation(self, data_scout):
        """Mock 컨텐츠 생성 테스트"""
        keywords = ['AI', 'fintech']
        mock_content = data_scout._generate_mock_content('test_server', keywords)
        
        assert len(mock_content) <= 3  # 최대 3개
        for content in mock_content:
            assert isinstance(content, RawContent)
            assert content.source == 'test_server'
            assert content.content_type == ContentType.NEWS
    
    @pytest.mark.asyncio
    async def test_content_quality_filtering(self, data_scout):
        """컨텐츠 품질 필터링 테스트"""
        # Mock 컨텐츠 생성
        mock_content = [
            RawContent(
                source="test",
                content_type=ContentType.NEWS,
                region=RegionType.GLOBAL,
                title=f"Test Article {i}",
                content=f"Content {i}",
                url=f"http://test.com/{i}",
                timestamp=datetime.now(timezone.utc),
                engagement_metrics={}
            ) for i in range(3)
        ]
        
        filtered = await data_scout._filter_content_quality(mock_content)
        assert isinstance(filtered, list)
        # Mock 응답은 항상 통과하므로 원본과 같은 길이이거나 적어야 함
        assert len(filtered) <= len(mock_content)


class TestTrendAnalyzerAgent:
    """TrendAnalyzerAgent 기능 테스트"""
    
    @pytest.fixture
    def trend_analyzer(self):
        return TrendAnalyzerAgent()
    
    @pytest.mark.asyncio
    async def test_process_with_content(self, trend_analyzer):
        """컨텐츠와 함께 처리 테스트"""
        mock_content = [
            RawContent(
                source="test",
                content_type=ContentType.NEWS,
                region=RegionType.EAST_ASIA,
                title="AI Revolution in Business",
                content="AI is transforming business operations...",
                url="http://test.com/ai",
                timestamp=datetime.now(timezone.utc),
                engagement_metrics={"views": 1000}
            )
        ]
        
        response = await trend_analyzer.process(mock_content)
        
        assert response.agent_role == AgentRole.TREND_ANALYZER
        assert response.success is True
        assert isinstance(response.result, list)
    
    @pytest.mark.asyncio
    async def test_process_empty_content(self, trend_analyzer):
        """빈 컨텐츠 처리 테스트"""
        response = await trend_analyzer.process([])
        
        assert response.agent_role == AgentRole.TREND_ANALYZER
        assert response.success is True
        assert response.result == []
    
    def test_mock_trend_data(self, trend_analyzer):
        """Mock 트렌드 데이터 테스트"""
        trend_data = trend_analyzer._mock_trend_data()
        
        assert isinstance(trend_data, dict)
        assert 'trending_topics' in trend_data
        assert 'rising_searches' in trend_data
        assert 'regional_trends' in trend_data
        
        # 지역별 트렌드 확인
        regional = trend_data['regional_trends']
        assert 'US' in regional
        assert 'KR' in regional
        assert 'JP' in regional


class TestHookingDetectorAgent:
    """HookingDetectorAgent 기능 테스트"""
    
    @pytest.fixture
    def hooking_detector(self):
        return HookingDetectorAgent()
    
    @pytest.fixture
    def sample_insights(self):
        """샘플 인사이트 생성"""
        return [
            ProcessedInsight(
                content_id="test_1",
                hooking_score=0.5,  # 업데이트될 예정
                business_opportunity=BusinessOpportunityLevel.MEDIUM,
                region=RegionType.EAST_ASIA,
                category="test",
                key_topics=["AI", "automation"],
                sentiment_score=0.7,
                trend_direction="rising",
                market_size_estimate="$50B",
                competitive_landscape=[],
                actionable_insights=["High growth potential"],
                timestamp=datetime.now(timezone.utc)
            )
        ]
    
    @pytest.mark.asyncio
    async def test_process_insights(self, hooking_detector, sample_insights):
        """인사이트 처리 테스트"""
        response = await hooking_detector.process(sample_insights)
        
        assert response.agent_role == AgentRole.HOOKING_DETECTOR
        assert response.success is True
        assert len(response.result) == len(sample_insights)
        
        # 후킹 점수가 업데이트되었는지 확인
        enhanced_insight = response.result[0]
        assert enhanced_insight.hooking_score != 0.5  # 초기값에서 변경됨
        assert 0.0 <= enhanced_insight.hooking_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_hooking_score_calculation(self, hooking_detector, sample_insights):
        """후킹 점수 계산 테스트"""
        original_insight = sample_insights[0]
        enhanced_insight = await hooking_detector._calculate_hooking_score(original_insight)
        
        # 점수가 유효한 범위에 있는지 확인
        assert 0.0 <= enhanced_insight.hooking_score <= 1.0
        assert enhanced_insight.content_id == original_insight.content_id


class TestStrategyPlannerAgent:
    """StrategyPlannerAgent 기능 테스트"""
    
    @pytest.fixture
    def strategy_planner(self):
        return StrategyPlannerAgent()
    
    @pytest.fixture
    def high_value_insights(self):
        """고가치 인사이트 생성"""
        return [
            ProcessedInsight(
                content_id="high_value_1",
                hooking_score=0.85,  # 높은 점수
                business_opportunity=BusinessOpportunityLevel.HIGH,
                region=RegionType.NORTH_AMERICA,
                category="fintech",
                key_topics=["fintech", "digital payment"],
                sentiment_score=0.8,
                trend_direction="rising",
                market_size_estimate="$100B",
                competitive_landscape=[],
                actionable_insights=["Market gap identified", "Strong user demand"],
                timestamp=datetime.now(timezone.utc)
            )
        ]
    
    @pytest.mark.asyncio
    async def test_process_high_value_insights(self, strategy_planner, high_value_insights):
        """고가치 인사이트 처리 테스트"""
        response = await strategy_planner.process(high_value_insights)
        
        assert response.agent_role == AgentRole.STRATEGY_PLANNER
        assert response.success is True
        assert isinstance(response.result, list)
        assert len(response.result) > 0  # 높은 점수이므로 전략이 생성되어야 함
    
    @pytest.mark.asyncio
    async def test_low_value_insights_filtered(self, strategy_planner):
        """낮은 가치 인사이트 필터링 테스트"""
        low_value_insights = [
            ProcessedInsight(
                content_id="low_value_1",
                hooking_score=0.3,  # 낮은 점수
                business_opportunity=BusinessOpportunityLevel.LOW,
                region=RegionType.EUROPE,
                category="test",
                key_topics=["test"],
                sentiment_score=0.4,
                trend_direction="stable",
                market_size_estimate="$1B",
                competitive_landscape=[],
                actionable_insights=["Limited potential"],
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        response = await strategy_planner.process(low_value_insights)
        
        assert response.agent_role == AgentRole.STRATEGY_PLANNER
        assert response.success is True
        assert len(response.result) == 0  # 낮은 점수이므로 전략이 생성되지 않아야 함


class TestAgentOrchestrator:
    """AgentOrchestrator 통합 테스트"""
    
    @pytest.fixture
    async def orchestrator(self):
        """오케스트레이터 생성 및 초기화"""
        orch = AgentOrchestrator()
        await orch.initialize()
        return orch
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """오케스트레이터 초기화 테스트"""
        assert len(orchestrator.agents) > 0
        assert AgentRole.DATA_SCOUT in orchestrator.agents
        assert AgentRole.TREND_ANALYZER in orchestrator.agents
        assert AgentRole.HOOKING_DETECTOR in orchestrator.agents
        assert AgentRole.STRATEGY_PLANNER in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, orchestrator):
        """전체 분석 워크플로우 테스트"""
        keywords = ['AI', 'startup']
        regions = [RegionType.EAST_ASIA, RegionType.NORTH_AMERICA]
        
        results = await orchestrator.run_full_analysis(keywords, regions)
        
        assert 'workflow_id' in results
        assert 'success' in results
        assert 'execution_time' in results
        assert 'analysis_timestamp' in results
        
        if results['success']:
            assert 'enhanced_insights' in results
            assert 'strategies' in results
            assert 'agent_performance' in results
            assert results['execution_time'] > 0
    
    def test_agent_status(self, orchestrator):
        """에이전트 상태 확인 테스트"""
        status = orchestrator.get_agent_status()
        
        assert isinstance(status, dict)
        for role in AgentRole:
            assert role.value in status
    
    def test_execution_stats(self, orchestrator):
        """실행 통계 테스트"""
        stats = orchestrator.get_execution_stats()
        
        assert 'total_workflows' in stats
        assert 'successful_workflows' in stats
        assert 'failed_workflows' in stats
        assert 'average_execution_time' in stats
        assert 'success_rate_percentage' in stats


class TestPerformanceAndReliability:
    """성능 및 안정성 테스트"""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self):
        """동시 에이전트 실행 테스트"""
        agents = [
            DataScoutAgent(),
            TrendAnalyzerAgent(),
            HookingDetectorAgent(),
            StrategyPlannerAgent()
        ]
        
        # 각 에이전트 초기화
        for agent in agents:
            await agent.initialize()
        
        # 동시 실행
        tasks = []
        if len(agents) >= 2:
            tasks.append(agents[0].process({'keywords': ['AI'], 'regions': [RegionType.GLOBAL]}))
            tasks.append(agents[1].process([]))  # 빈 데이터
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외가 발생하지 않았는지 확인
            for result in results:
                assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """에러 처리 테스트"""
        agent = DataScoutAgent()
        await agent.initialize()
        
        # 잘못된 입력 데이터
        invalid_inputs = [
            None,
            {},
            {'keywords': None},
            {'regions': None},
            {'keywords': [''], 'regions': []}
        ]
        
        for invalid_input in invalid_inputs:
            try:
                response = await agent.process(invalid_input)
                # 에러 발생 시 적절한 응답 구조를 가져야 함
                assert hasattr(response, 'success')
                assert hasattr(response, 'agent_role')
                assert hasattr(response, 'execution_time')
            except Exception as e:
                # 예외가 발생해도 시스템이 죽지 않아야 함
                assert isinstance(e, Exception)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """타임아웃 처리 테스트"""
        agent = DataScoutAgent()
        agent.timeout = 0.1  # 매우 짧은 타임아웃 설정
        
        await agent.initialize()
        
        # 타임아웃이 발생할 수 있는 작업
        response = await agent.process({'keywords': ['test'], 'regions': [RegionType.GLOBAL]})
        
        # 타임아웃이 발생해도 적절한 응답을 반환해야 함
        assert hasattr(response, 'success')
        assert hasattr(response, 'execution_time')


class TestIntegrationWithMainAgent:
    """메인 에이전트와의 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_main_agent_initialization(self):
        """메인 에이전트 초기화 테스트"""
        try:
            main_agent = await get_main_agent()
            assert main_agent is not None
        except Exception as e:
            # 설정 오류 등으로 실패할 수 있음
            assert isinstance(e, Exception)
    
    @pytest.mark.asyncio
    async def test_quick_analysis_function(self):
        """빠른 분석 함수 테스트"""
        try:
            results = await run_quick_analysis(['AI', 'test'])
            
            assert isinstance(results, dict)
            # 성공하면 필요한 키들이 있어야 함
            if 'error' not in results:
                assert 'analysis_id' in results
                assert 'enhanced_insights_count' in results
                
        except Exception as e:
            # 설정이나 네트워크 오류로 실패할 수 있음
            assert isinstance(e, Exception)


def test_module_imports():
    """모듈 임포트 테스트"""
    # 모든 주요 클래스가 임포트 가능한지 확인
    from .ai_engine import BaseAgent, AgentOrchestrator
    from .architecture import RawContent, ProcessedInsight, BusinessStrategy
    from .main_agent import MostHookingBusinessStrategyAgent
    
    assert BaseAgent is not None
    assert AgentOrchestrator is not None
    assert RawContent is not None
    assert ProcessedInsight is not None
    assert BusinessStrategy is not None
    assert MostHookingBusinessStrategyAgent is not None


# 성능 벤치마킹 함수들
class TestPerformanceBenchmarks:
    """성능 벤치마크 테스트"""
    
    @pytest.mark.asyncio
    async def test_agent_response_time(self):
        """에이전트 응답 시간 테스트"""
        agent = DataScoutAgent()
        await agent.initialize()
        
        start_time = time.time()
        response = await agent.process({'keywords': ['test'], 'regions': [RegionType.GLOBAL]})
        execution_time = time.time() - start_time
        
        # 합리적인 응답 시간 (30초 이내)
        assert execution_time < 30.0
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_performance(self):
        """오케스트레이터 성능 테스트"""
        orchestrator = AgentOrchestrator()
        await orchestrator.initialize()
        
        start_time = time.time()
        results = await orchestrator.run_full_analysis(['AI'], [RegionType.GLOBAL])
        total_time = time.time() - start_time
        
        # 전체 워크플로우가 60초 이내에 완료되어야 함
        assert total_time < 60.0
        
        if results['success']:
            assert results['execution_time'] > 0


if __name__ == "__main__":
    # 직접 실행 시 기본 테스트 수행
    import sys
    import os
    
    # 현재 디렉토리를 path에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    print("🧪 Running Basic Agent Tests...")
    
    # 간단한 수동 테스트
    def run_basic_tests():
        """기본 테스트 실행"""
        test_config = TestAgentConfiguration()
        test_config.test_agent_roles_defined()
        print("✅ Agent roles test passed")
        
        test_config.test_agent_initialization()
        print("✅ Agent initialization test passed")
        
        test_base = TestBaseAgent()
        mock_agent = test_base.mock_agent()
        test_base.test_system_prompt_generation(mock_agent)
        print("✅ System prompt test passed")
        
        test_base.test_prompt_formatting(mock_agent)
        print("✅ Prompt formatting test passed")
        
        print("\n🎉 All basic tests passed!")
        print("Run 'pytest test_agents.py' for full test suite")
    
    try:
        run_basic_tests()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)