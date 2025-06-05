"""
Main Agent for Most Hooking Business Strategy Agent

This is the central orchestrator that connects all components and manages
the complete business intelligence workflow from data collection to Notion documentation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
import schedule
import json

from .config import get_config
from .architecture import RegionType, BusinessOpportunityLevel, get_architecture
from .ai_engine import get_orchestrator, AgentRole
from .notion_integration import get_notion_integration
from .mcp_layer import get_mcp_manager, MCPRequest, DataCollectorFactory

logger = logging.getLogger(__name__)


class AdditionalMCPServers:
    """추가 MCP 서버 설정 및 관리"""
    
    ADDITIONAL_SERVERS = {
        # 소셜 미디어 확장
        "social_tiktok": {
            "base_url": "https://api.tiktok.com/v1/",
            "type": "social_media",
            "region": "global",
            "rate_limit": 100,
            "endpoints": {
                "trending": "trending/hashtags",
                "search": "search/videos"
            }
        },
        "social_instagram": {
            "base_url": "https://graph.instagram.com/v1/",
            "type": "social_media", 
            "region": "global",
            "rate_limit": 200,
            "endpoints": {
                "hashtags": "hashtags/{hashtag}/media",
                "search": "hashtags/search"
            }
        },
        "social_youtube": {
            "base_url": "https://www.googleapis.com/youtube/v3/",
            "type": "social_media",
            "region": "global", 
            "rate_limit": 10000,
            "endpoints": {
                "trending": "videos",
                "search": "search"
            }
        },
        
        # 비즈니스 데이터 확장
        "business_producthunt": {
            "base_url": "https://api.producthunt.com/v2/api/",
            "type": "business_data",
            "region": "global",
            "rate_limit": 100,
            "endpoints": {
                "posts": "posts",
                "trending": "posts?order=votes_count"
            }
        },
        "business_angellist": {
            "base_url": "https://api.angel.co/1/",
            "type": "business_data", 
            "region": "global",
            "rate_limit": 1000,
            "endpoints": {
                "startups": "startups",
                "funding": "startups/{id}/funding_rounds"
            }
        },
        "business_ycombinator": {
            "base_url": "https://api.ycombinator.com/v1/",
            "type": "business_data",
            "region": "global", 
            "rate_limit": 100,
            "endpoints": {
                "companies": "companies",
                "batches": "batches/current"
            }
        },
        
        # 시장 분석 도구
        "market_similarweb": {
            "base_url": "https://api.similarweb.com/v1/",
            "type": "market_analysis",
            "region": "global",
            "rate_limit": 100,
            "endpoints": {
                "traffic": "website/{domain}/traffic",
                "trending": "trending/websites"
            }
        },
        "market_semrush": {
            "base_url": "https://api.semrush.com/",
            "type": "market_analysis",
            "region": "global",
            "rate_limit": 10000,
            "endpoints": {
                "keywords": "reports/v1/keywords",
                "trends": "reports/v1/trends"
            }
        },
        
        # 동아시아 특화 소스
        "asia_kakaotalk": {
            "base_url": "https://kapi.kakao.com/v1/",
            "type": "social_media",
            "region": "east_asia",
            "rate_limit": 300,
            "endpoints": {
                "talk": "api/talk",
                "story": "api/story"
            }
        },
        "asia_line": {
            "base_url": "https://api.line.me/v2/",
            "type": "social_media", 
            "region": "east_asia",
            "rate_limit": 1000,
            "endpoints": {
                "timeline": "timeline",
                "today": "today"
            }
        },
        "asia_baidu": {
            "base_url": "https://api.baidu.com/",
            "type": "search_trends",
            "region": "east_asia",
            "rate_limit": 1000,
            "endpoints": {
                "trends": "trends/search",
                "index": "trends/index"
            }
        },
        
        # 경제 데이터
        "economic_yahoo_finance": {
            "base_url": "https://query1.finance.yahoo.com/v1/",
            "type": "economic_data",
            "region": "global",
            "rate_limit": 2000,
            "endpoints": {
                "quote": "quote",
                "trending": "finance/trending/US"
            }
        },
        "economic_alpha_vantage": {
            "base_url": "https://www.alphavantage.co/",
            "type": "economic_data",
            "region": "global", 
            "rate_limit": 5,
            "endpoints": {
                "forex": "query?function=FX_DAILY",
                "crypto": "query?function=DIGITAL_CURRENCY_DAILY"
            }
        },
        
        # 컨설팅 & 리포트
        "consulting_mckinsey": {
            "base_url": "https://api.mckinsey.com/v1/",
            "type": "industry_reports",
            "region": "global",
            "rate_limit": 50,
            "endpoints": {
                "insights": "insights",
                "reports": "reports"
            }
        },
        "consulting_bcg": {
            "base_url": "https://api.bcg.com/v1/",
            "type": "industry_reports", 
            "region": "global",
            "rate_limit": 50,
            "endpoints": {
                "insights": "insights",
                "publications": "publications"
            }
        }
    }
    
    @classmethod
    async def register_additional_servers(cls, mcp_manager):
        """추가 MCP 서버들 등록"""
        logger.info("Registering additional MCP servers...")
        
        from .config import APIConfig
        
        registered_count = 0
        
        for server_name, server_config in cls.ADDITIONAL_SERVERS.items():
            try:
                api_config = APIConfig(
                    name=server_name,
                    base_url=server_config["base_url"],
                    rate_limit=server_config["rate_limit"],
                    timeout=30
                )
                
                success = await mcp_manager.register_server(server_name, api_config)
                if success:
                    registered_count += 1
                    logger.info(f"Registered additional server: {server_name}")
                else:
                    logger.warning(f"Failed to register server: {server_name}")
                    
            except Exception as e:
                logger.error(f"Error registering {server_name}: {e}")
        
        logger.info(f"Successfully registered {registered_count}/{len(cls.ADDITIONAL_SERVERS)} additional servers")
        return registered_count


class EnhancedDataCollector:
    """확장된 데이터 수집기 (추가 MCP 서버 활용)"""
    
    def __init__(self, mcp_manager):
        self.mcp_manager = mcp_manager
        
    async def collect_social_media_trends(self, keywords: List[str]) -> Dict[str, Any]:
        """소셜 미디어 트렌드 수집"""
        social_servers = [
            "social_tiktok", "social_instagram", "social_youtube",
            "social_twitter", "social_linkedin"
        ]
        
        trends_data = {}
        
        for server in social_servers:
            try:
                request = MCPRequest(
                    server_name=server,
                    endpoint="trending",
                    params={'keywords': keywords, 'limit': 20}
                )
                
                response = await self.mcp_manager.send_request(server, request)
                
                if response.status_code == 200:
                    trends_data[server] = response.data
                    logger.info(f"Collected trends from {server}")
                    
            except Exception as e:
                logger.warning(f"Failed to collect from {server}: {e}")
        
        return trends_data
    
    async def collect_startup_ecosystem_data(self, regions: List[RegionType]) -> Dict[str, Any]:
        """스타트업 생태계 데이터 수집"""
        startup_servers = [
            "business_producthunt", "business_angellist", 
            "business_ycombinator", "business_crunchbase"
        ]
        
        ecosystem_data = {}
        
        for server in startup_servers:
            try:
                request = MCPRequest(
                    server_name=server,
                    endpoint="trending" if "trending" in AdditionalMCPServers.ADDITIONAL_SERVERS.get(server, {}).get("endpoints", {}).values() else "companies",
                    params={'limit': 50, 'regions': [r.value for r in regions]}
                )
                
                response = await self.mcp_manager.send_request(server, request)
                
                if response.status_code == 200:
                    ecosystem_data[server] = response.data
                    logger.info(f"Collected startup data from {server}")
                    
            except Exception as e:
                logger.warning(f"Failed to collect startup data from {server}: {e}")
        
        return ecosystem_data
    
    async def collect_market_intelligence(self, keywords: List[str]) -> Dict[str, Any]:
        """시장 인텔리전스 수집"""
        market_servers = ["market_similarweb", "market_semrush"]
        economic_servers = ["economic_yahoo_finance", "economic_alpha_vantage"]
        
        market_data = {}
        
        # 시장 분석 데이터
        for server in market_servers:
            try:
                request = MCPRequest(
                    server_name=server,
                    endpoint="trending",
                    params={'keywords': keywords, 'timeframe': '30d'}
                )
                
                response = await self.mcp_manager.send_request(server, request)
                
                if response.status_code == 200:
                    market_data[server] = response.data
                    
            except Exception as e:
                logger.warning(f"Market data collection failed for {server}: {e}")
        
        # 경제 데이터
        for server in economic_servers:
            try:
                request = MCPRequest(
                    server_name=server,
                    endpoint="trending",
                    params={'symbols': ['SPY', 'QQQ', 'BTC-USD', 'ETH-USD']}
                )
                
                response = await self.mcp_manager.send_request(server, request)
                
                if response.status_code == 200:
                    market_data[server] = response.data
                    
            except Exception as e:
                logger.warning(f"Economic data collection failed for {server}: {e}")
        
        return market_data
    
    async def collect_regional_insights(self, region: RegionType, keywords: List[str]) -> Dict[str, Any]:
        """지역별 특화 인사이트 수집"""
        regional_data = {}
        
        if region == RegionType.EAST_ASIA:
            asia_servers = ["asia_kakaotalk", "asia_line", "asia_baidu", "news_naver", "social_weibo"]
            
            for server in asia_servers:
                try:
                    request = MCPRequest(
                        server_name=server,
                        endpoint="trending",
                        params={'keywords': keywords, 'language': 'ko,ja,zh'}
                    )
                    
                    response = await self.mcp_manager.send_request(server, request)
                    
                    if response.status_code == 200:
                        regional_data[server] = response.data
                        
                except Exception as e:
                    logger.warning(f"Regional data collection failed for {server}: {e}")
        
        return regional_data


class MostHookingBusinessStrategyAgent:
    """메인 비즈니스 전략 에이전트"""
    
    def __init__(self):
        self.config = get_config()
        self.architecture = None
        self.orchestrator = None
        self.notion_integration = None
        self.mcp_manager = None
        self.enhanced_collector = None
        
        self.is_running = False
        self.last_analysis = None
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'insights_generated': 0,
            'strategies_created': 0,
            'notion_pages_created': 0
        }
    
    async def initialize(self) -> bool:
        """에이전트 초기화"""
        try:
            logger.info("🚀 Initializing Most Hooking Business Strategy Agent...")
            
            # 핵심 컴포넌트 초기화
            self.architecture = get_architecture()
            await self.architecture.initialize()
            
            self.orchestrator = await get_orchestrator()
            self.notion_integration = await get_notion_integration()
            self.mcp_manager = await get_mcp_manager()
            
            # 추가 MCP 서버 등록
            await AdditionalMCPServers.register_additional_servers(self.mcp_manager)
            
            # 확장된 데이터 수집기 초기화
            self.enhanced_collector = EnhancedDataCollector(self.mcp_manager)
            
            self.is_running = True
            logger.info("✅ Agent initialization completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            return False
    
    async def run_comprehensive_analysis(self, 
                                       keywords: Optional[List[str]] = None,
                                       regions: Optional[List[RegionType]] = None) -> Dict[str, Any]:
        """종합적인 비즈니스 분석 실행"""
        if not self.is_running:
            await self.initialize()
        
        # 기본값 설정
        if not keywords:
            keywords = self.config.monitoring_keywords
        if not regions:
            regions = [RegionType.EAST_ASIA, RegionType.NORTH_AMERICA]
        
        analysis_start = datetime.now(timezone.utc)
        logger.info(f"🔍 Starting comprehensive analysis for {len(keywords)} keywords across {len(regions)} regions")
        
        try:
            # 1. 기본 데이터 수집 및 분석
            basic_results = await self.orchestrator.run_full_analysis(keywords, regions)
            
            # 2. 확장된 데이터 수집
            extended_data = await self._collect_extended_data(keywords, regions)
            
            # 3. 시장 인텔리전스 강화
            enhanced_insights = await self._enhance_with_market_intelligence(
                basic_results.get('enhanced_insights', []),
                extended_data
            )
            
            # 4. 지역별 맞춤화
            regional_strategies = await self._create_regional_strategies(
                enhanced_insights, regions
            )
            
            # 5. Notion 문서화
            notion_results = await self._document_to_notion(
                enhanced_insights, regional_strategies
            )
            
            # 6. 성과 지표 업데이트
            self._update_performance_metrics(basic_results, regional_strategies, notion_results)
            
            analysis_duration = (datetime.now(timezone.utc) - analysis_start).total_seconds()
            
            comprehensive_results = {
                'analysis_id': f"analysis_{int(analysis_start.timestamp())}",
                'timestamp': analysis_start.isoformat(),
                'duration_seconds': analysis_duration,
                'keywords': keywords,
                'regions': [r.value for r in regions],
                'basic_results': basic_results,
                'extended_data': extended_data,
                'enhanced_insights_count': len(enhanced_insights),
                'regional_strategies_count': len(regional_strategies),
                'notion_results': notion_results,
                'performance_metrics': self.performance_metrics.copy(),
                'top_hooking_opportunities': [
                    {
                        'score': insight.hooking_score,
                        'topics': insight.key_topics,
                        'region': insight.region.value,
                        'opportunity_level': insight.business_opportunity.value
                    }
                    for insight in sorted(enhanced_insights, key=lambda x: x.hooking_score, reverse=True)[:5]
                ]
            }
            
            self.last_analysis = comprehensive_results
            
            logger.info(f"✅ Comprehensive analysis completed in {analysis_duration:.2f}s")
            logger.info(f"📊 Results: {len(enhanced_insights)} insights, {len(regional_strategies)} strategies")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis failed: {e}")
            return {'error': str(e), 'timestamp': analysis_start.isoformat()}
    
    async def _collect_extended_data(self, keywords: List[str], regions: List[RegionType]) -> Dict[str, Any]:
        """확장된 데이터 수집"""
        logger.info("📡 Collecting extended market data...")
        
        extended_data = {}
        
        # 소셜 미디어 트렌드
        extended_data['social_trends'] = await self.enhanced_collector.collect_social_media_trends(keywords)
        
        # 스타트업 생태계
        extended_data['startup_ecosystem'] = await self.enhanced_collector.collect_startup_ecosystem_data(regions)
        
        # 시장 인텔리전스
        extended_data['market_intelligence'] = await self.enhanced_collector.collect_market_intelligence(keywords)
        
        # 지역별 인사이트
        for region in regions:
            region_key = f"regional_{region.value}"
            extended_data[region_key] = await self.enhanced_collector.collect_regional_insights(region, keywords)
        
        logger.info(f"📊 Extended data collection completed: {len(extended_data)} data sources")
        return extended_data
    
    async def _enhance_with_market_intelligence(self, insights, extended_data):
        """시장 인텔리전스로 인사이트 강화"""
        logger.info("🧠 Enhancing insights with market intelligence...")
        
        # TODO: AI 기반 인사이트 강화 로직
        # 실제 구현에서는 extended_data를 분석하여 기존 insights를 보강
        
        return insights  # 임시로 원본 반환
    
    async def _create_regional_strategies(self, insights, regions):
        """지역별 맞춤 전략 생성"""
        logger.info("🌏 Creating regional strategies...")
        
        regional_strategies = []
        
        # 지역별로 인사이트 필터링하여 전략 생성
        for region in regions:
            region_insights = [i for i in insights if i.region == region or i.region == RegionType.GLOBAL]
            
            if region_insights:
                # Cross-cultural advisor 에이전트 활용
                strategy_agent = self.orchestrator.agents.get(AgentRole.STRATEGY_PLANNER)
                if strategy_agent:
                    region_strategies = await strategy_agent.process(region_insights)
                    regional_strategies.extend(region_strategies)
        
        logger.info(f"🎯 Created {len(regional_strategies)} regional strategies")
        return regional_strategies
    
    async def _document_to_notion(self, insights, strategies):
        """Notion에 결과 문서화"""
        logger.info("📝 Documenting results to Notion...")
        
        notion_results = {
            'daily_insights_page': None,
            'strategy_pages': [],
            'weekly_summary': None
        }
        
        try:
            # 일일 인사이트 페이지
            if insights:
                daily_page_id = await self.notion_integration.create_daily_insights_page(insights)
                notion_results['daily_insights_page'] = daily_page_id
            
            # 전략별 페이지
            for strategy in strategies:
                strategy_page_id = await self.notion_integration.create_strategy_page(strategy)
                if strategy_page_id:
                    notion_results['strategy_pages'].append(strategy_page_id)
            
            # 주간 요약 (일주일에 한 번)
            today = datetime.now(timezone.utc)
            if today.weekday() == 6:  # 일요일
                weekly_summary_id = await self.notion_integration.create_weekly_summary(insights, strategies)
                notion_results['weekly_summary'] = weekly_summary_id
            
            logger.info(f"📄 Notion documentation completed: {len(notion_results['strategy_pages'])} pages created")
            
        except Exception as e:
            logger.error(f"Notion documentation failed: {e}")
        
        return notion_results
    
    def _update_performance_metrics(self, basic_results, strategies, notion_results):
        """성과 지표 업데이트"""
        self.performance_metrics['total_analyses'] += 1
        
        if not basic_results.get('error'):
            self.performance_metrics['successful_analyses'] += 1
            self.performance_metrics['insights_generated'] += basic_results.get('insights_count', 0)
            self.performance_metrics['strategies_created'] += len(strategies)
            
            if notion_results.get('daily_insights_page'):
                self.performance_metrics['notion_pages_created'] += 1
            self.performance_metrics['notion_pages_created'] += len(notion_results.get('strategy_pages', []))
    
    async def run_scheduled_analysis(self):
        """스케줄된 분석 실행"""
        logger.info("⏰ Running scheduled analysis...")
        
        try:
            # 기본 키워드로 분석 실행
            results = await self.run_comprehensive_analysis()
            
            # 긴급 기회 알림
            if results and not results.get('error'):
                await self._check_critical_opportunities(results)
            
        except Exception as e:
            logger.error(f"Scheduled analysis failed: {e}")
    
    async def _check_critical_opportunities(self, results):
        """긴급 비즈니스 기회 체크 및 알림"""
        critical_opportunities = []
        
        for opportunity in results.get('top_hooking_opportunities', []):
            if opportunity['score'] >= 0.9 and opportunity['opportunity_level'] == 'CRITICAL':
                critical_opportunities.append(opportunity)
        
        if critical_opportunities:
            logger.warning(f"🚨 {len(critical_opportunities)} CRITICAL opportunities detected!")
            # TODO: Slack/이메일 알림 기능 추가
    
    def start_scheduler(self):
        """스케줄러 시작"""
        # 매 5분마다 분석 실행 (프로덕션에서는 더 긴 간격 권장)
        schedule.every(5).minutes.do(lambda: asyncio.create_task(self.run_scheduled_analysis()))
        
        # 매일 아침 9시에 종합 분석
        schedule.every().day.at("09:00").do(lambda: asyncio.create_task(self.run_comprehensive_analysis()))
        
        logger.info("📅 Scheduler started - analyses will run every 5 minutes and daily at 9 AM")
    
    def get_status(self) -> Dict[str, Any]:
        """에이전트 상태 반환"""
        return {
            'is_running': self.is_running,
            'last_analysis': self.last_analysis.get('timestamp') if self.last_analysis else None,
            'performance_metrics': self.performance_metrics,
            'orchestrator_status': self.orchestrator.get_agent_status() if self.orchestrator else {},
            'mcp_servers': len(self.mcp_manager.servers) if self.mcp_manager else 0,
            'additional_servers_count': len(AdditionalMCPServers.ADDITIONAL_SERVERS)
        }
    
    async def shutdown(self):
        """에이전트 종료"""
        logger.info("🔄 Shutting down Most Hooking Business Strategy Agent...")
        
        self.is_running = False
        
        if self.notion_integration:
            await self.notion_integration.shutdown()
        
        if self.mcp_manager:
            await self.mcp_manager.shutdown()
        
        logger.info("✅ Agent shutdown completed")


# 글로벌 에이전트 인스턴스
main_agent = MostHookingBusinessStrategyAgent()


async def get_main_agent() -> MostHookingBusinessStrategyAgent:
    """메인 에이전트 인스턴스 반환"""
    if not main_agent.is_running:
        await main_agent.initialize()
    return main_agent


# 편의 함수들
async def run_quick_analysis(keywords: List[str] = None) -> Dict[str, Any]:
    """빠른 분석 실행"""
    agent = await get_main_agent()
    return await agent.run_comprehensive_analysis(keywords)


async def get_agent_status() -> Dict[str, Any]:
    """에이전트 상태 조회"""
    agent = await get_main_agent()
    return agent.get_status()