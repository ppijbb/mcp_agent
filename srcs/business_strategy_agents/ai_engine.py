"""
AI Engine for Most Hooking Business Strategy Agent

This module implements specialized AI agents with distinct roles for
business intelligence analysis and strategy generation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import openai
from enum import Enum
import time
import random

from .config import get_config
from .architecture import (
    RawContent, ProcessedInsight, BusinessStrategy, 
    ContentType, RegionType, BusinessOpportunityLevel
)
from .mcp_layer import get_mcp_manager, MCPRequest

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """에이전트 역할 정의"""
    DATA_SCOUT = "data_scout"              # 데이터 수집 및 1차 필터링
    TREND_ANALYZER = "trend_analyzer"      # 트렌드 분석 및 패턴 인식
    HOOKING_DETECTOR = "hooking_detector"  # 후킹 포인트 탐지
    MARKET_RESEARCHER = "market_researcher" # 시장 조사 및 경쟁 분석
    STRATEGY_PLANNER = "strategy_planner"  # 전략 수립 및 실행 계획
    CROSS_CULTURAL_ADVISOR = "cross_cultural_advisor" # 지역별 문화 적응
    ROI_CALCULATOR = "roi_calculator"      # ROI 예측 및 리스크 분석


@dataclass
class AgentCapability:
    """에이전트 능력 정의"""
    name: str
    description: str
    mcp_servers: List[str]  # 사용할 MCP 서버들
    specialized_prompts: Dict[str, str]
    output_format: str


@dataclass
class AgentResponse:
    """에이전트 응답 데이터"""
    agent_role: AgentRole
    success: bool
    result: Any
    execution_time: float
    token_usage: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None


class BaseAgent(ABC):
    """기본 에이전트 클래스 - OpenAI 가이드의 Agent 패턴 적용"""
    
    def __init__(self, role: AgentRole, config: Dict[str, Any] = None):
        self.role = role
        self.config = config or {}
        self.mcp_manager = None
        self.logger = logging.getLogger(f"{__name__}.{role.value}")
        self.conversation_history = []
        
        # OpenAI 클라이언트 초기화
        self.openai_client = None
        self._initialize_openai()
        
        # 에이전트별 설정
        self.max_retries = 3
        self.timeout = 30
        self.temperature = 0.7
        
    def _initialize_openai(self):
        """OpenAI 클라이언트 초기화"""
        try:
            config = get_config()
            api_key = config.apis.get('openai_api_key', 'your_openai_api_key_here')
            
            if api_key and api_key != 'your_openai_api_key_here':
                self.openai_client = openai.AsyncOpenAI(api_key=api_key)
                self.logger.info(f"OpenAI client initialized for {self.role.value}")
            else:
                self.logger.warning(f"OpenAI API key not configured for {self.role.value}")
                # 테스트용 mock 클라이언트
                self.openai_client = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
        
    async def initialize(self):
        """에이전트 초기화"""
        self.mcp_manager = await get_mcp_manager()
        self.logger.info(f"Agent {self.role.value} initialized")
    
    @abstractmethod
    async def process(self, input_data: Any) -> AgentResponse:
        """데이터 처리 (각 에이전트별 구현)"""
        pass
    
    async def call_llm(self, prompt: str, context: Dict[str, Any] = None, 
                      model: str = "gpt-4", max_tokens: int = 2000) -> str:
        """LLM 호출 - OpenAI 가이드의 패턴 적용"""
        start_time = time.time()
        
        try:
            system_prompt = self._get_system_prompt()
            user_prompt = self._format_prompt(prompt, context)
            
            # 실제 OpenAI 호출 또는 Mock 응답
            if self.openai_client:
                response = await self._call_openai_api(system_prompt, user_prompt, model, max_tokens)
            else:
                response = await self._mock_llm_response(system_prompt, user_prompt)
            
            execution_time = time.time() - start_time
            self.logger.info(f"LLM call completed in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"LLM call failed after {execution_time:.2f}s: {e}")
            return self._get_fallback_response()
    
    async def _call_openai_api(self, system_prompt: str, user_prompt: str, 
                              model: str, max_tokens: int) -> str:
        """실제 OpenAI API 호출"""
        for attempt in range(self.max_retries):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout
                )
                
                return response.choices[0].message.content
                
            except openai.RateLimitError:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                self.logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                
            except openai.APITimeoutError:
                self.logger.warning(f"API timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise
                    
            except Exception as e:
                self.logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        raise Exception("Max retries exceeded")
    
    async def _mock_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """Mock LLM 응답 (테스트용)"""
        await asyncio.sleep(0.5)  # API 지연 시뮬레이션
        
        # 에이전트 역할별 Mock 응답
        mock_responses = {
            AgentRole.DATA_SCOUT: self._mock_data_scout_response(),
            AgentRole.TREND_ANALYZER: self._mock_trend_analyzer_response(),
            AgentRole.HOOKING_DETECTOR: self._mock_hooking_detector_response(),
            AgentRole.STRATEGY_PLANNER: self._mock_strategy_planner_response()
        }
        
        return mock_responses.get(self.role, "Mock response generated successfully.")
    
    def _mock_data_scout_response(self) -> str:
        return json.dumps([
            {"index": 0, "score": 8, "reason": "High business potential in AI sector"},
            {"index": 1, "score": 7, "reason": "Strong market indicators for fintech"},
            {"index": 2, "score": 6, "reason": "Emerging sustainability trends"}
        ])
    
    def _mock_trend_analyzer_response(self) -> str:
        return json.dumps({
            "key_topics": ["AI automation", "fintech innovation", "sustainability"],
            "sentiment_score": 0.75,
            "trend_direction": "rising",
            "market_size_estimate": "$50B+ market opportunity",
            "actionable_insights": [
                "AI automation showing 300% growth in enterprise adoption",
                "Fintech regulations creating new market opportunities"
            ]
        })
    
    def _mock_hooking_detector_response(self) -> str:
        return json.dumps({
            "hooking_score": 0.85,
            "opportunity_level": "HIGH",
            "hooking_elements": ["viral potential", "timing advantage", "market gap"],
            "audience_appeal": "Strong appeal to tech-savvy entrepreneurs",
            "timing_factor": "Perfect timing with current market trends"
        })
    
    def _mock_strategy_planner_response(self) -> str:
        return json.dumps({
            "title": "AI-Powered Fintech Innovation Strategy",
            "description": "Leverage AI automation trends to create next-gen fintech solutions",
            "action_items": [
                {"task": "Market research and competitor analysis", "timeline": "1-2 weeks", "resources": "Research team"},
                {"task": "MVP development", "timeline": "4-6 weeks", "resources": "Development team + $50K budget"}
            ],
            "roi_prediction": {
                "expected_revenue": "$500K-1M in Year 1",
                "investment_required": "$200K initial investment",
                "roi_percentage": "250-400%",
                "payback_period": "8-12 months"
            },
            "risk_factors": ["Market competition", "Regulatory changes", "Technology adoption rate"],
            "success_metrics": ["User acquisition rate", "Revenue growth", "Market share"],
            "target_market": "Tech startups and SMBs seeking AI automation"
        })
    
    def _get_fallback_response(self) -> str:
        """LLM 호출 실패 시 폴백 응답"""
        fallback_responses = {
            AgentRole.DATA_SCOUT: "Data collection completed with basic filtering applied.",
            AgentRole.TREND_ANALYZER: json.dumps({
                "key_topics": ["market_trend"],
                "sentiment_score": 0.5,
                "trend_direction": "stable",
                "actionable_insights": ["Further analysis required"]
            }),
            AgentRole.HOOKING_DETECTOR: json.dumps({
                "hooking_score": 0.5,
                "opportunity_level": "MEDIUM",
                "hooking_elements": ["standard_opportunity"]
            }),
            AgentRole.STRATEGY_PLANNER: json.dumps({
                "title": "Standard Business Strategy",
                "description": "Basic strategy framework applied",
                "action_items": [{"task": "Manual review required", "timeline": "TBD", "resources": "TBD"}]
            })
        }
        
        return fallback_responses.get(self.role, "Fallback response generated.")
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 반환 - OpenAI 가이드의 Instructions 패턴"""
        base_prompt = f"""You are a specialized AI agent with the role of {self.role.value.replace('_', ' ').title()}.
Your primary responsibility is to analyze business data and provide insights for identifying hooking business opportunities.

Always respond in a structured, professional manner and focus on actionable insights.
Consider both East Asian and North American market contexts when applicable.

Key Guidelines:
1. Be precise and data-driven in your analysis
2. Focus on actionable business insights
3. Consider market timing and competitive landscape
4. Provide quantifiable recommendations when possible
5. Flag high-potential opportunities clearly"""
        
        role_specific_prompts = {
            AgentRole.DATA_SCOUT: """
You excel at data collection, content filtering, and initial quality assessment.
Focus on identifying high-value information sources and preliminary content scoring.
Rate content on business relevance (0-10 scale) and filter out low-quality data.""",
            
            AgentRole.TREND_ANALYZER: """
You specialize in pattern recognition, trend analysis, and market movement prediction.
Identify emerging trends, growth patterns, and market shifts with timing predictions.
Provide sentiment analysis and trend direction (rising/stable/falling).""",
            
            AgentRole.HOOKING_DETECTOR: """
You are expert at spotting business opportunities with high engagement potential.
Rate opportunities on a 0.0-1.0 scale and identify specific hooking elements.
Focus on viral potential, timing advantages, and audience appeal.""",
            
            AgentRole.STRATEGY_PLANNER: """
You create detailed business strategies with clear action items and implementation timelines.
Focus on practical, executable plans with resource requirements and success metrics.
Include ROI predictions and risk assessments.""",
        }
        
        return base_prompt + "\n\n" + role_specific_prompts.get(self.role, "")
    
    def _format_prompt(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """프롬프트 포맷팅"""
        if context:
            context_str = json.dumps(context, indent=2, ensure_ascii=False)
            return f"{prompt}\n\nContext:\n{context_str}"
        return prompt
    
    async def collaborate_with(self, other_agent: 'BaseAgent', message: str) -> str:
        """다른 에이전트와 협업 - Multi-agent 패턴"""
        self.logger.info(f"Collaborating with {other_agent.role.value}")
        
        collaboration_prompt = f"""
I am collaborating with a {other_agent.role.value.replace('_', ' ')} agent.
They sent me this message: {message}

Please provide a response that adds value from my {self.role.value.replace('_', ' ')} perspective.
Focus on how my expertise can enhance their analysis.
"""
        
        return await self.call_llm(collaboration_prompt)


class DataScoutAgent(BaseAgent):
    """데이터 수집 및 1차 필터링 에이전트"""
    
    def __init__(self):
        super().__init__(AgentRole.DATA_SCOUT)
        
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """데이터 수집 및 필터링"""
        start_time = time.time()
        
        try:
            keywords = input_data.get('keywords', [])
            regions = input_data.get('regions', [])
            
            self.logger.info(f"Scouting data for keywords: {keywords}")
            
            # MCP 서버들로부터 데이터 수집
            all_content = await self._collect_from_mcp_servers(keywords, regions)
            
            # AI 기반 품질 필터링
            filtered_content = await self._filter_content_quality(all_content)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_role=self.role,
                success=True,
                result=filtered_content,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Data scout processing failed: {e}")
            
            return AgentResponse(
                agent_role=self.role,
                success=False,
                result=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _collect_from_mcp_servers(self, keywords: List[str], regions: List[RegionType]) -> List[RawContent]:
        """MCP 서버에서 데이터 수집"""
        all_content = []
        
        # 뉴스 수집
        news_servers = ['news_reuters', 'news_bloomberg', 'news_naver']
        for server in news_servers:
            try:
                if self.mcp_manager:
                    request = MCPRequest(
                        server_name=server,
                        endpoint="search",
                        params={'q': ' OR '.join(keywords), 'limit': 50}
                    )
                    response = await self.mcp_manager.send_request(server, request)
                    
                    if response and response.status_code == 200:
                        for item in response.data.get('articles', []):
                            content = RawContent(
                                source=server,
                                content_type=ContentType.NEWS,
                                region=RegionType.GLOBAL,
                                title=item.get('title', ''),
                                content=item.get('description', ''),
                                url=item.get('url', ''),
                                timestamp=datetime.now(timezone.utc),
                                author=item.get('author'),
                                engagement_metrics={}
                            )
                            all_content.append(content)
                
                # Mock 데이터 생성 (실제 MCP 서버 없을 때)
                else:
                    mock_content = self._generate_mock_content(server, keywords)
                    all_content.extend(mock_content)
                        
            except Exception as e:
                self.logger.warning(f"Failed to collect from {server}: {e}")
        
        return all_content
    
    def _generate_mock_content(self, server: str, keywords: List[str]) -> List[RawContent]:
        """Mock 컨텐츠 생성"""
        mock_articles = []
        
        for i, keyword in enumerate(keywords[:3]):  # 최대 3개
            content = RawContent(
                source=server,
                content_type=ContentType.NEWS,
                region=RegionType.GLOBAL,
                title=f"Breaking: {keyword} Market Shows Strong Growth",
                content=f"Latest analysis shows {keyword} sector experiencing unprecedented growth...",
                url=f"https://{server}.com/article/{keyword}-{i}",
                timestamp=datetime.now(timezone.utc),
                author="Market Analyst",
                engagement_metrics={"views": 1000 + i * 100, "shares": 50 + i * 10}
            )
            mock_articles.append(content)
        
        return mock_articles
    
    async def _filter_content_quality(self, content_list: List[RawContent]) -> List[RawContent]:
        """AI 기반 컨텐츠 품질 필터링"""
        if not content_list:
            return []
        
        filtered = []
        batch_size = 10
        
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i+batch_size]
            
            prompt = f"""
Rate the business relevance and quality of these {len(batch)} content items on a scale of 0-10.
Only return items with score >= 6.

Content to evaluate:
{json.dumps([{'title': c.title, 'content': c.content[:200]} for c in batch], ensure_ascii=False)}

Return as JSON array with format: [{{"index": 0, "score": 8, "reason": "High business potential"}}, ...]
"""
            
            try:
                response = await self.call_llm(prompt)
                evaluations = json.loads(response)
                
                for eval_item in evaluations:
                    if eval_item.get('score', 0) >= 6:
                        idx = eval_item['index']
                        if idx < len(batch):
                            filtered.append(batch[idx])
                            
            except Exception as e:
                self.logger.warning(f"Quality filtering failed for batch: {e}")
                # 실패 시 원본 반환
                filtered.extend(batch)
        
        return filtered


class TrendAnalyzerAgent(BaseAgent):
    """트렌드 분석 에이전트"""
    
    def __init__(self):
        super().__init__(AgentRole.TREND_ANALYZER)
    
    async def process(self, input_data: Any) -> AgentResponse:
        """트렌드 분석 및 인사이트 생성"""
        start_time = time.time()
        
        try:
            content_list = input_data if isinstance(input_data, list) else []
            
            if not content_list:
                return AgentResponse(
                    agent_role=self.role,
                    success=True,
                    result=[],
                    execution_time=time.time() - start_time
                )
            
            self.logger.info(f"Analyzing trends from {len(content_list)} content items")
            
            insights = []
            
            # Google Trends 데이터 수집
            trend_data = await self._collect_trend_data()
            
            # 컨텐츠별 트렌드 분석
            for content in content_list:
                try:
                    insight = await self._analyze_single_content(content, trend_data)
                    if insight:
                        insights.append(insight)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze content {content.title}: {e}")
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_role=self.role,
                success=True,
                result=insights,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Trend analysis failed: {e}")
            
            return AgentResponse(
                agent_role=self.role,
                success=False,
                result=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _collect_trend_data(self) -> Dict[str, Any]:
        """Google Trends 데이터 수집"""
        try:
            if self.mcp_manager:
                request = MCPRequest(
                    server_name="trends_google_trends",
                    endpoint="trending",
                    params={'geo': 'US,KR,JP', 'timeframe': 'now 7-d'}
                )
                
                response = await self.mcp_manager.send_request("trends_google_trends", request)
                
                if response and response.status_code == 200:
                    return response.data
                else:
                    self.logger.warning("Failed to collect trend data")
                    return self._mock_trend_data()
            else:
                return self._mock_trend_data()
                
        except Exception as e:
            self.logger.error(f"Trend data collection failed: {e}")
            return self._mock_trend_data()
    
    def _mock_trend_data(self) -> Dict[str, Any]:
        """Mock 트렌드 데이터"""
        return {
            "trending_topics": ["AI", "fintech", "sustainability", "Web3"],
            "rising_searches": ["ChatGPT business", "green finance", "blockchain adoption"],
            "regional_trends": {
                "US": ["AI regulation", "fintech IPO"],
                "KR": ["AI startup", "digital transformation"],
                "JP": ["robot automation", "green tech"]
            }
        }
    
    async def _analyze_single_content(self, content: RawContent, trend_data: Dict) -> Optional[ProcessedInsight]:
        """단일 컨텐츠 트렌드 분석"""
        prompt = f"""
Analyze this content for business trends and opportunities:

Title: {content.title}
Content: {content.content}
Source: {content.source}
Region: {content.region.value}

Current trend data context:
{json.dumps(trend_data, indent=2)}

Provide analysis in this JSON format:
{{
    "key_topics": ["topic1", "topic2", "topic3"],
    "sentiment_score": 0.0-1.0,
    "trend_direction": "rising|stable|falling",
    "market_size_estimate": "description",
    "actionable_insights": ["insight1", "insight2"]
}}
"""
        
        try:
            response = await self.call_llm(prompt)
            analysis = json.loads(response)
            
            return ProcessedInsight(
                content_id=f"{content.source}_{hash(content.title)}",
                hooking_score=0.5,  # Will be refined by HookingDetectorAgent
                business_opportunity=BusinessOpportunityLevel.MEDIUM,
                region=content.region,
                category="trend_analysis",
                key_topics=analysis.get('key_topics', []),
                sentiment_score=analysis.get('sentiment_score', 0.0),
                trend_direction=analysis.get('trend_direction', 'stable'),
                market_size_estimate=analysis.get('market_size_estimate'),
                competitive_landscape=[],
                actionable_insights=analysis.get('actionable_insights', []),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return None


class HookingDetectorAgent(BaseAgent):
    """후킹 포인트 탐지 에이전트"""
    
    def __init__(self):
        super().__init__(AgentRole.HOOKING_DETECTOR)
    
    async def process(self, input_data: Any) -> AgentResponse:
        """후킹 포인트 탐지 및 점수 계산"""
        start_time = time.time()
        
        try:
            insights = input_data if isinstance(input_data, list) else []
            
            self.logger.info(f"Detecting hooking points from {len(insights)} insights")
            
            enhanced_insights = []
            
            for insight in insights:
                try:
                    enhanced = await self._calculate_hooking_score(insight)
                    enhanced_insights.append(enhanced)
                except Exception as e:
                    self.logger.warning(f"Hooking detection failed: {e}")
                    enhanced_insights.append(insight)  # 원본 유지
            
            # 후킹 점수 기준으로 정렬
            enhanced_insights.sort(key=lambda x: x.hooking_score, reverse=True)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_role=self.role,
                success=True,
                result=enhanced_insights,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Hooking detection failed: {e}")
            
            return AgentResponse(
                agent_role=self.role,
                success=False,
                result=input_data if isinstance(input_data, list) else [],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _calculate_hooking_score(self, insight: ProcessedInsight) -> ProcessedInsight:
        """후킹 점수 계산"""
        prompt = f"""
Calculate the "hooking score" (0.0-1.0) for this business insight.
Hooking score represents how likely this insight is to capture audience attention and generate business value.

Consider:
- Market timing and opportunity
- Audience engagement potential  
- Competitive advantage possibilities
- Implementation feasibility
- Trend momentum

Insight:
- Topics: {insight.key_topics}
- Sentiment: {insight.sentiment_score}
- Trend: {insight.trend_direction}
- Region: {insight.region.value}
- Current insights: {insight.actionable_insights}

Return JSON:
{{
    "hooking_score": 0.0-1.0,
    "opportunity_level": "CRITICAL|HIGH|MEDIUM|LOW",
    "hooking_elements": ["element1", "element2"],
    "audience_appeal": "description",
    "timing_factor": "description"
}}
"""
        
        try:
            response = await self.call_llm(prompt)
            analysis = json.loads(response)
            
            # 기존 insight 업데이트
            insight.hooking_score = analysis.get('hooking_score', 0.5)
            
            # opportunity_level 매핑
            level_mapping = {
                'CRITICAL': BusinessOpportunityLevel.CRITICAL,
                'HIGH': BusinessOpportunityLevel.HIGH,
                'MEDIUM': BusinessOpportunityLevel.MEDIUM,
                'LOW': BusinessOpportunityLevel.LOW
            }
            insight.business_opportunity = level_mapping.get(
                analysis.get('opportunity_level', 'MEDIUM'),
                BusinessOpportunityLevel.MEDIUM
            )
            
            # 추가 인사이트 확장
            hooking_elements = analysis.get('hooking_elements', [])
            insight.actionable_insights.extend([
                f"Hooking Element: {element}" for element in hooking_elements
            ])
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Hooking score calculation failed: {e}")
            return insight


class StrategyPlannerAgent(BaseAgent):
    """전략 기획 에이전트"""
    
    def __init__(self):
        super().__init__(AgentRole.STRATEGY_PLANNER)
    
    async def process(self, input_data: Any) -> AgentResponse:
        """비즈니스 전략 생성"""
        start_time = time.time()
        
        try:
            top_insights = input_data if isinstance(input_data, list) else []
            
            self.logger.info(f"Creating strategies from {len(top_insights)} insights")
            
            strategies = []
            
            # 후킹 점수 상위 인사이트들로 전략 생성
            high_value_insights = [i for i in top_insights if i.hooking_score >= 0.7]
            
            for insight in high_value_insights:
                try:
                    strategy = await self._create_strategy(insight)
                    if strategy:
                        strategies.append(strategy)
                except Exception as e:
                    self.logger.warning(f"Strategy creation failed: {e}")
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_role=self.role,
                success=True,
                result=strategies,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Strategy planning failed: {e}")
            
            return AgentResponse(
                agent_role=self.role,
                success=False,
                result=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _create_strategy(self, insight: ProcessedInsight) -> Optional[BusinessStrategy]:
        """단일 인사이트로부터 전략 생성"""
        prompt = f"""
Create a comprehensive business strategy based on this high-value insight:

Insight Details:
- Hooking Score: {insight.hooking_score}
- Opportunity Level: {insight.business_opportunity.value}
- Key Topics: {insight.key_topics}
- Region: {insight.region.value}
- Actionable Insights: {insight.actionable_insights}
- Trend Direction: {insight.trend_direction}

Create a detailed strategy in JSON format:
{{
    "title": "Clear strategy title",
    "description": "Detailed strategy description",
    "action_items": [
        {{"task": "task description", "timeline": "1-2 weeks", "resources": "required resources"}},
        {{"task": "task description", "timeline": "3-4 weeks", "resources": "required resources"}}
    ],
    "roi_prediction": {{
        "expected_revenue": "revenue estimate",
        "investment_required": "investment estimate", 
        "roi_percentage": "percentage estimate",
        "payback_period": "time estimate"
    }},
    "risk_factors": ["risk1", "risk2", "risk3"],
    "success_metrics": ["metric1", "metric2", "metric3"],
    "target_market": "market description"
}}
"""
        
        try:
            response = await self.call_llm(prompt)
            strategy_data = json.loads(response)
            
            return BusinessStrategy(
                strategy_id=f"strategy_{insight.content_id}_{int(datetime.now().timestamp())}",
                title=strategy_data.get('title', 'Untitled Strategy'),
                opportunity_level=insight.business_opportunity,
                region=insight.region,
                category=insight.category,
                description=strategy_data.get('description', ''),
                key_insights=insight.actionable_insights,
                action_items=strategy_data.get('action_items', []),
                timeline="4-8 weeks",  # 기본값
                resource_requirements={"budget": "TBD", "team": "TBD"},
                roi_prediction=strategy_data.get('roi_prediction', {}),
                risk_factors=strategy_data.get('risk_factors', []),
                success_metrics=strategy_data.get('success_metrics', []),
                related_trends=insight.key_topics,
                created_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Strategy creation failed: {e}")
            return None


class AgentOrchestrator:
    """에이전트 오케스트레이터 - OpenAI 가이드의 Orchestration 패턴 적용"""
    
    def __init__(self):
        self.agents = {}
        self.workflow_state = {}
        self.execution_stats = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_execution_time': 0.0
        }
        
    async def initialize(self):
        """모든 에이전트 초기화"""
        agent_classes = [
            DataScoutAgent,
            TrendAnalyzerAgent, 
            HookingDetectorAgent,
            StrategyPlannerAgent
        ]
        
        for agent_class in agent_classes:
            agent = agent_class()
            await agent.initialize()
            self.agents[agent.role] = agent
            
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def run_full_analysis(self, keywords: List[str], regions: List[RegionType]) -> Dict[str, Any]:
        """전체 분석 워크플로우 실행 - Sequential 패턴"""
        workflow_id = f"workflow_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"Starting workflow {workflow_id}: business strategy analysis")
        
        try:
            self.execution_stats['total_workflows'] += 1
            
            # 1. 데이터 수집
            scout_agent = self.agents[AgentRole.DATA_SCOUT]
            scout_input = {'keywords': keywords, 'regions': regions}
            scout_response = await scout_agent.process(scout_input)
            
            if not scout_response.success:
                raise Exception(f"Data collection failed: {scout_response.error_message}")
                
            raw_content = scout_response.result
            logger.info(f"Collected {len(raw_content)} content items")
            
            # 2. 트렌드 분석
            trend_agent = self.agents[AgentRole.TREND_ANALYZER]
            trend_response = await trend_agent.process(raw_content)
            
            if not trend_response.success:
                raise Exception(f"Trend analysis failed: {trend_response.error_message}")
                
            insights = trend_response.result
            logger.info(f"Generated {len(insights)} insights")
            
            # 3. 후킹 포인트 탐지
            hooking_agent = self.agents[AgentRole.HOOKING_DETECTOR]
            hooking_response = await hooking_agent.process(insights)
            
            if not hooking_response.success:
                raise Exception(f"Hooking detection failed: {hooking_response.error_message}")
                
            enhanced_insights = hooking_response.result
            logger.info(f"Enhanced {len(enhanced_insights)} insights with hooking scores")
            
            # 4. 전략 생성
            strategy_agent = self.agents[AgentRole.STRATEGY_PLANNER]
            strategy_response = await strategy_agent.process(enhanced_insights)
            
            if not strategy_response.success:
                logger.warning(f"Strategy planning failed: {strategy_response.error_message}")
                strategies = []
            else:
                strategies = strategy_response.result
                logger.info(f"Created {len(strategies)} business strategies")
            
            # 실행 통계 업데이트
            execution_time = time.time() - start_time
            self.execution_stats['successful_workflows'] += 1
            self._update_execution_stats(execution_time)
            
            return {
                'workflow_id': workflow_id,
                'success': True,
                'raw_content_count': len(raw_content),
                'insights_count': len(insights),
                'enhanced_insights': enhanced_insights,
                'strategies': strategies,
                'top_hooking_scores': [i.hooking_score for i in enhanced_insights[:5]],
                'execution_time': execution_time,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'agent_performance': {
                    'data_scout': {'time': scout_response.execution_time, 'success': scout_response.success},
                    'trend_analyzer': {'time': trend_response.execution_time, 'success': trend_response.success},
                    'hooking_detector': {'time': hooking_response.execution_time, 'success': hooking_response.success},
                    'strategy_planner': {'time': strategy_response.execution_time, 'success': strategy_response.success}
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_stats['failed_workflows'] += 1
            self._update_execution_stats(execution_time)
            
            logger.error(f"Workflow {workflow_id} failed after {execution_time:.2f}s: {e}")
            return {
                'workflow_id': workflow_id,
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _update_execution_stats(self, execution_time: float):
        """실행 통계 업데이트"""
        total = self.execution_stats['total_workflows']
        current_avg = self.execution_stats['average_execution_time']
        
        # 이동 평균 계산
        self.execution_stats['average_execution_time'] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    def get_agent_status(self) -> Dict[str, str]:
        """에이전트 상태 반환"""
        return {
            role.value: "active" if role in self.agents else "inactive"
            for role in AgentRole
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        success_rate = 0.0
        if self.execution_stats['total_workflows'] > 0:
            success_rate = (
                self.execution_stats['successful_workflows'] / 
                self.execution_stats['total_workflows']
            ) * 100
        
        return {
            **self.execution_stats,
            'success_rate_percentage': success_rate
        }


# 글로벌 오케스트레이터 인스턴스
orchestrator = AgentOrchestrator()


async def get_orchestrator() -> AgentOrchestrator:
    """오케스트레이터 인스턴스 반환"""
    if not orchestrator.agents:
        await orchestrator.initialize()
    return orchestrator