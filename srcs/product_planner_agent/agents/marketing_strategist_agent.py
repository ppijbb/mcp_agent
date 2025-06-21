"""
Marketing Strategist Agent
마케팅 전략 수립, Go-to-Market 계획 및 사용자 획득 전략을 관리하는 Agent
"""

from mcp_agent.agents.agent import Agent
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams


class MarketingStrategistAgent:
    """마케팅 전략 및 사용자 획득 전문 Agent"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.agent_instance = self.create_agent()

    async def develop_marketing_strategy(self, prd_content: Dict[str, Any], business_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        PRD와 비즈니스 계획을 바탕으로 Go-to-Market(GTM) 전략을 수립합니다.
        """
        if not self.llm:
            return {
                "target_audience": "10-20대 소셜 미디어 사용자",
                "channels": ["Instagram", "TikTok", "YouTube"],
                "initial_campaign": "인플루언서 협업을 통한 바이럴 마케팅",
                "status": "created_mockup"
            }

        prompt = f"""
        You are a senior marketing strategist. Based on the provided PRD and business plan, develop a Go-to-Market (GTM) strategy.

        **PRD Content:**
        {json.dumps(prd_content, indent=2, ensure_ascii=False)}

        **Business Plan:**
        {json.dumps(business_plan, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  **Target Audience:** Define the primary target audience and key personas.
        2.  **Positioning:** Create a compelling product positioning statement.
        3.  **Channel Strategy:** Recommend the most effective marketing channels (e.g., social media, content marketing, SEO).
        4.  **Launch Campaign:** Outline a creative concept for the initial launch campaign.

        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.6, response_format="json"))
            marketing_strategy = json.loads(result_str)
            marketing_strategy["status"] = "created_successfully"
            return marketing_strategy
        except Exception as e:
            print(f"Error developing marketing strategy: {e}")
            return {
                "error": str(e),
                "status": "creation_failed"
            }

    @staticmethod
    def create_agent() -> Agent:
        """
        마케팅 전략 Agent 생성
        
        Returns:
            Agent: 설정된 마케팅 전략 Agent
        """
        
        instruction = """
        You are a senior marketing strategist with expertise in product marketing and growth strategy. Develop comprehensive marketing strategies that drive user acquisition, engagement, and retention.

        **PRIMARY OBJECTIVES**:
        - Create comprehensive go-to-market strategy
        - Develop user acquisition and retention plans
        - Design brand positioning and messaging framework
        - Plan marketing channels and campaign strategies
        - Establish growth metrics and optimization processes

        **MARKETING STRATEGY FRAMEWORK**:
        1. **Market Analysis**:
           - Target market segmentation and sizing
           - Competitive landscape analysis
           - User persona development and validation
           - Market positioning and differentiation
           - Value proposition definition

        2. **Go-to-Market Strategy**:
           - Launch strategy and timeline
           - Channel strategy and partnership plans
           - Pricing strategy and monetization model
           - Sales enablement and support materials
           - Launch success metrics and milestones

        3. **Brand & Messaging**:
           - Brand identity and voice definition
           - Core messaging and value proposition
           - Content strategy and editorial calendar
           - Creative guidelines and asset requirements
           - Brand consistency across touchpoints

        4. **User Acquisition Strategy**:
           - Marketing channel mix and prioritization
           - Paid advertising strategy (SEM, social, display)
           - Content marketing and SEO strategy
           - Referral and viral growth mechanisms
           - Partnership and affiliate programs

        5. **Retention & Growth**:
           - Onboarding experience optimization
           - Email marketing and lifecycle campaigns
           - Product-led growth initiatives
           - Community building and engagement
           - Customer success and advocacy programs

        **MARKETING CHANNELS**:
        - Digital marketing (SEM, SEO, social media)
        - Content marketing and thought leadership
        - Email marketing and marketing automation
        - Partnership and influencer marketing
        - Event marketing and PR initiatives
        - Product-led growth and viral mechanics

        **CAMPAIGN PLANNING**:
        - Campaign strategy and creative concepts
        - Budget allocation and media planning
        - Performance tracking and optimization
        - A/B testing and experimentation framework
        - ROI measurement and attribution modeling

        **GROWTH FRAMEWORKS**:
        - AARRR metrics (Acquisition, Activation, Retention, Referral, Revenue)
        - Growth hacking and experimentation
        - Customer journey optimization
        - Conversion funnel analysis and optimization
        - Cohort analysis and lifetime value optimization

        **DELIVERABLES**:
        - Comprehensive marketing strategy document
        - Go-to-market plan with timeline and milestones
        - Brand guidelines and messaging framework
        - Channel strategy and budget allocation
        - Campaign plans and creative briefs
        - Performance measurement and optimization framework

        **OUTPUT FORMAT**:
        Create comprehensive marketing strategy including:
        - Executive marketing strategy summary
        - Target market and competitive analysis
        - Brand positioning and messaging framework
        - Detailed go-to-market plan
        - Channel strategy and campaign roadmap
        - Budget allocation and resource requirements
        - Success metrics and optimization processes

        Focus on creating data-driven, scalable marketing strategies that align with business objectives and drive sustainable growth."""
        
        return Agent(
            name="marketing_strategist",
            instruction=instruction,
            server_names=["filesystem"]
        )
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "📈 마케팅 전략 수립, Go-to-Market 계획 및 사용자 획득 전략을 관리하는 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "종합적인 Go-to-Market 전략 수립",
            "사용자 획득 및 유지 계획",
            "브랜드 포지셔닝 및 메시징 프레임워크",
            "마케팅 채널 및 캠페인 전략",
            "성장 지표 및 최적화 프로세스",
            "경쟁 분석 및 시장 포지셔닝"
        ]
    
    @staticmethod
    def get_strategy_components() -> dict[str, list[str]]:
        """마케팅 전략 구성 요소 반환"""
        return {
            "market_analysis": [
                "타겟 시장 세분화 및 규모 산정",
                "경쟁 환경 분석",
                "사용자 페르소나 개발",
                "시장 포지셔닝",
                "가치 제안 정의"
            ],
            "go_to_market": [
                "런칭 전략 및 일정",
                "채널 전략 및 파트너십",
                "가격 전략 및 수익화 모델",
                "영업 지원 자료",
                "런칭 성공 지표"
            ],
            "brand_messaging": [
                "브랜드 아이덴티티 및 보이스",
                "핵심 메시징",
                "콘텐츠 전략",
                "크리에이티브 가이드라인",
                "브랜드 일관성"
            ],
            "acquisition_strategy": [
                "마케팅 채널 믹스",
                "유료 광고 전략",
                "콘텐츠 마케팅 및 SEO",
                "추천 및 바이럴 성장",
                "파트너십 프로그램"
            ]
        }
    
    @staticmethod
    def get_marketing_channels() -> list[str]:
        """마케팅 채널 목록 반환"""
        return [
            "디지털 마케팅 (SEM, SEO, 소셜미디어)",
            "콘텐츠 마케팅 및 소트 리더십",
            "이메일 마케팅 및 마케팅 자동화",
            "파트너십 및 인플루언서 마케팅",
            "이벤트 마케팅 및 PR",
            "제품 주도 성장 및 바이럴 메커니즘"
        ]
    
    @staticmethod
    def get_growth_frameworks() -> list[str]:
        """성장 프레임워크 목록 반환"""
        return [
            "AARRR 지표 (획득, 활성화, 유지, 추천, 수익)",
            "그로스 해킹 및 실험",
            "고객 여정 최적화",
            "전환 퍼널 분석 및 최적화",
            "코호트 분석 및 생애 가치 최적화"
        ] 