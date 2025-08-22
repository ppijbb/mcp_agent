"""
Ethereum Analysis Chain

This chain handles data analysis and market research:
1. Technical Analysis
2. Fundamental Analysis
3. Sentiment Analysis
4. Pattern Recognition
"""

from typing import Dict, Any, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
import json

class AnalysisChain:
    """Chain for comprehensive market and data analysis"""
    
    def __init__(self, llm):
        self.llm = llm
        self._setup_analysis_chains()
    
    def _setup_analysis_chains(self):
        """Setup various analysis chains"""
        
        # Technical Analysis Chain
        technical_analysis_prompt = PromptTemplate(
            input_variables=["price_data", "volume_data", "indicators"],
            template="""
            Perform technical analysis on the provided data:
            
            Price Data: {price_data}
            Volume Data: {volume_data}
            Technical Indicators: {indicators}
            
            Analyze:
            1. Price trends and patterns
            2. Support and resistance levels
            3. Moving averages and crossovers
            4. RSI, MACD, Bollinger Bands
            5. Volume-price relationships
            6. Chart patterns (head & shoulders, triangles, etc.)
            
            Technical Analysis Results:
            """
        )
        
        self.technical_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=technical_analysis_prompt,
            output_parser=StrOutputParser()
        )
        
        # Fundamental Analysis Chain
        fundamental_analysis_prompt = PromptTemplate(
            input_variables=["token_metrics", "network_stats", "ecosystem_data"],
            template="""
            Perform fundamental analysis on the token and ecosystem:
            
            Token Metrics: {token_metrics}
            Network Statistics: {network_stats}
            Ecosystem Data: {ecosystem_data}
            
            Analyze:
            1. Token utility and use cases
            2. Network adoption and growth
            3. Developer activity and partnerships
            4. Tokenomics and supply dynamics
            5. Regulatory environment
            6. Competitive landscape
            
            Fundamental Analysis Results:
            """
        )
        
        self.fundamental_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=fundamental_analysis_prompt,
            output_parser=StrOutputParser()
        )
        
        # Sentiment Analysis Chain
        sentiment_analysis_prompt = PromptTemplate(
            input_variables=["social_media", "news_data", "community_sentiment"],
            template="""
            Analyze market sentiment from various sources:
            
            Social Media: {social_media}
            News Data: {news_data}
            Community Sentiment: {community_sentiment}
            
            Analyze:
            1. Overall market sentiment (bullish/bearish/neutral)
            2. Social media sentiment trends
            3. News impact and sentiment
            4. Community engagement and sentiment
            5. Influencer opinions and impact
            6. Sentiment changes over time
            
            Sentiment Analysis Results:
            """
        )
        
        self.sentiment_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=sentiment_analysis_prompt,
            output_parser=StrOutputParser()
        )
        
        # Pattern Recognition Chain
        pattern_recognition_prompt = PromptTemplate(
            input_variables=["historical_patterns", "current_market", "timeframe"],
            template="""
            Identify and analyze market patterns:
            
            Historical Patterns: {historical_patterns}
            Current Market: {current_market}
            Timeframe: {timeframe}
            
            Identify:
            1. Recurring price patterns
            2. Seasonal trends and cycles
            3. Market structure patterns
            4. Volume patterns
            5. Correlation patterns
            6. Anomaly detection
            
            Pattern Recognition Results:
            """
        )
        
        self.pattern_recognition_chain = LLMChain(
            llm=self.llm,
            prompt=pattern_recognition_prompt,
            output_parser=StrOutputParser()
        )
    
    async def execute_comprehensive_analysis(
        self,
        market_data: Dict[str, Any],
        analysis_type: str = "all"
    ) -> Dict[str, Any]:
        """Execute comprehensive market analysis"""
        
        try:
            results = {}
            
            if analysis_type in ["all", "technical"]:
                results["technical_analysis"] = await self.technical_analysis_chain.arun({
                    "price_data": str(market_data.get("price_data", {})),
                    "volume_data": str(market_data.get("volume_data", {})),
                    "indicators": str(market_data.get("indicators", {}))
                })
            
            if analysis_type in ["all", "fundamental"]:
                results["fundamental_analysis"] = await self.fundamental_analysis_chain.arun({
                    "token_metrics": str(market_data.get("token_metrics", {})),
                    "network_stats": str(market_data.get("network_stats", {})),
                    "ecosystem_data": str(market_data.get("ecosystem_data", {}))
                })
            
            if analysis_type in ["all", "sentiment"]:
                results["sentiment_analysis"] = await self.sentiment_analysis_chain.arun({
                    "social_media": str(market_data.get("social_media", {})),
                    "news_data": str(market_data.get("news_data", {})),
                    "community_sentiment": str(market_data.get("community_sentiment", {}))
                })
            
            if analysis_type in ["all", "patterns"]:
                results["pattern_recognition"] = await self.pattern_recognition_chain.arun({
                    "historical_patterns": str(market_data.get("historical_patterns", {})),
                    "current_market": str(market_data.get("current_market", {})),
                    "timeframe": market_data.get("timeframe", "1d")
                })
            
            # Generate summary analysis
            results["summary"] = await self._generate_summary(results)
            results["timestamp"] = self._get_timestamp()
            results["status"] = "completed"
            
            return results
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": self._get_timestamp()
            }
    
    async def _generate_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a summary of all analysis results"""
        
        summary_prompt = PromptTemplate(
            input_variables=["analysis_results"],
            template="""
            Provide a concise summary of the market analysis:
            
            {analysis_results}
            
            Summary should include:
            1. Key findings from each analysis type
            2. Overall market outlook
            3. Risk factors
            4. Opportunities
            5. Recommendations
            
            Summary:
            """
        )
        
        summary_chain = LLMChain(
            llm=self.llm,
            prompt=summary_prompt,
            output_parser=StrOutputParser()
        )
        
        return await summary_chain.arun({
            "analysis_results": str(analysis_results)
        })
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about analysis capabilities"""
        return {
            "chain_type": "AnalysisChain",
            "analysis_types": [
                "Technical Analysis",
                "Fundamental Analysis",
                "Sentiment Analysis",
                "Pattern Recognition"
            ],
            "output_format": "comprehensive_market_analysis",
            "customizable": True
        }
