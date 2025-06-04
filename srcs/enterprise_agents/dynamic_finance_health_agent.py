"""
Dynamic Personal Finance Health Agent - Enterprise Edition
엔터프라이즈급 동적 개인 금융 건강 에이전트

Clean, modular architecture with separated concerns:
- Models: Pydantic data structures
- Providers: Data source implementations  
- Agent: Core business logic

Features:
- Dynamic product discovery
- Real-time market data integration
- Multi-provider data aggregation
- Enterprise-grade scalability and reliability
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import our modular components
from srcs.enterprise_agents.models import (
    DynamicFinancialData,
    DynamicProduct, 
    MarketInsight,
    UserProfile,
    FinancialHealthResult,
    AssetCategory,
    RiskLevel
)

from srcs.enterprise_agents.providers import ProviderFactory
from srcs.enterprise_agents.models.providers import ProviderManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicFinanceHealthAgent:
    """Enterprise-grade dynamic personal finance health agent with modular architecture"""
    
    def __init__(self, provider_manager: Optional[ProviderManager] = None):
        self.agent_name = "Dynamic Finance Health Agent"
        self.version = "3.0.0"
        
        # Initialize provider manager
        self.provider_manager = provider_manager or ProviderFactory.create_default_manager()
        
        # Agent configuration
        self.premium_subscription_cost = 9900  # KRW per month
        self.supported_categories = ["stocks", "etf", "crypto", "real_estate", "savings"]
        
        logger.info(f"✅ {self.agent_name} v{self.version} initialized")
        logger.info(f"📊 Available providers: {self.provider_manager.get_available_providers()}")
    
    async def analyze_financial_health(self, user_profile_data: Dict[str, Any]) -> FinancialHealthResult:
        """Main entry point for financial health analysis"""
        try:
            # Validate and parse user profile
            user_profile = UserProfile(**user_profile_data)
            
            logger.info(f"🔍 Analyzing financial health for {user_profile.age}-year-old user")
            
            # Get comprehensive market data from all providers
            market_data = await self._get_comprehensive_market_data()
            
            # Search for suitable products across all categories
            suitable_products = await self._search_suitable_products(user_profile)
            
            # Get market insights from all sources
            market_insights = await self._get_market_insights()
            
            # Calculate dynamic health score
            health_score = await self._calculate_dynamic_health_score(
                user_profile, market_data, suitable_products
            )
            
            # Generate recommendations
            recommendations = await self._generate_dynamic_recommendations(
                user_profile, suitable_products, market_insights
            )
            
            # Create result
            result = FinancialHealthResult(
                health_score=health_score,
                grade=self._get_health_grade(health_score),
                analysis={
                    "user_profile": user_profile.model_dump(),
                    "market_data_sources": len(market_data),
                    "total_products_found": sum(len(products) for products in suitable_products.values()),
                    "insights_count": len(market_insights)
                },
                recommendations=recommendations,
                suitable_products=suitable_products,
                market_insights=market_insights
            )
            
            logger.info(f"✅ Analysis complete. Health score: {health_score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise
    
    async def _get_comprehensive_market_data(self) -> Dict[str, List[DynamicFinancialData]]:
        """Get market data from all available providers"""
        market_data = {}
        
        # Korean stocks
        korean_symbols = ["005930.KS", "000660.KS", "035420.KS", "051910.KS", "006400.KS"]
        
        # Get data from multiple providers
        for provider_name in self.provider_manager.get_available_providers():
            provider = self.provider_manager.get_provider(provider_name)
            if provider and "stocks" in provider.get_supported_categories():
                try:
                    data = await provider.get_market_data(korean_symbols)
                    if data:
                        market_data[provider_name] = data
                        logger.info(f"📈 Got {len(data)} market data points from {provider_name}")
                except Exception as e:
                    logger.warning(f"Failed to get market data from {provider_name}: {e}")
        
        return market_data
    
    async def _search_suitable_products(self, user_profile: UserProfile) -> Dict[str, List[DynamicProduct]]:
        """Search for suitable products across all providers and categories"""
        suitable_products = {}
        
        # Search criteria based on user profile
        risk_value = user_profile.risk_tolerance.value if hasattr(user_profile.risk_tolerance, 'value') else user_profile.risk_tolerance
        base_criteria = {
            "max_risk": risk_value,
            "min_investment": user_profile.monthly_surplus * 0.1,  # Max 10% of surplus per product
        }
        
        # Search each category
        for category in self.supported_categories:
            category_products = []
            
            # Get best provider for this category
            provider = await self.provider_manager.get_best_provider("KR", category)
            if provider:
                try:
                    # Category-specific criteria
                    criteria = base_criteria.copy()
                    if category == "crypto":
                        criteria["min_return"] = 15.0  # Higher return expectation for crypto
                    elif category == "savings":
                        criteria["min_rate"] = 2.0   # Minimum interest rate
                    
                    products = await provider.search_products(category, criteria)
                    category_products.extend(products)
                    logger.info(f"🔍 Found {len(products)} {category} products from {provider.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to search {category} products: {e}")
            
            # Filter products based on user preferences
            if user_profile.preferred_categories:
                preferred_categories = [cat.value if hasattr(cat, 'value') else cat for cat in user_profile.preferred_categories]
                if category not in preferred_categories:
                    category_products = category_products[:2]  # Limit non-preferred categories
            
            if category_products:
                suitable_products[category] = category_products
        
        return suitable_products
    
    async def _get_market_insights(self) -> List[MarketInsight]:
        """Aggregate market insights from all providers"""
        all_insights = []
        
        for provider_name in self.provider_manager.get_available_providers():
            provider = self.provider_manager.get_provider(provider_name)
            if provider:
                try:
                    insights = await provider.get_market_insights("KR")
                    all_insights.extend(insights)
                    logger.info(f"💡 Got {len(insights)} insights from {provider_name}")
                except Exception as e:
                    logger.warning(f"Failed to get insights from {provider_name}: {e}")
        
        # Sort by impact score and confidence
        all_insights.sort(key=lambda x: x.impact_score * x.confidence, reverse=True)
        return all_insights[:10]  # Top 10 insights
    
    async def _calculate_dynamic_health_score(
        self, 
        user_profile: UserProfile, 
        market_data: Dict[str, List[DynamicFinancialData]],
        suitable_products: Dict[str, List[DynamicProduct]]
    ) -> float:
        """Calculate dynamic health score based on current market conditions"""
        
        # Base score factors
        age_score = max(0, 100 - user_profile.age) * 0.3
        savings_rate_score = min(100, user_profile.savings_rate * 2) * 0.4
        diversification_score = len(suitable_products) * 10 * 0.3
        
        base_score = age_score + savings_rate_score + diversification_score
        
        # Market condition adjustments
        market_adjustment = 0
        total_providers = len(market_data)
        
        if total_providers > 0:
            # Positive market sentiment bonus
            positive_changes = 0
            total_data_points = 0
            
            for provider_data in market_data.values():
                for data_point in provider_data:
                    total_data_points += 1
                    if data_point.change_percent > 0:
                        positive_changes += 1
            
            if total_data_points > 0:
                market_sentiment = positive_changes / total_data_points
                market_adjustment = (market_sentiment - 0.5) * 20  # -10 to +10 adjustment
        
        # Product diversity bonus
        diversity_bonus = min(10, len(suitable_products) * 2)
        
        final_score = min(100, max(0, base_score + market_adjustment + diversity_bonus))
        
        logger.info(f"📊 Health score components: base={base_score:.1f}, market={market_adjustment:.1f}, diversity={diversity_bonus:.1f}")
        
        return final_score
    
    def _get_health_grade(self, score: float) -> str:
        """Convert health score to letter grade"""
        if score >= 90: return "A+"
        elif score >= 85: return "A"
        elif score >= 80: return "B+"
        elif score >= 75: return "B"
        elif score >= 70: return "C+"
        elif score >= 65: return "C"
        elif score >= 60: return "D+"
        elif score >= 55: return "D"
        else: return "F"
    
    async def _generate_dynamic_recommendations(
        self,
        user_profile: UserProfile,
        suitable_products: Dict[str, List[DynamicProduct]],
        market_insights: List[MarketInsight]
    ) -> List[str]:
        """Generate dynamic recommendations based on current market conditions"""
        
        recommendations = []
        
        # Savings rate recommendation
        if user_profile.savings_rate < 20:
            recommendations.append(f"💰 저축률 개선: 현재 {user_profile.savings_rate:.1f}%에서 20% 이상으로 늘려보세요")
        
        # Product-specific recommendations
        for category, products in suitable_products.items():
            if products:
                best_product = max(products, key=lambda p: p.expected_return * (6 - (p.risk_level.value if hasattr(p.risk_level, 'value') else p.risk_level)))
                risk_value = best_product.risk_level.value if hasattr(best_product.risk_level, 'value') else best_product.risk_level
                recommendations.append(
                    f"📈 {category.upper()} 추천: {best_product.name} "
                    f"(예상수익률: {best_product.expected_return:.1f}%, 위험도: {risk_value}/5)"
                )
        
        # Market insight-based recommendations
        high_impact_insights = [insight for insight in market_insights if insight.impact_score > 0.7]
        for insight in high_impact_insights[:2]:
            recommendations.append(f"🎯 시장 기회: {insight.title} - {insight.description[:50]}...")
        
        # Risk management
        high_risk_products = sum(1 for products in suitable_products.values() 
                               for product in products if (product.risk_level.value if hasattr(product.risk_level, 'value') else product.risk_level) >= 4)
        if high_risk_products > 3:
            recommendations.append("⚠️ 위험 관리: 고위험 상품 비중을 줄이고 안전 자산 비중을 늘려보세요")
        
        return recommendations
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all data providers"""
        health_results = await ProviderFactory.health_check_all()
        
        provider_status = {}
        for provider_name in self.provider_manager.get_available_providers():
            provider = self.provider_manager.get_provider(provider_name)
            if provider:
                provider_status[provider_name] = {
                    "enabled": provider.enabled,
                    "markets": provider.get_supported_markets(),
                    "categories": provider.get_supported_categories(),
                    "health": health_results.get(provider_name, False),
                    "cache_duration": provider.config.cache_duration
                }
        
        return provider_status
    
    async def search_dynamic_products(self, category: str, custom_criteria: Dict[str, Any]) -> List[DynamicProduct]:
        """Dynamic product search with custom criteria"""
        provider = await self.provider_manager.get_best_provider("KR", category)
        if not provider:
            logger.warning(f"No provider available for category: {category}")
            return []
        
        try:
            products = await provider.search_products(category, custom_criteria)
            logger.info(f"🔍 Dynamic search found {len(products)} {category} products")
            return products
        except Exception as e:
            logger.error(f"Dynamic product search failed: {e}")
            return []


async def main():
    """Demo the dynamic finance health agent"""
    agent = DynamicFinanceHealthAgent()
    
    # Show provider status
    logger.info("🔧 Provider Status Check:")
    status = await agent.get_provider_status()
    for provider, info in status.items():
        health_status = "✅ Healthy" if info["health"] else "❌ Unhealthy"
        logger.info(f"  {provider}: {health_status} | Markets: {info['markets']} | Categories: {info['categories']}")
    
    # Sample user profile
    sample_user = {
        "age": 32,
        "risk_tolerance": 3,
        "monthly_income": 4500000,
        "monthly_expense": 2800000,
        "current_assets": 150000000,
        "investment_experience": 3,
        "preferred_categories": ["stocks", "crypto", "savings"],
        "goals": ["장기투자", "안정적수익", "포트폴리오다양화"]
    }
    
    # Analyze financial health
    logger.info("🎯 Starting Financial Health Analysis...")
    result = await agent.analyze_financial_health(sample_user)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"🏆 Financial Health Analysis Results")
    print(f"{'='*60}")
    print(f"📊 Health Score: {result.health_score:.1f}/100 (Grade: {result.grade})")
    print(f"🔍 Analysis Summary:")
    print(f"  - Data Sources: {result.analysis['market_data_sources']}")
    print(f"  - Products Found: {result.analysis['total_products_found']}")
    print(f"  - Market Insights: {result.analysis['insights_count']}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n📈 Suitable Products by Category:")
    for category, products in result.suitable_products.items():
        print(f"  {category.upper()}: {len(products)} products available")
        for product in products[:2]:  # Show top 2 per category
            risk_value = product.risk_level.value if hasattr(product.risk_level, 'value') else product.risk_level
            print(f"    - {product.name} (Return: {product.expected_return:.1f}%, Risk: {risk_value}/5)")
    
    print(f"\n🎯 Top Market Insights:")
    for insight in result.market_insights[:3]:
        print(f"  📌 {insight.title}")
        print(f"     Impact: {insight.impact_score:.1f} | Confidence: {insight.confidence:.1f}")
        print(f"     {insight.description[:80]}...")
    
    print(f"\n{'='*60}")
    logger.info("✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 