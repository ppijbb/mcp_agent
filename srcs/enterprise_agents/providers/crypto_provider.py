"""
Cryptocurrency Provider
암호화폐 데이터 제공자

Specialized provider for cryptocurrency markets and products
"""

import logging
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from srcs.enterprise_agents.models.providers import DataProvider, ProviderConfig
from srcs.enterprise_agents.models.financial_data import (
    DynamicFinancialData, 
    DynamicProduct, 
    MarketInsight,
    AssetCategory,
    RiskLevel,
    Currency
)

logger = logging.getLogger(__name__)

class CryptoProvider(DataProvider):
    """Cryptocurrency markets specialized data provider"""
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                name="Crypto Markets",
                enabled=True,
                priority=3,
                supported_markets=["KR", "Global"],
                supported_categories=["crypto"],
                timeout=15,
                cache_duration=180  # 3분 캐시 (crypto 가격 변동이 빠름)
            )
        super().__init__(config)
        
    async def get_market_data(self, symbols: List[str]) -> List[DynamicFinancialData]:
        """Get cryptocurrency market data"""
        cache_key = f"crypto_market_data:{':'.join(symbols)}"
        cached_data = self.get_cache(cache_key)
        if cached_data:
            return cached_data
            
        market_data = []
        
        # Simulate Korean crypto exchange API calls
        # Upbit, Bithumb, Coinone API integration
        crypto_prices = {
            "BTC": {"name": "비트코인", "price": 52000000, "change": 2.8, "volume": 1500000000},
            "ETH": {"name": "이더리움", "price": 3200000, "change": -1.5, "volume": 800000000},
            "XRP": {"name": "리플", "price": 650, "change": 5.2, "volume": 450000000},
            "ADA": {"name": "카르다노", "price": 520, "change": 3.1, "volume": 200000000},
            "DOGE": {"name": "도지코인", "price": 105, "change": -2.3, "volume": 300000000},
        }
        
        for symbol in symbols:
            if symbol in crypto_prices:
                data = crypto_prices[symbol]
                market_data.append(DynamicFinancialData(
                    symbol=symbol,
                    name=data["name"],
                    price=data["price"],
                    change_percent=data["change"],
                    volume=data["volume"],
                    currency=Currency.KRW,
                    country="Global",
                    data_source="Korean Crypto Exchanges"
                ))
        
        self.set_cache(cache_key, market_data)
        return market_data
    
    async def search_products(self, category: str, criteria: Dict[str, Any]) -> List[DynamicProduct]:
        """Search cryptocurrency investment products"""
        cache_key = f"crypto_products:{category}:{hash(str(sorted(criteria.items())))}"
        cached_data = self.get_cache(cache_key)
        if cached_data:
            return cached_data
            
        products = []
        
        if category == "crypto":
            # 암호화폐 투자상품 
            crypto_products = [
                ("BTC", "비트코인", "메이저코인", 25.5, 5, 5000),
                ("ETH", "이더리움", "메이저코인", 28.2, 5, 5000),
                ("XRP", "리플", "알트코인", 35.8, 5, 1000),
                ("ADA", "카르다노", "알트코인", 42.1, 5, 1000),
                ("MATIC", "폴리곤", "알트코인", 55.3, 5, 1000),
                ("AVAX", "아발란체", "알트코인", 48.7, 5, 1000),
            ]
            
            max_risk = criteria.get('max_risk', 5)
            min_return = criteria.get('min_return', 0)
            
            for symbol, name, category_type, expected_return, risk, min_invest in crypto_products:
                if risk <= max_risk and expected_return >= min_return:
                    products.append(DynamicProduct(
                        product_id=symbol,
                        name=name,
                        category=AssetCategory.CRYPTO,
                        subcategory=category_type,
                        provider="Korean Crypto Exchanges",
                        expected_return=expected_return,
                        risk_level=RiskLevel(risk),
                        min_investment=min_invest,
                        features=["24시간거래", "높은변동성", "디지털자산"],
                        rating=3.8,
                        data_source="Upbit/Bithumb API"
                    ))
        
        self.set_cache(cache_key, products)
        return products
    
    async def get_market_insights(self, market: str = "KR") -> List[MarketInsight]:
        """Get cryptocurrency market insights"""
        cache_key = f"crypto_insights:{market}"
        cached_data = self.get_cache(cache_key)
        if cached_data:
            return cached_data
            
        insights = [
            MarketInsight(
                insight_type="regulatory_update",
                title="한국 비트코인 ETF 출시 임박",
                description="2025년 비트코인, 이더리움 ETF 출시로 기관투자 확대 예상. 암호화폐 시장 성숙도 증가",
                impact_score=0.8,
                confidence=0.7,
                related_symbols=["BTC", "ETH"],
                source="Korean Financial Regulators"
            ),
            MarketInsight(
                insight_type="market_trend",
                title="알트코인 시장 회복 신호",
                description="메이저 코인 대비 알트코인 상대적 저평가. 선별적 투자 기회 존재",
                impact_score=0.6,
                confidence=0.6,
                related_symbols=["XRP", "ADA", "MATIC"],
                source="Crypto Market Analysis"
            ),
            MarketInsight(
                insight_type="institutional_adoption",
                title="한국 기관투자자 암호화폐 진입 가속화",
                description="연기금, 보험사 등 기관투자자 암호화폐 포트폴리오 편입 검토 중",
                impact_score=0.9,
                confidence=0.8,
                related_symbols=["BTC", "ETH"],
                source="Institutional Research"
            )
        ]
        
        self.set_cache(cache_key, insights)
        return insights 