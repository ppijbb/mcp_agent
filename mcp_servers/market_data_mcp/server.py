#!/usr/bin/env python3
"""
Market Data Collection MCP Server
Provides market data collection capabilities for trading decisions
"""

import asyncio
import aiohttp
import json
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataMCP:
    def __init__(self):
        self.api_keys = {
            'coinbase': os.getenv('COINBASE_API_KEY'),
            'binance': os.getenv('BINANCE_API_KEY'),
            'coingecko': os.getenv('COINGECKO_API_KEY')
        }
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_ethereum_price(self) -> Dict[str, Any]:
        """Get current Ethereum price from multiple sources"""
        try:
            prices = {}
            
            # CoinGecko API (free, no key required)
            async with self.session.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd,eur,btc') as response:
                if response.status == 200:
                    data = await response.json()
                    prices['coingecko'] = {
                        'usd': data['ethereum']['usd'],
                        'eur': data['ethereum']['eur'],
                        'btc': data['ethereum']['btc']
                    }
            
            # Binance API
            if self.api_keys['binance']:
                async with self.session.get('https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT') as response:
                    if response.status == 200:
                        data = await response.json()
                        prices['binance'] = {'usd': float(data['price'])}
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "prices": prices,
                "primary_price_usd": prices.get('coingecko', {}).get('usd', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get Ethereum price: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_market_trends(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Get market trends and indicators"""
        try:
            # Get price change data
            async with self.session.get(f'https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=1') as response:
                if response.status == 200:
                    data = await response.json()
                    
                    prices = data['prices']
                    volumes = data['total_volumes']
                    
                    # Calculate trends
                    if len(prices) >= 2:
                        current_price = prices[-1][1]
                        previous_price = prices[0][1]
                        price_change = ((current_price - previous_price) / previous_price) * 100
                        
                        current_volume = volumes[-1][1]
                        previous_volume = volumes[0][1]
                        volume_change = ((current_volume - previous_volume) / previous_volume) * 100
                        
                        return {
                            "status": "success",
                            "timestamp": datetime.now().isoformat(),
                            "current_price_usd": current_price,
                            "price_change_24h_percent": round(price_change, 2),
                            "volume_change_24h_percent": round(volume_change, 2),
                            "trend": "bullish" if price_change > 0 else "bearish",
                            "volume_trend": "increasing" if volume_change > 0 else "decreasing"
                        }
            
            return {"status": "error", "message": "Failed to fetch market data"}
            
        except Exception as e:
            logger.error(f"Failed to get market trends: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_technical_indicators(self) -> Dict[str, Any]:
        """Get technical analysis indicators"""
        try:
            # Get historical data for technical analysis
            async with self.session.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=30') as response:
                if response.status == 200:
                    data = await response.json()
                    prices = [price[1] for price in data['prices']]
                    
                    if len(prices) >= 30:
                        # Simple moving averages
                        sma_7 = sum(prices[-7:]) / 7
                        sma_30 = sum(prices[-30:]) / 30
                        
                        # RSI calculation (simplified)
                        gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
                        losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
                        
                        avg_gain = sum(gains[-14:]) / 14
                        avg_loss = sum(losses[-14:]) / 14
                        
                        if avg_loss != 0:
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                        else:
                            rsi = 100
                        
                        current_price = prices[-1]
                        
                        return {
                            "status": "success",
                            "timestamp": datetime.now().isoformat(),
                            "current_price": current_price,
                            "sma_7": round(sma_7, 2),
                            "sma_30": round(sma_30, 2),
                            "rsi": round(rsi, 2),
                            "price_vs_sma7": round(((current_price - sma_7) / sma_7) * 100, 2),
                            "price_vs_sma30": round(((current_price - sma_30) / sma_30) * 100, 2),
                            "signal": self._generate_signal(current_price, sma_7, sma_30, rsi)
                        }
            
            return {"status": "error", "message": "Failed to fetch technical data"}
            
        except Exception as e:
            logger.error(f"Failed to get technical indicators: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generate_signal(self, current_price: float, sma_7: float, sma_30: float, rsi: float) -> str:
        """Generate trading signal based on technical indicators"""
        signals = []
        
        # Price vs SMA signals
        if current_price > sma_7:
            signals.append("price_above_sma7")
        if current_price > sma_30:
            signals.append("price_above_sma30")
        
        # RSI signals
        if rsi > 70:
            signals.append("rsi_overbought")
        elif rsi < 30:
            signals.append("rsi_oversold")
        
        # Overall signal
        if len(signals) >= 2 and "price_above_sma7" in signals and "price_above_sma30" in signals:
            return "strong_buy"
        elif "rsi_oversold" in signals:
            return "buy"
        elif "rsi_overbought" in signals:
            return "sell"
        else:
            return "hold"
    
    async def search_market_news(self, query: str = "ethereum") -> Dict[str, Any]:
        """Search for market news and sentiment"""
        try:
            # This would typically use a news API, but for demo purposes we'll return a placeholder
            # In production, you'd integrate with services like NewsAPI, CryptoPanic, etc.
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "news_count": 0,
                "message": "News search functionality requires additional API integration",
                "sentiment": "neutral"
            }
            
        except Exception as e:
            logger.error(f"Failed to search market news: {e}")
            return {"status": "error", "message": str(e)}

async def main():
    async with MarketDataMCP() as market_mcp:
        # Test functionality
        price = await market_mcp.get_ethereum_price()
        print(f"Ethereum Price: {price}")
        
        trends = await market_mcp.get_market_trends()
        print(f"Market Trends: {trends}")
        
        indicators = await market_mcp.get_technical_indicators()
        print(f"Technical Indicators: {indicators}")

if __name__ == "__main__":
    asyncio.run(main())
