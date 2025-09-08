"""
Practical Prediction Algorithm for Ethereum Trading

This module implements practical prediction methods:
1. Technical analysis-based predictions
2. Statistical trend analysis
3. Market sentiment indicators
4. Simple moving average strategies
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import logging
import math
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions"""
    PRICE = "price"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    TREND = "trend"

class AnalysisMethod(Enum):
    """Analysis method types"""
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    SENTIMENT = "sentiment"
    MOMENTUM = "momentum"

@dataclass
class PredictionConfig:
    """Prediction configuration"""
    lookback_period: int = 20  # 20 periods for analysis
    prediction_horizon: int = 5  # 5 periods ahead
    confidence_threshold: float = 0.7
    technical_indicators: bool = True
    sentiment_analysis: bool = True
    momentum_analysis: bool = True

class PredictionResult(TypedDict):
    """Prediction result structure"""
    prediction: float
    confidence: float
    method: str
    timestamp: str
    indicators_used: List[str]
    prediction_horizon: int

class PracticalPredictionAlgorithm:
    """Practical prediction algorithm for Ethereum trading"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.prediction_history: List[PredictionResult] = []
        self.performance_metrics: Dict[str, float] = {}
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        
    async def prepare_technical_indicators(
        self, 
        market_data: Dict[str, any], 
        historical_data: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Prepare technical indicators for analysis"""
        try:
            if not historical_data:
                return {}
            
            # Extract price and volume data
            prices = [d.get("price", 0) for d in historical_data]
            volumes = [d.get("volume", 0) for d in historical_data]
            
            if len(prices) < self.config.lookback_period:
                return {}
            
            # Calculate technical indicators
            indicators = {}
            
            # Moving averages
            indicators["sma_5"] = self._calculate_sma(prices, 5)
            indicators["sma_20"] = self._calculate_sma(prices, 20)
            indicators["ema_12"] = self._calculate_ema(prices, 12)
            indicators["ema_26"] = self._calculate_ema(prices, 26)
            
            # RSI
            indicators["rsi"] = self._calculate_rsi(prices)
            
            # MACD
            macd, macd_signal = self._calculate_macd(prices)
            indicators["macd"] = macd
            indicators["macd_signal"] = macd_signal
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
            indicators["bb_upper"] = bb_upper
            indicators["bb_middle"] = bb_middle
            indicators["bb_lower"] = bb_lower
            
            # Volume indicators
            if volumes:
                indicators["volume_sma"] = self._calculate_sma(volumes, 10)
                indicators["volume_ratio"] = volumes[-1] / indicators["volume_sma"] if indicators["volume_sma"] > 0 else 1.0
            
            # Price momentum
            indicators["price_momentum"] = self._calculate_momentum(prices)
            
            # Volatility
            indicators["volatility"] = self._calculate_volatility(prices)
            
            return indicators
                
        except Exception as e:
            logger.error(f"Technical indicators preparation failed: {e}")
            raise ValueError(f"Technical analysis failed: {str(e)}")
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return prices[-1] if prices else 0.0
            return sum(prices[-period:]) / period
        except Exception as e:
            logger.error(f"SMA calculation failed: {e}")
            return 0.0
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return prices[-1] if prices else 0.0
            
            multiplier = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
            return 0.0
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate MACD indicator"""
        try:
            if len(prices) < 26:
                return 0.0, 0.0
            
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            macd = ema_12 - ema_26
            
            # Calculate MACD signal (simplified)
            macd_signal = macd * 0.9  # Simplified signal calculation
            
            return macd, macd_signal
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if prices else 0.0
                return current_price, current_price, current_price
            
            sma = self._calculate_sma(prices, period)
            variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
            std_dev = math.sqrt(variance)
            
            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)
            
            return upper_band, sma, lower_band
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            current_price = prices[-1] if prices else 0.0
            return current_price, current_price, current_price
    
    def _calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """Calculate price momentum"""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            current_price = prices[-1]
            past_price = prices[-period-1]
            
            return ((current_price - past_price) / past_price) * 100
        except Exception as e:
            logger.error(f"Momentum calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility(self, prices: List[float], period: int = 20) -> float:
        """Calculate price volatility"""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            returns = []
            for i in range(1, min(len(prices), period + 1)):
                if prices[i-1] != 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if not returns:
                return 0.0
            
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            
            return math.sqrt(variance) * 100  # Return as percentage
        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            return 0.0
    
    async def predict_price(
        self, 
        market_data: Dict[str, any], 
        historical_data: List[Dict[str, any]]
    ) -> PredictionResult:
        """Predict future price using technical analysis"""
        try:
            # Get technical indicators
            indicators = await self.prepare_technical_indicators(market_data, historical_data)
            
            if not indicators:
                raise ValueError("Insufficient data for prediction")
            
            current_price = market_data.get("price", 0)
            
            # Technical analysis prediction
            prediction, confidence = self._technical_analysis_prediction(current_price, indicators)
            
            # Create prediction result
            result: PredictionResult = {
                "prediction": prediction,
                "confidence": confidence,
                "method": "technical_analysis",
                "timestamp": datetime.now().isoformat(),
                "indicators_used": list(indicators.keys()),
                "prediction_horizon": self.config.prediction_horizon
            }
            
            # Store prediction history
            self.prediction_history.append(result)
            
            # Update performance metrics
            await self._update_performance_metrics(confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Price prediction failed: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _technical_analysis_prediction(
        self, 
        current_price: float, 
        indicators: Dict[str, any]
    ) -> Tuple[float, float]:
        """Predict price using technical analysis"""
        try:
            # Get key indicators
            sma_5 = indicators.get("sma_5", current_price)
            sma_20 = indicators.get("sma_20", current_price)
            rsi = indicators.get("rsi", 50)
            macd = indicators.get("macd", 0)
            macd_signal = indicators.get("macd_signal", 0)
            bb_upper = indicators.get("bb_upper", current_price)
            bb_lower = indicators.get("bb_lower", current_price)
            momentum = indicators.get("price_momentum", 0)
            volatility = indicators.get("volatility", 0)
            
            # Calculate prediction based on multiple signals
            signals = []
            confidence_factors = []
            
            # Moving average signals
            if sma_5 > sma_20:
                signals.append(0.02)  # 2% upward bias
                confidence_factors.append(0.3)
            else:
                signals.append(-0.02)  # 2% downward bias
                confidence_factors.append(0.3)
            
            # RSI signals
            if rsi < 30:  # Oversold
                signals.append(0.03)  # 3% upward bias
                confidence_factors.append(0.4)
            elif rsi > 70:  # Overbought
                signals.append(-0.03)  # 3% downward bias
                confidence_factors.append(0.4)
            else:
                signals.append(0.0)
                confidence_factors.append(0.2)
            
            # MACD signals
            if macd > macd_signal:
                signals.append(0.015)  # 1.5% upward bias
                confidence_factors.append(0.3)
            else:
                signals.append(-0.015)  # 1.5% downward bias
                confidence_factors.append(0.3)
            
            # Bollinger Bands signals
            if current_price < bb_lower:
                signals.append(0.025)  # 2.5% upward bias (bounce from lower band)
                confidence_factors.append(0.5)
            elif current_price > bb_upper:
                signals.append(-0.025)  # 2.5% downward bias (rejection from upper band)
                confidence_factors.append(0.5)
            else:
                signals.append(0.0)
                confidence_factors.append(0.1)
            
            # Momentum signals
            if momentum > 5:  # Strong upward momentum
                signals.append(0.02)
                confidence_factors.append(0.4)
            elif momentum < -5:  # Strong downward momentum
                signals.append(-0.02)
                confidence_factors.append(0.4)
            else:
                signals.append(0.0)
                confidence_factors.append(0.1)
            
            # Calculate weighted prediction
            total_signal = sum(signals)
            avg_confidence = sum(confidence_factors) / len(confidence_factors)
            
            # Apply volatility adjustment
            volatility_factor = 1 + (volatility / 100)  # Higher volatility = larger moves
            prediction_change = total_signal * volatility_factor
            
            # Calculate final prediction
            prediction = current_price * (1 + prediction_change)
            
            # Calculate confidence (0.0 to 1.0)
            confidence = min(0.95, max(0.1, avg_confidence))
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Technical analysis prediction failed: {e}")
            return current_price, 0.5
    
    async def _update_performance_metrics(self, confidence: float):
        """Update performance metrics"""
        try:
            if "total_predictions" not in self.performance_metrics:
                self.performance_metrics["total_predictions"] = 0
                self.performance_metrics["total_confidence"] = 0.0
            
            self.performance_metrics["total_predictions"] += 1
            self.performance_metrics["total_confidence"] += confidence
            
            # Calculate average confidence
            self.performance_metrics["average_confidence"] = (
                self.performance_metrics["total_confidence"] / 
                self.performance_metrics["total_predictions"]
            )
                
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        try:
            return self.performance_metrics.copy()
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    async def analyze_sentiment(
        self, 
        news_data: List[Dict[str, any]], 
        social_data: List[Dict[str, any]]
    ) -> Dict[str, float]:
        """Analyze market sentiment from news and social media"""
        try:
            sentiment_scores = []
            
            # Analyze news sentiment (simplified)
            for news in news_data:
                title = news.get("title", "").lower()
                content = news.get("content", "").lower()
                
                # Simple keyword-based sentiment
                positive_words = ["bullish", "surge", "rally", "gain", "up", "rise", "positive"]
                negative_words = ["bearish", "crash", "fall", "drop", "down", "negative", "decline"]
                
                positive_count = sum(1 for word in positive_words if word in title or word in content)
                negative_count = sum(1 for word in negative_words if word in title or word in content)
                
                if positive_count + negative_count > 0:
                    sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                    sentiment_scores.append(sentiment)
            
            # Analyze social sentiment (simplified)
            for social in social_data:
                text = social.get("text", "").lower()
                
                # Simple emoji and keyword analysis
                positive_emojis = ["🚀", "📈", "💎", "🔥", "✅"]
                negative_emojis = ["📉", "💀", "❌", "⚠️", "🔻"]
                
                positive_count = sum(1 for emoji in positive_emojis if emoji in text)
                negative_count = sum(1 for emoji in negative_emojis if emoji in text)
                
                if positive_count + negative_count > 0:
                    sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                    sentiment_scores.append(sentiment)
            
            if not sentiment_scores:
                return {"sentiment": 0.0, "confidence": 0.0}
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = min(1.0, len(sentiment_scores) / 10)  # More data = higher confidence
            
            return {
                "sentiment": avg_sentiment,
                "confidence": confidence,
                "sources_analyzed": len(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": 0.0, "confidence": 0.0}
