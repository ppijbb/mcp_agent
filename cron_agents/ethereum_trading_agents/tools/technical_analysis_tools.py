"""
Advanced Technical Analysis Tools
Comprehensive technical analysis for Ethereum trading
"""

import logging
from typing import Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
# import talib  # Optional dependency

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"


@dataclass
class TechnicalAnalysisConfig:
    """Configuration for technical analysis"""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    stoch_k: int = 14
    stoch_d: int = 3
    williams_r: int = 14
    atr_period: int = 14
    adx_period: int = 14


class AdvancedTechnicalAnalyzer:
    """Advanced technical analysis with comprehensive indicators"""

    def __init__(self, config: TechnicalAnalysisConfig = None):
        self.config = config or TechnicalAnalysisConfig()

    def analyze_comprehensive(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {required_columns}")

            # Convert to numpy arrays for talib
            open_prices = df['open'].values.astype(float)
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)
            volumes = df['volume'].values.astype(float)

            analysis = {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "indicators": {},
                "signals": {},
                "patterns": {},
                "trend_analysis": {},
                "volatility_analysis": {},
                "volume_analysis": {},
                "momentum_analysis": {},
                "support_resistance": {},
                "risk_metrics": {}
            }

            # Calculate all indicators
            analysis["indicators"] = self._calculate_all_indicators(
                open_prices, high_prices, low_prices, close_prices, volumes
            )

            # Generate trading signals
            analysis["signals"] = self._generate_trading_signals(analysis["indicators"])

            # Identify chart patterns
            analysis["patterns"] = self._identify_chart_patterns(df)

            # Analyze trends
            analysis["trend_analysis"] = self._analyze_trends(df, analysis["indicators"])

            # Analyze volatility
            analysis["volatility_analysis"] = self._analyze_volatility(df, analysis["indicators"])

            # Analyze volume
            analysis["volume_analysis"] = self._analyze_volume(df, analysis["indicators"])

            # Analyze momentum
            analysis["momentum_analysis"] = self._analyze_momentum(df, analysis["indicators"])

            # Find support and resistance
            analysis["support_resistance"] = self._find_support_resistance(df)

            # Calculate risk metrics
            analysis["risk_metrics"] = self._calculate_risk_metrics(df, analysis["indicators"])

            return analysis

        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    def _calculate_all_indicators(self, open_prices: np.ndarray, high_prices: np.ndarray,
                                low_prices: np.ndarray, close_prices: np.ndarray,
                                volumes: np.ndarray) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}

        try:
            # Trend Indicators
            indicators["sma_7"] = talib.SMA(close_prices, timeperiod=7)
            indicators["sma_14"] = talib.SMA(close_prices, timeperiod=14)
            indicators["sma_21"] = talib.SMA(close_prices, timeperiod=21)
            indicators["sma_50"] = talib.SMA(close_prices, timeperiod=50)
            indicators["sma_100"] = talib.SMA(close_prices, timeperiod=100)
            indicators["sma_200"] = talib.SMA(close_prices, timeperiod=200)

            indicators["ema_7"] = talib.EMA(close_prices, timeperiod=7)
            indicators["ema_14"] = talib.EMA(close_prices, timeperiod=14)
            indicators["ema_21"] = talib.EMA(close_prices, timeperiod=21)
            indicators["ema_50"] = talib.EMA(close_prices, timeperiod=50)
            indicators["ema_100"] = talib.EMA(close_prices, timeperiod=100)
            indicators["ema_200"] = talib.EMA(close_prices, timeperiod=200)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices,
                                                    fastperiod=self.config.macd_fast,
                                                    slowperiod=self.config.macd_slow,
                                                    signalperiod=self.config.macd_signal)
            indicators["macd"] = macd
            indicators["macd_signal"] = macd_signal
            indicators["macd_histogram"] = macd_hist

            # ADX (Average Directional Index)
            indicators["adx"] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.config.adx_period)
            indicators["plus_di"] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=self.config.adx_period)
            indicators["minus_di"] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=self.config.adx_period)

            # Aroon
            aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=14)
            indicators["aroon_down"] = aroon_down
            indicators["aroon_up"] = aroon_up
            indicators["aroon_oscillator"] = aroon_up - aroon_down

            # Momentum Indicators
            indicators["rsi"] = talib.RSI(close_prices, timeperiod=self.config.rsi_period)
            indicators["stoch_k"], indicators["stoch_d"] = talib.STOCH(high_prices, low_prices, close_prices,
                                                                      fastk_period=self.config.stoch_k,
                                                                      slowk_period=self.config.stoch_d,
                                                                      slowd_period=self.config.stoch_d)
            indicators["williams_r"] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=self.config.williams_r)
            indicators["cci"] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            indicators["roc"] = talib.ROC(close_prices, timeperiod=10)

            # Volatility Indicators
            indicators["bb_upper"], indicators["bb_middle"], indicators["bb_lower"] = talib.BBANDS(
                close_prices, timeperiod=self.config.bb_period, nbdevup=self.config.bb_std, nbdevdn=self.config.bb_std
            )
            indicators["atr"] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.config.atr_period)
            indicators["natr"] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=self.config.atr_period)
            indicators["trange"] = talib.TRANGE(high_prices, low_prices, close_prices)

            # Volume Indicators
            indicators["obv"] = talib.OBV(close_prices, volumes)
            indicators["ad"] = talib.AD(high_prices, low_prices, close_prices, volumes)
            indicators["adosc"] = talib.ADOSC(high_prices, low_prices, close_prices, volumes, fastperiod=3, slowperiod=10)
            indicators["mfi"] = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=14)

            # Oscillators
            indicators["ultosc"] = talib.ULTOSC(high_prices, low_prices, close_prices, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            indicators["trix"] = talib.TRIX(close_prices, timeperiod=30)
            indicators["dx"] = talib.DX(high_prices, low_prices, close_prices, timeperiod=14)

            # Price Patterns
            indicators["doji"] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            indicators["hammer"] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            indicators["hanging_man"] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
            indicators["shooting_star"] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
            indicators["engulfing"] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            indicators["harami"] = talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)
            indicators["morning_star"] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            indicators["evening_star"] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)

            # Get latest values (remove NaN)
            for key, value in indicators.items():
                if isinstance(value, np.ndarray):
                    indicators[key] = value[~np.isnan(value)]
                    if len(indicators[key]) > 0:
                        indicators[key] = indicators[key][-1]  # Get last value
                    else:
                        indicators[key] = np.nan

            return indicators

        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return {}

    def _generate_trading_signals(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on indicators"""
        signals = {
            "buy_signals": [],
            "sell_signals": [],
            "hold_signals": [],
            "signal_strength": 0,
            "overall_signal": "hold"
        }

        try:
            # RSI signals
            rsi = indicators.get("rsi", 50)
            if rsi < 30:
                signals["buy_signals"].append({"indicator": "RSI", "strength": "strong", "value": rsi})
            elif rsi > 70:
                signals["sell_signals"].append({"indicator": "RSI", "strength": "strong", "value": rsi})

            # MACD signals
            macd = indicators.get("macd", 0)
            macd_signal = indicators.get("macd_signal", 0)
            if macd > macd_signal and macd > 0:
                signals["buy_signals"].append({"indicator": "MACD", "strength": "medium", "value": macd})
            elif macd < macd_signal and macd < 0:
                signals["sell_signals"].append({"indicator": "MACD", "strength": "medium", "value": macd})

            # Moving average signals
            sma_7 = indicators.get("sma_7", 0)
            sma_21 = indicators.get("sma_21", 0)
            sma_50 = indicators.get("sma_50", 0)

            if sma_7 > sma_21 > sma_50:
                signals["buy_signals"].append({"indicator": "MA_Alignment", "strength": "strong", "value": "bullish"})
            elif sma_7 < sma_21 < sma_50:
                signals["sell_signals"].append({"indicator": "MA_Alignment", "strength": "strong", "value": "bearish"})

            # Bollinger Bands signals
            bb_upper = indicators.get("bb_upper", 0)
            bb_lower = indicators.get("bb_lower", 0)
            bb_middle = indicators.get("bb_middle", 0)
            current_price = indicators.get("close", 0)

            if current_price < bb_lower:
                signals["buy_signals"].append({"indicator": "BB", "strength": "medium", "value": "oversold"})
            elif current_price > bb_upper:
                signals["sell_signals"].append({"indicator": "BB", "strength": "medium", "value": "overbought"})

            # Stochastic signals
            stoch_k = indicators.get("stoch_k", 50)
            stoch_d = indicators.get("stoch_d", 50)
            if stoch_k < 20 and stoch_d < 20:
                signals["buy_signals"].append({"indicator": "Stochastic", "strength": "medium", "value": stoch_k})
            elif stoch_k > 80 and stoch_d > 80:
                signals["sell_signals"].append({"indicator": "Stochastic", "strength": "medium", "value": stoch_k})

            # Calculate overall signal strength
            buy_count = len(signals["buy_signals"])
            sell_count = len(signals["sell_signals"])

            if buy_count > sell_count:
                signals["overall_signal"] = "buy"
                signals["signal_strength"] = min(buy_count - sell_count, 5) / 5
            elif sell_count > buy_count:
                signals["overall_signal"] = "sell"
                signals["signal_strength"] = min(sell_count - buy_count, 5) / 5
            else:
                signals["overall_signal"] = "hold"
                signals["signal_strength"] = 0

            return signals

        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return signals

    def _identify_chart_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify chart patterns"""
        patterns = {
            "head_and_shoulders": False,
            "double_top": False,
            "double_bottom": False,
            "triangle": False,
            "flag": False,
            "pennant": False,
            "wedge": False,
            "channel": False
        }

        try:
            if len(df) < 50:
                return patterns

            # Get recent data
            recent_data = df.tail(50)
            highs = recent_data["high"].values
            lows = recent_data["low"].values
            closes = recent_data["close"].values

            # Double Top Pattern
            if self._detect_double_top(highs):
                patterns["double_top"] = True

            # Double Bottom Pattern
            if self._detect_double_bottom(lows):
                patterns["double_bottom"] = True

            # Triangle Pattern
            if self._detect_triangle(highs, lows):
                patterns["triangle"] = True

            # Flag Pattern
            if self._detect_flag(highs, lows, closes):
                patterns["flag"] = True

            # Pennant Pattern
            if self._detect_pennant(highs, lows):
                patterns["pennant"] = True

            return patterns

        except Exception as e:
            logger.error(f"Failed to identify patterns: {e}")
            return patterns

    def _detect_double_top(self, highs: np.ndarray) -> bool:
        """Detect double top pattern"""
        try:
            if len(highs) < 20:
                return False

            # Find peaks
            peaks = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))

            if len(peaks) < 2:
                return False

            # Check if last two peaks are similar height
            last_two_peaks = peaks[-2:]
            height_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1]

            return height_diff < 0.02  # Within 2% of each other

        except Exception as e:
            logger.error(f"Failed to detect double top: {e}")
            return False

    def _detect_double_bottom(self, lows: np.ndarray) -> bool:
        """Detect double bottom pattern"""
        try:
            if len(lows) < 20:
                return False

            # Find troughs
            troughs = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))

            if len(troughs) < 2:
                return False

            # Check if last two troughs are similar depth
            last_two_troughs = troughs[-2:]
            depth_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1]

            return depth_diff < 0.02  # Within 2% of each other

        except Exception as e:
            logger.error(f"Failed to detect double bottom: {e}")
            return False

    def _detect_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Detect triangle pattern"""
        try:
            if len(highs) < 20:
                return False

            # Check if highs are decreasing and lows are increasing
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]

            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

            return high_trend < 0 and low_trend > 0

        except Exception as e:
            logger.error(f"Failed to detect triangle: {e}")
            return False

    def _detect_flag(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> bool:
        """Detect flag pattern"""
        try:
            if len(highs) < 20:
                return False

            # Check for consolidation after a strong move
            recent_closes = closes[-10:]
            price_range = max(recent_closes) - min(recent_closes)
            avg_price = np.mean(recent_closes)

            # Flag if price is consolidating in a small range
            return price_range / avg_price < 0.05  # Less than 5% range

        except Exception as e:
            logger.error(f"Failed to detect flag: {e}")
            return False

    def _detect_pennant(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Detect pennant pattern"""
        try:
            if len(highs) < 20:
                return False

            # Similar to triangle but with converging lines
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]

            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

            # Both lines should be converging
            return high_trend < 0 and low_trend > 0 and abs(high_trend) > abs(low_trend)

        except Exception as e:
            logger.error(f"Failed to detect pennant: {e}")
            return False

    def _analyze_trends(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        try:
            trend_analysis = {
                "short_term": "neutral",
                "medium_term": "neutral",
                "long_term": "neutral",
                "trend_strength": 0,
                "trend_direction": "sideways"
            }

            # Short-term trend (7-day)
            sma_7 = indicators.get("sma_7", 0)
            ema_7 = indicators.get("ema_7", 0)
            current_price = df["close"].iloc[-1]

            if current_price > sma_7 and current_price > ema_7:
                trend_analysis["short_term"] = "bullish"
            elif current_price < sma_7 and current_price < ema_7:
                trend_analysis["short_term"] = "bearish"

            # Medium-term trend (21-day)
            sma_21 = indicators.get("sma_21", 0)
            ema_21 = indicators.get("ema_21", 0)

            if current_price > sma_21 and current_price > ema_21:
                trend_analysis["medium_term"] = "bullish"
            elif current_price < sma_21 and current_price < ema_21:
                trend_analysis["medium_term"] = "bearish"

            # Long-term trend (50-day)
            sma_50 = indicators.get("sma_50", 0)
            ema_50 = indicators.get("ema_50", 0)

            if current_price > sma_50 and current_price > ema_50:
                trend_analysis["long_term"] = "bullish"
            elif current_price < sma_50 and current_price < ema_50:
                trend_analysis["long_term"] = "bearish"

            # Calculate overall trend strength
            bullish_count = sum([
                trend_analysis["short_term"] == "bullish",
                trend_analysis["medium_term"] == "bullish",
                trend_analysis["long_term"] == "bullish"
            ])

            bearish_count = sum([
                trend_analysis["short_term"] == "bearish",
                trend_analysis["medium_term"] == "bearish",
                trend_analysis["long_term"] == "bearish"
            ])

            if bullish_count > bearish_count:
                trend_analysis["trend_direction"] = "bullish"
                trend_analysis["trend_strength"] = bullish_count / 3
            elif bearish_count > bullish_count:
                trend_analysis["trend_direction"] = "bearish"
                trend_analysis["trend_strength"] = bearish_count / 3
            else:
                trend_analysis["trend_direction"] = "sideways"
                trend_analysis["trend_strength"] = 0

            return trend_analysis

        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")
            return {"trend_direction": "unknown", "trend_strength": 0}

    def _analyze_volatility(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility metrics"""
        try:
            volatility_analysis = {
                "current_volatility": "normal",
                "volatility_trend": "stable",
                "atr_value": 0,
                "bb_position": "middle",
                "volatility_percentile": 50
            }

            # ATR analysis
            atr = indicators.get("atr", 0)
            if atr > 0:
                volatility_analysis["atr_value"] = atr
                # Compare with historical ATR
                historical_atr = df["close"].rolling(20).std().iloc[-1]
                if atr > historical_atr * 1.5:
                    volatility_analysis["current_volatility"] = "high"
                elif atr < historical_atr * 0.5:
                    volatility_analysis["current_volatility"] = "low"

            # Bollinger Bands position
            bb_upper = indicators.get("bb_upper", 0)
            bb_lower = indicators.get("bb_lower", 0)
            bb_middle = indicators.get("bb_middle", 0)
            current_price = df["close"].iloc[-1]

            if current_price > bb_upper:
                volatility_analysis["bb_position"] = "upper"
            elif current_price < bb_lower:
                volatility_analysis["bb_position"] = "lower"
            else:
                volatility_analysis["bb_position"] = "middle"

            return volatility_analysis

        except Exception as e:
            logger.error(f"Failed to analyze volatility: {e}")
            return {"current_volatility": "unknown"}

    def _analyze_volume(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            volume_analysis = {
                "volume_trend": "normal",
                "volume_surge": False,
                "obv_trend": "neutral",
                "mfi_value": 50,
                "volume_price_trend": "normal"
            }

            # Volume trend analysis
            recent_volume = df["volume"].tail(10).mean()
            historical_volume = df["volume"].tail(50).mean()

            if recent_volume > historical_volume * 1.5:
                volume_analysis["volume_trend"] = "high"
                volume_analysis["volume_surge"] = True
            elif recent_volume < historical_volume * 0.5:
                volume_analysis["volume_trend"] = "low"

            # MFI analysis
            mfi = indicators.get("mfi", 50)
            volume_analysis["mfi_value"] = mfi

            if mfi > 80:
                volume_analysis["obv_trend"] = "overbought"
            elif mfi < 20:
                volume_analysis["obv_trend"] = "oversold"

            return volume_analysis

        except Exception as e:
            logger.error(f"Failed to analyze volume: {e}")
            return {"volume_trend": "unknown"}

    def _analyze_momentum(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        try:
            momentum_analysis = {
                "rsi_momentum": "neutral",
                "stochastic_momentum": "neutral",
                "macd_momentum": "neutral",
                "overall_momentum": "neutral"
            }

            # RSI momentum
            rsi = indicators.get("rsi", 50)
            if rsi > 70:
                momentum_analysis["rsi_momentum"] = "overbought"
            elif rsi < 30:
                momentum_analysis["rsi_momentum"] = "oversold"
            elif rsi > 50:
                momentum_analysis["rsi_momentum"] = "bullish"
            else:
                momentum_analysis["rsi_momentum"] = "bearish"

            # Stochastic momentum
            stoch_k = indicators.get("stoch_k", 50)
            stoch_d = indicators.get("stoch_d", 50)
            if stoch_k > 80 and stoch_d > 80:
                momentum_analysis["stochastic_momentum"] = "overbought"
            elif stoch_k < 20 and stoch_d < 20:
                momentum_analysis["stochastic_momentum"] = "oversold"
            elif stoch_k > stoch_d:
                momentum_analysis["stochastic_momentum"] = "bullish"
            else:
                momentum_analysis["stochastic_momentum"] = "bearish"

            # MACD momentum
            macd = indicators.get("macd", 0)
            macd_signal = indicators.get("macd_signal", 0)
            if macd > macd_signal and macd > 0:
                momentum_analysis["macd_momentum"] = "bullish"
            elif macd < macd_signal and macd < 0:
                momentum_analysis["macd_momentum"] = "bearish"

            # Overall momentum
            bullish_count = sum([
                momentum_analysis["rsi_momentum"] in ["bullish", "oversold"],
                momentum_analysis["stochastic_momentum"] in ["bullish", "oversold"],
                momentum_analysis["macd_momentum"] == "bullish"
            ])

            bearish_count = sum([
                momentum_analysis["rsi_momentum"] in ["bearish", "overbought"],
                momentum_analysis["stochastic_momentum"] in ["bearish", "overbought"],
                momentum_analysis["macd_momentum"] == "bearish"
            ])

            if bullish_count > bearish_count:
                momentum_analysis["overall_momentum"] = "bullish"
            elif bearish_count > bullish_count:
                momentum_analysis["overall_momentum"] = "bearish"

            return momentum_analysis

        except Exception as e:
            logger.error(f"Failed to analyze momentum: {e}")
            return {"overall_momentum": "unknown"}

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find support and resistance levels"""
        try:
            support_resistance = {
                "resistance_levels": [],
                "support_levels": [],
                "current_resistance": None,
                "current_support": None,
                "strength": "medium"
            }

            # Find local highs and lows
            highs = df["high"].values
            lows = df["low"].values

            # Find resistance levels (local highs)
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                    highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                    support_resistance["resistance_levels"].append(highs[i])

            # Find support levels (local lows)
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
                    lows[i] < lows[i-2] and lows[i] < lows[i+2]):
                    support_resistance["support_levels"].append(lows[i])

            # Get current levels
            if support_resistance["resistance_levels"]:
                support_resistance["current_resistance"] = max(support_resistance["resistance_levels"][-5:])

            if support_resistance["support_levels"]:
                support_resistance["current_support"] = min(support_resistance["support_levels"][-5:])

            return support_resistance

        except Exception as e:
            logger.error(f"Failed to find support/resistance: {e}")
            return {"resistance_levels": [], "support_levels": []}

    def _calculate_risk_metrics(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics"""
        try:
            risk_metrics = {
                "volatility_risk": "medium",
                "trend_risk": "medium",
                "momentum_risk": "medium",
                "overall_risk": "medium",
                "risk_score": 0.5
            }

            # Volatility risk
            atr = indicators.get("atr", 0)
            current_price = df["close"].iloc[-1]
            if atr > 0:
                volatility_pct = (atr / current_price) * 100
                if volatility_pct > 5:
                    risk_metrics["volatility_risk"] = "high"
                elif volatility_pct < 2:
                    risk_metrics["volatility_risk"] = "low"

            # Trend risk (based on trend strength)
            trend_strength = indicators.get("adx", 0)
            if trend_strength > 50:
                risk_metrics["trend_risk"] = "high"
            elif trend_strength < 25:
                risk_metrics["trend_risk"] = "low"

            # Momentum risk
            rsi = indicators.get("rsi", 50)
            if rsi > 80 or rsi < 20:
                risk_metrics["momentum_risk"] = "high"
            elif 30 < rsi < 70:
                risk_metrics["momentum_risk"] = "low"

            # Calculate overall risk score
            risk_scores = {
                "high": 0.8,
                "medium": 0.5,
                "low": 0.2
            }

            avg_risk = (
                risk_scores[risk_metrics["volatility_risk"]] +
                risk_scores[risk_metrics["trend_risk"]] +
                risk_scores[risk_metrics["momentum_risk"]]
            ) / 3

            risk_metrics["risk_score"] = avg_risk

            if avg_risk > 0.6:
                risk_metrics["overall_risk"] = "high"
            elif avg_risk < 0.4:
                risk_metrics["overall_risk"] = "low"

            return risk_metrics

        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {"overall_risk": "unknown", "risk_score": 0.5}
