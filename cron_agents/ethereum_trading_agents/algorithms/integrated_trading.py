"""
Integrated Advanced Trading Algorithm for Ethereum

This module integrates all advanced algorithms:
1. AMM (Automated Market Making)
2. AI Prediction
3. Parallel EVM Optimization
4. Advanced Risk Management
5. Real-time strategy adaptation
"""

import numpy as np
from typing import Dict, List, TypedDict
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

from .amm_algorithm import AMMAlgorithm, AMMConfig
from .technical_analysis import TechnicalAnalysisAlgorithm, TechnicalAnalysisConfig
from .parallel_evm import ParallelEVMAlgorithm, ParallelEVMConfig, Transaction, TransactionPriority
from .advanced_risk import AdvancedRiskAlgorithm, RiskConfig

logger = logging.getLogger(__name__)


class TradingStrategy(Enum):
    """Trading strategy types"""
    AMM = "amm"
    TECHNICAL_ANALYSIS = "technical_analysis"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    HEDGE = "hedge"


class MarketCondition(Enum):
    """Market condition classifications"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"


@dataclass
class IntegratedTradingConfig:
    """Integrated trading configuration"""
    # AMM settings
    amm_enabled: bool = True
    amm_liquidity_threshold: float = 10000.0

    # Technical analysis settings
    technical_analysis_enabled: bool = True
    prediction_confidence_threshold: float = 0.7

    # Parallel EVM settings
    parallel_execution_enabled: bool = True
    max_parallel_trades: int = 5

    # Risk management settings
    risk_management_enabled: bool = True
    max_portfolio_risk: float = 0.05

    # Strategy selection
    primary_strategy: TradingStrategy = TradingStrategy.TECHNICAL_ANALYSIS

    # Market adaptation
    market_adaptation_enabled: bool = True
    strategy_switch_threshold: float = 0.1


class TradingDecision(TypedDict):
    """Trading decision structure"""
    action: str  # "BUY", "SELL", "HOLD", "AMM_PROVIDE", "AMM_REMOVE"
    amount: float
    price: float
    strategy: TradingStrategy
    confidence: float
    risk_score: float
    expected_return: float
    max_loss: float
    execution_priority: TransactionPriority
    gas_strategy: str
    timestamp: str


class IntegratedTradingAlgorithm:
    """Integrated advanced trading algorithm for Ethereum"""

    def __init__(self, config: IntegratedTradingConfig):
        self.config = config

        # Initialize sub-algorithms
        self.amm_algorithm = AMMAlgorithm(AMMConfig())
        self.technical_analysis = TechnicalAnalysisAlgorithm(TechnicalAnalysisConfig())
        self.parallel_evm = ParallelEVMAlgorithm(ParallelEVMConfig())
        self.risk_management = AdvancedRiskAlgorithm(RiskConfig())

        # State tracking
        self.current_strategy: TradingStrategy = config.primary_strategy
        self.market_condition: MarketCondition = MarketCondition.SIDEWAYS
        self.portfolio_state: Dict[str, any] = {}
        self.trading_history: List[TradingDecision] = []
        self.performance_metrics: Dict[str, float] = {}

    async def analyze_market_condition(
        self,
        market_data: Dict[str, any]
    ) -> MarketCondition:
        """Analyze current market condition"""
        try:
            price = market_data.get("price", 0)
            volatility = market_data.get("volatility", 0)
            volume = market_data.get("volume", 0)
            price_change_24h = market_data.get("price_change_24h", 0)

            # Determine market condition based on multiple factors
            if volatility > 0.1:  # High volatility
                condition = MarketCondition.VOLATILE
            elif abs(price_change_24h) > 0.05:  # Significant price change
                if price_change_24h > 0:
                    condition = MarketCondition.BULL
                else:
                    condition = MarketCondition.BEAR
            elif abs(price_change_24h) < 0.01:  # Low price change
                condition = MarketCondition.SIDEWAYS
            else:
                condition = MarketCondition.TRENDING

            self.market_condition = condition
            logger.info(f"Market condition analyzed: {condition.value}")

            return condition

        except Exception as e:
            logger.error(f"Market condition analysis failed: {e}")
            raise ValueError(f"Market condition analysis failed: {e}")

    async def select_optimal_strategy(
        self,
        market_data: Dict[str, any],
        portfolio_data: Dict[str, any]
    ) -> TradingStrategy:
        """Select optimal trading strategy based on market conditions"""
        try:
            if not self.config.market_adaptation_enabled:
                return self.current_strategy

            # Analyze market condition
            market_condition = await self.analyze_market_condition(market_data)

            # Get risk metrics
            risk_metrics = await self.risk_management.calculate_comprehensive_risk_metrics(portfolio_data)

            # Strategy selection logic
            if market_condition == MarketCondition.VOLATILE:
                # High volatility - use AMM for stability
                if self.config.amm_enabled:
                    selected_strategy = TradingStrategy.AMM
                else:
                    selected_strategy = TradingStrategy.HEDGE

            elif market_condition in [MarketCondition.BULL, MarketCondition.BEAR]:
                # Trending market - use technical analysis
                if self.config.technical_analysis_enabled:
                    selected_strategy = TradingStrategy.TECHNICAL_ANALYSIS
                else:
                    selected_strategy = TradingStrategy.MOMENTUM

            elif market_condition == MarketCondition.SIDEWAYS:
                # Sideways market - use mean reversion
                selected_strategy = TradingStrategy.MEAN_REVERSION

            else:  # TRENDING
                # Use momentum strategy
                selected_strategy = TradingStrategy.MOMENTUM

            # Check if strategy switch is needed
            if selected_strategy != self.current_strategy:
                logger.info(f"Strategy switched from {self.current_strategy.value} to {selected_strategy.value}")
                self.current_strategy = selected_strategy

            return selected_strategy

        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return self.config.primary_strategy

    async def execute_amm_strategy(
        self,
        market_data: Dict[str, any],
        portfolio_data: Dict[str, any]
    ) -> TradingDecision:
        """Execute AMM strategy"""
        try:
            token0 = "ETH"
            token1 = "USDC"
            current_price = market_data.get("price", 0)
            volatility = market_data.get("volatility", 0)
            volume_24h = market_data.get("volume_24h", 0)

            # Execute AMM strategy
            amm_result = await self.amm_algorithm.execute_amm_strategy(
                token0, token1, current_price, volatility, volume_24h
            )

            # Create trading decision
            decision: TradingDecision = {
                "action": "AMM_PROVIDE",
                "amount": amm_result["position"]["liquidity"],
                "price": current_price,
                "strategy": TradingStrategy.AMM,
                "confidence": amm_result["confidence"],
                "risk_score": 0.3,  # AMM is generally lower risk
                "expected_return": amm_result["net_expected_return"],
                "max_loss": amm_result["expected_il"],
                "execution_priority": TransactionPriority.MEDIUM,
                "gas_strategy": "dynamic",
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"AMM strategy executed: {decision['action']}")

            return decision

        except Exception as e:
            logger.error(f"AMM strategy execution failed: {e}")
            raise ValueError(f"AMM strategy failed: {str(e)}")

    async def execute_technical_analysis_strategy(
        self,
        market_data: Dict[str, any],
        portfolio_data: Dict[str, any]
    ) -> TradingDecision:
        """Execute technical analysis strategy"""
        try:
            # Get price analysis using technical analysis
            historical_data = portfolio_data.get("historical_data", [])
            analysis_result = await self.technical_analysis.analyze_price(market_data, historical_data)

            # Determine action based on analysis
            current_price = market_data.get("price", 0)
            predicted_price = analysis_result["prediction"]
            confidence = analysis_result["confidence"]

            if confidence < self.config.prediction_confidence_threshold:
                action = "HOLD"
                amount = 0.0
            elif predicted_price > current_price * 1.02:  # 2% upside
                action = "BUY"
                amount = self._calculate_position_size(current_price, confidence)
            elif predicted_price < current_price * 0.98:  # 2% downside
                action = "SELL"
                amount = self._calculate_position_size(current_price, confidence)
            else:
                action = "HOLD"
                amount = 0.0

            # Calculate risk metrics
            risk_score = 1 - confidence  # Higher confidence = lower risk
            expected_return = abs(predicted_price - current_price) / current_price
            max_loss = expected_return * 0.5  # Assume 50% of expected return as max loss

            decision: TradingDecision = {
                "action": action,
                "amount": amount,
                "price": current_price,
                "strategy": TradingStrategy.TECHNICAL_ANALYSIS,
                "confidence": confidence,
                "risk_score": risk_score,
                "expected_return": expected_return,
                "max_loss": max_loss,
                "execution_priority": TransactionPriority.HIGH if confidence > 0.8 else TransactionPriority.MEDIUM,
                "gas_strategy": "aggressive" if confidence > 0.9 else "dynamic",
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"AI prediction strategy executed: {decision['action']}")

            return decision

        except Exception as e:
            logger.error(f"AI prediction strategy execution failed: {e}")
            raise ValueError(f"AI prediction strategy failed: {str(e)}")

    async def execute_momentum_strategy(
        self,
        market_data: Dict[str, any],
        portfolio_data: Dict[str, any]
    ) -> TradingDecision:
        """Execute momentum strategy"""
        try:
            current_price = market_data.get("price", 0)
            price_change_24h = market_data.get("price_change_24h", 0)
            volume_24h = market_data.get("volume_24h", 0)

            # Calculate momentum score
            momentum_score = price_change_24h * (volume_24h / 1000000)  # Normalize volume

            # Determine action
            if momentum_score > 0.1:  # Strong positive momentum
                action = "BUY"
                amount = self._calculate_position_size(current_price, 0.8)
                confidence = min(0.9, abs(momentum_score) * 2)
            elif momentum_score < -0.1:  # Strong negative momentum
                action = "SELL"
                amount = self._calculate_position_size(current_price, 0.8)
                confidence = min(0.9, abs(momentum_score) * 2)
            else:
                action = "HOLD"
                amount = 0.0
                confidence = 0.5

            decision: TradingDecision = {
                "action": action,
                "amount": amount,
                "price": current_price,
                "strategy": TradingStrategy.MOMENTUM,
                "confidence": confidence,
                "risk_score": 0.4,  # Medium risk
                "expected_return": abs(momentum_score),
                "max_loss": abs(momentum_score) * 0.3,
                "execution_priority": TransactionPriority.MEDIUM,
                "gas_strategy": "dynamic",
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Momentum strategy executed: {decision['action']}")

            return decision

        except Exception as e:
            logger.error(f"Momentum strategy execution failed: {e}")
            raise ValueError(f"Momentum strategy failed: {str(e)}")

    async def execute_mean_reversion_strategy(
        self,
        market_data: Dict[str, any],
        portfolio_data: Dict[str, any]
    ) -> TradingDecision:
        """Execute mean reversion strategy"""
        try:
            current_price = market_data.get("price", 0)
            historical_prices = [d.get("price", 0) for d in portfolio_data.get("historical_data", [])]

            if len(historical_prices) < 20:
                raise ValueError("Insufficient historical data for mean reversion")

            # Calculate moving average
            ma_20 = np.mean(historical_prices[-20:])
            ma_50 = np.mean(historical_prices[-50:]) if len(historical_prices) >= 50 else ma_20

            # Calculate deviation from mean
            deviation = (current_price - ma_20) / ma_20

            # Determine action
            if deviation > 0.05:  # Price above mean by 5%
                action = "SELL"  # Expect reversion down
                amount = self._calculate_position_size(current_price, 0.7)
                confidence = min(0.8, abs(deviation) * 5)
            elif deviation < -0.05:  # Price below mean by 5%
                action = "BUY"  # Expect reversion up
                amount = self._calculate_position_size(current_price, 0.7)
                confidence = min(0.8, abs(deviation) * 5)
            else:
                action = "HOLD"
                amount = 0.0
                confidence = 0.5

            decision: TradingDecision = {
                "action": action,
                "amount": amount,
                "price": current_price,
                "strategy": TradingStrategy.MEAN_REVERSION,
                "confidence": confidence,
                "risk_score": 0.5,  # Medium-high risk
                "expected_return": abs(deviation),
                "max_loss": abs(deviation) * 0.4,
                "execution_priority": TransactionPriority.MEDIUM,
                "gas_strategy": "conservative",
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Mean reversion strategy executed: {decision['action']}")

            return decision

        except Exception as e:
            logger.error(f"Mean reversion strategy execution failed: {e}")
            raise ValueError(f"Mean reversion strategy failed: {str(e)}")

    def _calculate_position_size(
        self,
        current_price: float,
        confidence: float
    ) -> float:
        """Calculate position size based on confidence and risk management"""
        try:
            # Base position size
            base_size = 0.01  # 1% of portfolio

            # Adjust for confidence
            confidence_multiplier = confidence * 2  # 0.5 to 2.0

            # Calculate final position size
            position_size = base_size * confidence_multiplier

            # Apply risk management constraints
            max_position = 0.1  # 10% max position
            position_size = min(position_size, max_position)

            return position_size

        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.01

    async def execute_trading_decision(
        self,
        decision: TradingDecision,
        portfolio_data: Dict[str, any]
    ) -> Dict[str, any]:
        """Execute trading decision with risk management"""
        try:
            # Risk management check
            if self.config.risk_management_enabled:
                should_reduce, recommendations = await self.risk_management.should_reduce_risk(
                    portfolio_data.get("risk_metrics", {})
                )

                if should_reduce and decision["action"] in ["BUY", "AMM_PROVIDE"]:
                    logger.warning(f"Risk management override: {recommendations}")
                    decision["action"] = "HOLD"
                    decision["amount"] = 0.0

            # Create transaction if action is not HOLD
            if decision["action"] != "HOLD":
                transaction = await self._create_transaction(decision)

                # Execute with parallel EVM optimization
                if self.config.parallel_execution_enabled:
                    execution_result = await self.parallel_evm.execute_parallel_transactions([transaction])
                else:
                    execution_result = await self.parallel_evm._execute_sequential([transaction])

                # Update trading history
                self.trading_history.append(decision)

                return {
                    "decision": decision,
                    "execution_result": execution_result[0] if execution_result else None,
                    "success": execution_result[0]["success"] if execution_result else False
                }
            else:
                return {
                    "decision": decision,
                    "execution_result": None,
                    "success": True
                }

        except Exception as e:
            logger.error(f"Trading decision execution failed: {e}")
            return {
                "decision": decision,
                "execution_result": None,
                "success": False,
                "error": str(e)
            }

    async def _create_transaction(self, decision: TradingDecision) -> Transaction:
        """Create transaction from trading decision"""
        try:
            # Calculate gas price and limit
            gas_price = 20  # Base gas price in Gwei
            gas_limit = 21000  # Base gas limit

            if decision["action"] in ["BUY", "SELL"]:
                gas_limit = 100000  # Higher gas for complex transactions

            # Create transaction
            transaction: Transaction = {
                "to": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",  # Example address
                "value": int(decision["amount"] * 1e18),  # Convert to wei
                "data": "0x",  # Empty data for simple transfers
                "gas_limit": gas_limit,
                "gas_price": int(gas_price * 1e9),  # Convert to wei
                "nonce": 0,  # Will be set by wallet
                "priority": decision["execution_priority"],
                "deadline": int((datetime.now() + timedelta(minutes=10)).timestamp()),
                "max_fee_per_gas": int(gas_price * 1.2 * 1e9),
                "max_priority_fee_per_gas": int(gas_price * 0.1 * 1e9)
            }

            return transaction

        except Exception as e:
            logger.error(f"Transaction creation failed: {e}")
            raise ValueError(f"Transaction creation failed: {str(e)}")

    async def run_integrated_trading_cycle(
        self,
        market_data: Dict[str, any],
        portfolio_data: Dict[str, any]
    ) -> Dict[str, any]:
        """Run complete integrated trading cycle"""
        try:
            # Select optimal strategy
            strategy = await self.select_optimal_strategy(market_data, portfolio_data)

            # Execute strategy
            if strategy == TradingStrategy.AMM:
                decision = await self.execute_amm_strategy(market_data, portfolio_data)
            elif strategy == TradingStrategy.TECHNICAL_ANALYSIS:
                decision = await self.execute_technical_analysis_strategy(market_data, portfolio_data)
            elif strategy == TradingStrategy.MOMENTUM:
                decision = await self.execute_momentum_strategy(market_data, portfolio_data)
            elif strategy == TradingStrategy.MEAN_REVERSION:
                decision = await self.execute_mean_reversion_strategy(market_data, portfolio_data)
            else:
                # Fallback to primary strategy
                decision = await self.execute_technical_analysis_strategy(market_data, portfolio_data)

            # Execute trading decision
            execution_result = await self.execute_trading_decision(decision, portfolio_data)

            # Update performance metrics
            await self._update_performance_metrics(execution_result)

            return {
                "strategy_used": strategy.value,
                "market_condition": self.market_condition.value,
                "decision": decision,
                "execution_result": execution_result,
                "performance_metrics": self.performance_metrics
            }

        except Exception as e:
            logger.error(f"Integrated trading cycle failed: {e}")
            raise ValueError(f"Trading cycle failed: {str(e)}")

    async def _update_performance_metrics(self, execution_result: Dict[str, any]):
        """Update performance metrics"""
        try:
            if "decision" not in execution_result:
                return

            decision = execution_result["decision"]

            # Update basic metrics
            if "total_trades" not in self.performance_metrics:
                self.performance_metrics["total_trades"] = 0
                self.performance_metrics["successful_trades"] = 0
                self.performance_metrics["total_return"] = 0.0

            self.performance_metrics["total_trades"] += 1

            if execution_result.get("success", False):
                self.performance_metrics["successful_trades"] += 1
                self.performance_metrics["total_return"] += decision.get("expected_return", 0.0)

            # Calculate success rate
            success_rate = (
                self.performance_metrics["successful_trades"] /
                self.performance_metrics["total_trades"]
            )
            self.performance_metrics["success_rate"] = success_rate

            logger.info(f"Performance metrics updated: {self.performance_metrics}")

        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")

    async def get_algorithm_status(self) -> Dict[str, any]:
        """Get current algorithm status"""
        try:
            return {
                "current_strategy": self.current_strategy.value,
                "market_condition": self.market_condition.value,
                "total_trades": len(self.trading_history),
                "performance_metrics": self.performance_metrics,
                "config": {
                    "amm_enabled": self.config.amm_enabled,
                    "technical_analysis_enabled": self.config.technical_analysis_enabled,
                    "parallel_execution_enabled": self.config.parallel_execution_enabled,
                    "risk_management_enabled": self.config.risk_management_enabled
                }
            }

        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {}
