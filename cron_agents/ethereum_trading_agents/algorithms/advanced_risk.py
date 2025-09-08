"""
Advanced Risk Management Algorithm for Ethereum Trading

This module implements sophisticated risk management strategies:
1. Dynamic position sizing
2. Real-time risk monitoring
3. Portfolio-level risk assessment
4. Stress testing and scenario analysis
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

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(Enum):
    """Types of risks"""
    MARKET = "market"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    CONCENTRATION = "concentration"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_portfolio_risk: float = 0.05  # 5% max portfolio risk
    max_position_size: float = 0.1    # 10% max position size
    stop_loss_threshold: float = 0.05  # 5% stop loss
    take_profit_threshold: float = 0.15  # 15% take profit
    var_confidence_level: float = 0.95  # 95% VaR
    max_drawdown: float = 0.1  # 10% max drawdown
    correlation_threshold: float = 0.7  # 70% correlation threshold
    stress_test_scenarios: int = 1000

class RiskMetrics(TypedDict):
    """Risk metrics structure"""
    portfolio_var: float
    portfolio_es: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float

class PositionRisk(TypedDict):
    """Position risk assessment"""
    position_id: str
    risk_score: float
    risk_level: RiskLevel
    var_contribution: float
    expected_loss: float
    max_loss: float
    correlation_risk: float
    liquidity_risk: float

class AdvancedRiskAlgorithm:
    """Advanced risk management algorithm for Ethereum trading"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.risk_metrics: Dict[str, RiskMetrics] = {}
        self.position_risks: Dict[str, PositionRisk] = {}
        self.portfolio_history: List[Dict[str, float]] = []
        self.correlation_matrix: np.ndarray = np.array([])
        self.stress_test_results: Dict[str, float] = {}
        
    async def calculate_portfolio_var(
        self, 
        portfolio_returns: List[float], 
        confidence_level: float = None
    ) -> float:
        """Calculate Value at Risk (VaR) for portfolio"""
        try:
            if confidence_level is None:
                confidence_level = self.config.var_confidence_level
            
            if not portfolio_returns:
                raise ValueError("No portfolio returns data available")
            
            # Convert to numpy array
            returns = np.array(portfolio_returns)
            
            # Calculate VaR using historical simulation
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(returns, var_percentile)
            
            # Ensure VaR is negative (loss)
            var_value = min(var_value, 0)
            
            logger.info(f"Portfolio VaR calculated: {var_value:.4f} at {confidence_level*100}% confidence")
            
            return float(var_value)
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise ValueError(f"VaR calculation failed: {str(e)}")
    
    async def calculate_expected_shortfall(
        self, 
        portfolio_returns: List[float], 
        confidence_level: float = None
    ) -> float:
        """Calculate Expected Shortfall (ES) for portfolio"""
        try:
            if confidence_level is None:
                confidence_level = self.config.var_confidence_level
            
            if not portfolio_returns:
                raise ValueError("No portfolio returns data available")
            
            # Calculate VaR first
            var_value = await self.calculate_portfolio_var(portfolio_returns, confidence_level)
            
            # Calculate ES as average of returns below VaR
            returns = np.array(portfolio_returns)
            tail_returns = returns[returns <= var_value]
            
            if len(tail_returns) == 0:
                es_value = var_value
            else:
                es_value = np.mean(tail_returns)
            
            logger.info(f"Portfolio ES calculated: {es_value:.4f}")
            
            return float(es_value)
            
        except Exception as e:
            logger.error(f"ES calculation failed: {e}")
            raise ValueError(f"ES calculation failed: {str(e)}")
    
    async def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if not portfolio_values:
                raise ValueError("No portfolio values available")
            
            values = np.array(portfolio_values)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(values)
            
            # Calculate drawdown
            drawdown = (values - running_max) / running_max
            
            # Get maximum drawdown
            max_dd = np.min(drawdown)
            
            logger.info(f"Maximum drawdown calculated: {max_dd:.4f}")
            
            return float(max_dd)
            
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            raise ValueError(f"Max drawdown calculation failed: {str(e)}")
    
    async def calculate_sharpe_ratio(
        self, 
        portfolio_returns: List[float], 
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not portfolio_returns:
                raise ValueError("No portfolio returns available")
            
            returns = np.array(portfolio_returns)
            
            # Calculate excess returns
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            
            # Calculate Sharpe ratio
            if np.std(returns) == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            
            logger.info(f"Sharpe ratio calculated: {sharpe_ratio:.4f}")
            
            return float(sharpe_ratio)
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation failed: {e}")
            raise ValueError(f"Sharpe ratio calculation failed: {str(e)}")
    
    async def calculate_sortino_ratio(
        self, 
        portfolio_returns: List[float], 
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if not portfolio_returns:
                raise ValueError("No portfolio returns available")
            
            returns = np.array(portfolio_returns)
            
            # Calculate excess returns
            excess_returns = returns - risk_free_rate / 252
            
            # Calculate downside deviation
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                sortino_ratio = float('inf')
            else:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            
            logger.info(f"Sortino ratio calculated: {sortino_ratio:.4f}")
            
            return float(sortino_ratio)
            
        except Exception as e:
            logger.error(f"Sortino ratio calculation failed: {e}")
            raise ValueError(f"Sortino ratio calculation failed: {str(e)}")
    
    async def calculate_correlation_risk(
        self, 
        position_returns: Dict[str, List[float]]
    ) -> float:
        """Calculate correlation risk across positions"""
        try:
            if len(position_returns) < 2:
                return 0.0
            
            # Convert to DataFrame
            df = pd.DataFrame(position_returns)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr().values
            
            # Calculate average correlation (excluding diagonal)
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            avg_correlation = np.mean(correlation_matrix[mask])
            
            # Calculate correlation risk as deviation from independence
            correlation_risk = abs(avg_correlation)
            
            logger.info(f"Correlation risk calculated: {correlation_risk:.4f}")
            
            return float(correlation_risk)
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return 0.0
    
    async def calculate_concentration_risk(
        self, 
        position_weights: Dict[str, float]
    ) -> float:
        """Calculate concentration risk using Herfindahl-Hirschman Index"""
        try:
            if not position_weights:
                return 0.0
            
            weights = np.array(list(position_weights.values()))
            
            # Calculate HHI
            hhi = np.sum(weights ** 2)
            
            # Normalize to 0-1 scale
            max_hhi = 1.0  # Maximum when all weight in one position
            concentration_risk = hhi / max_hhi
            
            logger.info(f"Concentration risk calculated: {concentration_risk:.4f}")
            
            return float(concentration_risk)
            
        except Exception as e:
            logger.error(f"Concentration risk calculation failed: {e}")
            return 0.0
    
    async def calculate_liquidity_risk(
        self, 
        position_sizes: Dict[str, float], 
        market_volumes: Dict[str, float]
    ) -> float:
        """Calculate liquidity risk for positions"""
        try:
            if not position_sizes or not market_volumes:
                return 0.0
            
            liquidity_risks = []
            
            for position_id, size in position_sizes.items():
                if position_id in market_volumes:
                    volume = market_volumes[position_id]
                    
                    # Calculate position size as percentage of daily volume
                    size_ratio = size / volume if volume > 0 else 1.0
                    
                    # Higher ratio = higher liquidity risk
                    liquidity_risk = min(size_ratio, 1.0)
                    liquidity_risks.append(liquidity_risk)
            
            if not liquidity_risks:
                return 0.0
            
            avg_liquidity_risk = np.mean(liquidity_risks)
            
            logger.info(f"Liquidity risk calculated: {avg_liquidity_risk:.4f}")
            
            return float(avg_liquidity_risk)
            
        except Exception as e:
            logger.error(f"Liquidity risk calculation failed: {e}")
            return 0.0
    
    async def perform_stress_test(
        self, 
        portfolio_positions: Dict[str, float], 
        market_scenarios: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Perform stress testing on portfolio"""
        try:
            if not portfolio_positions or not market_scenarios:
                return {}
            
            stress_results = {}
            
            for i, scenario in enumerate(market_scenarios):
                scenario_loss = 0.0
                
                for position_id, position_value in portfolio_positions.items():
                    if position_id in scenario:
                        # Calculate loss under scenario
                        price_change = scenario[position_id]
                        position_loss = position_value * price_change
                        scenario_loss += position_loss
                
                stress_results[f"scenario_{i}"] = scenario_loss
            
            # Calculate stress test statistics
            losses = list(stress_results.values())
            
            stress_test_summary = {
                "worst_case_loss": min(losses),
                "average_loss": np.mean(losses),
                "loss_std": np.std(losses),
                "var_95": np.percentile(losses, 5),
                "var_99": np.percentile(losses, 1)
            }
            
            self.stress_test_results = stress_test_summary
            
            logger.info(f"Stress test completed: {len(market_scenarios)} scenarios")
            
            return stress_test_summary
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            raise ValueError(f"Stress test failed: {str(e)}")
    
    async def calculate_optimal_position_size(
        self, 
        position_expected_return: float, 
        position_volatility: float,
        portfolio_value: float,
        risk_budget: float = None
    ) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            if risk_budget is None:
                risk_budget = self.config.max_portfolio_risk
            
            # Kelly Criterion: f = (bp - q) / b
            # where b = odds, p = probability of win, q = probability of loss
            
            # Estimate win probability from expected return
            win_probability = max(0.5, min(0.9, 0.5 + position_expected_return))
            loss_probability = 1 - win_probability
            
            # Calculate Kelly fraction
            kelly_fraction = (position_expected_return * win_probability - loss_probability) / position_expected_return
            
            # Apply risk budget constraint
            max_position_fraction = min(
                kelly_fraction,
                risk_budget / position_volatility,
                self.config.max_position_size
            )
            
            # Calculate position size
            optimal_size = portfolio_value * max_position_fraction
            
            # Ensure minimum viable position
            min_size = portfolio_value * 0.001  # 0.1% minimum
            optimal_size = max(optimal_size, min_size)
            
            logger.info(f"Optimal position size calculated: {optimal_size:.2f}")
            
            return float(optimal_size)
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            raise ValueError(f"Position sizing failed: {str(e)}")
    
    async def assess_position_risk(
        self, 
        position_id: str, 
        position_value: float, 
        position_volatility: float,
        portfolio_correlation: float = 0.0
    ) -> PositionRisk:
        """Assess risk for individual position"""
        try:
            # Calculate risk score (0-1)
            volatility_risk = min(position_volatility, 1.0)
            correlation_risk = abs(portfolio_correlation)
            size_risk = min(position_value / 1000000, 1.0)  # Normalize to 1M
            
            risk_score = (volatility_risk + correlation_risk + size_risk) / 3
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level = RiskLevel.LOW
            elif risk_score < 0.6:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 0.8:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            # Calculate VaR contribution
            var_contribution = position_value * position_volatility * 1.96  # 95% VaR
            
            # Calculate expected and maximum loss
            expected_loss = position_value * position_volatility * 0.5
            max_loss = position_value * position_volatility * 2.0
            
            position_risk: PositionRisk = {
                "position_id": position_id,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "var_contribution": var_contribution,
                "expected_loss": expected_loss,
                "max_loss": max_loss,
                "correlation_risk": correlation_risk,
                "liquidity_risk": 0.0  # Will be calculated separately
            }
            
            self.position_risks[position_id] = position_risk
            
            logger.info(f"Position risk assessed for {position_id}: {risk_level.value}")
            
            return position_risk
            
        except Exception as e:
            logger.error(f"Position risk assessment failed: {e}")
            raise ValueError(f"Position risk assessment failed: {str(e)}")
    
    async def calculate_comprehensive_risk_metrics(
        self, 
        portfolio_data: Dict[str, any]
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics for portfolio"""
        try:
            # Extract data
            portfolio_returns = portfolio_data.get("returns", [])
            portfolio_values = portfolio_data.get("values", [])
            position_returns = portfolio_data.get("position_returns", {})
            position_weights = portfolio_data.get("position_weights", {})
            position_sizes = portfolio_data.get("position_sizes", {})
            market_volumes = portfolio_data.get("market_volumes", {})
            
            # Calculate individual metrics
            portfolio_var = await self.calculate_portfolio_var(portfolio_returns)
            portfolio_es = await self.calculate_expected_shortfall(portfolio_returns)
            max_drawdown = await self.calculate_max_drawdown(portfolio_values)
            sharpe_ratio = await self.calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio = await self.calculate_sortino_ratio(portfolio_returns)
            
            # Calculate additional metrics
            correlation_risk = await self.calculate_correlation_risk(position_returns)
            concentration_risk = await self.calculate_concentration_risk(position_weights)
            liquidity_risk = await self.calculate_liquidity_risk(position_sizes, market_volumes)
            
            # Calculate Calmar ratio
            calmar_ratio = 0.0
            if max_drawdown != 0:
                annual_return = np.mean(portfolio_returns) * 252
                calmar_ratio = annual_return / abs(max_drawdown)
            
            # Calculate beta (simplified)
            beta = 1.0  # Placeholder - would need market returns
            
            # Create comprehensive risk metrics
            risk_metrics: RiskMetrics = {
                "portfolio_var": portfolio_var,
                "portfolio_es": portfolio_es,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "beta": beta,
                "correlation_risk": correlation_risk,
                "concentration_risk": concentration_risk,
                "liquidity_risk": liquidity_risk
            }
            
            # Store metrics
            timestamp = datetime.now().isoformat()
            self.risk_metrics[timestamp] = risk_metrics
            
            logger.info("Comprehensive risk metrics calculated")
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            raise ValueError(f"Risk metrics calculation failed: {str(e)}")
    
    async def should_reduce_risk(self, risk_metrics: RiskMetrics) -> Tuple[bool, List[str]]:
        """Determine if risk should be reduced and provide recommendations"""
        try:
            recommendations = []
            should_reduce = False
            
            # Check VaR threshold
            if risk_metrics["portfolio_var"] < -self.config.max_portfolio_risk:
                recommendations.append("Portfolio VaR exceeds maximum risk threshold")
                should_reduce = True
            
            # Check maximum drawdown
            if risk_metrics["max_drawdown"] < -self.config.max_drawdown:
                recommendations.append("Maximum drawdown exceeds threshold")
                should_reduce = True
            
            # Check concentration risk
            if risk_metrics["concentration_risk"] > 0.5:
                recommendations.append("Portfolio concentration too high")
                should_reduce = True
            
            # Check correlation risk
            if risk_metrics["correlation_risk"] > self.config.correlation_threshold:
                recommendations.append("High correlation between positions")
                should_reduce = True
            
            # Check liquidity risk
            if risk_metrics["liquidity_risk"] > 0.3:
                recommendations.append("High liquidity risk detected")
                should_reduce = True
            
            if should_reduce:
                logger.warning(f"Risk reduction recommended: {recommendations}")
            else:
                logger.info("Portfolio risk within acceptable limits")
            
            return should_reduce, recommendations
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return False, [f"Risk assessment error: {str(e)}"]
