"""
Advanced Portfolio Management Tools
Comprehensive portfolio management for Ethereum trading
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class PortfolioAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"
    HEDGE = "hedge"


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management"""
    max_positions: int = 10
    min_position_size: float = 0.01  # 1% minimum
    max_position_size: float = 0.25  # 25% maximum
    rebalance_threshold: float = 0.05  # 5% deviation threshold
    target_volatility: float = 0.15  # 15% target volatility
    max_correlation: float = 0.7  # 70% max correlation
    rebalance_frequency: int = 7  # Days between rebalancing


class AdvancedPortfolioManager:
    """Advanced portfolio management with comprehensive strategies"""

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def manage_portfolio(self, portfolio_data: Dict[str, Any],
                        market_data: Dict[str, Any],
                        risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive portfolio management"""
        try:
            portfolio_analysis = {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "current_portfolio": portfolio_data,
                "analysis": {},
                "recommendations": [],
                "actions": [],
                "rebalance_needed": False
            }

            # Analyze current portfolio
            portfolio_analysis["analysis"] = self._analyze_portfolio(portfolio_data, market_data)

            # Generate recommendations
            portfolio_analysis["recommendations"] = self._generate_portfolio_recommendations(
                portfolio_analysis["analysis"], risk_metrics
            )

            # Determine actions
            portfolio_analysis["actions"] = self._determine_portfolio_actions(
                portfolio_analysis["analysis"], portfolio_analysis["recommendations"]
            )

            # Check if rebalancing is needed
            portfolio_analysis["rebalance_needed"] = self._check_rebalance_needed(
                portfolio_analysis["analysis"]
            )

            return portfolio_analysis

        except Exception as e:
            logger.error(f"Portfolio management failed: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_portfolio(self, portfolio_data: Dict[str, Any],
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current portfolio state"""
        try:
            analysis = {
                "total_value": portfolio_data.get("total_value", 0),
                "cash_balance": portfolio_data.get("cash_balance", 0),
                "positions": portfolio_data.get("positions", []),
                "diversification": {},
                "performance": {},
                "risk_metrics": {},
                "allocation": {},
                "correlation_matrix": {}
            }

            # Analyze diversification
            analysis["diversification"] = self._analyze_diversification(portfolio_data)

            # Analyze performance
            analysis["performance"] = self._analyze_performance(portfolio_data)

            # Analyze risk metrics
            analysis["risk_metrics"] = self._analyze_portfolio_risk(portfolio_data, market_data)

            # Analyze allocation
            analysis["allocation"] = self._analyze_allocation(portfolio_data)

            # Calculate correlation matrix
            analysis["correlation_matrix"] = self._calculate_correlation_matrix(portfolio_data)

            return analysis

        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {}

    def _analyze_diversification(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio diversification"""
        try:
            positions = portfolio_data.get("positions", [])
            total_value = portfolio_data.get("total_value", 0)

            if not positions or total_value == 0:
                return {"score": 0, "hhi": 0, "effective_positions": 0}

            # Calculate position weights
            weights = [pos.get("value", 0) / total_value for pos in positions]

            # Calculate Herfindahl-Hirschman Index
            hhi = sum(w ** 2 for w in weights)

            # Calculate effective number of positions
            effective_positions = 1 / hhi if hhi > 0 else 0

            # Calculate diversification score
            max_positions = self.config.max_positions
            diversification_score = min(effective_positions / max_positions, 1.0)

            # Analyze by asset type
            asset_types = {}
            for position in positions:
                asset_type = position.get("asset_type", "unknown")
                value = position.get("value", 0)
                if asset_type not in asset_types:
                    asset_types[asset_type] = 0
                asset_types[asset_type] += value

            # Calculate type diversification
            type_weights = [value / total_value for value in asset_types.values()]
            type_hhi = sum(w ** 2 for w in type_weights)
            type_diversification = 1 / type_hhi if type_hhi > 0 else 0

            return {
                "score": diversification_score,
                "hhi": hhi,
                "effective_positions": effective_positions,
                "asset_types": asset_types,
                "type_diversification": type_diversification,
                "recommendation": "good" if diversification_score > 0.7 else "needs_improvement"
            }

        except Exception as e:
            logger.error(f"Diversification analysis failed: {e}")
            return {"score": 0, "hhi": 0, "effective_positions": 0}

    def _analyze_performance(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio performance"""
        try:
            total_value = portfolio_data.get("total_value", 0)
            initial_value = portfolio_data.get("initial_value", total_value)
            positions = portfolio_data.get("positions", [])

            # Calculate total return
            total_return = (total_value - initial_value) / initial_value if initial_value > 0 else 0

            # Calculate position-level performance
            position_performance = []
            for position in positions:
                entry_price = position.get("entry_price", 0)
                current_price = position.get("current_price", 0)
                amount = position.get("amount", 0)

                if entry_price > 0 and current_price > 0:
                    position_return = (current_price - entry_price) / entry_price
                    position_value = amount * current_price
                    position_performance.append({
                        "asset": position.get("asset", ""),
                        "return": position_return,
                        "value": position_value,
                        "pnl": position_value - (amount * entry_price)
                    })

            # Calculate best and worst performers
            if position_performance:
                best_performer = max(position_performance, key=lambda x: x["return"])
                worst_performer = min(position_performance, key=lambda x: x["return"])
            else:
                best_performer = {"asset": "N/A", "return": 0, "value": 0, "pnl": 0}
                worst_performer = {"asset": "N/A", "return": 0, "value": 0, "pnl": 0}

            # Calculate volatility (simplified)
            returns = [p["return"] for p in position_performance]
            volatility = np.std(returns) if len(returns) > 1 else 0

            # Calculate Sharpe ratio (simplified)
            risk_free_rate = 0.02  # 2% risk-free rate
            sharpe_ratio = (total_return - risk_free_rate) / volatility if volatility > 0 else 0

            return {
                "total_return": total_return,
                "total_pnl": total_value - initial_value,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "best_performer": best_performer,
                "worst_performer": worst_performer,
                "position_count": len(positions),
                "performance_rating": "excellent" if total_return > 0.2 else "good" if total_return > 0.1 else "poor"
            }

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"total_return": 0, "total_pnl": 0, "volatility": 0}

    def _analyze_portfolio_risk(self, portfolio_data: Dict[str, Any],
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio risk metrics"""
        try:
            positions = portfolio_data.get("positions", [])
            total_value = portfolio_data.get("total_value", 0)

            if not positions or total_value == 0:
                return {"overall_risk": "low", "risk_score": 0}

            # Calculate portfolio volatility
            portfolio_volatility = 0
            for position in positions:
                weight = position.get("value", 0) / total_value
                asset_volatility = position.get("volatility", 0.2)
                portfolio_volatility += (weight * asset_volatility) ** 2

            portfolio_volatility = math.sqrt(portfolio_volatility)

            # Calculate concentration risk
            weights = [pos.get("value", 0) / total_value for pos in positions]
            max_weight = max(weights) if weights else 0
            hhi = sum(w ** 2 for w in weights)

            # Calculate correlation risk
            avg_correlation = self._calculate_average_correlation(positions)

            # Calculate overall risk score
            risk_factors = [
                min(portfolio_volatility / 0.3, 1.0),  # Volatility risk
                min(max_weight / 0.25, 1.0),  # Concentration risk
                min(hhi, 1.0),  # Diversification risk
                min(avg_correlation, 1.0)  # Correlation risk
            ]

            risk_score = np.mean(risk_factors)

            # Determine risk level
            if risk_score > 0.7:
                risk_level = "high"
            elif risk_score > 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"

            return {
                "overall_risk": risk_level,
                "risk_score": risk_score,
                "portfolio_volatility": portfolio_volatility,
                "max_position_weight": max_weight,
                "concentration_hhi": hhi,
                "avg_correlation": avg_correlation,
                "risk_factors": {
                    "volatility": risk_factors[0],
                    "concentration": risk_factors[1],
                    "diversification": risk_factors[2],
                    "correlation": risk_factors[3]
                }
            }

        except Exception as e:
            logger.error(f"Portfolio risk analysis failed: {e}")
            return {"overall_risk": "unknown", "risk_score": 0.5}

    def _analyze_allocation(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio allocation"""
        try:
            positions = portfolio_data.get("positions", [])
            total_value = portfolio_data.get("total_value", 0)
            cash_balance = portfolio_data.get("cash_balance", 0)

            if total_value == 0:
                return {"allocation": {}, "cash_ratio": 0}

            # Calculate allocation by asset
            allocation = {}
            for position in positions:
                asset = position.get("asset", "unknown")
                value = position.get("value", 0)
                allocation[asset] = value / total_value

            # Calculate cash ratio
            cash_ratio = cash_balance / (total_value + cash_balance) if (total_value + cash_balance) > 0 else 0

            # Calculate allocation by type
            type_allocation = {}
            for position in positions:
                asset_type = position.get("asset_type", "unknown")
                value = position.get("value", 0)
                if asset_type not in type_allocation:
                    type_allocation[asset_type] = 0
                type_allocation[asset_type] += value / total_value

            # Check allocation balance
            balanced = True
            for weight in allocation.values():
                if weight > self.config.max_position_size or weight < self.config.min_position_size:
                    balanced = False
                    break

            return {
                "allocation": allocation,
                "type_allocation": type_allocation,
                "cash_ratio": cash_ratio,
                "balanced": balanced,
                "max_weight": max(allocation.values()) if allocation else 0,
                "min_weight": min(allocation.values()) if allocation else 0
            }

        except Exception as e:
            logger.error(f"Allocation analysis failed: {e}")
            return {"allocation": {}, "cash_ratio": 0}

    def _calculate_correlation_matrix(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation matrix for portfolio positions"""
        try:
            positions = portfolio_data.get("positions", [])

            if len(positions) < 2:
                return {"matrix": {}, "avg_correlation": 0}

            # Create correlation matrix
            assets = [pos.get("asset", f"asset_{i}") for i, pos in enumerate(positions)]
            correlation_matrix = {}

            for i, asset1 in enumerate(assets):
                correlation_matrix[asset1] = {}
                for j, asset2 in enumerate(assets):
                    if i == j:
                        correlation_matrix[asset1][asset2] = 1.0
                    else:
                        # Simplified correlation calculation
                        correlation = self._calculate_asset_correlation(positions[i], positions[j])
                        correlation_matrix[asset1][asset2] = correlation

            # Calculate average correlation
            correlations = []
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    correlations.append(correlation_matrix[assets[i]][assets[j]])

            avg_correlation = np.mean(correlations) if correlations else 0

            return {
                "matrix": correlation_matrix,
                "avg_correlation": avg_correlation,
                "max_correlation": max(correlations) if correlations else 0
            }

        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            return {"matrix": {}, "avg_correlation": 0}

    def _calculate_asset_correlation(self, asset1: Dict[str, Any], asset2: Dict[str, Any]) -> float:
        """Calculate correlation between two assets"""
        try:
            # Simplified correlation based on asset types
            type1 = asset1.get("asset_type", "unknown")
            type2 = asset2.get("asset_type", "unknown")

            # Define correlation matrix for asset types
            correlation_matrix = {
                ("ethereum", "ethereum"): 1.0,
                ("ethereum", "bitcoin"): 0.7,
                ("ethereum", "defi"): 0.8,
                ("ethereum", "stablecoin"): 0.1,
                ("bitcoin", "bitcoin"): 1.0,
                ("bitcoin", "defi"): 0.6,
                ("bitcoin", "stablecoin"): 0.1,
                ("defi", "defi"): 0.9,
                ("defi", "stablecoin"): 0.2,
                ("stablecoin", "stablecoin"): 1.0
            }

            # Look up correlation
            key1 = (type1, type2)
            key2 = (type2, type1)

            correlation = correlation_matrix.get(key1) or correlation_matrix.get(key2)
            return correlation if correlation is not None else 0.5

        except Exception as e:
            logger.error(f"Asset correlation calculation failed: {e}")
            return 0.5

    def _calculate_average_correlation(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate average correlation across all positions"""
        try:
            if len(positions) < 2:
                return 0.0

            correlations = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    correlation = self._calculate_asset_correlation(positions[i], positions[j])
                    correlations.append(correlation)

            return np.mean(correlations) if correlations else 0.0

        except Exception as e:
            logger.error(f"Average correlation calculation failed: {e}")
            return 0.0

    def _generate_portfolio_recommendations(self, analysis: Dict[str, Any],
                                          risk_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate portfolio management recommendations"""
        try:
            recommendations = []

            # Diversification recommendations
            diversification = analysis.get("diversification", {})
            if diversification.get("score", 0) < 0.5:
                recommendations.append({
                    "type": "diversification",
                    "priority": "high",
                    "message": "Portfolio is not well diversified",
                    "action": "Add more positions or rebalance existing ones",
                    "score": diversification.get("score", 0)
                })

            # Allocation recommendations
            allocation = analysis.get("allocation", {})
            if not allocation.get("balanced", True):
                recommendations.append({
                    "type": "allocation",
                    "priority": "medium",
                    "message": "Portfolio allocation is not balanced",
                    "action": "Rebalance positions to meet size constraints",
                    "max_weight": allocation.get("max_weight", 0),
                    "min_weight": allocation.get("min_weight", 0)
                })

            # Risk recommendations
            risk_metrics_portfolio = analysis.get("risk_metrics", {})
            if risk_metrics_portfolio.get("overall_risk") == "high":
                recommendations.append({
                    "type": "risk_management",
                    "priority": "high",
                    "message": "Portfolio risk is too high",
                    "action": "Reduce position sizes or add hedging",
                    "risk_score": risk_metrics_portfolio.get("risk_score", 0)
                })

            # Correlation recommendations
            correlation_matrix = analysis.get("correlation_matrix", {})
            if correlation_matrix.get("avg_correlation", 0) > self.config.max_correlation:
                recommendations.append({
                    "type": "correlation",
                    "priority": "medium",
                    "message": "Portfolio positions are too correlated",
                    "action": "Add uncorrelated assets",
                    "avg_correlation": correlation_matrix.get("avg_correlation", 0)
                })

            # Performance recommendations
            performance = analysis.get("performance", {})
            if performance.get("total_return", 0) < 0:
                recommendations.append({
                    "type": "performance",
                    "priority": "high",
                    "message": "Portfolio is underperforming",
                    "action": "Review and potentially exit losing positions",
                    "total_return": performance.get("total_return", 0)
                })

            return recommendations

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []

    def _determine_portfolio_actions(self, analysis: Dict[str, Any],
                                   recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine specific portfolio actions"""
        try:
            actions = []

            # Process recommendations into actions
            for rec in recommendations:
                if rec["type"] == "diversification":
                    actions.append({
                        "action": "add_position",
                        "priority": rec["priority"],
                        "reason": rec["message"],
                        "details": "Add new position to improve diversification"
                    })

                elif rec["type"] == "allocation":
                    actions.append({
                        "action": "rebalance",
                        "priority": rec["priority"],
                        "reason": rec["message"],
                        "details": "Rebalance existing positions"
                    })

                elif rec["type"] == "risk_management":
                    actions.append({
                        "action": "reduce_exposure",
                        "priority": rec["priority"],
                        "reason": rec["message"],
                        "details": "Reduce position sizes or add hedging"
                    })

                elif rec["type"] == "correlation":
                    actions.append({
                        "action": "add_uncorrelated",
                        "priority": rec["priority"],
                        "reason": rec["message"],
                        "details": "Add uncorrelated assets"
                    })

                elif rec["type"] == "performance":
                    actions.append({
                        "action": "review_positions",
                        "priority": rec["priority"],
                        "reason": rec["message"],
                        "details": "Review underperforming positions"
                    })

            return actions

        except Exception as e:
            logger.error(f"Action determination failed: {e}")
            return []

    def _check_rebalance_needed(self, analysis: Dict[str, Any]) -> bool:
        """Check if portfolio rebalancing is needed"""
        try:
            allocation = analysis.get("allocation", {})
            diversification = analysis.get("diversification", {})
            risk_metrics = analysis.get("risk_metrics", {})

            # Check allocation balance
            if not allocation.get("balanced", True):
                return True

            # Check diversification
            if diversification.get("score", 0) < 0.6:
                return True

            # Check risk levels
            if risk_metrics.get("overall_risk") == "high":
                return True

            # Check correlation
            correlation_matrix = analysis.get("correlation_matrix", {})
            if correlation_matrix.get("avg_correlation", 0) > self.config.max_correlation:
                return True

            return False

        except Exception as e:
            logger.error(f"Rebalance check failed: {e}")
            return False
