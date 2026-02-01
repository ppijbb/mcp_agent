"""
Advanced Risk Management Tools
Comprehensive risk management for Ethereum trading
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class RiskType(Enum):
    MARKET = "market"
    LIQUIDITY = "liquidity"
    CREDIT = "credit"
    OPERATIONAL = "operational"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"


@dataclass
class RiskManagementConfig:
    """Configuration for risk management"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.05   # 5% daily loss limit
    max_drawdown: float = 0.15     # 15% max drawdown
    volatility_threshold: float = 0.3  # 30% volatility limit
    correlation_threshold: float = 0.7  # 70% correlation limit
    var_confidence: float = 0.95   # 95% VaR confidence level
    stress_test_scenarios: int = 1000
    lookback_period: int = 252     # 1 year of trading days


class AdvancedRiskManager:
    """Advanced risk management with comprehensive metrics"""

    def __init__(self, config: RiskManagementConfig):
        self.config = config

    def calculate_comprehensive_risk(self, portfolio_data: Dict[str, Any],
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            risk_analysis = {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "risk_metrics": {},
                "risk_scores": {},
                "risk_alerts": [],
                "recommendations": [],
                "overall_risk_level": "medium"
            }

            # Calculate individual risk metrics
            risk_analysis["risk_metrics"] = self._calculate_risk_metrics(portfolio_data, market_data)

            # Calculate risk scores
            risk_analysis["risk_scores"] = self._calculate_risk_scores(risk_analysis["risk_metrics"])

            # Generate risk alerts
            risk_analysis["risk_alerts"] = self._generate_risk_alerts(risk_analysis["risk_metrics"])

            # Generate recommendations
            risk_analysis["recommendations"] = self._generate_recommendations(risk_analysis["risk_metrics"])

            # Calculate overall risk level
            risk_analysis["overall_risk_level"] = self._calculate_overall_risk_level(risk_analysis["risk_scores"])

            return risk_analysis

        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return {"status": "error", "error": str(e)}

    def _calculate_risk_metrics(self, portfolio_data: Dict[str, Any],
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate individual risk metrics"""
        try:
            risk_metrics = {}

            # Market risk metrics
            risk_metrics["market_risk"] = self._calculate_market_risk(portfolio_data, market_data)

            # Liquidity risk metrics
            risk_metrics["liquidity_risk"] = self._calculate_liquidity_risk(portfolio_data, market_data)

            # Concentration risk metrics
            risk_metrics["concentration_risk"] = self._calculate_concentration_risk(portfolio_data)

            # Volatility risk metrics
            risk_metrics["volatility_risk"] = self._calculate_volatility_risk(portfolio_data, market_data)

            # Drawdown risk metrics
            risk_metrics["drawdown_risk"] = self._calculate_drawdown_risk(portfolio_data)

            # Value at Risk (VaR)
            risk_metrics["var"] = self._calculate_var(portfolio_data, market_data)

            # Expected Shortfall (ES)
            risk_metrics["expected_shortfall"] = self._calculate_expected_shortfall(portfolio_data, market_data)

            # Stress test results
            risk_metrics["stress_test"] = self._perform_stress_test(portfolio_data, market_data)

            # Correlation risk
            risk_metrics["correlation_risk"] = self._calculate_correlation_risk(portfolio_data)

            return risk_metrics

        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {}

    def _calculate_market_risk(self, portfolio_data: Dict[str, Any],
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market risk metrics"""
        try:
            portfolio_value = portfolio_data.get("total_value", 0)
            positions = portfolio_data.get("positions", [])

            if not positions or portfolio_value == 0:
                return {"risk_score": 0, "beta": 0, "market_exposure": 0}

            # Calculate portfolio beta
            total_beta = 0
            total_weight = 0

            for position in positions:
                weight = position.get("value", 0) / portfolio_value
                beta = position.get("beta", 1.0)  # Default beta of 1.0
                total_beta += weight * beta
                total_weight += weight

            portfolio_beta = total_beta / total_weight if total_weight > 0 else 1.0

            # Calculate market exposure
            market_exposure = sum(pos.get("value", 0) for pos in positions) / portfolio_value

            # Calculate risk score
            risk_score = min(portfolio_beta * market_exposure, 1.0)

            return {
                "risk_score": risk_score,
                "beta": portfolio_beta,
                "market_exposure": market_exposure,
                "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to calculate market risk: {e}")
            return {"risk_score": 0, "beta": 0, "market_exposure": 0}

    def _calculate_liquidity_risk(self, portfolio_data: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate liquidity risk metrics"""
        try:
            positions = portfolio_data.get("positions", [])
            total_value = portfolio_data.get("total_value", 0)

            if not positions or total_value == 0:
                return {"risk_score": 0, "liquidity_ratio": 0}

            # Calculate liquidity ratio for each position
            liquidity_scores = []

            for position in positions:
                asset = position.get("asset", "")
                value = position.get("value", 0)

                # Get liquidity metrics for the asset
                liquidity_metrics = self._get_asset_liquidity_metrics(asset, market_data)

                # Calculate position liquidity score
                position_liquidity = liquidity_metrics.get("liquidity_score", 0.5)
                position_weight = value / total_value

                liquidity_scores.append(position_liquidity * position_weight)

            # Calculate weighted average liquidity
            avg_liquidity = sum(liquidity_scores) if liquidity_scores else 0.5

            # Calculate risk score (inverse of liquidity)
            risk_score = 1 - avg_liquidity

            return {
                "risk_score": risk_score,
                "liquidity_ratio": avg_liquidity,
                "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to calculate liquidity risk: {e}")
            return {"risk_score": 0, "liquidity_ratio": 0}

    def _calculate_concentration_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate concentration risk metrics"""
        try:
            positions = portfolio_data.get("positions", [])
            total_value = portfolio_data.get("total_value", 0)

            if not positions or total_value == 0:
                return {"risk_score": 0, "hhi": 0, "max_position": 0}

            # Calculate position weights
            weights = [pos.get("value", 0) / total_value for pos in positions]

            # Calculate Herfindahl-Hirschman Index (HHI)
            hhi = sum(w ** 2 for w in weights)

            # Calculate maximum position weight
            max_position = max(weights) if weights else 0

            # Calculate concentration risk score
            risk_score = min(hhi, 1.0)

            return {
                "risk_score": risk_score,
                "hhi": hhi,
                "max_position": max_position,
                "position_count": len(positions),
                "risk_level": "high" if risk_score > 0.5 else "medium" if risk_score > 0.25 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to calculate concentration risk: {e}")
            return {"risk_score": 0, "hhi": 0, "max_position": 0}

    def _calculate_volatility_risk(self, portfolio_data: Dict[str, Any],
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate volatility risk metrics"""
        try:
            positions = portfolio_data.get("positions", [])
            total_value = portfolio_data.get("total_value", 0)

            if not positions or total_value == 0:
                return {"risk_score": 0, "portfolio_volatility": 0}

            # Calculate portfolio volatility
            portfolio_volatility = 0

            for position in positions:
                weight = position.get("value", 0) / total_value
                asset_volatility = position.get("volatility", 0.2)  # Default 20% volatility
                portfolio_volatility += (weight * asset_volatility) ** 2

            portfolio_volatility = math.sqrt(portfolio_volatility)

            # Calculate risk score
            risk_score = min(portfolio_volatility / self.config.volatility_threshold, 1.0)

            return {
                "risk_score": risk_score,
                "portfolio_volatility": portfolio_volatility,
                "volatility_threshold": self.config.volatility_threshold,
                "risk_level": "high" if risk_score > 0.8 else "medium" if risk_score > 0.5 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to calculate volatility risk: {e}")
            return {"risk_score": 0, "portfolio_volatility": 0}

    def _calculate_drawdown_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate drawdown risk metrics"""
        try:
            historical_values = portfolio_data.get("historical_values", [])

            if len(historical_values) < 2:
                return {"risk_score": 0, "max_drawdown": 0, "current_drawdown": 0}

            # Calculate drawdowns
            peak = historical_values[0]
            max_drawdown = 0
            current_drawdown = 0

            for value in historical_values:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                    current_drawdown = drawdown

            # Calculate risk score
            risk_score = min(max_drawdown / self.config.max_drawdown, 1.0)

            return {
                "risk_score": risk_score,
                "max_drawdown": max_drawdown,
                "current_drawdown": current_drawdown,
                "max_drawdown_threshold": self.config.max_drawdown,
                "risk_level": "high" if risk_score > 0.8 else "medium" if risk_score > 0.5 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to calculate drawdown risk: {e}")
            return {"risk_score": 0, "max_drawdown": 0, "current_drawdown": 0}

    def _calculate_var(self, portfolio_data: Dict[str, Any],
                      market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR)"""
        try:
            historical_returns = portfolio_data.get("historical_returns", [])

            if len(historical_returns) < 30:
                return {"var_95": 0, "var_99": 0, "confidence_level": 0.95}

            # Convert to numpy array for calculations
            returns = np.array(historical_returns)

            # Calculate VaR at different confidence levels
            var_95 = np.percentile(returns, 5)  # 95% VaR
            var_99 = np.percentile(returns, 1)  # 99% VaR

            # Calculate VaR for the configured confidence level
            var_confidence = (1 - self.config.var_confidence) * 100
            var_value = np.percentile(returns, var_confidence)

            return {
                "var_95": var_95,
                "var_99": var_99,
                "var_value": var_value,
                "confidence_level": self.config.var_confidence,
                "risk_level": "high" if abs(var_value) > 0.05 else "medium" if abs(var_value) > 0.02 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to calculate VaR: {e}")
            return {"var_95": 0, "var_99": 0, "confidence_level": 0.95}

    def _calculate_expected_shortfall(self, portfolio_data: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Expected Shortfall (ES)"""
        try:
            historical_returns = portfolio_data.get("historical_returns", [])

            if len(historical_returns) < 30:
                return {"es_95": 0, "es_99": 0, "confidence_level": 0.95}

            # Convert to numpy array
            returns = np.array(historical_returns)

            # Calculate ES at different confidence levels
            var_95 = np.percentile(returns, 5)
            es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0

            var_99 = np.percentile(returns, 1)
            es_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0

            # Calculate ES for the configured confidence level
            var_confidence = (1 - self.config.var_confidence) * 100
            var_value = np.percentile(returns, var_confidence)
            es_value = returns[returns <= var_value].mean() if len(returns[returns <= var_value]) > 0 else 0

            return {
                "es_95": es_95,
                "es_99": es_99,
                "es_value": es_value,
                "confidence_level": self.config.var_confidence,
                "risk_level": "high" if abs(es_value) > 0.08 else "medium" if abs(es_value) > 0.03 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to calculate Expected Shortfall: {e}")
            return {"es_95": 0, "es_99": 0, "confidence_level": 0.95}

    def _perform_stress_test(self, portfolio_data: Dict[str, Any],
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress testing scenarios"""
        try:
            positions = portfolio_data.get("positions", [])
            total_value = portfolio_data.get("total_value", 0)

            if not positions or total_value == 0:
                return {"scenarios": [], "worst_case": 0, "average_loss": 0}

            # Define stress test scenarios
            scenarios = [
                {"name": "Market Crash", "price_change": -0.2, "volatility": 0.5},
                {"name": "Moderate Decline", "price_change": -0.1, "volatility": 0.3},
                {"name": "High Volatility", "price_change": 0.0, "volatility": 0.4},
                {"name": "Liquidity Crisis", "price_change": -0.15, "volatility": 0.6},
                {"name": "Flash Crash", "price_change": -0.3, "volatility": 0.8}
            ]

            scenario_results = []

            for scenario in scenarios:
                scenario_loss = 0

                for position in positions:
                    weight = position.get("value", 0) / total_value
                    price_change = scenario["price_change"]
                    volatility_impact = scenario["volatility"] * 0.1  # Additional volatility impact

                    position_loss = weight * (price_change + volatility_impact)
                    scenario_loss += position_loss

                scenario_results.append({
                    "name": scenario["name"],
                    "loss_percentage": abs(scenario_loss),
                    "loss_amount": abs(scenario_loss) * total_value
                })

            # Calculate summary statistics
            losses = [result["loss_percentage"] for result in scenario_results]
            worst_case = max(losses) if losses else 0
            average_loss = np.mean(losses) if losses else 0

            return {
                "scenarios": scenario_results,
                "worst_case": worst_case,
                "average_loss": average_loss,
                "risk_level": "high" if worst_case > 0.2 else "medium" if worst_case > 0.1 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to perform stress test: {e}")
            return {"scenarios": [], "worst_case": 0, "average_loss": 0}

    def _calculate_correlation_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation risk metrics"""
        try:
            positions = portfolio_data.get("positions", [])

            if len(positions) < 2:
                return {"risk_score": 0, "max_correlation": 0, "avg_correlation": 0}

            # Calculate correlations between positions
            correlations = []

            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    correlation = self._calculate_asset_correlation(pos1, pos2)
                    if correlation is not None:
                        correlations.append(abs(correlation))

            if not correlations:
                return {"risk_score": 0, "max_correlation": 0, "avg_correlation": 0}

            max_correlation = max(correlations)
            avg_correlation = np.mean(correlations)

            # Calculate risk score
            risk_score = min(max_correlation / self.config.correlation_threshold, 1.0)

            return {
                "risk_score": risk_score,
                "max_correlation": max_correlation,
                "avg_correlation": avg_correlation,
                "correlation_threshold": self.config.correlation_threshold,
                "risk_level": "high" if risk_score > 0.8 else "medium" if risk_score > 0.5 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to calculate correlation risk: {e}")
            return {"risk_score": 0, "max_correlation": 0, "avg_correlation": 0}

    def _calculate_asset_correlation(self, asset1: Dict[str, Any], asset2: Dict[str, Any]) -> Optional[float]:
        """Calculate correlation between two assets"""
        try:
            # This is a simplified correlation calculation
            # In practice, you would use historical price data
            asset1_type = asset1.get("asset", "").lower()
            asset2_type = asset2.get("asset", "").lower()

            # Define correlation matrix for different asset types
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
            key1 = (asset1_type, asset2_type)
            key2 = (asset2_type, asset1_type)

            correlation = correlation_matrix.get(key1) or correlation_matrix.get(key2)

            return correlation if correlation is not None else 0.5  # Default correlation

        except Exception as e:
            logger.error(f"Failed to calculate asset correlation: {e}")
            return None

    def _get_asset_liquidity_metrics(self, asset: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get liquidity metrics for an asset"""
        try:
            # This is a simplified liquidity calculation
            # In practice, you would use real market data

            liquidity_scores = {
                "ethereum": 0.9,
                "bitcoin": 0.95,
                "usdc": 0.98,
                "usdt": 0.97,
                "defi_token": 0.6,
                "small_cap": 0.3
            }

            # Determine asset type
            asset_lower = asset.lower()
            if "eth" in asset_lower or "ethereum" in asset_lower:
                asset_type = "ethereum"
            elif "btc" in asset_lower or "bitcoin" in asset_lower:
                asset_type = "bitcoin"
            elif "usdc" in asset_lower:
                asset_type = "usdc"
            elif "usdt" in asset_lower:
                asset_type = "usdt"
            elif "defi" in asset_lower:
                asset_type = "defi_token"
            else:
                asset_type = "small_cap"

            liquidity_score = liquidity_scores.get(asset_type, 0.5)

            return {
                "liquidity_score": liquidity_score,
                "asset_type": asset_type,
                "risk_level": "low" if liquidity_score > 0.8 else "medium" if liquidity_score > 0.5 else "high"
            }

        except Exception as e:
            logger.error(f"Failed to get asset liquidity metrics: {e}")
            return {"liquidity_score": 0.5, "asset_type": "unknown"}

    def _calculate_risk_scores(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate normalized risk scores"""
        try:
            risk_scores = {}

            # Extract risk scores from each metric
            for metric_name, metric_data in risk_metrics.items():
                if isinstance(metric_data, dict) and "risk_score" in metric_data:
                    risk_scores[metric_name] = metric_data["risk_score"]

            # Calculate weighted average risk score
            weights = {
                "market_risk": 0.25,
                "liquidity_risk": 0.15,
                "concentration_risk": 0.15,
                "volatility_risk": 0.20,
                "drawdown_risk": 0.15,
                "correlation_risk": 0.10
            }

            weighted_score = 0
            total_weight = 0

            for metric_name, score in risk_scores.items():
                weight = weights.get(metric_name, 0.1)
                weighted_score += score * weight
                total_weight += weight

            overall_score = weighted_score / total_weight if total_weight > 0 else 0.5

            risk_scores["overall"] = overall_score

            return risk_scores

        except Exception as e:
            logger.error(f"Failed to calculate risk scores: {e}")
            return {"overall": 0.5}

    def _generate_risk_alerts(self, risk_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on metrics"""
        try:
            alerts = []

            # Check each risk metric for alerts
            for metric_name, metric_data in risk_metrics.items():
                if isinstance(metric_data, dict) and "risk_score" in metric_data:
                    risk_score = metric_data["risk_score"]
                    risk_level = metric_data.get("risk_level", "medium")

                    if risk_score > 0.8:
                        alerts.append({
                            "type": "high_risk",
                            "metric": metric_name,
                            "score": risk_score,
                            "message": f"High {metric_name} risk detected: {risk_score:.2f}",
                            "severity": "critical"
                        })
                    elif risk_score > 0.6:
                        alerts.append({
                            "type": "medium_risk",
                            "metric": metric_name,
                            "score": risk_score,
                            "message": f"Medium {metric_name} risk detected: {risk_score:.2f}",
                            "severity": "warning"
                        })

            return alerts

        except Exception as e:
            logger.error(f"Failed to generate risk alerts: {e}")
            return []

    def _generate_recommendations(self, risk_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk management recommendations"""
        try:
            recommendations = []

            # Check concentration risk
            if "concentration_risk" in risk_metrics:
                conc_risk = risk_metrics["concentration_risk"]
                if conc_risk.get("risk_score", 0) > 0.5:
                    recommendations.append({
                        "type": "diversification",
                        "priority": "high",
                        "message": "Consider diversifying portfolio to reduce concentration risk",
                        "action": "Reduce position sizes and add more assets"
                    })

            # Check volatility risk
            if "volatility_risk" in risk_metrics:
                vol_risk = risk_metrics["volatility_risk"]
                if vol_risk.get("risk_score", 0) > 0.7:
                    recommendations.append({
                        "type": "volatility_management",
                        "priority": "high",
                        "message": "High volatility detected, consider risk management strategies",
                        "action": "Implement stop-loss orders or reduce position sizes"
                    })

            # Check drawdown risk
            if "drawdown_risk" in risk_metrics:
                dd_risk = risk_metrics["drawdown_risk"]
                if dd_risk.get("risk_score", 0) > 0.8:
                    recommendations.append({
                        "type": "drawdown_management",
                        "priority": "critical",
                        "message": "Maximum drawdown threshold approaching",
                        "action": "Consider reducing exposure or implementing hedging strategies"
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []

    def _calculate_overall_risk_level(self, risk_scores: Dict[str, Any]) -> str:
        """Calculate overall risk level"""
        try:
            overall_score = risk_scores.get("overall", 0.5)

            if overall_score > 0.8:
                return "high"
            elif overall_score > 0.6:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.error(f"Failed to calculate overall risk level: {e}")
            return "unknown"
