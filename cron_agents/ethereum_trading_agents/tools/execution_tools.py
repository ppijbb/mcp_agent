"""
Advanced Trading Execution Tools
Comprehensive trade execution and order management for Ethereum trading
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ExecutionConfig:
    """Configuration for trade execution"""
    max_slippage: float = 0.005  # 0.5% max slippage
    max_gas_price: float = 100.0  # 100 Gwei max gas price
    execution_timeout: int = 300  # 5 minutes timeout
    retry_attempts: int = 3
    retry_delay: float = 1.0  # 1 second delay
    min_trade_size: float = 0.001  # 0.001 ETH minimum
    max_trade_size: float = 10.0  # 10 ETH maximum

class AdvancedExecutionManager:
    """Advanced trade execution with comprehensive order management"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.active_orders = {}
        self.order_history = []
        
    def execute_trade(self, trade_decision: Dict[str, Any], 
                     market_data: Dict[str, Any],
                     account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on decision"""
        try:
            execution_result = {
                "timestamp": datetime.now().isoformat(),
                "status": "pending",
                "order_id": None,
                "trade_details": {},
                "execution_details": {},
                "risk_checks": {},
                "errors": []
            }
            
            # Validate trade decision
            validation_result = self._validate_trade_decision(trade_decision, account_info)
            if not validation_result["valid"]:
                execution_result["status"] = "rejected"
                execution_result["errors"] = validation_result["errors"]
                return execution_result
            
            # Perform risk checks
            risk_checks = self._perform_risk_checks(trade_decision, market_data, account_info)
            execution_result["risk_checks"] = risk_checks
            
            if not risk_checks["passed"]:
                execution_result["status"] = "rejected"
                execution_result["errors"] = risk_checks["errors"]
                return execution_result
            
            # Create order
            order = self._create_order(trade_decision, market_data)
            execution_result["order_id"] = order["order_id"]
            execution_result["trade_details"] = order
            
            # Execute order
            execution_details = self._execute_order(order, market_data)
            execution_result["execution_details"] = execution_details
            
            if execution_details["success"]:
                execution_result["status"] = "filled"
                # Update order history
                self.order_history.append({
                    "order_id": order["order_id"],
                    "timestamp": execution_result["timestamp"],
                    "status": "filled",
                    "details": order,
                    "execution": execution_details
                })
            else:
                execution_result["status"] = "failed"
                execution_result["errors"] = execution_details["errors"]
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def _validate_trade_decision(self, trade_decision: Dict[str, Any], 
                               account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade decision before execution"""
        try:
            errors = []
            
            # Check required fields
            required_fields = ["action", "amount_eth", "target_price"]
            for field in required_fields:
                if field not in trade_decision:
                    errors.append(f"Missing required field: {field}")
            
            if errors:
                return {"valid": False, "errors": errors}
            
            # Validate action
            action = trade_decision.get("action", "")
            if action not in ["buy", "sell"]:
                errors.append("Invalid action. Must be 'buy' or 'sell'")
            
            # Validate amount
            amount = trade_decision.get("amount_eth", 0)
            if amount <= 0:
                errors.append("Amount must be positive")
            elif amount < self.config.min_trade_size:
                errors.append(f"Amount too small. Minimum: {self.config.min_trade_size} ETH")
            elif amount > self.config.max_trade_size:
                errors.append(f"Amount too large. Maximum: {self.config.max_trade_size} ETH")
            
            # Validate price
            target_price = trade_decision.get("target_price", 0)
            if target_price <= 0:
                errors.append("Target price must be positive")
            
            # Check account balance for buy orders
            if action == "buy":
                balance = account_info.get("balance_eth", 0)
                required_balance = amount * target_price
                if balance < required_balance:
                    errors.append(f"Insufficient balance. Required: {required_balance:.4f} ETH, Available: {balance:.4f} ETH")
            
            # Check stop loss and take profit
            stop_loss = trade_decision.get("stop_loss", 0)
            take_profit = trade_decision.get("take_profit", 0)
            
            if action == "buy" and stop_loss > 0 and stop_loss >= target_price:
                errors.append("Stop loss must be below target price for buy orders")
            elif action == "sell" and stop_loss > 0 and stop_loss <= target_price:
                errors.append("Stop loss must be above target price for sell orders")
            
            if action == "buy" and take_profit > 0 and take_profit <= target_price:
                errors.append("Take profit must be above target price for buy orders")
            elif action == "sell" and take_profit > 0 and take_profit >= target_price:
                errors.append("Take profit must be below target price for sell orders")
            
            return {"valid": len(errors) == 0, "errors": errors}
            
        except Exception as e:
            logger.error(f"Trade validation failed: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    def _perform_risk_checks(self, trade_decision: Dict[str, Any], 
                           market_data: Dict[str, Any],
                           account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk checks before execution"""
        try:
            risk_checks = {
                "passed": True,
                "checks": {},
                "errors": []
            }
            
            # Check market volatility
            volatility = market_data.get("volatility", 0)
            if volatility > 0.5:  # 50% volatility threshold
                risk_checks["checks"]["volatility"] = "high"
                risk_checks["errors"].append("Market volatility too high")
                risk_checks["passed"] = False
            else:
                risk_checks["checks"]["volatility"] = "acceptable"
            
            # Check gas price
            gas_price = market_data.get("gas_price_gwei", 0)
            if gas_price > self.config.max_gas_price:
                risk_checks["checks"]["gas_price"] = "high"
                risk_checks["errors"].append(f"Gas price too high: {gas_price} Gwei")
                risk_checks["passed"] = False
            else:
                risk_checks["checks"]["gas_price"] = "acceptable"
            
            # Check slippage
            current_price = market_data.get("price_usd", 0)
            target_price = trade_decision.get("target_price", 0)
            if current_price > 0 and target_price > 0:
                slippage = abs(current_price - target_price) / current_price
                if slippage > self.config.max_slippage:
                    risk_checks["checks"]["slippage"] = "high"
                    risk_checks["errors"].append(f"Slippage too high: {slippage:.2%}")
                    risk_checks["passed"] = False
                else:
                    risk_checks["checks"]["slippage"] = "acceptable"
            
            # Check position size
            amount = trade_decision.get("amount_eth", 0)
            total_value = account_info.get("total_value", 0)
            if total_value > 0:
                position_ratio = (amount * target_price) / total_value
                if position_ratio > 0.25:  # 25% max position
                    risk_checks["checks"]["position_size"] = "high"
                    risk_checks["errors"].append(f"Position size too large: {position_ratio:.2%}")
                    risk_checks["passed"] = False
                else:
                    risk_checks["checks"]["position_size"] = "acceptable"
            
            # Check daily trading limits
            daily_trades = account_info.get("daily_trades", 0)
            if daily_trades >= 10:  # Max 10 trades per day
                risk_checks["checks"]["daily_limits"] = "exceeded"
                risk_checks["errors"].append("Daily trading limit exceeded")
                risk_checks["passed"] = False
            else:
                risk_checks["checks"]["daily_limits"] = "acceptable"
            
            return risk_checks
            
        except Exception as e:
            logger.error(f"Risk checks failed: {e}")
            return {"passed": False, "checks": {}, "errors": [str(e)]}
    
    def _create_order(self, trade_decision: Dict[str, Any], 
                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create order from trade decision"""
        try:
            order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            order = {
                "order_id": order_id,
                "timestamp": datetime.now().isoformat(),
                "action": trade_decision.get("action", ""),
                "amount_eth": trade_decision.get("amount_eth", 0),
                "target_price": trade_decision.get("target_price", 0),
                "stop_loss": trade_decision.get("stop_loss", 0),
                "take_profit": trade_decision.get("take_profit", 0),
                "order_type": OrderType.LIMIT.value,
                "status": OrderStatus.PENDING.value,
                "reason": trade_decision.get("reason", ""),
                "risk_level": trade_decision.get("risk_level", "medium"),
                "expected_return": trade_decision.get("expected_return", "0%")
            }
            
            # Store active order
            self.active_orders[order_id] = order
            
            return order
            
        except Exception as e:
            logger.error(f"Order creation failed: {e}")
            return {}
    
    def _execute_order(self, order: Dict[str, Any], 
                      market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the order"""
        try:
            execution_details = {
                "success": False,
                "execution_price": 0,
                "execution_amount": 0,
                "slippage": 0,
                "gas_used": 0,
                "gas_price": 0,
                "transaction_hash": "",
                "execution_time": 0,
                "errors": []
            }
            
            # Simulate order execution
            # In production, this would interact with actual trading APIs
            
            current_price = market_data.get("price_usd", 0)
            target_price = order.get("target_price", 0)
            amount = order.get("amount_eth", 0)
            
            if current_price == 0 or target_price == 0:
                execution_details["errors"].append("Invalid price data")
                return execution_details
            
            # Check if order can be filled
            action = order.get("action", "")
            if action == "buy" and current_price <= target_price:
                execution_details["success"] = True
                execution_details["execution_price"] = current_price
                execution_details["execution_amount"] = amount
                execution_details["slippage"] = abs(current_price - target_price) / target_price
            elif action == "sell" and current_price >= target_price:
                execution_details["success"] = True
                execution_details["execution_price"] = current_price
                execution_details["execution_amount"] = amount
                execution_details["slippage"] = abs(current_price - target_price) / target_price
            else:
                execution_details["errors"].append("Order cannot be filled at current market price")
                return execution_details
            
            # Simulate gas usage
            execution_details["gas_used"] = 21000  # Standard ETH transfer
            execution_details["gas_price"] = market_data.get("gas_price_gwei", 20)
            execution_details["transaction_hash"] = f"0x{''.join([str(np.random.randint(0, 16)) for _ in range(64)])}"
            execution_details["execution_time"] = np.random.uniform(0.5, 2.0)  # 0.5-2 seconds
            
            # Update order status
            order["status"] = OrderStatus.FILLED.value
            order["execution_details"] = execution_details
            
            return execution_details
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {"success": False, "errors": [str(e)]}
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an order"""
        try:
            if order_id in self.active_orders:
                return {
                    "status": "success",
                    "order": self.active_orders[order_id]
                }
            else:
                # Check order history
                for order in self.order_history:
                    if order["order_id"] == order_id:
                        return {
                            "status": "success",
                            "order": order
                        }
                
                return {
                    "status": "error",
                    "message": "Order not found"
                }
                
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return {"status": "error", "error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an active order"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order["status"] = OrderStatus.CANCELLED.value
                order["cancelled_at"] = datetime.now().isoformat()
                
                # Move to history
                self.order_history.append(order)
                del self.active_orders[order_id]
                
                return {
                    "status": "success",
                    "message": "Order cancelled successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": "Order not found or already processed"
                }
                
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_active_orders(self) -> Dict[str, Any]:
        """Get all active orders"""
        try:
            return {
                "status": "success",
                "active_orders": list(self.active_orders.values()),
                "count": len(self.active_orders)
            }
        except Exception as e:
            logger.error(f"Failed to get active orders: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_order_history(self, limit: int = 100) -> Dict[str, Any]:
        """Get order history"""
        try:
            recent_orders = self.order_history[-limit:] if self.order_history else []
            
            return {
                "status": "success",
                "orders": recent_orders,
                "count": len(recent_orders),
                "total_orders": len(self.order_history)
            }
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            return {"status": "error", "error": str(e)}
    
    def calculate_execution_metrics(self) -> Dict[str, Any]:
        """Calculate execution performance metrics"""
        try:
            if not self.order_history:
                return {"status": "success", "metrics": {}}
            
            # Calculate metrics
            total_orders = len(self.order_history)
            filled_orders = len([o for o in self.order_history if o["status"] == "filled"])
            cancelled_orders = len([o for o in self.order_history if o["status"] == "cancelled"])
            
            fill_rate = filled_orders / total_orders if total_orders > 0 else 0
            
            # Calculate average execution time
            execution_times = []
            for order in self.order_history:
                if "execution_details" in order and "execution_time" in order["execution_details"]:
                    execution_times.append(order["execution_details"]["execution_time"])
            
            avg_execution_time = np.mean(execution_times) if execution_times else 0
            
            # Calculate average slippage
            slippages = []
            for order in self.order_history:
                if "execution_details" in order and "slippage" in order["execution_details"]:
                    slippages.append(order["execution_details"]["slippage"])
            
            avg_slippage = np.mean(slippages) if slippages else 0
            
            return {
                "status": "success",
                "metrics": {
                    "total_orders": total_orders,
                    "filled_orders": filled_orders,
                    "cancelled_orders": cancelled_orders,
                    "fill_rate": fill_rate,
                    "avg_execution_time": avg_execution_time,
                    "avg_slippage": avg_slippage,
                    "success_rate": fill_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate execution metrics: {e}")
            return {"status": "error", "error": str(e)}
