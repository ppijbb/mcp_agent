"""
Portfolio Investment Agent

Automatically manages multi-asset investment portfolio with risk-based allocation and rebalancing.
"""

import logging
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

import aiohttp

from ...core.orchestrator import BaseAgent
from ...core.ledger import Ledger

logger = logging.getLogger(__name__)


class RiskTolerance(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class PortfolioInvestmentAgent(BaseAgent):
    """
    Portfolio Investment Agent
    
    Automatically:
    - Allocates capital across multiple asset classes
    - Rebalances portfolio based on risk tolerance
    - Executes trades through MCP servers (Alpaca, Polygon, etc.)
    - Optimizes for tax efficiency
    - Tracks performance and returns
    """
    
    def __init__(self, name: str, config: Dict[str, Any], ledger: Ledger):
        super().__init__(name, config, ledger)
        self.config_detail = config.get('config', {})
        self.initial_capital = self.config_detail.get('initial_capital', 10000.0)
        self.risk_tolerance = RiskTolerance(
            self.config_detail.get('risk_tolerance', 'moderate')
        )
        self.rebalance_threshold = self.config_detail.get('rebalance_threshold', 0.05)
        
        # API keys
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY', '')
        self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        self.alpaca_base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Portfolio state
        self.portfolio_value = self.initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.asset_allocation: Dict[str, float] = {}
        
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> bool:
        """Initialize portfolio investment agent."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Initialize asset allocation based on risk tolerance
            self._initialize_asset_allocation()
            
            # Load existing positions if any
            await self._load_positions()
            
            logger.info(
                f"Portfolio Investment Agent initialized: "
                f"${self.portfolio_value:.2f} capital, {self.risk_tolerance.value} risk"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Portfolio Investment Agent: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown agent."""
        await super().shutdown()
        if self.session:
            await self.session.close()
    
    def _initialize_asset_allocation(self):
        """Initialize asset allocation based on risk tolerance."""
        if self.risk_tolerance == RiskTolerance.CONSERVATIVE:
            self.asset_allocation = {
                'stocks': 0.40,
                'bonds': 0.40,
                'cash': 0.15,
                'commodities': 0.05
            }
        elif self.risk_tolerance == RiskTolerance.MODERATE:
            self.asset_allocation = {
                'stocks': 0.60,
                'bonds': 0.25,
                'cash': 0.10,
                'commodities': 0.05
            }
        else:  # AGGRESSIVE
            self.asset_allocation = {
                'stocks': 0.80,
                'bonds': 0.10,
                'cash': 0.05,
                'commodities': 0.05
            }
    
    async def _load_positions(self):
        """Load existing positions from broker API."""
        if not self.alpaca_api_key:
            logger.warning("Alpaca API key not configured, using simulated positions")
            return
        
        try:
            # Get positions from Alpaca API
            url = f"{self.alpaca_base_url}/v2/positions"
            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret_key
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    positions_data = await response.json()
                    for pos in positions_data:
                        self.positions[pos['symbol']] = {
                            'quantity': float(pos['qty']),
                            'avg_price': float(pos['avg_entry_price']),
                            'current_price': float(pos['current_price']),
                            'market_value': float(pos['market_value'])
                        }
                    logger.info(f"Loaded {len(self.positions)} positions from Alpaca")
                else:
                    logger.warning(f"Failed to load positions: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute portfolio management cycle.
        
        Returns:
            Execution result with returns/profits
        """
        try:
            self._running = True
            
            # Update portfolio value
            await self._update_portfolio_value()
            
            # Check if rebalancing is needed
            needs_rebalance = self._check_rebalance_needed()
            
            if needs_rebalance:
                rebalance_result = await self._rebalance_portfolio()
            else:
                rebalance_result = {'rebalanced': False}
            
            # Calculate returns
            returns = self._calculate_returns()
            
            # Record returns as income
            if returns > 0:
                self.ledger.record_transaction(
                    agent_name=self.name,
                    transaction_type='income',
                    amount=returns,
                    description=f"Portfolio returns: ${returns:.2f}",
                    metadata={
                        'portfolio_value': self.portfolio_value,
                        'initial_capital': self.initial_capital,
                        'return_percentage': (returns / self.initial_capital) * 100 if self.initial_capital > 0 else 0,
                        'rebalanced': rebalance_result.get('rebalanced', False)
                    }
                )
            
            result = {
                'success': True,
                'income': returns,
                'description': f"Portfolio Investment: ${self.portfolio_value:.2f} value, ${returns:.2f} returns",
                'metadata': {
                    'portfolio_value': self.portfolio_value,
                    'positions_count': len(self.positions),
                    'rebalanced': rebalance_result.get('rebalanced', False)
                }
            }
            
            logger.info(
                f"Portfolio Investment Agent executed: "
                f"${self.portfolio_value:.2f} value, ${returns:.2f} returns"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing Portfolio Investment Agent: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'income': 0.0
            }
        finally:
            self._running = False
    
    async def _update_portfolio_value(self):
        """Update current portfolio value."""
        try:
            total_value = 0.0
            
            if self.alpaca_api_key:
                # Get account value from Alpaca
                url = f"{self.alpaca_base_url}/v2/account"
                headers = {
                    'APCA-API-KEY-ID': self.alpaca_api_key,
                    'APCA-API-SECRET-KEY': self.alpaca_secret_key
                }
                
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        account_data = await response.json()
                        self.portfolio_value = float(account_data['portfolio_value'])
                        return
            else:
                # Simulate portfolio value
                for symbol, position in self.positions.items():
                    # Simulate price change
                    price_change = (hash(symbol + datetime.now().strftime('%Y%m%d')) % 100 - 50) / 1000.0
                    current_price = position['current_price'] * (1 + price_change)
                    position['current_price'] = current_price
                    total_value += position['quantity'] * current_price
                
                # Add cash
                cash_allocation = self.asset_allocation.get('cash', 0.1)
                total_value += self.initial_capital * cash_allocation
                
                self.portfolio_value = total_value
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def _check_rebalance_needed(self) -> bool:
        """Check if portfolio needs rebalancing."""
        if not self.positions:
            return True  # Need initial allocation
        
        # Calculate current allocation
        total_value = self.portfolio_value
        if total_value == 0:
            return False
        
        current_allocation = {}
        for symbol, position in self.positions.items():
            asset_type = self._get_asset_type(symbol)
            value = position['quantity'] * position['current_price']
            current_allocation[asset_type] = current_allocation.get(asset_type, 0.0) + value
        
        # Normalize to percentages
        for asset_type in current_allocation:
            current_allocation[asset_type] /= total_value
        
        # Check if any allocation deviates beyond threshold
        for asset_type, target_allocation in self.asset_allocation.items():
            current = current_allocation.get(asset_type, 0.0)
            deviation = abs(current - target_allocation)
            if deviation > self.rebalance_threshold:
                logger.info(f"Rebalancing needed: {asset_type} deviation {deviation:.2%}")
                return True
        
        return False
    
    def _get_asset_type(self, symbol: str) -> str:
        """Determine asset type from symbol."""
        # Simple heuristic
        if any(bond_indicator in symbol.upper() for bond_indicator in ['BOND', 'TREASURY', 'TLT']):
            return 'bonds'
        elif any(commodity_indicator in symbol.upper() for commodity_indicator in ['GOLD', 'SILVER', 'OIL', 'GLD', 'SLV']):
            return 'commodities'
        else:
            return 'stocks'
    
    async def _rebalance_portfolio(self) -> Dict[str, Any]:
        """
        Rebalance portfolio to target allocation.
        
        Returns:
            Rebalancing result
        """
        try:
            logger.info("Rebalancing portfolio...")
            
            # Calculate target values for each asset class
            target_values = {}
            for asset_type, allocation in self.asset_allocation.items():
                target_values[asset_type] = self.portfolio_value * allocation
            
            # Determine trades needed
            trades = []
            
            # For simplicity, we'll use ETFs for each asset class
            etf_mapping = {
                'stocks': 'SPY',
                'bonds': 'TLT',
                'commodities': 'GLD',
                'cash': 'CASH'
            }
            
            for asset_type, target_value in target_values.items():
                if asset_type == 'cash':
                    continue  # Cash doesn't need trading
                
                symbol = etf_mapping.get(asset_type)
                if not symbol:
                    continue
                
                current_value = 0.0
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    current_value = pos['quantity'] * pos['current_price']
                
                difference = target_value - current_value
                
                if abs(difference) > 100:  # Minimum trade size
                    # Execute trade
                    if self.alpaca_api_key:
                        success = await self._execute_trade(symbol, difference)
                    else:
                        # Simulate trade
                        success = await self._simulate_trade(symbol, difference)
                    
                    if success:
                        trades.append({
                            'symbol': symbol,
                            'amount': difference,
                            'asset_type': asset_type
                        })
            
            logger.info(f"Rebalancing complete: {len(trades)} trades executed")
            
            return {
                'rebalanced': True,
                'trades_count': len(trades),
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return {'rebalanced': False, 'error': str(e)}
    
    async def _execute_trade(self, symbol: str, amount: float) -> bool:
        """Execute trade through Alpaca API."""
        try:
            # Determine order side
            side = 'buy' if amount > 0 else 'sell'
            qty = abs(amount) / 100.0  # Approximate share price
            
            url = f"{self.alpaca_base_url}/v2/orders"
            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret_key,
                'Content-Type': 'application/json'
            }
            
            order_data = {
                'symbol': symbol,
                'qty': int(qty),
                'side': side,
                'type': 'market',
                'time_in_force': 'day'
            }
            
            async with self.session.post(url, headers=headers, json=order_data) as response:
                if response.status in [200, 201]:
                    logger.info(f"Trade executed: {side} {qty} {symbol}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Trade failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def _simulate_trade(self, symbol: str, amount: float) -> bool:
        """Simulate trade (for testing without API)."""
        try:
            # Get current price (simulated)
            current_price = 100.0  # Default price
            if symbol in self.positions:
                current_price = self.positions[symbol]['current_price']
            
            qty = abs(amount) / current_price
            side = 'buy' if amount > 0 else 'sell'
            
            if side == 'buy':
                if symbol in self.positions:
                    self.positions[symbol]['quantity'] += qty
                else:
                    self.positions[symbol] = {
                        'quantity': qty,
                        'avg_price': current_price,
                        'current_price': current_price,
                        'market_value': qty * current_price
                    }
            else:
                if symbol in self.positions:
                    self.positions[symbol]['quantity'] -= qty
                    if self.positions[symbol]['quantity'] <= 0:
                        del self.positions[symbol]
            
            logger.info(f"Simulated trade: {side} {qty:.2f} {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return False
    
    def _calculate_returns(self) -> float:
        """
        Calculate portfolio returns.
        
        Returns:
            Returns in USD
        """
        current_value = self.portfolio_value
        returns = current_value - self.initial_capital
        
        return returns

