"""
Debt Management Agent

Automatically tracks debts, calculates optimal repayment strategies, and saves on interest.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio

from ...core.orchestrator import BaseAgent
from ...core.ledger import Ledger

logger = logging.getLogger(__name__)


class DebtAccount:
    """Represents a single debt account."""
    
    def __init__(
        self,
        name: str,
        balance: float,
        interest_rate: float,
        minimum_payment: float,
        due_date: Optional[str] = None
    ):
        self.name = name
        self.balance = balance
        self.interest_rate = interest_rate
        self.minimum_payment = minimum_payment
        self.due_date = due_date
        self.monthly_interest = balance * (interest_rate / 12 / 100)
    
    def calculate_savings(self, extra_payment: float) -> float:
        """Calculate interest savings from extra payment."""
        if self.balance <= 0:
            return 0.0
        
        # Simple calculation: extra payment reduces principal, saving interest
        months_to_payoff = self.balance / (self.minimum_payment + extra_payment)
        interest_saved = self.monthly_interest * months_to_payoff * (extra_payment / (self.minimum_payment + extra_payment))
        
        return interest_saved


class DebtManagementAgent(BaseAgent):
    """
    Debt Management Agent
    
    Automatically:
    - Tracks all debts
    - Calculates optimal repayment order
    - Identifies refinancing opportunities
    - Saves on interest payments
    """
    
    def __init__(self, name: str, config: Dict[str, Any], ledger: Ledger):
        super().__init__(name, config, ledger)
        self.debts: List[DebtAccount] = []
        self.plaid_enabled = config.get('config', {}).get('plaid_enabled', False)
        self.auto_refinance = config.get('config', {}).get('optimization', {}).get('auto_refinance', False)
        self.min_savings_threshold = config.get('config', {}).get('optimization', {}).get('min_savings_threshold', 100.0)
    
    async def initialize(self) -> bool:
        """Initialize debt management agent."""
        try:
            # Load debts from configuration
            accounts_config = self.config.get('config', {}).get('accounts', [])
            
            for account_data in accounts_config:
                debt = DebtAccount(
                    name=account_data.get('name', 'Unknown'),
                    balance=account_data.get('balance', 0.0),
                    interest_rate=account_data.get('interest_rate', 0.0),
                    minimum_payment=account_data.get('minimum_payment', 0.0),
                    due_date=account_data.get('due_date')
                )
                self.debts.append(debt)
            
            # TODO: If Plaid enabled, fetch debts from bank API
            if self.plaid_enabled:
                await self._fetch_debts_from_plaid()
            
            logger.info(f"Debt Management Agent initialized with {len(self.debts)} debts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Debt Management Agent: {e}")
            return False
    
    async def _fetch_debts_from_plaid(self):
        """Fetch debts from Plaid API (placeholder)."""
        # TODO: Implement Plaid integration
        logger.info("Plaid integration not yet implemented")
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute debt management cycle.
        
        Returns:
            Execution result with savings calculated
        """
        try:
            self._running = True
            
            if not self.debts:
                logger.debug("No debts to manage")
                return {
                    'success': True,
                    'income': 0.0,
                    'description': 'No debts to manage'
                }
            
            # Calculate optimal repayment strategy
            strategy = self._calculate_optimal_strategy()
            
            # Calculate potential savings
            total_savings = strategy.get('total_savings', 0.0)
            
            # Check for refinancing opportunities
            refinancing_opportunities = self._find_refinancing_opportunities()
            
            # Record savings as "income" (money saved)
            if total_savings > 0:
                # Only record if savings exceed threshold
                if total_savings >= self.min_savings_threshold:
                    self.ledger.record_transaction(
                        agent_name=self.name,
                        transaction_type='income',
                        amount=total_savings,
                        description=f"Interest savings from debt optimization",
                        metadata={
                            'strategy': strategy,
                            'refinancing_opportunities': refinancing_opportunities
                        }
                    )
            
            result = {
                'success': True,
                'income': total_savings,
                'description': f"Debt optimization: ${total_savings:.2f} in interest savings",
                'metadata': {
                    'debts_managed': len(self.debts),
                    'strategy': strategy,
                    'refinancing_opportunities': len(refinancing_opportunities)
                }
            }
            
            logger.info(
                f"Debt Management Agent executed: "
                f"${total_savings:.2f} in potential savings"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing Debt Management Agent: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'income': 0.0
            }
        finally:
            self._running = False
    
    def _calculate_optimal_strategy(self) -> Dict[str, Any]:
        """
        Calculate optimal debt repayment strategy (Avalanche method).
        
        Returns:
            Strategy dictionary with recommendations
        """
        if not self.debts:
            return {'total_savings': 0.0, 'recommendations': []}
        
        # Sort debts by interest rate (highest first - Avalanche method)
        sorted_debts = sorted(
            self.debts,
            key=lambda d: d.interest_rate,
            reverse=True
        )
        
        recommendations = []
        total_savings = 0.0
        
        # Calculate savings from paying off highest interest debt first
        for i, debt in enumerate(sorted_debts):
            if debt.balance <= 0:
                continue
            
            # Calculate if we pay extra on this debt
            extra_payment = 100.0  # Example: $100 extra payment
            savings = debt.calculate_savings(extra_payment)
            
            if savings > 0:
                recommendations.append({
                    'debt_name': debt.name,
                    'current_balance': debt.balance,
                    'interest_rate': debt.interest_rate,
                    'recommended_extra_payment': extra_payment,
                    'estimated_savings': savings,
                    'priority': i + 1
                })
                total_savings += savings
        
        return {
            'total_savings': total_savings,
            'method': 'avalanche',
            'recommendations': recommendations
        }
    
    def _find_refinancing_opportunities(self) -> List[Dict[str, Any]]:
        """
        Find debt refinancing opportunities.
        
        Returns:
            List of refinancing opportunities
        """
        opportunities = []
        
        # Simple heuristic: if interest rate > 10%, consider refinancing
        for debt in self.debts:
            if debt.interest_rate > 10.0 and debt.balance > 1000.0:
                # Estimate potential savings from refinancing to 8% (example)
                current_annual_interest = debt.balance * (debt.interest_rate / 100)
                new_annual_interest = debt.balance * (8.0 / 100)
                annual_savings = current_annual_interest - new_annual_interest
                
                if annual_savings > self.min_savings_threshold:
                    opportunities.append({
                        'debt_name': debt.name,
                        'current_rate': debt.interest_rate,
                        'suggested_rate': 8.0,
                        'annual_savings': annual_savings,
                        'balance': debt.balance
                    })
        
        return opportunities

