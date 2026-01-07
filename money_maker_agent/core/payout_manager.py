"""
Payout Manager for Automatic Money Transfers

Handles daily/weekly/monthly automatic payouts to user's bank account.
"""

import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import asyncio

from .ledger import Ledger
from .account_manager import AccountManager

logger = logging.getLogger(__name__)


class PayoutManager:
    """
    Automatic payout manager.
    
    Features:
    - Daily/weekly/monthly payout scheduling
    - Threshold-based payout triggers
    - Retry logic for failed payouts
    - Payout history tracking
    """
    
    def __init__(
        self,
        ledger: Ledger,
        account_manager: AccountManager,
        threshold: float = 100.0,
        schedule: str = 'daily',
        payout_time: str = '23:00'
    ):
        """
        Initialize payout manager.
        
        Args:
            ledger: Ledger instance for transaction tracking
            account_manager: AccountManager instance for account info
            threshold: Minimum amount for payout (default: $100)
            schedule: Payout schedule ('daily', 'weekly', 'monthly')
            payout_time: Time to execute payout (HH:MM format, UTC)
        """
        self.ledger = ledger
        self.account_manager = account_manager
        self.threshold = threshold
        self.schedule = schedule
        self.payout_time = payout_time
        
        self._last_payout_date: Optional[datetime] = None
        self._enabled = True
    
    def set_config(
        self,
        threshold: Optional[float] = None,
        schedule: Optional[str] = None,
        payout_time: Optional[str] = None,
        enabled: Optional[bool] = None
    ):
        """Update payout configuration."""
        if threshold is not None:
            self.threshold = threshold
        if schedule is not None:
            self.schedule = schedule
        if payout_time is not None:
            self.payout_time = payout_time
        if enabled is not None:
            self._enabled = enabled
        
        logger.info(
            f"Payout config updated: threshold=${self.threshold}, "
            f"schedule={self.schedule}, time={self.payout_time}, enabled={self._enabled}"
        )
    
    def is_payout_due(self) -> bool:
        """
        Check if payout is due based on schedule.
        
        Returns:
            True if payout is due
        """
        if not self._enabled:
            return False
        
        now = datetime.utcnow()
        
        # Check if we've already paid out today
        if self._last_payout_date:
            if self.schedule == 'daily':
                if (now - self._last_payout_date).days < 1:
                    return False
            elif self.schedule == 'weekly':
                if (now - self._last_payout_date).days < 7:
                    return False
            elif self.schedule == 'monthly':
                if (now - self._last_payout_date).days < 30:
                    return False
        
        # Check if it's the right time
        payout_hour, payout_minute = map(int, self.payout_time.split(':'))
        payout_time_obj = time(payout_hour, payout_minute)
        current_time_obj = now.time()
        
        # Allow 1 hour window for payout time
        if current_time_obj < payout_time_obj:
            return False
        
        # Check if we're within 1 hour of payout time
        time_diff = (
            datetime.combine(now.date(), current_time_obj) -
            datetime.combine(now.date(), payout_time_obj)
        ).total_seconds() / 3600
        
        if time_diff > 1.0:
            return False
        
        return True
    
    def calculate_available_balance(self) -> float:
        """
        Calculate available balance for payout.
        
        Returns:
            Available balance in USD
        """
        assets = self.ledger.get_total_assets()
        return assets.get('USD', 0.0)
    
    def should_payout(self) -> Tuple[bool, float]:
        """
        Check if payout should be executed.
        
        Returns:
            (should_payout, available_balance)
        """
        if not self.is_payout_due():
            return False, 0.0
        
        balance = self.calculate_available_balance()
        
        if balance < self.threshold:
            logger.debug(
                f"Payout threshold not met: ${balance:.2f} < ${self.threshold:.2f}"
            )
            return False, balance
        
        return True, balance
    
    async def execute_payout(self, amount: Optional[float] = None) -> Tuple[bool, str]:
        """
        Execute payout to user's bank account.
        
        Args:
            amount: Payout amount (uses available balance if None)
        
        Returns:
            (success, message)
        """
        # Validate account
        is_valid, error = self.account_manager.validate_payout_account()
        if not is_valid:
            payout_id = self.ledger.record_payout(
                0.0, 'failed', error_message=error
            )
            return False, f"Account validation failed: {error}"
        
        # Get payout amount
        if amount is None:
            should_pay, balance = self.should_payout()
            if not should_pay:
                return False, f"Payout conditions not met: balance=${balance:.2f}"
            amount = balance
        else:
            available = self.calculate_available_balance()
            if amount > available:
                return False, f"Insufficient balance: ${available:.2f} < ${amount:.2f}"
        
        # Get account info
        account = self.account_manager.get_payout_account()
        if not account:
            return False, "Payout account not configured"
        
        # Record payout as pending
        payout_id = self.ledger.record_payout(amount, 'pending')
        
        try:
            # TODO: Implement actual bank transfer API
            # For now, simulate the transfer
            logger.info(
                f"Executing payout: ${amount:.2f} to {account['bank_name']} "
                f"account ending in {account['account_number'][-4:]}"
            )
            
            # Simulate API call delay
            await asyncio.sleep(1)
            
            # In production, this would call actual bank API:
            # transaction_id = await bank_api.transfer(
            #     account_number=account['account_number'],
            #     routing_number=account['routing_number'],
            #     amount=amount,
            #     description="Money Maker Agent Payout"
            # )
            
            # For now, simulate success
            transaction_id = f"TXN_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Record transaction in ledger
            self.ledger.record_transaction(
                agent_name='system',
                transaction_type='payout',
                amount=amount,
                description=f"Payout to {account['bank_name']}",
                metadata={'transaction_id': transaction_id, 'payout_id': payout_id}
            )
            
            # Update payout status
            with self.ledger._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE payouts
                    SET status = 'completed', transaction_id = ?
                    WHERE id = ?
                """, (transaction_id, payout_id))
            
            self._last_payout_date = datetime.utcnow()
            
            logger.info(f"Payout completed: ${amount:.2f} (Transaction: {transaction_id})")
            return True, f"Payout successful: ${amount:.2f} (Transaction: {transaction_id})"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Payout failed: {error_msg}")
            
            # Update payout status
            with self.ledger._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE payouts
                    SET status = 'failed', error_message = ?
                    WHERE id = ?
                """, (error_msg, payout_id))
            
            return False, f"Payout failed: {error_msg}"
    
    async def retry_failed_payouts(self, max_retries: int = 3) -> Dict[str, int]:
        """
        Retry failed payouts.
        
        Args:
            max_retries: Maximum number of retries
        
        Returns:
            Dictionary with retry statistics
        """
        with self.ledger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, amount
                FROM payouts
                WHERE status = 'failed'
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            
            failed_payouts = cursor.fetchall()
        
        stats = {'attempted': 0, 'succeeded': 0, 'failed': 0}
        
        for payout in failed_payouts:
            payout_id = payout['id']
            amount = payout['amount']
            
            stats['attempted'] += 1
            
            success, message = await self.execute_payout(amount)
            
            if success:
                stats['succeeded'] += 1
                logger.info(f"Retry successful for payout {payout_id}")
            else:
                stats['failed'] += 1
                logger.warning(f"Retry failed for payout {payout_id}: {message}")
        
        return stats
    
    def get_payout_history(
        self,
        limit: int = 10,
        status: Optional[str] = None
    ) -> list[Dict]:
        """
        Get payout history.
        
        Args:
            limit: Maximum number of records
            status: Filter by status ('pending', 'completed', 'failed')
        
        Returns:
            List of payout records
        """
        with self.ledger._get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM payouts
                    WHERE status = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM payouts
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            payouts = []
            for row in cursor.fetchall():
                payouts.append({
                    'id': row['id'],
                    'amount': row['amount'],
                    'status': row['status'],
                    'timestamp': row['timestamp'],
                    'transaction_id': row['transaction_id'],
                    'error_message': row['error_message']
                })
            
            return payouts

