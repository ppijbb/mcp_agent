"""
Ledger System for Transaction History and Accounting

Tracks all income, expenses, and payouts with SQLite database.
"""

import json
import logging
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Ledger:
    """
    Financial ledger system for tracking all transactions.
    
    Features:
    - Transaction recording (income, expense, payout)
    - Agent performance tracking
    - Daily/weekly/monthly reports
    - Asset tracking
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize ledger system.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    description TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Agent performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    revenue REAL DEFAULT 0,
                    expenses REAL DEFAULT 0,
                    net_profit REAL DEFAULT 0,
                    transactions_count INTEGER DEFAULT 0,
                    UNIQUE(agent_name, date)
                )
            """)
            
            # Payouts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS payouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    transaction_id TEXT,
                    error_message TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_agent 
                ON transactions(agent_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_type 
                ON transactions(transaction_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_timestamp 
                ON transactions(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_agent_date 
                ON agent_performance(agent_name, date)
            """)
            
            conn.commit()
            logger.info("Ledger database initialized")
    
    def record_transaction(
        self,
        agent_name: str,
        transaction_type: str,
        amount: float,
        currency: str = 'USD',
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Record a transaction.
        
        Args:
            agent_name: Name of the agent
            transaction_type: 'income', 'expense', or 'payout'
            amount: Transaction amount
            currency: Currency code (default: USD)
            description: Transaction description
            metadata: Additional metadata as dictionary
        
        Returns:
            Transaction ID
        """
        if transaction_type not in ['income', 'expense', 'payout']:
            raise ValueError(f"Invalid transaction type: {transaction_type}")
        
        metadata_str = json.dumps(metadata) if metadata else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions 
                (agent_name, transaction_type, amount, currency, description, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (agent_name, transaction_type, amount, currency, description, metadata_str))
            
            transaction_id = cursor.lastrowid
            
            # Update agent performance
            today = date.today()
            self._update_agent_performance(
                conn, agent_name, today, transaction_type, amount
            )
            
            logger.info(
                f"Recorded {transaction_type} transaction: "
                f"{agent_name} - ${amount:.2f} {currency}"
            )
            
            return transaction_id
    
    def _update_agent_performance(
        self,
        conn: sqlite3.Connection,
        agent_name: str,
        date_obj: date,
        transaction_type: str,
        amount: float
    ):
        """Update agent performance metrics."""
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute("""
            SELECT revenue, expenses, transactions_count
            FROM agent_performance
            WHERE agent_name = ? AND date = ?
        """, (agent_name, date_obj.isoformat()))
        
        row = cursor.fetchone()
        
        if row:
            revenue = row['revenue']
            expenses = row['expenses']
            count = row['transactions_count']
            
            if transaction_type == 'income':
                revenue += amount
            elif transaction_type == 'expense':
                expenses += amount
            
            count += 1
            net_profit = revenue - expenses
            
            cursor.execute("""
                UPDATE agent_performance
                SET revenue = ?, expenses = ?, net_profit = ?, transactions_count = ?
                WHERE agent_name = ? AND date = ?
            """, (revenue, expenses, net_profit, count, agent_name, date_obj.isoformat()))
        else:
            revenue = amount if transaction_type == 'income' else 0.0
            expenses = amount if transaction_type == 'expense' else 0.0
            net_profit = revenue - expenses
            
            cursor.execute("""
                INSERT INTO agent_performance
                (agent_name, date, revenue, expenses, net_profit, transactions_count)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (agent_name, date_obj.isoformat(), revenue, expenses, net_profit))
    
    def get_agent_performance(
        self,
        agent_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Args:
            agent_name: Agent name
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)
        
        Returns:
            Performance metrics dictionary
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            from datetime import timedelta
            start_date = end_date - timedelta(days=30)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    SUM(revenue) as total_revenue,
                    SUM(expenses) as total_expenses,
                    SUM(net_profit) as total_profit,
                    SUM(transactions_count) as total_transactions
                FROM agent_performance
                WHERE agent_name = ? AND date >= ? AND date <= ?
            """, (agent_name, start_date.isoformat(), end_date.isoformat()))
            
            row = cursor.fetchone()
            
            return {
                'agent_name': agent_name,
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_revenue': row['total_revenue'] or 0.0,
                'total_expenses': row['total_expenses'] or 0.0,
                'total_profit': row['total_profit'] or 0.0,
                'total_transactions': row['total_transactions'] or 0
            }
    
    def get_total_assets(self) -> Dict[str, float]:
        """
        Get total assets (sum of all income minus expenses and payouts).
        
        Returns:
            Dictionary with currency -> amount
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    currency,
                    SUM(CASE WHEN transaction_type = 'income' THEN amount ELSE 0 END) as income,
                    SUM(CASE WHEN transaction_type = 'expense' THEN amount ELSE 0 END) as expenses,
                    SUM(CASE WHEN transaction_type = 'payout' THEN amount ELSE 0 END) as payouts
                FROM transactions
                GROUP BY currency
            """)
            
            assets = {}
            for row in cursor.fetchall():
                currency = row['currency']
                net = (row['income'] or 0.0) - (row['expenses'] or 0.0) - (row['payouts'] or 0.0)
                assets[currency] = net
            
            return assets
    
    def get_daily_summary(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Get daily summary of all transactions.
        
        Args:
            target_date: Target date (default: today)
        
        Returns:
            Daily summary dictionary
        """
        if target_date is None:
            target_date = date.today()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get transactions for the day
            cursor.execute("""
                SELECT 
                    agent_name,
                    transaction_type,
                    SUM(amount) as total_amount,
                    currency,
                    COUNT(*) as count
                FROM transactions
                WHERE DATE(timestamp) = ?
                GROUP BY agent_name, transaction_type, currency
            """, (target_date.isoformat(),))
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append({
                    'agent_name': row['agent_name'],
                    'transaction_type': row['transaction_type'],
                    'total_amount': row['total_amount'],
                    'currency': row['currency'],
                    'count': row['count']
                })
            
            # Calculate totals
            total_income = sum(
                t['total_amount'] for t in transactions 
                if t['transaction_type'] == 'income'
            )
            total_expenses = sum(
                t['total_amount'] for t in transactions 
                if t['transaction_type'] == 'expense'
            )
            total_payouts = sum(
                t['total_amount'] for t in transactions 
                if t['transaction_type'] == 'payout'
            )
            
            return {
                'date': target_date.isoformat(),
                'total_income': total_income,
                'total_expenses': total_expenses,
                'total_payouts': total_payouts,
                'net_profit': total_income - total_expenses - total_payouts,
                'transactions': transactions
            }
    
    def record_payout(
        self,
        amount: float,
        status: str,
        transaction_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> int:
        """
        Record a payout transaction.
        
        Args:
            amount: Payout amount
            status: 'pending', 'completed', or 'failed'
            transaction_id: External transaction ID
            error_message: Error message if failed
        
        Returns:
            Payout record ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO payouts (amount, status, transaction_id, error_message)
                VALUES (?, ?, ?, ?)
            """, (amount, status, transaction_id, error_message))
            
            payout_id = cursor.lastrowid
            logger.info(f"Recorded payout: ${amount:.2f} - {status}")
            
            return payout_id

