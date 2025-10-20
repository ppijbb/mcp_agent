"""
Database module for storing agent execution records
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class TradingDatabase:
    def __init__(self, db_path: str = "ethereum_trading.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Agent execution records - supports ETH/BTC
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        cryptocurrency TEXT NOT NULL DEFAULT 'ethereum',
                        execution_time TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        input_data TEXT,
                        output_data TEXT,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Trading decisions - supports ETH/BTC
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        cryptocurrency TEXT NOT NULL DEFAULT 'ethereum',
                        decision_type TEXT NOT NULL,
                        decision_data TEXT NOT NULL,
                        market_conditions TEXT,
                        reasoning TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                # Market data snapshots - supports ETH/BTC
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        cryptocurrency TEXT NOT NULL DEFAULT 'ethereum',
                        price_usd REAL,
                        price_change_24h REAL,
                        volume_24h REAL,
                        technical_indicators TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                # Risk management records - supports ETH/BTC
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS risk_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        cryptocurrency TEXT NOT NULL DEFAULT 'ethereum',
                        daily_trades_count INTEGER,
                        daily_loss_amount REAL,
                        risk_level TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                # Bitcoin-specific tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS bitcoin_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        tx_hash TEXT UNIQUE,
                        from_address TEXT,
                        to_address TEXT,
                        amount_btc REAL,
                        amount_sats INTEGER,
                        fee_rate REAL,
                        confirmation_count INTEGER,
                        status TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS bitcoin_market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        price_usd REAL,
                        price_btc REAL,
                        market_cap REAL,
                        volume_24h REAL,
                        dominance_percentage REAL,
                        active_addresses INTEGER,
                        transaction_count INTEGER,
                        timestamp TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                # Portfolio tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        ethereum_balance REAL,
                        bitcoin_balance REAL,
                        ethereum_value_usd REAL,
                        bitcoin_value_usd REAL,
                        total_value_usd REAL,
                        allocation_eth_percent REAL,
                        allocation_btc_percent REAL,
                        timestamp TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def record_agent_execution(self, agent_name: str, status: str, cryptocurrency: str = "ethereum",
                              input_data: Dict = None, output_data: Dict = None, error_message: str = None) -> int:
        """Record agent execution and return execution ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO agent_executions (agent_name, cryptocurrency, execution_time, status, input_data, output_data, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_name,
                    cryptocurrency,
                    datetime.now().isoformat(),
                    status,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    error_message
                ))
                
                execution_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Recorded execution for agent {agent_name} ({cryptocurrency}) with ID {execution_id}")
                return execution_id
                
        except Exception as e:
            logger.error(f"Failed to record agent execution: {e}")
            raise
    
    def record_trading_decision(self, execution_id: int, decision_type: str, decision_data: Dict,
                               market_conditions: Dict = None, reasoning: str = None):
        """Record trading decision"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trading_decisions (execution_id, decision_type, decision_data, market_conditions, reasoning)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    execution_id,
                    decision_type,
                    json.dumps(decision_data),
                    json.dumps(market_conditions) if market_conditions else None,
                    reasoning
                ))
                
                conn.commit()
                logger.info(f"Recorded trading decision for execution {execution_id}")
                
        except Exception as e:
            logger.error(f"Failed to record trading decision: {e}")
            raise
    
    def record_market_snapshot(self, execution_id: int, price_usd: float, price_change_24h: float,
                              volume_24h: float, technical_indicators: Dict = None):
        """Record market data snapshot"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO market_snapshots (execution_id, price_usd, price_change_24h, volume_24h, technical_indicators, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    execution_id,
                    price_usd,
                    price_change_24h,
                    volume_24h,
                    json.dumps(technical_indicators) if technical_indicators else None,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                logger.info(f"Recorded market snapshot for execution {execution_id}")
                
        except Exception as e:
            logger.error(f"Failed to record market snapshot: {e}")
            raise
    
    def get_last_execution_data(self, agent_name: str) -> Optional[Dict]:
        """Get last execution data for an agent"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM agent_executions 
                    WHERE agent_name = ? AND status = 'success'
                    ORDER BY execution_time DESC 
                    LIMIT 1
                """, (agent_name,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'agent_name': row[1],
                        'execution_time': row[2],
                        'status': row[3],
                        'input_data': json.loads(row[4]) if row[4] else None,
                        'output_data': json.loads(row[5]) if row[5] else None,
                        'error_message': row[6],
                        'created_at': row[7]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get last execution data: {e}")
            return None
    
    def get_market_trends(self, hours: int = 24) -> List[Dict]:
        """Get market trends over specified hours"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT price_usd, price_change_24h, volume_24h, timestamp
                    FROM market_snapshots 
                    WHERE timestamp >= datetime('now', '-{} hours')
                    ORDER BY timestamp ASC
                """.format(hours))
                
                rows = cursor.fetchall()
                return [
                    {
                        'price_usd': row[0],
                        'price_change_24h': row[1],
                        'volume_24h': row[2],
                        'timestamp': row[3]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get market trends: {e}")
            return []
    
    def get_daily_trading_summary(self) -> Dict:
        """Get daily trading summary for risk management"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT COUNT(*) as trade_count, 
                           SUM(CASE WHEN decision_type = 'sell' THEN 1 ELSE 0 END) as sell_count,
                           SUM(CASE WHEN decision_type = 'buy' THEN 1 ELSE 0 END) as buy_count
                    FROM trading_decisions 
                    WHERE DATE(created_at) = DATE('now')
                """)
                
                row = cursor.fetchone()
                return {
                    'total_trades': row[0],
                    'sell_count': row[1],
                    'buy_count': row[2],
                    'date': datetime.now().date().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get daily trading summary: {e}")
            return {'total_trades': 0, 'sell_count': 0, 'buy_count': 0, 'date': datetime.now().date().isoformat()}
    
    # Bitcoin-specific methods
    def record_bitcoin_transaction(self, execution_id: int, tx_hash: str, from_address: str,
                                 to_address: str, amount_btc: float, amount_sats: int,
                                 fee_rate: float, confirmation_count: int = 0, status: str = "pending"):
        """Record Bitcoin transaction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO bitcoin_transactions (execution_id, tx_hash, from_address, to_address, 
                                                   amount_btc, amount_sats, fee_rate, confirmation_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (execution_id, tx_hash, from_address, to_address, amount_btc, amount_sats, 
                      fee_rate, confirmation_count, status))
                
                conn.commit()
                logger.info(f"Recorded Bitcoin transaction {tx_hash}")
                
        except Exception as e:
            logger.error(f"Failed to record Bitcoin transaction: {e}")
            raise
    
    def record_bitcoin_market_data(self, execution_id: int, price_usd: float, price_btc: float,
                                 market_cap: float, volume_24h: float, dominance_percentage: float,
                                 active_addresses: int, transaction_count: int):
        """Record Bitcoin market data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO bitcoin_market_data (execution_id, price_usd, price_btc, market_cap,
                                                   volume_24h, dominance_percentage, active_addresses, 
                                                   transaction_count, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (execution_id, price_usd, price_btc, market_cap, volume_24h, dominance_percentage,
                      active_addresses, transaction_count, datetime.now().isoformat()))
                
                conn.commit()
                logger.info(f"Recorded Bitcoin market data for execution {execution_id}")
                
        except Exception as e:
            logger.error(f"Failed to record Bitcoin market data: {e}")
            raise
    
    def record_portfolio_snapshot(self, execution_id: int, ethereum_balance: float, bitcoin_balance: float,
                                ethereum_value_usd: float, bitcoin_value_usd: float, total_value_usd: float,
                                allocation_eth_percent: float, allocation_btc_percent: float):
        """Record portfolio snapshot"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO portfolio_snapshots (execution_id, ethereum_balance, bitcoin_balance,
                                                   ethereum_value_usd, bitcoin_value_usd, total_value_usd,
                                                   allocation_eth_percent, allocation_btc_percent, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (execution_id, ethereum_balance, bitcoin_balance, ethereum_value_usd, bitcoin_value_usd,
                      total_value_usd, allocation_eth_percent, allocation_btc_percent, datetime.now().isoformat()))
                
                conn.commit()
                logger.info(f"Recorded portfolio snapshot for execution {execution_id}")
                
        except Exception as e:
            logger.error(f"Failed to record portfolio snapshot: {e}")
            raise
    
    def get_bitcoin_transaction_history(self, hours: int = 24) -> List[Dict]:
        """Get Bitcoin transaction history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT tx_hash, from_address, to_address, amount_btc, amount_sats, 
                           fee_rate, confirmation_count, status, created_at
                    FROM bitcoin_transactions 
                    WHERE created_at >= datetime('now', '-{} hours')
                    ORDER BY created_at DESC
                """.format(hours))
                
                rows = cursor.fetchall()
                return [
                    {
                        'tx_hash': row[0],
                        'from_address': row[1],
                        'to_address': row[2],
                        'amount_btc': row[3],
                        'amount_sats': row[4],
                        'fee_rate': row[5],
                        'confirmation_count': row[6],
                        'status': row[7],
                        'created_at': row[8]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get Bitcoin transaction history: {e}")
            return []
    
    def get_cross_crypto_analysis(self, hours: int = 24) -> Dict:
        """Get cross-cryptocurrency analysis data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get latest portfolio snapshot
                cursor.execute("""
                    SELECT ethereum_balance, bitcoin_balance, ethereum_value_usd, bitcoin_value_usd,
                           total_value_usd, allocation_eth_percent, allocation_btc_percent
                    FROM portfolio_snapshots 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                
                portfolio_row = cursor.fetchone()
                
                # Get market data trends
                cursor.execute("""
                    SELECT cryptocurrency, AVG(price_usd) as avg_price, COUNT(*) as data_points
                    FROM market_snapshots 
                    WHERE timestamp >= datetime('now', '-{} hours')
                    GROUP BY cryptocurrency
                """.format(hours))
                
                market_rows = cursor.fetchall()
                
                return {
                    'portfolio': {
                        'ethereum_balance': portfolio_row[0] if portfolio_row else 0,
                        'bitcoin_balance': portfolio_row[1] if portfolio_row else 0,
                        'ethereum_value_usd': portfolio_row[2] if portfolio_row else 0,
                        'bitcoin_value_usd': portfolio_row[3] if portfolio_row else 0,
                        'total_value_usd': portfolio_row[4] if portfolio_row else 0,
                        'allocation_eth_percent': portfolio_row[5] if portfolio_row else 0,
                        'allocation_btc_percent': portfolio_row[6] if portfolio_row else 0
                    },
                    'market_trends': {
                        row[0]: {'avg_price': row[1], 'data_points': row[2]} 
                        for row in market_rows
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get cross-crypto analysis: {e}")
            return {'portfolio': {}, 'market_trends': {}}
