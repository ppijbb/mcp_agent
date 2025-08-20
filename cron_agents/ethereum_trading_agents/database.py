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
                
                # Agent execution records
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        execution_time TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        input_data TEXT,
                        output_data TEXT,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Trading decisions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        decision_type TEXT NOT NULL,
                        decision_data TEXT NOT NULL,
                        market_conditions TEXT,
                        reasoning TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                # Market data snapshots
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        price_usd REAL,
                        price_change_24h REAL,
                        volume_24h REAL,
                        technical_indicators TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                # Risk management records
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS risk_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id INTEGER,
                        daily_trades_count INTEGER,
                        daily_loss_eth REAL,
                        risk_level TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (execution_id) REFERENCES agent_executions (id)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def record_agent_execution(self, agent_name: str, status: str, input_data: Dict = None, 
                              output_data: Dict = None, error_message: str = None) -> int:
        """Record agent execution and return execution ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO agent_executions (agent_name, execution_time, status, input_data, output_data, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    agent_name,
                    datetime.now().isoformat(),
                    status,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    error_message
                ))
                
                execution_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Recorded execution for agent {agent_name} with ID {execution_id}")
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
