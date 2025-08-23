"""
Configuration for Ethereum Trading Agents
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    
    # Ethereum Configuration
    ETHEREUM_RPC_URL = os.getenv('ETHEREUM_RPC_URL', 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID')
    ETHEREUM_PRIVATE_KEY = os.getenv('ETHEREUM_PRIVATE_KEY')
    ETHEREUM_ADDRESS = os.getenv('ETHEREUM_ADDRESS')
    
    # Trading Configuration
    MIN_TRADE_AMOUNT_ETH = float(os.getenv('MIN_TRADE_AMOUNT_ETH', '0.01'))
    MAX_TRADE_AMOUNT_ETH = float(os.getenv('MAX_TRADE_AMOUNT_ETH', '1.0'))
    STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', '5.0'))
    TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', '10.0'))
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///ethereum_trading.db')
    
    # MCP Server URLs
    MCP_ETHEREUM_TRADING_URL = os.getenv('MCP_ETHEREUM_TRADING_URL', 'http://localhost:3005')
    MCP_MARKET_DATA_URL = os.getenv('MCP_MARKET_DATA_URL', 'http://localhost:3006')
    
    # Agent Configuration
    AGENT_EXECUTION_INTERVAL_MINUTES = 5
    MAX_CONCURRENT_AGENTS = 3
    
    # Risk Management
    MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '10'))
    MAX_DAILY_LOSS_ETH = float(os.getenv('MAX_DAILY_LOSS_ETH', '0.1'))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'ethereum_trading.log')
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_vars = [
            'GEMINI_API_KEY',
            'ETHEREUM_PRIVATE_KEY',
            'ETHEREUM_ADDRESS'
        ]
        
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True
