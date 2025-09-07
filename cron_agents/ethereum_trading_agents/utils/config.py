"""
Configuration for Ethereum Trading Agents
Enhanced with security features and LangGraph integration
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    
    # Ethereum Configuration - NO FALLBACKS
    ETHEREUM_RPC_URL = os.getenv('ETHEREUM_RPC_URL')
    ETHEREUM_ADDRESS = os.getenv('ETHEREUM_ADDRESS')
    
    # Security Configuration - NO FALLBACKS
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
    HARDWARE_WALLET_ENABLED = os.getenv('HARDWARE_WALLET_ENABLED')
    HARDWARE_WALLET_PATH = os.getenv('HARDWARE_WALLET_PATH')
    
    # Encrypted private key (if not using hardware wallet)
    _encrypted_private_key = os.getenv('ETHEREUM_PRIVATE_KEY_ENCRYPTED')
    
    # Trading Configuration - NO FALLBACKS
    MIN_TRADE_AMOUNT_ETH = os.getenv('MIN_TRADE_AMOUNT_ETH')
    MAX_TRADE_AMOUNT_ETH = os.getenv('MAX_TRADE_AMOUNT_ETH')
    STOP_LOSS_PERCENT = os.getenv('STOP_LOSS_PERCENT')
    TAKE_PROFIT_PERCENT = os.getenv('TAKE_PROFIT_PERCENT')
    
    # Database Configuration - NO FALLBACKS
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    # MCP Server URLs - NO FALLBACKS
    MCP_ETHEREUM_TRADING_URL = os.getenv('MCP_ETHEREUM_TRADING_URL')
    MCP_MARKET_DATA_URL = os.getenv('MCP_MARKET_DATA_URL')
    
    # Agent Configuration - Fixed values only
    AGENT_EXECUTION_INTERVAL_MINUTES = 5
    MAX_CONCURRENT_AGENTS = 3
    
    # Risk Management - NO FALLBACKS
    MAX_DAILY_TRADES = os.getenv('MAX_DAILY_TRADES')
    MAX_DAILY_LOSS_ETH = os.getenv('MAX_DAILY_LOSS_ETH')
    
    # Logging - NO FALLBACKS
    LOG_LEVEL = os.getenv('LOG_LEVEL')
    LOG_FILE = os.getenv('LOG_FILE')
    
    @classmethod
    def get_private_key(cls) -> str:
        """Get decrypted private key with security validation"""
        if cls.HARDWARE_WALLET_ENABLED:
            return cls._get_hardware_wallet_key()
        else:
            return cls._decrypt_private_key()
    
    @classmethod
    def _get_hardware_wallet_key(cls) -> str:
        """Get private key from hardware wallet"""
        try:
            import ledgereth
            # Hardware wallet integration
            if not cls.HARDWARE_WALLET_PATH:
                raise ValueError("Hardware wallet path not configured")
            
            # This would integrate with actual hardware wallet
            # For now, return a placeholder
            raise NotImplementedError("Hardware wallet integration not implemented yet")
        except ImportError:
            raise ImportError("ledgereth package required for hardware wallet support")
    
    @classmethod
    def _decrypt_private_key(cls) -> str:
        """Decrypt private key using encryption key"""
        if not cls._encrypted_private_key:
            raise ValueError("No encrypted private key found")
        
        if not cls.ENCRYPTION_KEY:
            raise ValueError("Encryption key not configured")
        
        try:
            # Derive key from password
            password = cls.ENCRYPTION_KEY.encode()
            salt = b'ethereum_trading_salt'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Decrypt private key
            fernet = Fernet(key)
            decrypted_key = fernet.decrypt(cls._encrypted_private_key.encode())
            return decrypted_key.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt private key: {e}")
    
    @classmethod
    def encrypt_private_key(cls, private_key: str, encryption_key: str) -> str:
        """Encrypt private key for secure storage"""
        password = encryption_key.encode()
        salt = b'ethereum_trading_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        fernet = Fernet(key)
        encrypted_key = fernet.encrypt(private_key.encode())
        return encrypted_key.decode()
    
    @classmethod
    def validate(cls):
        """Validate ALL required configuration - NO FALLBACKS ALLOWED"""
        required_vars = [
            'GEMINI_API_KEY',
            'ETHEREUM_RPC_URL',
            'ETHEREUM_ADDRESS',
            'DATABASE_URL',
            'MCP_ETHEREUM_TRADING_URL',
            'MCP_MARKET_DATA_URL',
            'MIN_TRADE_AMOUNT_ETH',
            'MAX_TRADE_AMOUNT_ETH',
            'STOP_LOSS_PERCENT',
            'TAKE_PROFIT_PERCENT',
            'MAX_DAILY_TRADES',
            'MAX_DAILY_LOSS_ETH',
            'LOG_LEVEL',
            'LOG_FILE'
        ]
        
        # Security validation - NO FALLBACKS
        if not cls.HARDWARE_WALLET_ENABLED:
            if not cls._encrypted_private_key:
                raise ValueError("ETHEREUM_PRIVATE_KEY_ENCRYPTED is required when hardware wallet is disabled")
            if not cls.ENCRYPTION_KEY:
                raise ValueError("ENCRYPTION_KEY is required for private key decryption")
        else:
            if not cls.HARDWARE_WALLET_PATH:
                raise ValueError("HARDWARE_WALLET_PATH is required when hardware wallet is enabled")
        
        # Check all required variables
        missing_vars = []
        for var in required_vars:
            value = getattr(cls, var)
            if value is None or value == '':
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Type validation for numeric values
        try:
            float(cls.MIN_TRADE_AMOUNT_ETH)
            float(cls.MAX_TRADE_AMOUNT_ETH)
            float(cls.STOP_LOSS_PERCENT)
            float(cls.TAKE_PROFIT_PERCENT)
            int(cls.MAX_DAILY_TRADES)
            float(cls.MAX_DAILY_LOSS_ETH)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric configuration values: {e}")
        
        return True
