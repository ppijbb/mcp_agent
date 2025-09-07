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
    
    # Ethereum Configuration
    ETHEREUM_RPC_URL = os.getenv('ETHEREUM_RPC_URL', 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID')
    ETHEREUM_ADDRESS = os.getenv('ETHEREUM_ADDRESS')
    
    # Security Configuration
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
    HARDWARE_WALLET_ENABLED = os.getenv('HARDWARE_WALLET_ENABLED', 'false').lower() == 'true'
    HARDWARE_WALLET_PATH = os.getenv('HARDWARE_WALLET_PATH')
    
    # Encrypted private key (if not using hardware wallet)
    _encrypted_private_key = os.getenv('ETHEREUM_PRIVATE_KEY_ENCRYPTED')
    
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
        """Validate required configuration with security checks"""
        required_vars = [
            'GEMINI_API_KEY',
            'ETHEREUM_ADDRESS'
        ]
        
        # Security validation
        if not cls.HARDWARE_WALLET_ENABLED:
            if not cls._encrypted_private_key:
                raise ValueError("Either hardware wallet must be enabled or encrypted private key must be provided")
            if not cls.ENCRYPTION_KEY:
                raise ValueError("Encryption key required for private key decryption")
        else:
            if not cls.HARDWARE_WALLET_PATH:
                raise ValueError("Hardware wallet path required when hardware wallet is enabled")
        
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True
