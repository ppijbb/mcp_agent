"""
Account Manager for Secure Account Information Storage

Handles encrypted storage of account information for payouts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from cryptography.fernet import Fernet
import base64
import os

logger = logging.getLogger(__name__)


class AccountManager:
    """
    Secure account information manager with encryption.
    
    Features:
    - Encrypted storage of account details
    - Account validation
    - Connection testing
    """
    
    def __init__(self, accounts_file: Path, encryption_key: Optional[bytes] = None):
        """
        Initialize account manager.
        
        Args:
            accounts_file: Path to encrypted accounts JSON file
            encryption_key: Fernet encryption key (generates if None)
        """
        self.accounts_file = Path(accounts_file)
        self.accounts_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate or use provided encryption key
        if encryption_key is None:
            # Try to load from environment or generate new
            key_env = os.getenv('MONEY_MAKER_ENCRYPTION_KEY')
            if key_env:
                self.encryption_key = base64.urlsafe_b64decode(key_env.encode())
            else:
                self.encryption_key = Fernet.generate_key()
                logger.warning(
                    f"Generated new encryption key. Save this to MONEY_MAKER_ENCRYPTION_KEY: "
                    f"{base64.urlsafe_b64encode(self.encryption_key).decode()}"
                )
        else:
            self.encryption_key = encryption_key
        
        self.cipher = Fernet(self.encryption_key)
        self._accounts: Dict[str, Dict] = {}
        self._load_accounts()
    
    def _load_accounts(self):
        """Load encrypted accounts from file."""
        if self.accounts_file.exists():
            try:
                with open(self.accounts_file, 'rb') as f:
                    encrypted_data = f.read()
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    self._accounts = json.loads(decrypted_data.decode())
                logger.info(f"Loaded accounts from {self.accounts_file}")
            except Exception as e:
                logger.error(f"Failed to load accounts: {e}")
                self._accounts = {}
        else:
            self._accounts = {}
            logger.info(f"Accounts file not found, starting with empty accounts")
    
    def _save_accounts(self):
        """Save encrypted accounts to file."""
        try:
            data = json.dumps(self._accounts, indent=2).encode()
            encrypted_data = self.cipher.encrypt(data)
            
            with open(self.accounts_file, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"Saved accounts to {self.accounts_file}")
        except Exception as e:
            logger.error(f"Failed to save accounts: {e}")
            raise
    
    def set_payout_account(
        self,
        bank_name: str,
        account_number: str,
        routing_number: str,
        account_holder_name: Optional[str] = None,
        account_type: Optional[str] = None
    ):
        """
        Set payout account information.
        
        Args:
            bank_name: Bank name
            account_number: Account number
            routing_number: Routing number (for US banks)
            account_holder_name: Account holder name
            account_type: Account type (checking, savings, etc.)
        """
        self._accounts['payout_account'] = {
            'bank_name': bank_name,
            'account_number': account_number,
            'routing_number': routing_number,
            'account_holder_name': account_holder_name,
            'account_type': account_type
        }
        self._save_accounts()
        logger.info("Payout account information updated")
    
    def get_payout_account(self) -> Optional[Dict]:
        """Get payout account information."""
        return self._accounts.get('payout_account')
    
    def validate_payout_account(self) -> tuple[bool, Optional[str]]:
        """
        Validate payout account information.
        
        Returns:
            (is_valid, error_message)
        """
        account = self.get_payout_account()
        
        if not account:
            return False, "Payout account not configured"
        
        required_fields = ['bank_name', 'account_number', 'routing_number']
        for field in required_fields:
            if not account.get(field):
                return False, f"Missing required field: {field}"
        
        # Basic validation
        if len(account['account_number']) < 4:
            return False, "Account number too short"
        
        if len(account['routing_number']) != 9:
            return False, "Routing number must be 9 digits"
        
        return True, None
    
    def test_connection(self) -> tuple[bool, Optional[str]]:
        """
        Test account connection (placeholder for actual bank API integration).
        
        Returns:
            (success, error_message)
        """
        is_valid, error = self.validate_payout_account()
        if not is_valid:
            return False, error
        
        # TODO: Implement actual bank API connection test
        # For now, just validate the format
        logger.info("Account connection test passed (format validation only)")
        return True, None

