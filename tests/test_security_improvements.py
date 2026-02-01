"""
Test suite for MCP Agent security improvements.
"""
import os
import tempfile
import unittest
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.core.security.crypto import (
    validate_encryption_key, 
    get_encryption_key, 
    encrypt_file, 
    decrypt_file_content,
    generate_key
)
from srcs.core.errors import EncryptionError, SecurityError


class TestSecurityImprovements(unittest.TestCase):
    """Test security module improvements."""
    
    def setUp(self):
        """Set up test fixtures."""
        from cryptography.fernet import Fernet
        self.test_key = Fernet.generate_key().decode()
        
    def test_validate_encryption_key_valid(self):
        """Test valid key validation."""
        self.assertTrue(validate_encryption_key(self.test_key))
        
    def test_validate_encryption_key_invalid(self):
        """Test invalid key validation."""
        self.assertFalse(validate_encryption_key("short"))
        self.assertFalse(validate_encryption_key(""))
        self.assertFalse(validate_encryption_key("invalid_key_123"))
        
    def test_generate_key(self):
        """Test key generation."""
        import io
        from contextlib import redirect_stdout
        
        # Capture print output
        f = io.StringIO()
        with redirect_stdout(f):
            key = generate_key()
        
        output = f.getvalue()
        self.assertIn("새로운 암호화 키가 생성되었습니다", output)
        self.assertIsNotNone(key)
        
    def test_encrypt_file_missing_key(self):
        """Test encryption fails without key."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write("test content")
            tmp_path = tmp.name
            
        # Remove key from environment
        original_key = os.environ.get("MCP_SECRET_KEY")
        if "MCP_SECRET_KEY" in os.environ:
            del os.environ["MCP_SECRET_KEY"]
            
        try:
            with self.assertRaises(EncryptionError) as context:
                encrypt_file(tmp_path)
                
            self.assertIn("Failed to encrypt file", str(context.exception))
            
        finally:
            # Restore key
            if original_key:
                os.environ["MCP_SECRET_KEY"] = original_key
            os.unlink(tmp_path)
            
    def test_encrypt_decrypt_roundtrip(self):
        """Test encrypt and decrypt roundtrip."""
        # Set valid key
        os.environ["MCP_SECRET_KEY"] = self.test_key
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            test_content = "This is test content for encryption"
            tmp.write(test_content)
            tmp_path = tmp.name
            
        encrypted_path = None
        try:
            # Encrypt file
            encrypted_path = encrypt_file(tmp_path)
            self.assertTrue(encrypted_path.endswith('.enc'))
            self.assertTrue(Path(encrypted_path).exists())
            
            # Decrypt content
            decrypted_content = decrypt_file_content(encrypted_path)
            self.assertEqual(decrypted_content.decode('utf-8'), test_content)
            
        finally:
            # Cleanup
            for path in [tmp_path, encrypted_path]:
                if path and Path(path).exists():
                    os.unlink(path)
                    

if __name__ == '__main__':
    unittest.main()