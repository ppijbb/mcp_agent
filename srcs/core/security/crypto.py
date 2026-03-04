"""
Cryptography utilities for MCP Agent system.

Provides encryption and decryption functionality using Fernet (AES 128-bit)
for securing sensitive configuration files and data.

Functions:
    validate_encryption_key: Validate Fernet encryption key format
    get_encryption_key: Retrieve and validate encryption key from environment
    get_cipher_suite: Create Fernet encryption object
    encrypt_file: Encrypt a file and save to output path
    decrypt_file_content: Decrypt file content and return as bytes
    decrypt_file: Decrypt an encrypted file and save it
    generate_key: Generate a new encryption key
"""

import os
from cryptography.fernet import Fernet, InvalidToken
from typing import Optional


def validate_encryption_key(key: str) -> bool:
    """
    Validate encryption key format and strength.
    
    Args:
        key: Encryption key string to validate
        
    Returns:
        bool: True if key is valid Fernet key, False otherwise
    """
    if not key:
        return False
    try:
        Fernet(key.encode() if isinstance(key, str) else key)
        return True
    except Exception:
        return False


def get_encryption_key() -> str:
    """
    Get and validate encryption key from environment.
    
    Retrieves MCP_SECRET_KEY from environment variables and validates it.
    
    Returns:
        str: Validated encryption key
        
    Raises:
        ValueError: If key is not set or invalid
    """
    key = os.getenv("MCP_SECRET_KEY")
    if not key:
        raise ValueError("MCP_SECRET_KEY environment variable is not set. Cannot use encryption.")
    
    if not validate_encryption_key(key):
        raise ValueError("MCP_SECRET_KEY is invalid. Key must be a 32-byte base64-encoded string.")
    
    return key


ENCRYPTION_KEY = None  # Will be loaded on demand


def get_cipher_suite() -> Fernet:
    """
    Create Fernet encryption object using key from environment variables.
    
    Returns:
        Fernet: Configured Fernet cipher suite
        
    Raises:
        ValueError: If encryption key is not set or invalid
    """
    try:
        key = get_encryption_key()
    except ValueError:
        raise ValueError(
            "MCP_SECRET_KEY environment variable is not set. "
            "Set it to enable encryption functionality."
        )
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except (ValueError, TypeError) as e:
        raise ValueError(f"MCP_SECRET_KEY is invalid: {e}")


def encrypt_file(file_path: str, output_path: Optional[str] = None) -> str:
    """Encrypt a file and save to output path.
    
    Args:
        file_path: Path to the file to encrypt
        output_path: Optional output path (defaults to file_path + '.enc')
        
    Returns:
        Path to the encrypted file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        EncryptionError: If encryption fails
    """
    if not output_path:
        output_path = f"{file_path}.enc"

    try:
        cipher = get_cipher_suite()

        with open(file_path, "rb") as f:
            plaintext = f.read()

        encrypted_data = cipher.encrypt(plaintext)

        with open(output_path, "wb") as f:
            f.write(encrypted_data)

        print(f"Successfully encrypted file: {file_path} -> {output_path}")
        return output_path
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except Exception as e:
        # Deferred import to avoid circular dependency
        try:
            from srcs.core.errors import EncryptionError
            raise EncryptionError(f"Failed to encrypt file: {str(e)}")
        except ImportError:
            # Fallback if import fails
            raise RuntimeError(f"Failed to encrypt file: {str(e)}")


def decrypt_file_content(encrypted_path: str) -> bytes:
    """
    Decrypt the content of an encrypted file and return as bytes.
    
    Args:
        encrypted_path: Path to the encrypted file
        
    Returns:
        bytes: Decrypted file content
        
    Raises:
        ValueError: If decryption fails due to invalid key or corrupted file
    """
    cipher = get_cipher_suite()

    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()

    try:
        decrypted_data = cipher.decrypt(encrypted_data)
        return decrypted_data
    except InvalidToken:
        raise ValueError("Cannot decrypt file. The key may be incorrect or the file is corrupted.")


def decrypt_file(encrypted_path: str, output_path: str | None = None) -> None:
    """
    Decrypt an encrypted file and save it.
    
    Args:
        encrypted_path: Path to the encrypted file
        output_path: Optional output path (defaults to encrypted_path with .enc removed)
        
    Raises:
        ValueError: If output_path not specified and file doesn't end with .enc
    """
    if not output_path:
        if not encrypted_path.endswith(".enc"):
            raise ValueError("Output file path must be specified.")
        output_path = encrypted_path[:-4]  # Remove .enc extension

    decrypted_content = decrypt_file_content(encrypted_path)

    with open(output_path, "wb") as f:
        f.write(decrypted_content)

    print(f"Successfully decrypted file: {encrypted_path} -> {output_path}")


def generate_key() -> str:
    """
    Generate a new encryption key.
    
    Returns:
        str: Newly generated Fernet key
        
    Note:
        Store the generated key in MCP_SECRET_KEY environment variable
    """
    key = Fernet.generate_key()
    key_str = key.decode()
    print("New encryption key generated. Store this in MCP_SECRET_KEY environment variable:")
    print(f"   {key_str}")
    return key_str
