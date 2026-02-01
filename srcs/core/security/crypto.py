import os
import re
from cryptography.fernet import Fernet, InvalidToken
from typing import Optional


def validate_encryption_key(key: str) -> bool:
    """Validate encryption key format and strength."""
    if not key:
        return False
    # Fernet key should be 32 bytes base64-encoded
    try:
        from cryptography.fernet import Fernet
        Fernet(key)  # This will raise an error if key is invalid
        return True
    except Exception:
        return False


def get_encryption_key() -> str:
    """Get and validate encryption key from environment."""
    key = os.getenv("MCP_SECRET_KEY")
    if not key:
        raise ValueError("MCP_SECRET_KEY 환경 변수가 설정되지 않았습니다. 암호화 기능을 사용할 수 없습니다.")
    
    if not validate_encryption_key(key):
        raise ValueError("MCP_SECRET_KEY가 유효하지 않습니다. 키는 32자 이상의 base64 문자열이어야 합니다.")
    
    return key


ENCRYPTION_KEY = None  # Will be loaded on demand


def get_cipher_suite():
    """환경 변수에서 키를 가져와 Fernet 암호화 객체를 생성합니다."""
    key = get_encryption_key()
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except (ValueError, TypeError) as e:
        raise ValueError(f"MCP_SECRET_KEY가 유효하지 않습니다: {e}")


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
    from srcs.core.errors import EncryptionError
    
    if not output_path:
        output_path = f"{file_path}.enc"

    try:
        cipher = get_cipher_suite()

        with open(file_path, "rb") as f:
            plaintext = f.read()

        encrypted_data = cipher.encrypt(plaintext)

        with open(output_path, "wb") as f:
            f.write(encrypted_data)

        print(f"✅ 파일이 성공적으로 암호화되었습니다: {file_path} -> {output_path}")
        return output_path
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except Exception as e:
        raise EncryptionError(f"Failed to encrypt file: {str(e)}")


def decrypt_file_content(encrypted_path: str) -> bytes:
    """암호화된 파일의 내용을 복호화하여 바이트로 반환합니다."""
    cipher = get_cipher_suite()

    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()

    try:
        decrypted_data = cipher.decrypt(encrypted_data)
        return decrypted_data
    except InvalidToken:
        raise ValueError("암호화된 파일을 복호화할 수 없습니다. 키가 잘못되었거나 파일이 손상되었습니다.")


def decrypt_file(encrypted_path: str, output_path: str | None = None):
    """암호화된 파일을 복호화하여 저장합니다."""
    if not output_path:
        if not encrypted_path.endswith(".enc"):
            raise ValueError("출력 파일 경로를 지정해야 합니다.")
        output_path = encrypted_path[:-4]  # .enc 확장자 제거

    decrypted_content = decrypt_file_content(encrypted_path)

    with open(output_path, "wb") as f:
        f.write(decrypted_content)

    print(f"✅ 파일이 성공적으로 복호화되었습니다: {encrypted_path} -> {output_path}")


def generate_key() -> str:
    """새로운 암호화 키를 생성합니다."""
    key = Fernet.generate_key()
    key_str = key.decode()
    print("🔑 새로운 암호화 키가 생성되었습니다. 이 키를 MCP_SECRET_KEY 환경 변수에 안전하게 저장하세요.")
    print(f"   {key_str}")
    return key_str
