import argparse
import sys
import os

# 스크립트가 프로젝트 루트에서 실행되도록 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srcs.core.security.crypto import generate_key, encrypt_file, decrypt_file

def main():
    parser = argparse.ArgumentParser(description="""
    비밀 키 생성 및 설정 파일 암호화/복호화 관리 스크립트.
    
    사용법:
    1. 새로운 비밀 키 생성:
       python scripts/manage_secrets.py generate-key
       
    2. 설정 파일 암호화 (예: production.yaml -> production.yaml.enc):
       python scripts/manage_secrets.py encrypt --file configs/production.yaml
       
    3. 암호화된 설정 파일 복호화 (예: production.yaml.enc -> production.yaml):
       python scripts/manage_secrets.py decrypt --file configs/production.yaml.enc
    """)
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="실행할 명령어")

    # generate-key 명령어
    subparsers.add_parser("generate-key", help="새로운 Fernet 암호화 키를 생성합니다.")

    # encrypt 명령어
    encrypt_parser = subparsers.add_parser("encrypt", help="파일을 암호화합니다.")
    encrypt_parser.add_argument("--file", required=True, help="암호화할 파일 경로")
    encrypt_parser.add_argument("--output", help="암호화된 파일의 출력 경로 (기본값: [입력파일].enc)")

    # decrypt 명령어
    decrypt_parser = subparsers.add_parser("decrypt", help="파일을 복호화합니다.")
    decrypt_parser.add_argument("--file", required=True, help="복호화할 파일 경로 (.enc)")
    decrypt_parser.add_argument("--output", help="복호화된 파일의 출력 경로 (기본값: .enc 확장자 제거)")

    args = parser.parse_args()

    if args.command == "generate-key":
        generate_key()
    elif args.command == "encrypt":
        if not os.path.exists(args.file):
            print(f"❌ 오류: 파일이 존재하지 않습니다 - {args.file}")
            sys.exit(1)
        encrypt_file(args.file, args.output)
    elif args.command == "decrypt":
        if not os.path.exists(args.file):
            print(f"❌ 오류: 파일이 존재하지 않습니다 - {args.file}")
            sys.exit(1)
        decrypt_file(args.file, args.output)

if __name__ == "__main__":
    main() 