#!/usr/bin/env python3
"""
🏥 SEO Doctor 런처

사이트 응급처치 + 경쟁사 스파이 AI
모바일 친화적 Streamlit 앱

실행: python seo_doctor_launcher.py
"""

import sys
import os
import subprocess
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_dependencies():
    """필요한 패키지 확인 및 설치"""
    
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"🔧 필요한 패키지를 설치합니다: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} 설치 완료")
            except subprocess.CalledProcessError as e:
                print(f"❌ {package} 설치 실패: {e}")
                return False
    
    return True

def launch_seo_doctor():
    """SEO Doctor 앱 실행"""
    
    print("🏥 SEO Doctor 시작 중...")
    print("=" * 50)
    print("📱 모바일 친화적 SEO 응급처치 + 경쟁사 스파이")
    print("🎯 목표: MAU 10만+ 달성!")
    print("=" * 50)
    
    # 의존성 확인
    if not check_dependencies():
        print("❌ 의존성 설치 실패. 수동으로 설치해주세요:")
        print("pip install streamlit plotly pandas")
        return
    
    # Streamlit 앱 경로
    app_path = os.path.join(project_root, 'srcs', 'seo_doctor', 'seo_doctor_app.py')
    
    if not os.path.exists(app_path):
        print(f"❌ 앱 파일을 찾을 수 없습니다: {app_path}")
        return
    
    # Streamlit 설정
    streamlit_config = {
        '--server.port': '8502',
        '--server.headless': 'true',
        '--browser.gatherUsageStats': 'false',
        '--theme.base': 'light',
        '--theme.primaryColor': '#ff4757',
        '--theme.backgroundColor': '#ffffff',
        '--theme.secondaryBackgroundColor': '#f8f9fa'
    }
    
    # 실행 명령어 구성
    cmd = [sys.executable, '-m', 'streamlit', 'run', app_path]
    
    for key, value in streamlit_config.items():
        cmd.extend([key, value])
    
    print("🚀 SEO Doctor 실행 중...")
    print(f"📱 브라우저에서 http://localhost:8502 를 열어주세요")
    print("🔥 모바일에서도 접속 가능!")
    print("")
    print("종료하려면 Ctrl+C 를 눌러주세요")
    print("=" * 50)
    
    try:
        # Streamlit 앱 실행
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 SEO Doctor를 종료합니다.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 실행 중 오류: {e}")
        print("수동 실행: streamlit run srcs/seo_doctor/seo_doctor_app.py --server.port 8502")

def show_banner():
    """시작 배너"""
    banner = """
    
🏥🏥🏥 SEO DOCTOR 🏥🏥🏥

    ╔══════════════════════════════════╗
    ║        사이트 응급처치            ║
    ║             +                    ║  
    ║        경쟁사 스파이              ║
    ╠══════════════════════════════════╣
    ║  🎯 목표: MAU 10만+              ║
    ║  📱 모바일 친화적                ║
    ║  🚀 바이럴 기능 탑재             ║
    ╚══════════════════════════════════╝

    """
    print(banner)

def main():
    """메인 실행 함수"""
    
    show_banner()
    
    # 실행 환경 체크
    print("🔍 실행 환경 체크...")
    print(f"Python: {sys.version}")
    print(f"작업 디렉토리: {os.getcwd()}")
    print(f"프로젝트 루트: {project_root}")
    print("")
    
    # SEO Doctor 실행
    launch_seo_doctor()

if __name__ == "__main__":
    main() 