#!/usr/bin/env python3
"""
Streamlit App Runner for Most Hooking Business Strategy Agent

This script runs the Streamlit web interface for testing and interacting
with the business strategy agent system.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """필수 의존성 확인"""
    try:
        import streamlit
        import pandas
        import plotly
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requriements.txt")
        return False

def setup_environment():
    """환경 설정"""
    # 현재 디렉토리를 Python 경로에 추가
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # 환경 변수 설정
    os.environ['PYTHONPATH'] = str(current_dir)
    
    print(f"📁 Working directory: {current_dir}")
    return current_dir

def run_streamlit_app(port: int = 8501, debug: bool = False):
    """Streamlit 앱 실행"""
    app_file = "srcs/business_strategy_agents/streamlit_app.py"
    
    if not os.path.exists(app_file):
        print(f"❌ Streamlit app file not found: {app_file}")
        return False
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port", str(port),
        "--server.headless", "true" if not debug else "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    if debug:
        cmd.extend(["--logger.level", "debug"])
    
    try:
        print(f"🚀 Starting Streamlit app on port {port}...")
        print(f"🌐 Open your browser to: http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Streamlit 실행
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️ Streamlit server stopped")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run Streamlit: {e}")
        return False
    except FileNotFoundError:
        print("❌ Streamlit not installed. Please run: pip install streamlit")
        return False

def run_tests():
    """테스트 실행"""
    test_file = "srcs/business_strategy_agents/test_agents.py"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        print("🧪 Running basic tests...")
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, check=False)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Basic tests passed!")
            return True
        else:
            print(f"❌ Tests failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def run_pytest():
    """Pytest 실행"""
    try:
        print("🧪 Running comprehensive tests with pytest...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "srcs/business_strategy_agents/test_agents.py",
            "-v", "--tb=short"
        ], check=False)
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("❌ pytest not installed. Please run: pip install pytest")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Run Most Hooking Business Strategy Agent Streamlit App"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8501,
        help="Port to run Streamlit on (default: 8501)"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Run in debug mode"
    )
    parser.add_argument(
        "--test", "-t", action="store_true",
        help="Run tests instead of starting the app"
    )
    parser.add_argument(
        "--pytest", action="store_true",
        help="Run comprehensive tests with pytest"
    )
    parser.add_argument(
        "--check", "-c", action="store_true",
        help="Check dependencies and environment"
    )
    
    args = parser.parse_args()
    
    print("🎯 Most Hooking Business Strategy Agent")
    print("=" * 50)
    
    # 환경 설정
    current_dir = setup_environment()
    
    # 의존성 확인
    if not check_dependencies():
        return 1
    
    # 체크 모드
    if args.check:
        print("✅ Environment check completed successfully")
        return 0
    
    # 테스트 모드
    if args.test:
        if run_tests():
            return 0
        else:
            return 1
    
    # Pytest 모드
    if args.pytest:
        if run_pytest():
            return 0
        else:
            return 1
    
    # Streamlit 앱 실행
    try:
        if run_streamlit_app(args.port, args.debug):
            return 0
        else:
            return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 