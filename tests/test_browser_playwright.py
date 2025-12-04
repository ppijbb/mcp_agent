#!/usr/bin/env python3
"""
Playwright 브라우저 테스트 스크립트
페이지를 열고 콘텐츠를 확인합니다.
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_playwright_browser(url: str = "https://www.google.com"):
    """Playwright를 사용하여 브라우저를 열고 페이지를 확인합니다."""
    
    try:
        from playwright.async_api import async_playwright
        
        print(f"🌐 Playwright 브라우저 시작 중...")
        print(f"📄 URL: {url}")
        
        async with async_playwright() as p:
            # 브라우저 실행 (headless=False로 설정하여 브라우저 창 표시)
            print("🚀 Chromium 브라우저 실행 중...")
            browser = await p.chromium.launch(
                headless=False,  # 브라우저 창 표시
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            # 새 페이지 생성
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            page = await context.new_page()
            
            # 페이지 이동
            print(f"📥 페이지 로딩 중: {url}")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 페이지 정보 확인
            title = await page.title()
            url_actual = page.url
            
            print(f"\n✅ 페이지 로드 완료!")
            print(f"📌 제목: {title}")
            print(f"🔗 실제 URL: {url_actual}")
            
            # 페이지 콘텐츠 일부 추출
            content = await page.content()
            print(f"📄 페이지 크기: {len(content)} bytes")
            
            # 텍스트 콘텐츠 추출
            text_content = await page.inner_text('body')
            print(f"📝 텍스트 콘텐츠 길이: {len(text_content)} characters")
            print(f"\n📋 페이지 텍스트 미리보기 (처음 500자):")
            print("-" * 80)
            print(text_content[:500])
            print("-" * 80)
            
            # 스크린샷 저장
            screenshot_path = project_root / "browser_screenshot.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"\n📸 스크린샷 저장: {screenshot_path}")
            
            # 5초 대기 (사용자가 브라우저를 확인할 수 있도록)
            print("\n⏳ 5초 후 브라우저를 닫습니다...")
            await asyncio.sleep(5)
            
            # 브라우저 닫기
            await browser.close()
            print("✅ 브라우저 종료 완료")
            
            return {
                "success": True,
                "title": title,
                "url": url_actual,
                "content_length": len(content),
                "text_length": len(text_content),
                "screenshot": str(screenshot_path)
            }
            
    except ImportError:
        print("❌ Playwright가 설치되지 않았습니다.")
        print("설치 방법: pip install playwright")
        print("브라우저 설치: playwright install chromium")
        return {"success": False, "error": "Playwright not installed"}
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_browser_manager():
    """BrowserManager를 사용하여 페이지를 확인합니다."""
    
    try:
        from src.automation.browser_manager import BrowserManager
        
        print("\n" + "=" * 80)
        print("🔧 BrowserManager 테스트")
        print("=" * 80)
        
        browser_manager = BrowserManager()
        
        # 브라우저 초기화
        print("🚀 브라우저 초기화 중...")
        initialized = await browser_manager.initialize_browser()
        
        if not initialized:
            print("❌ 브라우저 초기화 실패")
            return {"success": False, "error": "Browser initialization failed"}
        
        print("✅ 브라우저 초기화 완료")
        
        # 페이지 탐색 및 콘텐츠 추출
        url = "https://www.google.com"
        print(f"\n📥 페이지 탐색 중: {url}")
        
        result = await browser_manager.navigate_and_extract(
            url=url,
            extraction_goal="extract_all_content"
        )
        
        if result.get("success"):
            print("✅ 콘텐츠 추출 완료")
            print(f"📄 추출된 콘텐츠 길이: {len(result.get('content', ''))} characters")
            print(f"\n📋 콘텐츠 미리보기 (처음 500자):")
            print("-" * 80)
            print(result.get('content', '')[:500])
            print("-" * 80)
        else:
            print(f"❌ 콘텐츠 추출 실패: {result.get('error', 'Unknown error')}")
        
        # 정리
        await browser_manager.cleanup()
        print("\n✅ BrowserManager 정리 완료")
        
        return result
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def main():
    """메인 함수"""
    
    print("=" * 80)
    print("🧪 Playwright 브라우저 테스트")
    print("=" * 80)
    
    # URL 인자 확인
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.google.com"
    
    # 1. 직접 Playwright 테스트
    print("\n[테스트 1] 직접 Playwright 사용")
    result1 = await test_playwright_browser(url)
    
    # 2. BrowserManager 테스트
    print("\n[테스트 2] BrowserManager 사용")
    result2 = await test_browser_manager()
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)
    print(f"테스트 1 (직접 Playwright): {'✅ 성공' if result1.get('success') else '❌ 실패'}")
    print(f"테스트 2 (BrowserManager): {'✅ 성공' if result2.get('success') else '❌ 실패'}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

