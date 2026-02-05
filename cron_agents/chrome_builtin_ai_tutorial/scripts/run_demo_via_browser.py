#!/usr/bin/env python3
"""
실험용: Playwright로 Chrome을 띄워 Built-in AI 데모 페이지를 엽니다.

주의:
- Chrome 내장 AI(Gemini Nano)는 브라우저 전용 API이므로, cron 등 백그라운드에서
  이 스크립트를 돌려도 환경(Chrome 경로, 플래그, 사용자 활성화, headless 미지원 등)에 따라
  API가 동작하지 않을 수 있습니다.
- cron 연동 시 "가능하면 시도해 보는" 수준으로만 사용하고, 실패 시 대체 수단을 두세요.
- 데모를 수동으로 사용하려면 README의 '로컬 브라우저에서 데모 실행' 절차를 따르세요.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOG = logging.getLogger(__name__)

# 프로젝트 루트: cron_agents/chrome_builtin_ai_tutorial/scripts -> demo/
SCRIPT_DIR = Path(__file__).resolve().parent
TUTORIAL_ROOT = SCRIPT_DIR.parent
DEMO_DIR = TUTORIAL_ROOT / "demo"
DEFAULT_PORT = 8765


async def run_with_playwright(headless: bool, port: int, timeout_seconds: float) -> int:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        LOG.error("playwright가 필요합니다: pip install playwright && playwright install chromium")
        return 2

    url = f"http://127.0.0.1:{port}"
    LOG.info("데모 URL: %s (서버가 이미 떠 있어야 함)", url)
    LOG.info("headless=%s (Built-in AI는 headless에서 동작하지 않을 수 있음)", headless)

    exit_code = 0
    async with async_playwright() as p:
        # Chromium으로 열면 Gemini Nano가 없을 수 있음. channel="chrome" 이면 시스템 Chrome 사용.
        try:
            browser = await p.chromium.launch(headless=headless, channel="chrome")
        except Exception as e:
            LOG.warning("Chrome 채널 실패, Chromium 사용: %s", e)
            browser = await p.chromium.launch(headless=headless)

        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, timeout=int(timeout_seconds * 1000))
            LOG.info("페이지 로드 완료. 브라우저에서 직접 조작하세요.")
            await asyncio.sleep(max(0, timeout_seconds - 2))
        except Exception as e:
            LOG.error("페이지 로드 실패: %s", e)
            exit_code = 1
        finally:
            await browser.close()

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chrome Built-in AI 데모를 브라우저에서 엽니다 (실험용)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless 모드 (Built-in AI는 보통 동작하지 않음)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"로컬 서버 포트 (기본: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="서버를 띄우지 않음 (이미 다른 터미널에서 http.server 실행 중일 때)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="페이지 대기 시간(초)",
    )
    args = parser.parse_args()

    if not DEMO_DIR.is_dir():
        LOG.error("데모 디렉터리를 찾을 수 없음: %s", DEMO_DIR)
        return 1

    server_process = None
    if not args.no_serve:
        LOG.info("데모 디렉터리에서 HTTP 서버 시작 (포트 %s)", args.port)
        server_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(args.port)],
            cwd=str(DEMO_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            import time
            time.sleep(1)
        except Exception:
            pass

    try:
        exit_code = asyncio.run(
            run_with_playwright(args.headless, args.port, args.timeout)
        )
        return exit_code
    finally:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
