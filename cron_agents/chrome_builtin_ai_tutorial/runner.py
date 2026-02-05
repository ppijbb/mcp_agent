"""
Chrome Built-in AI(Gemini Nano) 작업을 Playwright로 실행하는 러너.

다른 cron agent에서 이 모듈을 import해 사용할 수 있습니다.
예: cron → (다른 agent) → run_builtin_ai_task("prompt", {"text": "..."})

주의: Built-in AI는 브라우저·Chrome 환경에 의존하므로, headless나 서버 환경에서는
동작하지 않을 수 있습니다. 호출 측에서 실패 시 fallback을 두는 것을 권장합니다.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

_TUTORIAL_ROOT = Path(__file__).resolve().parent
_DEMO_DIR = _TUTORIAL_ROOT / "demo"
_DEFAULT_PORT = 8766  # run_demo_via_browser와 구분

# 데모 페이지에서 사용하는 로딩 문구 (이게 사라질 때까지 대기)
_LOADING_PHRASES = ("처리 중", "요약 중", "번역 중", "감지 중", "Download")


async def run_builtin_ai_task(
    task_type: str,
    input_data: dict[str, Any],
    *,
    base_url: str | None = None,
    timeout_seconds: float = 90.0,
    headless: bool = False,
    serve: bool = True,
    port: int = _DEFAULT_PORT,
) -> dict[str, Any]:
    """
    Chrome Built-in AI 데모 페이지를 Playwright로 열고, 지정한 작업을 실행한 뒤 결과를 반환합니다.

    다른 agent에서 cron 등으로 호출할 때 사용하세요.

    Args:
        task_type: "prompt" | "summarize" | "translate" | "detect_language"
        input_data: 작업별 입력
            - prompt: {"text": "질문 문자열"}
            - summarize: {"text": "요약할 긴 텍스트"}
            - translate: {"text": "번역할 텍스트", "target_language": "en"}
            - detect_language: {"text": "언어 감지할 텍스트"}
        base_url: 데모 페이지 URL (예: "http://127.0.0.1:8766"). None이면 serve=True일 때 로컬 서버 사용.
        timeout_seconds: AI 응답 대기 시간(초).
        headless: True면 headless 브라우저 (Built-in AI는 보통 동작하지 않음).
        serve: base_url이 None일 때 데모 디렉터리에서 HTTP 서버를 띄울지 여부.
        port: serve=True일 때 사용할 포트.

    Returns:
        {"success": bool, "result": str | None, "error": str | None}
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {
            "success": False,
            "result": None,
            "error": "playwright 미설치. pip install playwright && playwright install chrome",
        }

    if not _DEMO_DIR.is_dir():
        return {
            "success": False,
            "result": None,
            "error": f"데모 디렉터리 없음: {_DEMO_DIR}",
        }

    server_process = None
    url = base_url
    if url is None and serve:
        url = f"http://127.0.0.1:{port}"
        server_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port)],
            cwd=str(_DEMO_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.sleep(1.2)
    elif url is None:
        return {"success": False, "result": None, "error": "base_url 또는 serve=True 필요"}

    exit_result: dict[str, Any] = {"success": False, "result": None, "error": None}

    try:
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=headless, channel="chrome")
            except Exception as e:
                LOG.warning("Chrome 채널 실패, Chromium 사용: %s", e)
                browser = await p.chromium.launch(headless=headless)

            try:
                context = await browser.new_context()
                page = await context.new_page()
                await page.goto(url, timeout=15000)
                await page.wait_for_load_state("networkidle", timeout=5000)

                # 사용자 활성화 유사 동작: body 클릭
                await page.click("body", timeout=2000)
                await asyncio.sleep(0.3)

                out_selector, loading_done = await _run_task_on_page(
                    page, task_type, input_data, timeout_seconds
                )
                if out_selector is None:
                    exit_result["error"] = loading_done
                    return exit_result

                text = await page.text_content(out_selector)
                text = (text or "").strip()
                if text and not any(phrase in text for phrase in _LOADING_PHRASES):
                    if text.startswith("오류:"):
                        exit_result["error"] = text
                    else:
                        exit_result["success"] = True
                        exit_result["result"] = text
                else:
                    exit_result["error"] = f"응답 없음 또는 타임아웃 (수신: {text!r})"
            finally:
                await browser.close()
    except Exception as e:
        exit_result["error"] = str(e)
        LOG.exception("run_builtin_ai_task 실패")
    finally:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=5)

    return exit_result


async def _run_task_on_page(
    page: Any,
    task_type: str,
    input_data: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str | None, str]:
    """
    페이지에서 작업을 실행하고, 결과가 찍힌 output selector를 반환.
    실패 시 (None, error_message) 반환.
    """
    task_type = task_type.strip().lower()
    if task_type == "prompt":
        inp = page.locator("#promptInput")
        btn = page.locator("#promptBtn")
        out_sel = "#promptOut"
    elif task_type == "summarize":
        inp = page.locator("#summarizeInput")
        btn = page.locator("#summarizeBtn")
        out_sel = "#summarizeOut"
    elif task_type == "translate":
        inp = page.locator("#translateInput")
        await page.fill("#translateTarget", input_data.get("target_language", "en"))
        btn = page.locator("#translateBtn")
        out_sel = "#translateOut"
    elif task_type == "detect_language":
        inp = page.locator("#translateInput")
        btn = page.locator("#detectBtn")
        out_sel = "#translateOut"
    else:
        return None, f"지원하지 않는 task_type: {task_type}"

    text = input_data.get("text") or input_data.get("input") or ""
    if not text:
        return None, "input_data에 'text' 또는 'input' 필요"

    await inp.fill("")
    await inp.fill(text)
    await asyncio.sleep(0.2)
    await btn.click()
    await asyncio.sleep(0.5)

    timeout_ms = int(timeout_seconds * 1000)
    out_loc = page.locator(out_sel)
    try:
        await out_loc.wait_for(
            state="visible",
            timeout=timeout_ms,
        )
        # 로딩 문구가 사라지고 실제 결과가 나올 때까지 폴링
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_seconds:
            content = await out_loc.text_content()
            content = (content or "").strip()
            if content and not any(phrase in content for phrase in _LOADING_PHRASES):
                return out_sel, ""
            await asyncio.sleep(0.8)
        return None, "결과 대기 타임아웃"
    except Exception as e:
        return None, str(e)


def run_builtin_ai_task_sync(
    task_type: str,
    input_data: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """run_builtin_ai_task의 동기 래퍼. 다른 agent에서 asyncio 없이 호출할 때 사용."""
    return asyncio.run(run_builtin_ai_task(task_type, input_data, **kwargs))
