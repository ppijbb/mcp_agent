"""
Playwright browser tests.
"""

import pytest


@pytest.mark.asyncio
async def test_playwright_browser_loads_page():
    """Verify Playwright can launch a headless browser and load a page."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        pytest.skip("playwright not installed")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()
        await page.goto("https://www.google.com", wait_until="networkidle", timeout=30000)

        title = await page.title()
        content = await page.content()

        assert "Google" in title
        assert len(content) > 0

        await browser.close()


@pytest.mark.asyncio
async def test_playwright_screenshot():
    """Verify Playwright can take a screenshot."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        pytest.skip("playwright not installed")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = await browser.new_page()
        await page.goto("https://www.google.com", wait_until="networkidle", timeout=30000)

        screenshot = await page.screenshot(full_page=False)
        assert len(screenshot) > 0

        await browser.close()
