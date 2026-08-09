# Tests

- **Pytest suite**: `core/`, `test_*.py` — run from repo root: `pytest tests/` or `python -m pytest tests/`
- **Security module**: `test_security_improvements.py` — validates crypto key handling, encryption round-trips, and missing-key failures.
- **Browser automation**: `test_browser_playwright.py` — Playwright-based browser interaction checks.
