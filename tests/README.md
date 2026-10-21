# Tests

- **Pytest suite**: `core/`, `test_*.py` — run from repo root: `pytest tests/` or `python -m pytest tests/`
- Config lives in `pytest.ini` at the repo root (`testpaths`, `pythonpath`, `asyncio_mode`).

## Layout

| Path | Covers |
| --- | --- |
| `core/agent/test_base.py` | `BaseAgent` initialization |
| `core/config/test_loader.py` | Config merge precedence (`base.yaml` vs. environment file) |
| `test_common_infrastructure.py` | Connection pool lifecycle, sensitive-data logging filter |
| `test_security_improvements.py` | Encryption / key validation helpers |
| `test_browser_playwright.py` | Browser automation (skipped by default) |

Browser automation tests need a real browser, a display and network access.
They are skipped unless `MCP_RUN_BROWSER_TESTS=1` is set:

```bash
MCP_RUN_BROWSER_TESTS=1 pytest tests/test_browser_playwright.py
```

Some modules require optional dependencies (`mcp-agent`, `playwright`, `streamlit`).
Install the full dependency set with `pip install -r requirements.txt` before
running the whole suite.
