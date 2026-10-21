"""
Test suite for the shared infrastructure helpers.

Covers the connection pool lifecycle (shutdown must not block) and the
sensitive-data logging filter (secrets must be masked without corrupting
the log record).
"""
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from srcs.common.connection_pool import ImprovedConnectionPool
from srcs.common.logging_utils import OptimizedHTTPErrorFilter


class TestConnectionPoolShutdown:
    """The background cleanup thread must not delay shutdown()."""

    def test_shutdown_returns_quickly(self):
        pool = ImprovedConnectionPool(pool_size=2, max_idle_time=300)
        start = time.monotonic()
        pool.shutdown()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"shutdown() took {elapsed:.2f}s"

    def test_shutdown_is_idempotent(self):
        pool = ImprovedConnectionPool(pool_size=1)
        pool.shutdown()
        pool.shutdown()

    def test_pooled_connection_is_disposed(self):
        pool = ImprovedConnectionPool(pool_size=2)
        connection = pool.get_connection("m", "p", lambda: object())
        pool.return_connection("m", "p", connection)
        pool.shutdown()
        assert pool.get_stats()["total_pooled_connections"] == 0
        assert pool.get_stats()["total_active_connections"] == 0


class TestSensitiveDataFilter:
    """The filter must mask secrets without breaking log formatting."""

    @staticmethod
    def _render(msg, args=None):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, msg, args, None)
        OptimizedHTTPErrorFilter().filter(record)
        return record.getMessage()

    def test_masks_inline_secret(self):
        rendered = self._render("api_key=abcdef")
        assert "abcdef" not in rendered
        assert "api_key=" in rendered

    def test_masks_secret_passed_as_positional_arg(self):
        rendered = self._render("token=%s", ("supersecret123",))
        assert "supersecret123" not in rendered
        assert rendered.startswith("token=")

    def test_masks_secret_passed_as_mapping_arg(self):
        rendered = self._render("token=%(t)s", ({"t": "supersecret123"},))
        assert "supersecret123" not in rendered
        assert rendered.startswith("token=")

    def test_positional_placeholder_is_not_corrupted(self):
        """The %s placeholder must survive filtering."""
        rendered = self._render("value: %s", ("harmless",))
        assert rendered == "value: harmless"

    def test_mapping_placeholder_is_not_corrupted(self):
        rendered = self._render("value: %(v)s", ({"v": "harmless"},))
        assert rendered == "value: harmless"

    def test_secret_inside_argument_is_masked(self):
        rendered = self._render("payload: %s", ("password=hunter2",))
        assert "hunter2" not in rendered

    def test_non_sensitive_message_is_untouched(self):
        assert self._render("nothing to see here") == "nothing to see here"

    def test_filter_does_not_raise_on_malformed_record(self):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "%s", (), None)
        record.args = ("too", "many")
        OptimizedHTTPErrorFilter().filter(record)

    def test_long_messages_do_not_share_cache_entries(self):
        """Two long messages sharing a prefix must not be masked identically."""
        filt = OptimizedHTTPErrorFilter()
        prefix = "x" * 600
        first = logging.LogRecord("t", logging.INFO, __file__, 1, prefix + " token=AAA", None, None)
        second = logging.LogRecord("t", logging.INFO, __file__, 1, prefix + " token=BBB", None, None)
        filt.filter(first)
        filt.filter(second)
        assert "AAA" not in first.getMessage()
        assert "BBB" not in second.getMessage()
