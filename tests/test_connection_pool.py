"""
Regression tests for ImprovedConnectionPool lifecycle handling.

Covers the expired-connection leak fix: connections that exceed
max_idle_time must be disposed (not silently dropped) when a reuse
attempt pops them from the pool.
"""
import time

from srcs.common.connection_pool import ImprovedConnectionPool


class FakeConn:
    """Connection stub that tracks whether close() was called."""

    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def test_reuse_valid_connection():
    pool = ImprovedConnectionPool(pool_size=5, max_idle_time=300)
    try:
        conn = FakeConn("c1")
        assert pool.get_connection("m", "p", lambda: conn) is conn
        pool.return_connection("m", "p", conn)

        assert pool.get_connection("m", "p", lambda: FakeConn("c2")) is conn
        assert conn.closed is False

        stats = pool.get_stats()
        assert stats["connection_stats"]["p:m"]["reused"] == 1
        assert stats["connection_stats"]["p:m"]["expired"] == 0
    finally:
        pool.shutdown()


def test_expired_connection_is_disposed_on_reuse():
    pool = ImprovedConnectionPool(pool_size=5, max_idle_time=1)
    try:
        conn = FakeConn("expired")
        assert pool.get_connection("m", "p", lambda: conn) is conn
        pool.return_connection("m", "p", conn)
        assert conn.closed is False

        # Let the pooled connection exceed max_idle_time
        time.sleep(1.1)

        fresh = FakeConn("fresh")
        result = pool.get_connection("m", "p", lambda: fresh)

        # Expired connection must have been disposed, not leaked
        assert conn.closed is True
        assert result is fresh
        assert fresh.closed is False

        stats = pool.get_stats()
        assert stats["connection_stats"]["p:m"]["expired"] == 1
    finally:
        pool.shutdown()
