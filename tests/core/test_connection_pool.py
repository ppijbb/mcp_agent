"""
Test suite for ImprovedConnectionPool resource lifecycle.
"""
import time
import pytest

from srcs.common.connection_pool import ImprovedConnectionPool


class _TrackedConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_connection_reuse():
    """Connections are reused while still valid."""
    pool = ImprovedConnectionPool(pool_size=5, max_idle_time=300, enable_monitoring=False)
    conn = pool.get_connection("m", "p", lambda: _TrackedConn())
    pool.return_connection("m", "p", conn)

    reused = pool.get_connection("m", "p", lambda: _TrackedConn())
    assert reused is conn
    pool.shutdown()


def test_expired_connection_is_disposed():
    """An expired pooled connection must be disposed, not leaked."""
    pool = ImprovedConnectionPool(pool_size=5, max_idle_time=1, enable_monitoring=False)
    created = []
    disposed = []

    def create():
        created.append(1)
        return _TrackedConn()

    conn = pool.get_connection("m", "p", create)
    pool.return_connection("m", "p", conn)

    original_dispose = pool._dispose_connection

    def tracked_dispose(connection, pool_key):
        disposed.append(connection)
        original_dispose(connection, pool_key)

    pool._dispose_connection = tracked_dispose

    time.sleep(1.2)  # exceed max_idle_time
    new_conn = pool.get_connection("m", "p", create)

    assert new_conn is not conn
    assert len(created) == 2
    assert disposed == [conn]
    assert conn.closed is True
    assert pool.connection_stats["p:m"]["expired"] == 1
    pool.shutdown()


def test_invalid_connection_is_not_returned():
    """Invalid connections are disposed instead of pooled."""
    pool = ImprovedConnectionPool(pool_size=5, max_idle_time=300, enable_monitoring=False)
    conn = pool.get_connection("m", "p", lambda: _TrackedConn())

    original_validate = pool._validate_connection
    pool._validate_connection = lambda c: False

    disposed = []
    original_dispose = pool._dispose_connection

    def tracked_dispose(connection, pool_key):
        disposed.append(connection)
        original_dispose(connection, pool_key)

    pool._dispose_connection = tracked_dispose
    pool.return_connection("m", "p", conn)

    assert disposed == [conn]
    assert conn.closed is True
    pool.shutdown()


def test_shutdown_disposes_pooled_connections():
    """Shutdown disposes all idle and active connections."""
    pool = ImprovedConnectionPool(pool_size=5, max_idle_time=300, enable_monitoring=False)
    idle = pool.get_connection("m", "p", lambda: _TrackedConn())
    active = pool.get_connection("m", "p", lambda: _TrackedConn())
    pool.return_connection("m", "p", idle)

    pool.shutdown()
    assert idle.closed is True
    assert active.closed is True
    assert not pool._active_connections["p:m"]
    assert not pool._pools["p:m"]
