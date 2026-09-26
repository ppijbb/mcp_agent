"""
Regression tests for ImprovedConnectionPool resource lifecycle.

Covers two defects:

1. A pooled connection that had been idle longer than ``max_idle_time`` was
   popped from the pool but never disposed, leaking the resources it held
   (sockets / file handles).
2. ``shutdown()`` joined the background cleanup thread while holding the pool
   lock, and the thread slept for a full ``cleanup_interval`` before it could
   observe the shutdown flag -- so every shutdown stalled for the join timeout.
"""

import time

import pytest

from srcs.common.connection_pool import ImprovedConnectionPool


class FakeConnection:
    """Minimal connection that records whether it was released."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def pool():
    p = ImprovedConnectionPool(pool_size=2, max_idle_time=300)
    yield p
    p.shutdown()


# --------------------------------------------------------------------------
# Expired connections must be disposed, not dropped
# --------------------------------------------------------------------------

def test_expired_connection_is_disposed_on_checkout():
    """A stale pooled connection must be closed, not silently dropped."""
    p = ImprovedConnectionPool(pool_size=2, max_idle_time=0)
    stale = FakeConnection()
    p.return_connection("m", "prov", stale)

    p.get_connection("m", "prov", FakeConnection)

    assert stale.closed is True
    assert p.connection_stats["prov:m"]["expired"] == 1
    p.shutdown()


def test_valid_connection_is_still_reused():
    """A non-expired connection must be reused rather than recreated."""
    p = ImprovedConnectionPool(pool_size=2, max_idle_time=300)
    conn = FakeConnection()
    p.return_connection("m", "prov", conn)

    got = p.get_connection("m", "prov", FakeConnection)

    assert got is conn
    assert conn.closed is False
    assert p.connection_stats["prov:m"]["reused"] == 1
    p.shutdown()


def test_repeated_checkouts_do_not_accumulate_leaked_connections():
    """Each expired checkout disposes its connection and counts it."""
    p = ImprovedConnectionPool(pool_size=2, max_idle_time=0)
    for _ in range(5):
        conn = FakeConnection()
        p.return_connection("m", "prov", conn)
        p.get_connection("m", "prov", FakeConnection)
        assert conn.closed is True

    assert p.connection_stats["prov:m"]["expired"] == 5
    p.shutdown()


def test_expiry_sweep_disposes_stale_and_keeps_fresh():
    """The periodic sweep must close aged-out connections and retain fresh ones."""
    p = ImprovedConnectionPool(pool_size=5, max_idle_time=300)
    aged, fresh = FakeConnection(), FakeConnection()
    p.return_connection("m", "prov", aged)
    p.return_connection("m", "prov", fresh)

    stale_time = time.time() - 10_000
    p._pools["prov:m"][0]["last_used"] = stale_time
    p._pools["prov:m"][0]["created_at"] = stale_time

    p._cleanup_old_connections()

    assert aged.closed is True
    assert fresh.closed is False
    assert p.connection_stats["prov:m"]["expired"] == 1
    p.shutdown()


def test_idle_time_is_measured_from_last_use():
    """max_idle_time applies to idle time, not total connection lifetime."""
    p = ImprovedConnectionPool(pool_size=2, max_idle_time=300)
    conn = FakeConnection()
    p.return_connection("m", "prov", conn)

    # Old connection, but returned to the pool just now -> not idle yet.
    ancient = time.time() - 10_000
    p._pools["prov:m"][0]["created_at"] = ancient
    p._pools["prov:m"][0]["last_used"] = time.time()

    assert p.get_connection("m", "prov", FakeConnection) is conn
    assert conn.closed is False
    p.shutdown()


# --------------------------------------------------------------------------
# Shutdown must not stall
# --------------------------------------------------------------------------

def test_shutdown_returns_promptly():
    """shutdown() must not block for the join timeout."""
    p = ImprovedConnectionPool(pool_size=2, max_idle_time=300)
    time.sleep(0.2)  # let the background thread settle into its wait

    start = time.perf_counter()
    p.shutdown()
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"shutdown() stalled for {elapsed:.2f}s"


def test_shutdown_stops_background_thread():
    """The cleanup thread must actually exit on shutdown."""
    p = ImprovedConnectionPool(pool_size=2, max_idle_time=300)
    p.shutdown()
    assert not p._cleanup_thread.is_alive()


def test_shutdown_is_idempotent():
    """Calling shutdown() twice must be safe and still fast."""
    p = ImprovedConnectionPool(pool_size=2, max_idle_time=300)
    p.shutdown()
    start = time.perf_counter()
    p.shutdown()
    assert time.perf_counter() - start < 1.0


def test_shutdown_disposes_pooled_and_active_connections(pool):
    """Everything still tracked must be released during shutdown."""
    pooled, active = FakeConnection(), FakeConnection()
    pool.return_connection("m", "prov", pooled)
    pool.get_connection("m", "prov", FakeConnection)  # leaves one active
    pool._active_connections["prov:m"].append(active)

    pool.shutdown()

    assert pooled.closed is True
    assert active.closed is True
    assert pool.get_stats()["total_pooled_connections"] == 0
    assert pool.get_stats()["total_active_connections"] == 0


def test_pool_respects_max_size(pool):
    """Connections beyond pool_size are disposed instead of pooled."""
    conns = [FakeConnection() for _ in range(4)]
    for conn in conns:
        pool.return_connection("m", "prov", conn)

    assert pool.get_stats()["pools"]["prov:m"] == pool.pool_size
    assert sum(1 for c in conns if c.closed) == 2
