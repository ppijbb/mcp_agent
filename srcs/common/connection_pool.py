"""
Improved connection pool management for LLM Manager.

Fixes memory leaks and provides proper connection lifecycle management.

Features:
- Thread-safe connection pooling with weak references
- Automatic connection cleanup and validation
- Memory leak prevention through proper disposal
- Performance monitoring and statistics
- Background cleanup with graceful shutdown

Example:
    >>> from srcs.common.connection_pool import create_improved_connection_pool
    >>> pool = create_improved_connection_pool(pool_size=10)
    >>> connection = pool.get_connection("model", "provider", create_func)
    >>> pool.return_connection("model", "provider", connection)
"""

import time
import threading
import weakref
import gc
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ImprovedConnectionPool:
    """
    Improved connection pool with proper memory management and lifecycle control.
    
    Features:
    - Automatic connection cleanup with weak references
    - Proper resource disposal
    - Thread-safe operations
    - Memory leak prevention
    - Performance monitoring
    """
    
    def __init__(self, pool_size: int = 5, max_idle_time: int = 300, enable_monitoring: bool = True):
        """
        Initialize improved connection pool.
        
        Args:
            pool_size: Maximum number of connections per model
            max_idle_time: Maximum idle time before connection cleanup (seconds)
            enable_monitoring: Enable performance monitoring
        """
        self.pool_size = pool_size
        self.max_idle_time = max_idle_time
        self.enable_monitoring = enable_monitoring
        
        # Thread-safe connection storage
        self._lock = threading.RLock()
        self._pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._active_connections: Dict[str, List[Any]] = defaultdict(list)
        
        # Connection lifecycle tracking
        self.connection_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"created": 0, "reused": 0, "expired": 0, "errors": 0}
        )
        
        # Weak reference tracking for cleanup
        self._weak_refs: List[weakref.ref] = []
        
        # Monitoring and cleanup
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # seconds
        self._shutdown = False
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def get_connection(self, model_name: str, provider: str, create_func: Callable) -> Any:
        """
        Get or create a connection from the pool with proper lifecycle management.
        
        Args:
            model_name: Model name
            provider: Provider name  
            create_func: Function to create new connection
            
        Returns:
            Connection object
        """
        pool_key = f"{provider}:{model_name}"
        current_time = time.time()
        
        with self._lock:
            # Periodic cleanup check
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_connections()
                self.last_cleanup = current_time
            
            # Try to reuse existing connection
            if self._pools[pool_key]:
                conn_info = self._pools[pool_key].pop()
                if current_time - conn_info["created_at"] < self.max_idle_time:
                    # Validate connection is still usable
                    if self._validate_connection(conn_info["connection"]):
                        self.connection_stats[pool_key]["reused"] += 1
                        self._active_connections[pool_key].append(conn_info["connection"])
                        
                        # Add weak reference for tracking
                        self._add_weak_ref(conn_info["connection"], pool_key)
                        
                        logger.debug(f"Reusing connection for {pool_key}")
                        return conn_info["connection"]
                    else:
                        # Connection is invalid, dispose properly
                        self._dispose_connection(conn_info["connection"], pool_key)
                        self.connection_stats[pool_key]["errors"] += 1
            
            # Create new connection
            try:
                connection = create_func()
                self.connection_stats[pool_key]["created"] += 1
                self._active_connections[pool_key].append(connection)
                
                # Add weak reference for tracking
                self._add_weak_ref(connection, pool_key)
                
                logger.debug(f"Created new connection for {pool_key}")
                return connection
                
            except Exception as e:
                logger.error(f"Failed to create connection for {pool_key}: {e}")
                self.connection_stats[pool_key]["errors"] += 1
                raise
    
    def return_connection(self, model_name: str, provider: str, connection: Any) -> None:
        """
        Return a connection to the pool with proper validation.
        
        Args:
            model_name: Model name
            provider: Provider name
            connection: Connection object to return
        """
        pool_key = f"{provider}:{model_name}"
        
        with self._lock:
            # Remove from active connections
            if connection in self._active_connections[pool_key]:
                self._active_connections[pool_key].remove(connection)
            
            # Validate connection before returning to pool
            if not self._validate_connection(connection):
                logger.warning(f"Connection validation failed for {pool_key}, disposing")
                self._dispose_connection(connection, pool_key)
                self.connection_stats[pool_key]["errors"] += 1
                return
            
            # Check pool size limit
            if len(self._pools[pool_key]) < self.pool_size:
                self._pools[pool_key].append({
                    "connection": connection,
                    "created_at": time.time(),
                    "last_used": time.time()
                })
                logger.debug(f"Returned connection to pool for {pool_key}")
            else:
                # Pool is full, dispose of excess connection
                self._dispose_connection(connection, pool_key)
                logger.debug(f"Pool full for {pool_key}, disposed connection")
    
    def _validate_connection(self, connection: Any) -> bool:
        """
        Validate if connection is still usable.
        
        Args:
            connection: Connection to validate
            
        Returns:
            True if connection is valid
        """
        try:
            # Basic validation - check if connection object exists and has expected attributes
            if connection is None:
                return False
            
            # Add provider-specific validation if needed
            # This is a basic implementation - extend based on your connection types
            return True
            
        except Exception as e:
            logger.debug(f"Connection validation failed: {e}")
            return False
    
    def _dispose_connection(self, connection: Any, pool_key: str) -> None:
        """
        Properly dispose of a connection to prevent memory leaks.
        
        Args:
            connection: Connection to dispose
            pool_key: Pool key for tracking
        """
        try:
            # Provider-specific cleanup
            if hasattr(connection, 'close'):
                connection.close()
            elif hasattr(connection, 'disconnect'):
                connection.disconnect()
            elif hasattr(connection, '__exit__'):
                connection.__exit__(None, None, None)
            
            # Remove from weak references
            self._remove_weak_ref(connection)
            
            logger.debug(f"Disposed connection for {pool_key}")
            
        except Exception as e:
            logger.warning(f"Error disposing connection for {pool_key}: {e}")
        finally:
            # Force garbage collection for the connection object
            try:
                del connection
            except:
                pass
    
    def _add_weak_ref(self, connection: Any, pool_key: str) -> None:
        """
        Add weak reference for connection tracking.

        Args:
            connection: Connection object to track
            pool_key: Pool key for logging purposes
        """
        def cleanup_callback(ref):
            with self._lock:
                self._remove_weak_ref(ref)
                logger.debug(f"Connection garbage collected for {pool_key}")
        
        try:
            weak_ref = weakref.ref(connection, cleanup_callback)
            self._weak_refs.append(weak_ref)
        except Exception as e:
            logger.debug(f"Failed to create weak reference: {e}")
    
    def _remove_weak_ref(self, connection_or_ref: Any) -> None:
        """Remove weak reference from tracking."""
        try:
            if isinstance(connection_or_ref, weakref.ref):
                ref = connection_or_ref
            else:
                # Find the weak reference for this connection
                ref = None
                for weak_ref in self._weak_refs:
                    if weak_ref() is connection_or_ref:
                        ref = weak_ref
                        break
            
            if ref and ref in self._weak_refs:
                self._weak_refs.remove(ref)
        except Exception as e:
            logger.debug(f"Failed to remove weak reference: {e}")
    
    def _cleanup_old_connections(self) -> None:
        """Clean up expired connections from pools."""
        current_time = time.time()
        
        with self._lock:
            for pool_key, connections in list(self._pools.items()):
                valid_connections = []
                
                for conn_info in connections:
                    age = current_time - conn_info["created_at"]
                    
                    if age < self.max_idle_time:
                        valid_connections.append(conn_info)
                    else:
                        # Connection expired
                        self._dispose_connection(conn_info["connection"], pool_key)
                        self.connection_stats[pool_key]["expired"] += 1
                        logger.debug(f"Cleaned up expired connection for {pool_key}")
                
                self._pools[pool_key] = valid_connections
            
            # Clean up dead weak references
            self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]
    
    def _background_cleanup(self) -> None:
        """Background thread for periodic cleanup."""
        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval)
                if not self._shutdown:
                    self._cleanup_old_connections()
                    
                    # Periodic garbage collection
                    if len(self._weak_refs) > 100:  # Threshold for forcing GC
                        gc.collect()
                        
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            total_pooled = sum(len(conns) for conns in self._pools.values())
            total_active = sum(len(conns) for conns in self._active_connections.values())
            
            return {
                "total_pooled_connections": total_pooled,
                "total_active_connections": total_active,
                "pool_size_limit": self.pool_size,
                "max_idle_time": self.max_idle_time,
                "weak_refs_count": len(self._weak_refs),
                "connection_stats": dict(self.connection_stats),
                "pools": {key: len(conns) for key, conns in self._pools.items()}
            }
    
    def shutdown(self) -> None:
        """Shutdown connection pool and clean up all resources."""
        logger.info("Shutting down connection pool...")
        
        with self._lock:
            self._shutdown = True
            
            # Dispose all pooled connections
            for pool_key, connections in self._pools.items():
                for conn_info in connections:
                    self._dispose_connection(conn_info["connection"], pool_key)
            
            # Dispose all active connections
            for pool_key, connections in self._active_connections.items():
                for connection in connections:
                    self._dispose_connection(connection, pool_key)
            
            # Clear all tracking structures
            self._pools.clear()
            self._active_connections.clear()
            self.connection_stats.clear()
            self._weak_refs.clear()
            
            # Wait for cleanup thread to finish
            if self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5)
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Connection pool shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if not hasattr(self, '_shutdown') or not self._shutdown:
                self.shutdown()
        except:
            pass  # Ignore errors during destruction


# Factory function to replace the original ConnectionPool
def create_improved_connection_pool(pool_size: int = 5, max_idle_time: int = 300) -> ImprovedConnectionPool:
    """
    Create an improved connection pool instance.
    
    Args:
        pool_size: Maximum number of connections per model
        max_idle_time: Maximum idle time before connection cleanup (seconds)
        
    Returns:
        ImprovedConnectionPool instance
    """
    return ImprovedConnectionPool(pool_size, max_idle_time)