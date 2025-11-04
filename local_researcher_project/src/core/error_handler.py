#!/usr/bin/env python3
"""
Error Handler - Type-Specific Error Handling

Provides specialized error handling strategies for different exception types
to improve success rate and system reliability.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Error handler with type-specific strategies."""
    
    def __init__(
        self,
        default_max_retries: int = 3,
        timeout_increase_factor: float = 1.5,
        connection_retry_delay: float = 2.0
    ):
        """
        Initialize error handler.
        
        Args:
            default_max_retries: Default maximum retry attempts
            timeout_increase_factor: Factor to increase timeout on retry
            connection_retry_delay: Base delay for connection retries
        """
        self.default_max_retries = default_max_retries
        self.timeout_increase_factor = timeout_increase_factor
        self.connection_retry_delay = connection_retry_delay
        
        # Error handling statistics
        self.stats = {
            'timeout_errors': 0,
            'connection_errors': 0,
            'value_errors': 0,
            'runtime_errors': 0,
            'total_errors': 0,
            'successful_recoveries': 0
        }
        
        logger.info("ErrorHandler initialized")
    
    async def handle_timeout_error(
        self,
        error: asyncio.TimeoutError,
        func: Callable,
        *args,
        current_timeout: Optional[float] = None,
        max_retries: int = 3,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Handle TimeoutError: Increase timeout or retry immediately.
        
        Args:
            error: The TimeoutError exception
            func: Function to retry
            args: Function arguments
            current_timeout: Current timeout value (if applicable)
            max_retries: Maximum retry attempts
            kwargs: Function keyword arguments
        
        Returns:
            Tuple of (result, success)
        """
        self.stats['timeout_errors'] += 1
        self.stats['total_errors'] += 1
        
        logger.warning(f"TimeoutError occurred: {error}")
        
        # Increase timeout if current_timeout is provided
        if current_timeout is not None:
            new_timeout = current_timeout * self.timeout_increase_factor
            logger.info(f"Increasing timeout from {current_timeout}s to {new_timeout}s")
            
            # If function accepts timeout parameter, update it
            if 'timeout' in kwargs:
                kwargs['timeout'] = new_timeout
            elif len(args) > 0 and isinstance(args[-1], dict):
                # Try to update timeout in last positional arg if it's a dict
                pass  # Would need to modify args, which is immutable
        
        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                if attempt > 0:
                    logger.debug(f"Retrying after timeout (attempt {attempt + 1}/{max_retries}) in {wait_time}s")
                    await asyncio.sleep(wait_time)
                
                result = await func(*args, **kwargs)
                self.stats['successful_recoveries'] += 1
                logger.info(f"Successfully recovered from TimeoutError after {attempt + 1} retries")
                return result, True
                
            except asyncio.TimeoutError as e:
                if attempt == max_retries - 1:
                    logger.error(f"TimeoutError persists after {max_retries} retries")
                    return None, False
                continue
            except Exception as e:
                logger.error(f"Unexpected error during timeout retry: {e}")
                return None, False
        
        return None, False
    
    async def handle_connection_error(
        self,
        error: ConnectionError,
        func: Callable,
        *args,
        max_retries: int = 5,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Handle ConnectionError: Connection retry with exponential backoff.
        
        Args:
            error: The ConnectionError exception
            func: Function to retry
            args: Function arguments
            max_retries: Maximum retry attempts
            kwargs: Function keyword arguments
        
        Returns:
            Tuple of (result, success)
        """
        self.stats['connection_errors'] += 1
        self.stats['total_errors'] += 1
        
        logger.warning(f"ConnectionError occurred: {error}")
        
        # Exponential backoff retry
        for attempt in range(max_retries):
            try:
                wait_time = self.connection_retry_delay * (2 ** attempt)
                if attempt > 0:
                    logger.debug(f"Retrying connection (attempt {attempt + 1}/{max_retries}) in {wait_time}s")
                    await asyncio.sleep(wait_time)
                
                result = await func(*args, **kwargs)
                self.stats['successful_recoveries'] += 1
                logger.info(f"Successfully recovered from ConnectionError after {attempt + 1} retries")
                return result, True
                
            except (ConnectionError, OSError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"ConnectionError persists after {max_retries} retries")
                    return None, False
                continue
            except Exception as e:
                logger.error(f"Unexpected error during connection retry: {e}")
                return None, False
        
        return None, False
    
    async def handle_value_error(
        self,
        error: ValueError,
        func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Handle ValueError: Enhanced input validation before retry.
        
        Args:
            error: The ValueError exception
            func: Function to retry (usually won't retry)
            args: Function arguments
            kwargs: Function keyword arguments
        
        Returns:
            Tuple of (result, success) - usually (None, False) as no retry
        """
        self.stats['value_errors'] += 1
        self.stats['total_errors'] += 1
        
        logger.warning(f"ValueError (validation error): {error}")
        logger.info("ValueError indicates input validation issue - no automatic retry")
        
        # ValueErrors are typically validation errors, so we don't retry
        # But we log it for analysis
        return None, False
    
    async def handle_runtime_error(
        self,
        error: RuntimeError,
        func: Callable,
        *args,
        max_retries: int = 3,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Handle RuntimeError: Standard exponential backoff retry.
        
        Args:
            error: The RuntimeError exception
            func: Function to retry
            args: Function arguments
            max_retries: Maximum retry attempts
            kwargs: Function keyword arguments
        
        Returns:
            Tuple of (result, success)
        """
        self.stats['runtime_errors'] += 1
        self.stats['total_errors'] += 1
        
        error_msg = str(error)
        logger.warning(f"RuntimeError occurred: {error_msg}")
        
        # Check if error message indicates non-retryable error
        non_retryable_patterns = [
            'circuit breaker',
            'not found',
            'invalid',
            'unauthorized',
            'forbidden'
        ]
        
        if any(pattern in error_msg.lower() for pattern in non_retryable_patterns):
            logger.info(f"RuntimeError appears non-retryable based on error message")
            return None, False
        
        # Standard exponential backoff retry
        for attempt in range(max_retries):
            try:
                wait_time = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
                if attempt > 0:
                    logger.debug(f"Retrying after RuntimeError (attempt {attempt + 1}/{max_retries}) in {wait_time}s")
                    await asyncio.sleep(wait_time)
                
                result = await func(*args, **kwargs)
                self.stats['successful_recoveries'] += 1
                logger.info(f"Successfully recovered from RuntimeError after {attempt + 1} retries")
                return result, True
                
            except RuntimeError as e:
                if attempt == max_retries - 1:
                    logger.error(f"RuntimeError persists after {max_retries} retries")
                    return None, False
                continue
            except Exception as e:
                logger.error(f"Unexpected error during runtime retry: {e}")
                return None, False
        
        return None, False
    
    async def handle_error(
        self,
        error: Exception,
        func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Handle any error with appropriate strategy.
        
        Args:
            error: The exception
            func: Function to retry
            args: Function arguments
            kwargs: Function keyword arguments
        
        Returns:
            Tuple of (result, success)
        """
        if isinstance(error, asyncio.TimeoutError):
            return await self.handle_timeout_error(error, func, *args, **kwargs)
        elif isinstance(error, (ConnectionError, OSError)):
            return await self.handle_connection_error(error, func, *args, **kwargs)
        elif isinstance(error, ValueError):
            return await self.handle_value_error(error, func, *args, **kwargs)
        elif isinstance(error, RuntimeError):
            return await self.handle_runtime_error(error, func, *args, **kwargs)
        else:
            # Unknown error type - use RuntimeError handler
            logger.warning(f"Unknown error type: {type(error).__name__}, using RuntimeError handler")
            return await self.handle_runtime_error(RuntimeError(str(error)), func, *args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        recovery_rate = (
            self.stats['successful_recoveries'] / self.stats['total_errors']
            if self.stats['total_errors'] > 0 else 0.0
        )
        
        return {
            **self.stats,
            'recovery_rate': recovery_rate,
            'timestamp': datetime.now().isoformat()
        }


# Global error handler instance
_error_handler_instance: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _error_handler_instance
    if _error_handler_instance is None:
        _error_handler_instance = ErrorHandler()
    return _error_handler_instance

