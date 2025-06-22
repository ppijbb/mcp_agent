"""Simple async retry helper with exponential backoff."""
from __future__ import annotations
import asyncio, logging, random
from typing import Callable, Type, Tuple, Any

logger = logging.getLogger("retry")

async def async_retry(
    func: Callable[..., Any],
    *args,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retries: int = 3,
    min_delay: float = 0.5,
    max_delay: float = 2.0,
    **kwargs,
):
    """Retry an async func up to `retries` times on specified exceptions."""
    attempt = 0
    while True:
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            attempt += 1
            if attempt > retries:
                logger.error("Retry failed (%s/%s): %s", attempt, retries, e)
                raise
            delay = random.uniform(min_delay, max_delay) * (2 ** (attempt - 1))
            logger.warning("Retrying %s (attempt %s/%s) after %.2fs due to %s", func.__name__, attempt, retries, delay, e)
            await asyncio.sleep(delay) 