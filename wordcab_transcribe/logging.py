# Copyright 2023 The Wordcab Team. All rights reserved.
#
# Licensed under the Wordcab Transcribe License 0.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Wordcab/wordcab-transcribe/blob/main/LICENSE
#
# Except as expressly provided otherwise herein, and to the fullest
# extent permitted by law, Licensor provides the Software (and each
# Contributor provides its Contributions) AS IS, and Licensor
# disclaims all warranties or guarantees of any kind, express or
# implied, whether arising under any law or from any usage in trade,
# or otherwise including but not limited to the implied warranties
# of merchantability, non-infringement, quiet enjoyment, fitness
# for a particular purpose, or otherwise.
#
# See the License for the specific language governing permissions
# and limitations under the License.

"""Logging module to add a logging middleware to the Wordcab Transcribe API."""

import asyncio
import sys
import time
from functools import wraps
from typing import Awaitable, Callable

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from wordcab_transcribe.config import settings


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests, responses, errors and execution time."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware."""
        super().__init__(app)
        logger.remove()
        logger.add(
            sys.stdout,
            level="DEBUG"
            if settings.debug
            else "WARNING",  # Avoid logging debug messages in prod
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Dispatch a request and log it, along with the response and any errors.

        Args:
            request: The request to dispatch.
            call_next: The next middleware to call.

        Returns:
            The response from the next middleware.
        """
        start_time = time.time()
        logger.debug(f"Request: {request.method} {request.url}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.debug(
            f"Response status: {response.status_code}, Process Time: {process_time:.4f} secs"
        )

        return response


def time_and_tell(func: Callable) -> Callable:
    """
    This decorator logs the execution time of a function only if the debug setting is True.

    Args:
        func: The function to decorate.

    Returns:
        The appropriate wrapper for the function.
    """

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Callable:
        """Sync wrapper for the decorated function."""
        if settings.debug:
            start_time = time.time()

            result = func(*args, **kwargs)

            process_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {process_time:.4f} secs")
        else:
            result = func(*args, **kwargs)

        return result

    async def async_wrapper(*args, **kwargs) -> Awaitable:
        """Async wrapper for the decorated function."""
        if settings.debug:
            start_time = time.time()

            result = await func(*args, **kwargs)

            process_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {process_time:.4f} secs")
        else:
            result = await func(*args, **kwargs)

        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
