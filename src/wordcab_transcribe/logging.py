# Copyright 2024 The Wordcab Team. All rights reserved.
#
# Licensed under the MIT License (the "License");
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
import uuid
from functools import partial
from typing import Any, Awaitable, Callable, Tuple

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests, responses, errors and execution time."""

    def __init__(self, app: ASGIApp, debug_mode: bool) -> None:
        """Initialize the middleware."""
        super().__init__(app)
        logger.remove()
        logger.add(
            sys.stdout,
            level=(
                "DEBUG" if debug_mode else "INFO"
            ),  # Avoid logging debug messages in prod
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
        tracing_id = uuid.uuid4()

        if request.method == "POST":
            logger.info(f"Task [{tracing_id}] | {request.method} {request.url}")
        else:
            logger.info(f"{request.method} {request.url}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"Task [{tracing_id}] | Status: {response.status_code}, Time:"
            f" {process_time:.4f} secs"
        )

        return response


def time_and_tell(
    func: Callable, func_name: str, debug_mode: bool
) -> Tuple[Any, float]:
    """
    This decorator logs the execution time of a function only if the debug setting is True.

    Args:
        func: The function to call in the wrapper.
        func_name: The name of the function for logging purposes.
        debug_mode: The debug setting for logging purposes.

    Returns:
        The appropriate wrapper for the function.
    """
    start_time = time.time()
    result = func
    process_time = time.time() - start_time

    if debug_mode:
        logger.debug(f"{func_name} executed in {process_time:.4f} secs")

    return result, process_time


async def time_and_tell_async(
    func: Callable, func_name: str, debug_mode: bool
) -> Tuple[Any, float]:
    """
    This decorator logs the execution time of an async function only if the debug setting is True.

    Args:
        func: The function to call in the wrapper.
        func_name: The name of the function for logging purposes.
        debug_mode: The debug setting for logging purposes.

    Returns:
        The appropriate wrapper for the function.
    """
    start_time = time.time()

    if asyncio.iscoroutinefunction(func) or asyncio.iscoroutine(func):
        result = await func
    else:
        loop = asyncio.get_event_loop()
        if isinstance(func, partial):
            result = await loop.run_in_executor(
                None, func.func, *func.args, **func.keywords
            )
        else:
            result = await loop.run_in_executor(None, func)

    process_time = time.time() - start_time

    if debug_mode:
        logger.debug(f"{func_name} executed in {process_time:.4f} secs")

    return result, process_time
