# Copyright 2023 The Wordcab Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main API module of the Wordcab Transcribe."""

import asyncio

from fastapi import FastAPI, Depends
from fastapi import status as http_status
from fastapi.responses import HTMLResponse
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.router.authentication import get_current_user
from wordcab_transcribe.router.v1.endpoints import (
    api_router,
    auth_router,
    cortex_router,
)
from wordcab_transcribe.utils import retrieve_user_platform


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    debug=settings.debug,
)

if settings.debug is False:
    app.include_router(auth_router, tags=["authentication"])
    app.include_router(api_router, prefix=settings.api_prefix, dependencies=[Depends(get_current_user)])
else:
    app.include_router(api_router, prefix=settings.api_prefix)

if settings.cortex_endpoint:
    app.include_router(cortex_router, tags=["cortex"])



@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.debug("Starting up...")

    if retrieve_user_platform() != "linux":
        logger.warning(
            "You are not running the application on Linux.\n"
            "The application was tested on Ubuntu 22.04, so we cannot guarantee that it will work on other OS.\n"
            "Report any issues with your env specs to: https://github.com/Wordcab/wordcab-transcribe/issues"
        )

    asyncio.create_task(asr.runner())


@app.get("/", tags=["status"])
async def home() -> HTMLResponse:
    """Root endpoint returning a simple HTML page with the project info."""
    content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.project_name}</title>
        <link href="https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="h-screen mx-auto text-center flex min-h-full items-center justify-center bg-gray-100 text-gray-700">
        <div class="container mx-auto p-4">
            <h1 class="text-4xl font-medium">{settings.project_name}</h1>
            <p class="text-gray-600">Version: {settings.version}</p>
            <p class="text-gray-600">{settings.description}</p>
            <p class="mt-16 text-gray-500">If you find any issues, please report them to:</p>
                <a class="text-blue-400 text-underlined" href="https://github.com/Wordcab/wordcab-transcribe/issues">
                    wordcab/wordcab-transcribe
                </a>
            </p>
            <a href="/docs">
                <button class="mt-8 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Docs</button>
            </a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content, media_type="text/html")


@app.get("/healthz", status_code=http_status.HTTP_200_OK, tags=["status"])
async def health() -> dict:
    """Health check endpoint. Important for Kubernetes liveness probe."""
    return {"status": "ok"}
