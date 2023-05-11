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

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi import status as http_status
from fastapi.responses import HTMLResponse
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import CortexPayload, CortexResponse, DataRequest
from wordcab_transcribe.router.v1.audio_url_endpoint import inference_with_audio_url
from wordcab_transcribe.router.v1.endpoints import api_router
from wordcab_transcribe.router.v1.youtube_endpoint import inference_with_youtube
from wordcab_transcribe.utils import retrieve_user_platform


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    debug=settings.debug,
)

app.include_router(api_router, prefix=settings.api_prefix)


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


@app.post("/", tags=["cortex"])
async def run_cortex(payload: CortexPayload, request: Request):
    """Endpoint for Cortex."""
    logger.debug("Received a request from Cortex.")

    if payload.ping:
        return

    url = payload.url
    url_type = payload.url_type
    source_lang = payload.source_lang
    timestamps = payload.timestamps
    data_request = DataRequest(source_lang=source_lang, timestamps=timestamps)

    job_name = payload.job_name
    request_id = request.headers.get("x-cortex-request-id", None)

    if url_type == "audio_url":
        utterances = await inference_with_audio_url(
            background_tasks=BackgroundTasks(),
            url=url,
            data=data_request,
        )
    elif url_type == "youtube":
        utterances = await inference_with_youtube(
            background_tasks=BackgroundTasks(),
            url=url,
            data=data_request,
        )
    else:
        return

    return CortexResponse(
        utterances=utterances,
        source_lang=source_lang,
        timestamps=timestamps,
        job_name=job_name,
        request_id=request_id,
    )


@app.get("/", tags=["status"])
async def home():
    """Health check endpoint and default home screen."""
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
async def health():
    """Health check endpoint for Cortex."""
    return {"status": "ok"}
