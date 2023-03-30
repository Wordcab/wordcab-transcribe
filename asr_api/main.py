# Copyright (c) 2023, The Wordcab team. All rights reserved.
"""Main API module of the Wordcab ASR API."""

import asyncio
import io
from loguru import logger

from fastapi import FastAPI, File, UploadFile
from fastapi import status as http_status
from fastapi.responses import HTMLResponse

from asr_api.config import settings
from asr_api.models import ASRRequest, ASRResponse
from asr_api.service import ASRService


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    debug=settings.debug,
)

service = ASRService()


@app.on_event("startup")
async def startup_event():
    logger.debug("Starting up...")
    asyncio.create_task(service.runner())


@app.get("/", tags=["status"])
async def health_check():
    """Health check endpoint"""
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
            <p class="mt-16 text-gray-500">Want access? Contact us: <a class="text-blue-400 text-underlined" href="mailto:info@wordcab.com?subject=Access">info@wordcab.com</a></p>
            <a href="/docs">
                <button class="mt-8 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Docs</button>
            </a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content, media_type="text/html")


@app.post(
    f"{settings.api_prefix}/inference",
    tags=["inference"],
    response_model=ASRResponse,
    status_code=http_status.HTTP_200_OK
)
async def inference(data: ASRRequest, file: UploadFile = File(...)):
    """
    Inference endpoint.

    Args:
        data (ASRRequest): Request data.
        file (UploadFile): Audio file.

    Returns:
        ASRResponse: Response data.

    Examples:
        >>> # Using a local audio file
        >>> import requests
        >>> files = {"file": ("test.wav", open("test.wav", "rb"))}
        >>> response = requests.post("url.../api/v1/inference", files=files)
        >>> print(response.json())

        >>> # Using a YouTube link
        >>> import requests
        >>> data = {"url": "https://www.youtube.com/watch?v=..."}
        >>> response = requests.post("url.../api/v1/inference", data=data)
        >>> print(response.json())
    """
    if data.url:
        # TODO: Implement URL inference
        return ASRResponse(text=[{"text": "YouTube links are not implemented yet."}])
    else:
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.seek(0)

    output = await service.process_input(audio_file)

    return ASRResponse(text=output)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7680, reload=True)
