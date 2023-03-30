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
"""Main API module of the Wordcab ASR API."""

import aiofiles
import asyncio
import random
from loguru import logger

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi import status as http_status
from fastapi.responses import HTMLResponse

from asr_api.config import settings
from asr_api.models import ASRResponse
from asr_api.service import ASRService
from asr_api.utils import convert_file_to_wav, delete_file, download_file_from_youtube


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    debug=settings.debug,
)

asr = ASRService()


@app.on_event("startup")
async def startup_event():
    logger.debug("Starting up...")
    asyncio.create_task(asr.runner())


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
    f"{settings.api_prefix}/audio",
    tags=["inference"],
    response_model=ASRResponse,
    status_code=http_status.HTTP_200_OK
)
async def inference_with_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_speakers: int | None = None,
):
    """
    Inference endpoint.

    Args:
        background_tasks (BackgroundTasks): Background tasks dependency.
        file (UploadFile): Audio file.
        num_speakers (int): Number of speakers in the audio file. Default: 0.

    Returns:
        ASRResponse: Response data.

    Examples:

        # Example using a local audio file
        import requests
        filepath = "sample_1.mp3"
        with open(file, "rb") as f:
            files = {"file": (filepath, f)}
            response = requests.post("url/api/v1/audio", files=files)
            print(response.json())
    """
    num_speakers = num_speakers or 0
    extension = file.filename.split(".")[-1]

    filename = f"audio_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=32))}.{extension}"
    async with aiofiles.open(filename, "wb") as f:
        audio_bytes = await file.read()
        await f.write(audio_bytes)

    if extension != "wav":
        filepath = await convert_file_to_wav(filename)
        background_tasks.add_task(delete_file, filepath=filename)
    else:
        filepath = filename

    utterances = await asr.process_input(filepath, num_speakers)
    utterances = [
        {
            "start": float(utterance["start"]),
            "text": str(utterance["text"]),
            "end": float(utterance["end"]),
            "speaker": int(utterance["speaker"]),
        }
        for utterance in utterances
    ]

    background_tasks.add_task(delete_file, filepath=filename)

    return ASRResponse(utterances=utterances)


@app.post(
    f"{settings.api_prefix}/youtube",
    tags=["inference"],
    response_model=ASRResponse,
    status_code=http_status.HTTP_200_OK
)
async def inference_with_youtube(
    background_tasks: BackgroundTasks,
    url: str,
    num_speakers: int | None = None,
):
    """
    Inference endpoint.

    Args:
        background_tasks (BackgroundTasks): Background tasks dependency.
        url (str): Youtube URL.
        num_speakers (int): Number of speakers in the audio file. Default: 0.

    Returns:
        ASRResponse: Response data.

    Examples:

        # Example using a local audio file
        import requests
        url = "https://www.youtube.com/..."
        r = requests.post(f"http://localhost:5001/api/v1/youtube?url={url}")
    """
    num_speakers = num_speakers or 0

    filename = f"audio_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=32))}"
    filepath = await download_file_from_youtube(url, filename)

    utterances = await asr.process_input(filepath, num_speakers)
    utterances = [
        {
            "start": float(utterance["start"]),
            "text": str(utterance["text"]),
            "end": float(utterance["end"]),
            "speaker": int(utterance["speaker"]),
        }
        for utterance in utterances
    ]

    background_tasks.add_task(delete_file, filepath=filename)

    return ASRResponse(utterances=utterances)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7680, reload=True)
