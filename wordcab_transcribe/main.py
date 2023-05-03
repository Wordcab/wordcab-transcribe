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
import json
from typing import Optional

import aiofiles
import shortuuid
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, WebSocket
from fastapi import status as http_status
from fastapi.responses import HTMLResponse
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.models import ASRResponse, DataRequest, LiveDataRequest
from wordcab_transcribe.service import ASRService
from wordcab_transcribe.utils import (
    convert_file_to_wav,
    delete_file,
    download_audio_file,
    download_file_from_youtube,
    format_punct,
    is_empty_string,
    retrieve_user_platform,
)


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    debug=settings.debug,
)

asr = ASRService()


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
async def health_check():
    """Health check endpoint."""
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
            <p class="mt-16 text-gray-500">Want access? Contact us:
                <a class="text-blue-400 text-underlined" href="mailto:info@wordcab.com?subject=Access">
                    info@wordcab.com
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


@app.post(
    f"{settings.api_prefix}/audio",
    tags=["inference"],
    response_model=ASRResponse,
    status_code=http_status.HTTP_200_OK,
)
async def inference_with_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008
    data: Optional[DataRequest] = None,
) -> ASRResponse:
    """Inference endpoint with audio file."""
    extension = file.filename.split(".")[-1]
    filename = f"audio_{shortuuid.ShortUUID().random(length=32)}.{extension}"

    async with aiofiles.open(filename, "wb") as f:
        audio_bytes = await file.read()
        await f.write(audio_bytes)

    if extension != "wav":
        filepath = await convert_file_to_wav(filename)
        background_tasks.add_task(delete_file, filepath=filename)
    else:
        filepath = filename

    if data is None:
        data = DataRequest()
    else:
        data = DataRequest(**data.dict())

    raw_utterances = await asr.process_input(
        filepath, data.num_speakers, data.source_lang, data.timestamps
    )
    utterances = [
        {
            "text": format_punct(utterance["text"]),
            "start": utterance["start"],
            "end": utterance["end"],
            "speaker": int(utterance["speaker"]),
        }
        for utterance in raw_utterances
        if not is_empty_string(utterance["text"])
    ]

    background_tasks.add_task(delete_file, filepath=filepath)

    return ASRResponse(utterances=utterances)


@app.post(
    f"{settings.api_prefix}/youtube",
    tags=["inference"],
    response_model=ASRResponse,
    status_code=http_status.HTTP_200_OK,
)
async def inference_with_youtube(
    background_tasks: BackgroundTasks,
    url: str,
    data: Optional[DataRequest] = None,
) -> ASRResponse:
    """Inference endpoint with YouTube url."""
    filename = f"yt_{shortuuid.ShortUUID().random(length=32)}"
    filepath = await download_file_from_youtube(url, filename)

    if data is None:
        data = DataRequest()
    else:
        data = DataRequest(**data.dict())

    raw_utterances = await asr.process_input(
        filepath, data.num_speakers, data.source_lang, data.timestamps
    )
    utterances = [
        {
            "text": format_punct(utterance["text"]),
            "start": utterance["start"],
            "end": utterance["end"],
            "speaker": int(utterance["speaker"]),
        }
        for utterance in raw_utterances
        if not is_empty_string(utterance["text"])
    ]

    background_tasks.add_task(delete_file, filepath=filepath)

    return ASRResponse(utterances=utterances)


@app.post(
    f"{settings.api_prefix}/audio-url",
    tags=["inference"],
    response_model=ASRResponse,
    status_code=http_status.HTTP_200_OK,
)
async def inference_with_audio_url(
    background_tasks: BackgroundTasks,
    url: str,
    data: Optional[DataRequest] = None,
) -> ASRResponse:
    """Inference endpoint with audio url."""
    filename = f"audio_url_{shortuuid.ShortUUID().random(length=32)}"
    filepath = await download_audio_file(url, filename)
    extension = filepath.split(".")[-1]

    if extension != "wav":
        filepath = await convert_file_to_wav(filepath)
        background_tasks.add_task(delete_file, filepath=f"{filename}.{extension}")
    else:
        filepath = filename

    if data is None:
        data = DataRequest()
    else:
        data = DataRequest(**data.dict())

    raw_utterances = await asr.process_input(
        filepath, data.num_speakers, data.source_lang, data.timestamps
    )
    utterances = [
        {
            "text": format_punct(utterance["text"]),
            "start": utterance["start"],
            "end": utterance["end"],
            "speaker": int(utterance["speaker"]),
        }
        for utterance in raw_utterances
        if not is_empty_string(utterance["text"])
    ]

    background_tasks.add_task(delete_file, filepath=filepath)

    return ASRResponse(utterances=utterances)


@app.websocket(f"{settings.api_prefix}/live")
async def websocket_transcription(websocket: WebSocket):
    """Websocket endpoint for live transcription."""
    await websocket.accept()

    try:
        while True:
            audio_data_json = await websocket.receive_text()
            audio_data = LiveDataRequest(**json.loads(audio_data_json))
            source_lang = audio_data.source_lang
            if audio_data.audio_bytes:
                result = await asr.process_input(
                    filepath="",
                    num_speakers=1,
                    source_lang=source_lang,
                    timestamps="seconds",
                    live=True,
                    audio_bytes=audio_data.audio_bytes,
                )
                print(f"YO: {result}")
                await websocket.send_json(result)

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
