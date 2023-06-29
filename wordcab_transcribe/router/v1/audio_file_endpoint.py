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
"""Audio file endpoint for the Wordcab Transcribe API."""

import asyncio
from typing import Union

import shortuuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi import status as http_status
from loguru import logger

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import AudioRequest, AudioResponse
from wordcab_transcribe.utils import (
    convert_file_to_wav,
    delete_file,
    save_file_locally,
    split_dual_channel_file,
)


router = APIRouter()


@router.post(
    "", response_model=Union[AudioResponse, str], status_code=http_status.HTTP_200_OK
)
async def inference_with_audio(
    background_tasks: BackgroundTasks,
    alignment: bool = Form(False),  # noqa: B008
    diarization: bool = Form(False),  # noqa: B008
    dual_channel: bool = Form(False),  # noqa: B008
    source_lang: str = Form("en"),  # noqa: B008
    timestamps: str = Form("s"),  # noqa: B008
    use_batch: bool = Form(False),  # noqa: B008
    word_timestamps: bool = Form(False),  # noqa: B008
    file: UploadFile = File(...),  # noqa: B008
) -> AudioResponse:
    """Inference endpoint with audio file."""
    if file.filename is not None:
        extension = file.filename.split(".")[-1]
    else:
        extension = "wav"

    filename = f"audio_{shortuuid.ShortUUID().random(length=32)}.{extension}"

    await save_file_locally(filename=filename, file=file)

    data = AudioRequest(
        alignment=alignment,
        diarization=diarization,
        source_lang=source_lang,
        timestamps=timestamps,
        use_batch=use_batch,
        word_timestamps=word_timestamps,
        dual_channel=dual_channel,
    )

    if data.dual_channel:
        try:
            filepath = await split_dual_channel_file(filename)
        except Exception as e:
            logger.error(f"{e}\nFallback to single channel mode.")
            data.dual_channel = False

    try:
        filepath = await convert_file_to_wav(filename)

    except Exception as e:
        raise HTTPException(  # noqa: B904
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Process failed: {e}",
        )

    background_tasks.add_task(delete_file, filepath=filename)

    task = asyncio.create_task(
        asr.process_input(
            filepath=filepath,
            alignment=data.alignment,
            diarization=data.diarization,
            dual_channel=data.dual_channel,
            source_lang=data.source_lang,
            timestamps_format=data.timestamps,
            use_batch=data.use_batch,
            word_timestamps=data.word_timestamps,
        )
    )
    utterances = await task

    background_tasks.add_task(delete_file, filepath=filepath)

    if isinstance(utterances, Exception):
        logger.error(f"Error: {utterances}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(utterances),
        )
    else:
        return AudioResponse(
            utterances=utterances,
            alignment=data.alignment,
            diarization=data.diarization,
            dual_channel=data.dual_channel,
            source_lang=data.source_lang,
            timestamps=data.timestamps,
            use_batch=data.use_batch,
            word_timestamps=data.word_timestamps,
        )
