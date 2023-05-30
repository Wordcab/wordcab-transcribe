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
"""Audio url endpoint for the Wordcab Transcribe API."""

import asyncio
from typing import Optional

import shortuuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi import status as http_status

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import AudioRequest, AudioResponse
from wordcab_transcribe.utils import (
    convert_file_to_wav,
    delete_file,
    download_audio_file,
    split_dual_channel_file,
)


router = APIRouter()


@router.post("", response_model=AudioResponse, status_code=http_status.HTTP_200_OK)
async def inference_with_audio_url(
    background_tasks: BackgroundTasks,
    url: str,
    data: Optional[AudioRequest] = None,
) -> AudioResponse:
    """Inference endpoint with audio url."""
    filename = f"audio_url_{shortuuid.ShortUUID().random(length=32)}"

    data = AudioRequest() if data is None else AudioRequest(**data.dict())

    filepath, extension = await download_audio_file(url, filename)

    if data.dual_channel:
        filepath = await split_dual_channel_file(filepath)
    else:
        filepath = await convert_file_to_wav(filepath)

    background_tasks.add_task(delete_file, filepath=f"{filename}.{extension}")

    try:
        task = asyncio.create_task(
            asr.process_input(
                filepath=filepath,
                alignment=data.alignment,
                diarization=data.diarization,
                dual_channel=data.dual_channel,
                source_lang=data.source_lang,
                timestamps_format=data.timestamps,
                word_timestamps=data.word_timestamps,
            )
        )
        utterances = await task

        return AudioResponse(
            utterances=utterances,
            alignment=data.alignment,
            diarization=data.diarization,
            dual_channel=data.dual_channel,
            source_lang=data.source_lang,
            timestamps=data.timestamps,
            word_timestamps=data.word_timestamps,
        )

    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e

    finally:
        background_tasks.add_task(delete_file, filepath=filepath)
