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
"""YouTube endpoint for the Wordcab Transcribe API."""

import asyncio
from typing import Optional

import shortuuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi import status as http_status
from loguru import logger

from wordcab_transcribe.dependencies import asr, io_executor
from wordcab_transcribe.models import BaseRequest, YouTubeResponse
from wordcab_transcribe.utils import delete_file, download_file_from_youtube


router = APIRouter()


@router.post("", response_model=YouTubeResponse, status_code=http_status.HTTP_200_OK)
async def inference_with_youtube(
    background_tasks: BackgroundTasks,
    url: str,
    data: Optional[BaseRequest] = None,
) -> YouTubeResponse:
    """Inference endpoint with YouTube url."""
    filename = f"yt_{shortuuid.ShortUUID().random(length=32)}"
    filepath = await asyncio.get_running_loop().run_in_executor(
        io_executor, download_file_from_youtube, url, filename
    )

    data = BaseRequest() if data is None else BaseRequest(**data.dict())

    task = asyncio.create_task(
        asr.process_input(
            filepath=filepath,
            alignment=data.alignment,
            diarization=data.diarization,
            dual_channel=False,
            source_lang=data.source_lang,
            timestamps_format=data.timestamps,
            use_batch=data.use_batch,
            word_timestamps=data.word_timestamps,
        )
    )
    utterances, audio_duration = await task

    background_tasks.add_task(delete_file, filepath=filepath)

    if isinstance(utterances, Exception):
        logger.error(f"Error: {utterances}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(utterances),
        )
    else:
        return YouTubeResponse(
            utterances=utterances,
            alignment=data.alignment,
            diarization=data.diarization,
            source_lang=data.source_lang,
            timestamps=data.timestamps,
            use_batch=data.use_batch,
            word_timestamps=data.word_timestamps,
            video_url=url,
        )
