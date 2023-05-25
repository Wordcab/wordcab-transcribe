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
from fastapi import APIRouter, BackgroundTasks
from fastapi import status as http_status

from wordcab_transcribe.dependencies import asr, io_executor
from wordcab_transcribe.models import BaseRequest, YouTubeResponse
from wordcab_transcribe.utils import (
    convert_timestamp,
    delete_file,
    download_file_from_youtube,
    format_punct,
    is_empty_string,
)


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

    raw_utterances = await asr.process_input(
        filepath,
        alignment=data.alignment,
        diarization=data.diarization,
        dual_channel=False,
        source_lang=data.source_lang,
        word_timestamps=data.word_timestamps,
    )

    timestamps_format = data.timestamps
    utterances = [
        {
            "text": format_punct(utterance["text"]),
            "start": convert_timestamp(utterance["start"], timestamps_format),
            "end": convert_timestamp(utterance["end"], timestamps_format),
            "speaker": int(utterance["speaker"]) if data.diarization else None,
            "words": utterance["words"] if data.word_timestamps else [],
        }
        for utterance in raw_utterances
        if not is_empty_string(utterance["text"])
    ]

    background_tasks.add_task(delete_file, filepath=filepath)

    return YouTubeResponse(
        utterances=utterances,
        alignment=data.alignment,
        diarization=data.diarization,
        source_lang=data.source_lang,
        timestamps=data.timestamps,
        word_timestamps=data.word_timestamps,
        video_url=url,
    )
