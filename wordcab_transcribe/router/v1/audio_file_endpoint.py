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

from typing import Optional

import aiofiles
import asyncio
import shortuuid
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi import status as http_status

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import ASRResponse, DataRequest
from wordcab_transcribe.utils import (
    convert_file_to_wav,
    convert_timestamp,
    delete_file,
    enhance_audio,
    format_punct,
    is_empty_string,
    split_dual_channel_file,
)


router = APIRouter()


@router.post("", response_model=ASRResponse, status_code=http_status.HTTP_200_OK)
async def inference_with_audio(
    background_tasks: BackgroundTasks,
    alignment: Optional[bool] = Form(False),  # noqa: B008
    dual_channel: Optional[bool] = Form(False),  # noqa: B008
    source_lang: Optional[str] = Form("en"),  # noqa: B008
    timestamps: Optional[str] = Form("s"),  # noqa: B008
    file: UploadFile = File(...),  # noqa: B008
) -> ASRResponse:
    """Inference endpoint with audio file."""
    extension = file.filename.split(".")[-1]
    filename = f"audio_{shortuuid.ShortUUID().random(length=32)}.{extension}"

    async with aiofiles.open(filename, "wb") as f:
        audio_bytes = await file.read()
        await f.write(audio_bytes)

    if dual_channel is False:
        filepath = await convert_file_to_wav(filename)
        background_tasks.add_task(delete_file, filepath=filename)
    else:
        enhanced_audio_filepath = asyncio.run_in_executor(
            None, enhance_audio, filename
        )
        filepath = await split_dual_channel_file(enhanced_audio_filepath)
        background_tasks.add_task(delete_file, filepath=enhanced_audio_filepath)

    data = DataRequest(
        alignment=alignment,
        dual_channel=dual_channel,
        source_lang=source_lang,
        timestamps=timestamps,
    )

    raw_utterances = await asr.process_input(
        filepath,
        alignment=data.alignment,
        dual_channel=data.dual_channel,
        source_lang=data.source_lang,
    )

    timestamps_format = data.timestamps
    utterances = [
        {
            "text": format_punct(utterance["text"]),
            "start": convert_timestamp(utterance["start"], timestamps_format),
            "end": convert_timestamp(utterance["end"], timestamps_format),
            "speaker": int(utterance["speaker"]),
        }
        for utterance in raw_utterances
        if not is_empty_string(utterance["text"])
    ]

    background_tasks.add_task(delete_file, filepath=filepath)

    return ASRResponse(
        utterances=utterances,
        alignment=data.alignment,
        dual_channel=data.dual_channel,
        source_lang=data.source_lang,
        timestamps=data.timestamps,
    )
