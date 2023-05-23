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
import shortuuid
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi import status as http_status

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import AudioRequest, AudioResponse
from wordcab_transcribe.utils import (
    convert_file_to_wav,
    convert_timestamp,
    delete_file,
    format_punct,
    is_empty_string,
    split_dual_channel_file,
)


router = APIRouter()


@router.post("", response_model=AudioResponse, status_code=http_status.HTTP_200_OK)
async def inference_with_audio(
    background_tasks: BackgroundTasks,
    alignment: Optional[bool] = Form(False),  # noqa: B008
    diarization: Optional[bool] = Form(False),  # noqa: B008
    dual_channel: Optional[bool] = Form(False),  # noqa: B008
    source_lang: Optional[str] = Form("en"),  # noqa: B008
    timestamps: Optional[str] = Form("s"),  # noqa: B008
    word_timestamps: Optional[bool] = Form(False),  # noqa: B008
    file: UploadFile = File(...),  # noqa: B008
) -> AudioResponse:
    """Inference endpoint with audio file."""
    extension = file.filename.split(".")[-1]
    filename = f"audio_{shortuuid.ShortUUID().random(length=32)}.{extension}"

    async with aiofiles.open(filename, "wb") as f:
        audio_bytes = await file.read()
        await f.write(audio_bytes)

    data = AudioRequest(
        alignment=alignment,
        diarization=diarization,
        source_lang=source_lang,
        timestamps=timestamps,
        word_timestamps=word_timestamps,
        dual_channel=dual_channel,
    )

    if data.dual_channel:
        filepath = await split_dual_channel_file(filename)
    else:
        filepath = await convert_file_to_wav(filename)
        background_tasks.add_task(delete_file, filepath=filename)

    raw_utterances = await asr.process_input(
        filepath,
        alignment=data.alignment,
        diarization=data.diarization,
        dual_channel=data.dual_channel,
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

    return AudioResponse(
        utterances=utterances,
        alignment=data.alignment,
        diarization=data.diarization,
        dual_channel=data.dual_channel,
        source_lang=data.source_lang,
        timestamps=data.timestamps,
        word_timestamps=data.word_timestamps,
    )
