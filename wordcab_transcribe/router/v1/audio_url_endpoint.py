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

from typing import Optional

import shortuuid
from fastapi import APIRouter, BackgroundTasks
from fastapi import status as http_status

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import AudioRequest, AudioResponse
from wordcab_transcribe.utils import (
    convert_timestamp,
    delete_file,
    download_audio_file,
    format_punct,
    is_empty_string,
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

    if (
        data.dual_channel
    ):  # TODO: In AWS URLs extension seems to be in URL, also generally find extension from any file
        filepath = await download_audio_file(url, filename, guess_extension=False)
        filepath = await split_dual_channel_file(filepath)
        background_tasks.add_task(delete_file, filepath=f"{filename}.wav")
    else:
        filepath = await download_audio_file(url, filename, guess_extension=False)
        # extension = filepath.split(".")[-1]
        # filepath = await convert_file_to_wav(filepath)
        # TODO: As only Wordcab would really use this, and we convert to wav, commenting this out
        background_tasks.add_task(delete_file, filepath=f"{filename}.wav")

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
            "start": convert_timestamp(
                utterance["start"], timestamps_format, data.dual_channel
            ),
            "end": convert_timestamp(
                utterance["end"], timestamps_format, data.dual_channel
            ),
            "speaker": int(utterance["speaker"]),
        }
        for utterance in raw_utterances
        if not is_empty_string(utterance["text"])
    ]

    background_tasks.add_task(delete_file, filepath=filepath)

    return AudioResponse(
        utterances=utterances,
        alignment=data.alignment,
        dual_channel=data.dual_channel,
        source_lang=data.source_lang,
        timestamps=data.timestamps,
    )
