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

from typing import Optional

import shortuuid
from fastapi import APIRouter, BackgroundTasks
from fastapi import status as http_status

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import ASRResponse, DataRequest
from wordcab_transcribe.utils import (
    convert_timestamp,
    delete_file,
    download_file_from_youtube,
    format_punct,
    is_empty_string,
)


router = APIRouter()


# TODO: Are YT videos stereo at all? If not, then we can remove dual_channel here?
@router.post("", response_model=ASRResponse, status_code=http_status.HTTP_200_OK)
async def inference_with_youtube(
    background_tasks: BackgroundTasks,
    url: str,
    data: Optional[DataRequest] = None,
) -> ASRResponse:
    """Inference endpoint with YouTube url."""
    filename = f"yt_{shortuuid.ShortUUID().random(length=32)}"
    filepath = await download_file_from_youtube(url, filename)

    data = DataRequest() if data is None else DataRequest(**data.dict())

    raw_utterances = await asr.process_input(
        filepath,
        alignment=data.alignment,
        dual_channel=data.dual_channel,
        source_lang=data.source_lang,
    )

    timestamps_format = data.timestamps
    if not data.dual_channel:
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
    else:
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

    return ASRResponse(
        utterances=utterances,
        alignment=data.alignment,
        dual_channel=data.dual_channel,
        source_lang=data.source_lang,
        timestamps=data.timestamps,
    )
