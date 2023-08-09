# Copyright 2023 The Wordcab Team. All rights reserved.
#
# Licensed under the Wordcab Transcribe License 0.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Wordcab/wordcab-transcribe/blob/main/LICENSE
#
# Except as expressly provided otherwise herein, and to the fullest
# extent permitted by law, Licensor provides the Software (and each
# Contributor provides its Contributions) AS IS, and Licensor
# disclaims all warranties or guarantees of any kind, express or
# implied, whether arising under any law or from any usage in trade,
# or otherwise including but not limited to the implied warranties
# of merchantability, non-infringement, quiet enjoyment, fitness
# for a particular purpose, or otherwise.
#
# See the License for the specific language governing permissions
# and limitations under the License.
"""Audio url endpoint for the Wordcab Transcribe API."""

import asyncio
from typing import Optional

import shortuuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi import status as http_status
from loguru import logger

from wordcab_transcribe.dependencies import asr, download_limit
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

    async with download_limit:
        _filepath = await download_audio_file("url", url, filename)

        if data.dual_channel:
            try:
                filepath = await split_dual_channel_file(_filepath)
            except Exception as e:
                logger.error(f"{e}\nFallback to single channel mode.")
                data.dual_channel = False

        if not data.dual_channel:
            try:
                filepath = await convert_file_to_wav(_filepath)

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
                num_speakers=data.num_speakers,
                diarization=data.diarization,
                dual_channel=data.dual_channel,
                source_lang=data.source_lang,
                timestamps_format=data.timestamps,
                use_batch=data.use_batch,
                vocab=data.vocab,
                word_timestamps=data.word_timestamps,
                internal_vad=data.internal_vad,
                repetition_penalty=data.repetition_penalty,
            )
        )
        result = await task

    background_tasks.add_task(delete_file, filepath=filepath)

    if isinstance(result, Exception):
        logger.error(f"Error: {result}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(result),
        )
    else:
        utterances, audio_duration = result
        return AudioResponse(
            utterances=utterances,
            audio_duration=audio_duration,
            alignment=data.alignment,
            num_speakers=data.num_speakers,
            diarization=data.diarization,
            dual_channel=data.dual_channel,
            source_lang=data.source_lang,
            timestamps=data.timestamps,
            use_batch=data.use_batch,
            vocab=data.vocab,
            word_timestamps=data.word_timestamps,
            internal_vad=data.internal_vad,
            repetition_penalty=data.repetition_penalty,
        )
