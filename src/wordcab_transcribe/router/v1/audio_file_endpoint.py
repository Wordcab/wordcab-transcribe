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
"""Audio file endpoint for the Wordcab Transcribe API."""

import asyncio
from typing import List, Union

import shortuuid
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi import status as http_status
from loguru import logger

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import AudioRequest, AudioResponse
from wordcab_transcribe.utils import (
    check_num_channels,
    delete_file,
    process_audio_file,
    save_file_locally,
)

router = APIRouter()


@router.post(
    "", response_model=Union[AudioResponse, str], status_code=http_status.HTTP_200_OK
)
async def inference_with_audio(  # noqa: C901
    background_tasks: BackgroundTasks,
    offset_start: float = Form(None),  # noqa: B008
    offset_end: float = Form(None),  # noqa: B008
    num_speakers: int = Form(-1),  # noqa: B008
    diarization: bool = Form(False),  # noqa: B008
    multi_channel: bool = Form(False),  # noqa: B008
    source_lang: str = Form("en"),  # noqa: B008
    timestamps: str = Form("s"),  # noqa: B008
    vocab: List[str] = Form([]),  # noqa: B008
    word_timestamps: bool = Form(False),  # noqa: B008
    internal_vad: bool = Form(False),  # noqa: B008
    repetition_penalty: float = Form(1.2),  # noqa: B008
    compression_ratio_threshold: float = Form(2.4),  # noqa: B008
    log_prob_threshold: float = Form(-1.0),  # noqa: B008
    no_speech_threshold: float = Form(0.6),  # noqa: B008
    condition_on_previous_text: bool = Form(True),  # noqa: B008
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
        offset_start=offset_start,
        offset_end=offset_end,
        num_speakers=num_speakers,
        diarization=diarization,
        source_lang=source_lang,
        timestamps=timestamps,
        vocab=vocab,
        word_timestamps=word_timestamps,
        internal_vad=internal_vad,
        repetition_penalty=repetition_penalty,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        multi_channel=multi_channel,
    )

    num_channels = await check_num_channels(filename)
    print(f"num_channels: {num_channels}")
    if num_channels > 1 and data.multi_channel is False:
        num_channels = 1  # Force mono channel if more than 1 channel

    try:
        filepath: Union[str, List[str]] = await process_audio_file(
            filename, num_channels=num_channels
        )

    except Exception as e:
        raise HTTPException(  # noqa: B904
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Process failed: {e}",
        )

    background_tasks.add_task(delete_file, filepath=filename)

    task = asyncio.create_task(
        asr.process_input(
            filepath=filepath,
            offset_start=data.offset_start,
            offset_end=data.offset_end,
            num_speakers=data.num_speakers,
            diarization=data.diarization,
            multi_channel=data.multi_channel,
            source_lang=data.source_lang,
            timestamps_format=data.timestamps,
            vocab=data.vocab,
            word_timestamps=data.word_timestamps,
            internal_vad=data.internal_vad,
            repetition_penalty=data.repetition_penalty,
            compression_ratio_threshold=data.compression_ratio_threshold,
            log_prob_threshold=data.log_prob_threshold,
            no_speech_threshold=data.no_speech_threshold,
            condition_on_previous_text=data.condition_on_previous_text,
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
        utterances, process_times, audio_duration = result
        return AudioResponse(
            utterances=utterances,
            audio_duration=audio_duration,
            offset_start=data.offset_start,
            offset_end=data.offset_end,
            num_speakers=data.num_speakers,
            diarization=data.diarization,
            multi_channel=data.multi_channel,
            source_lang=data.source_lang,
            timestamps=data.timestamps,
            vocab=data.vocab,
            word_timestamps=data.word_timestamps,
            internal_vad=data.internal_vad,
            repetition_penalty=data.repetition_penalty,
            compression_ratio_threshold=data.compression_ratio_threshold,
            log_prob_threshold=data.log_prob_threshold,
            no_speech_threshold=data.no_speech_threshold,
            condition_on_previous_text=data.condition_on_previous_text,
            process_times=process_times,
        )
