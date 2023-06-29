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
"""Cortex endpoint module of the Wordcab Transcribe API."""

from datetime import datetime
from typing import Optional, Union

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi import status as http_status
from loguru import logger
from svix.api import MessageIn, SvixAsync

from wordcab_transcribe.config import settings
from wordcab_transcribe.models import (
    AudioRequest,
    AudioResponse,
    BaseRequest,
    CortexError,
    CortexPayload,
    CortexUrlResponse,
    CortexYoutubeResponse,
    PongResponse,
    YouTubeResponse,
)
from wordcab_transcribe.router.v1.audio_url_endpoint import inference_with_audio_url
from wordcab_transcribe.router.v1.youtube_endpoint import inference_with_youtube
from wordcab_transcribe.utils import remove_words_for_svix


router = APIRouter()


@router.post(
    "/",
    response_model=Union[
        CortexError, CortexUrlResponse, CortexYoutubeResponse, PongResponse
    ],
    status_code=http_status.HTTP_200_OK,
)
async def run_cortex(
    payload: CortexPayload, request: Request
) -> Union[CortexError, CortexUrlResponse, CortexYoutubeResponse, PongResponse]:
    """Root endpoint for Cortex."""
    logger.debug("Received a request from Cortex.")

    if payload.api_key != settings.cortex_api_key or not payload.api_key:
        return CortexError(message="Invalid API key.")

    if payload.ping:
        return PongResponse(message="pong")

    request_id = request.headers.get("x-cortex-request-id", None)

    try:
        if payload.url_type == "audio_url":
            data = AudioRequest(
                alignment=payload.alignment,
                diarization=payload.diarization,
                dual_channel=payload.dual_channel,
                source_lang=payload.source_lang,
                timestamps=payload.timestamps,
                use_batch=payload.use_batch,
                vocab=payload.vocab,
                word_timestamps=payload.word_timestamps,
            )
            utterances: AudioResponse = await inference_with_audio_url(
                background_tasks=BackgroundTasks(),
                url=payload.url,
                data=data,
            )

        elif payload.url_type == "youtube":
            data = BaseRequest(
                alignment=payload.alignment,
                diarization=payload.diarization,
                source_lang=payload.source_lang,
                timestamps=payload.timestamps,
                use_batch=payload.use_batch,
                vocab=payload.vocab,
                word_timestamps=payload.word_timestamps,
            )
            utterances: YouTubeResponse = await inference_with_youtube(
                background_tasks=BackgroundTasks(),
                url=payload.url,
                data=data,
            )

        else:
            return CortexError(
                message="Invalid url_type parameter. Supported values are 'audio_url' and 'youtube'"
            )

    except Exception as e:
        error_message = f"Error during transcription: {e}"
        logger.error(error_message)

        error_payload = {
            "error": error_message,
            "job_name": payload.job_name,
            "request_id": request_id,
        }

        await send_update_with_svix(payload.job_name, "error", error_payload)

        return CortexError(message=error_message)

    _cortex_response = {
        **utterances.dict(),
        "job_name": payload.job_name,
        "request_id": request_id,
    }

    await send_update_with_svix(
        payload.job_name, "finished", remove_words_for_svix(_cortex_response)
    )

    if payload.url_type == "youtube":
        return CortexYoutubeResponse(**_cortex_response)
    else:
        return CortexUrlResponse(**_cortex_response)


async def send_update_with_svix(
    job_name: str,
    status: str,
    payload: dict,
    payload_retention_period: Optional[int] = 5,
) -> None:
    """
    Send the status update to Svix.

    Args:
        job_name (str): The name of the job.
        status (str): The status of the job.
        payload (dict): The payload to send.
        payload_retention_period (Optional[int], optional): The payload retention period. Defaults to 5.
    """
    if settings.svix_api_key and settings.svix_app_id:
        svix = SvixAsync(settings.svix_api_key)
        await svix.message.create(
            settings.svix_app_id,
            MessageIn(
                event_type=f"async_job.wordcab_transcribe.{status}",
                event_id=f"wordcab_transcribe_{status}_{job_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}",
                payload_retention_period=payload_retention_period,
                payload=payload,
            ),
        )
    else:
        logger.warning(
            "Svix API key and app ID are not set. Cannot send the status update to Svix."
        )
