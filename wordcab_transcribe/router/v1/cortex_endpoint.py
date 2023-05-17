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
from typing import Union

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi import status as http_status
from loguru import logger

from wordcab_transcribe.config import settings
from wordcab_transcribe.models import (
    ASRResponse,
    CortexError,
    CortexPayload,
    CortexResponse,
    DataRequest,
    PongResponse,
)
from wordcab_transcribe.router.v1.audio_url_endpoint import inference_with_audio_url
from wordcab_transcribe.router.v1.youtube_endpoint import inference_with_youtube


router = APIRouter()


@router.post(
    "/",
    response_model=Union[CortexError, CortexResponse, PongResponse],
    status_code=http_status.HTTP_200_OK,
)
async def run_cortex(payload: CortexPayload, request: Request) -> CortexResponse:
    """Root endpoint for Cortex."""
    logger.debug("Received a request from Cortex.")

    if payload.api_key != settings.cortex_api_key or not payload.api_key:
        return CortexError(message="Invalid API key.")

    if payload.ping:
        return PongResponse(message="pong")

    request_id = request.headers.get("x-cortex-request-id", None)

    data = DataRequest(
        alignment=payload.alignment,
        dual_channel=payload.dual_channel,
        source_lang=payload.source_lang,
        timestamps=payload.timestamps,
    )

    try:
        if payload.url_type == "audio_url":
            utterances: ASRResponse = await inference_with_audio_url(
                background_tasks=BackgroundTasks(),
                url=payload.url,
                data=data,
            )
        elif payload.url_type == "youtube":
            utterances: ASRResponse = await inference_with_youtube(
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

        if settings.svix_api_key and settings.svix_app_id:
            from svix.api import MessageIn, Svix

            svix = Svix(settings.svix_api_key)
            svix.message.create(
                settings.svix_app_id,
                MessageIn(
                    event_type="async_job.wordcab_transcribe.error",
                    event_id=f"wordcab_transcribe_error_{payload.job_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}",
                    payload_retention_period=5,
                    payload={"error": error_message},
                ),
            )

        return CortexError(message=error_message)

    _cortex_response = {
        **utterances.dict(),
        "job_name": payload.job_name,
        "request_id": request_id,
    }

    if settings.svix_api_key and settings.svix_app_id:
        from svix.api import MessageIn, Svix

        svix = Svix(settings.svix_api_key)
        svix.message.create(
            settings.svix_app_id,
            MessageIn(
                event_type="async_job.wordcab_transcribe.finished",
                event_id=f"wordcab_transcribe_finished_{payload.job_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}",
                payload_retention_period=5,
                payload=_cortex_response,
            ),
        )

    return CortexResponse(**_cortex_response)
