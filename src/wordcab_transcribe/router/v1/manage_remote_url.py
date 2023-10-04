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
"""Add Remote URL endpoint for remote transcription or diarization."""

from typing import Union

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status
from loguru import logger
from pydantic import ValidationError

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import UrlRequest, UrlSchema
from wordcab_transcribe.services.asr_service import ProcessException

router = APIRouter()


@router.post(
    "/add",
    response_model=Union[UrlSchema, str],
    status_code=http_status.HTTP_200_OK,
)
async def add_url(data: UrlRequest) -> UrlSchema:
    """Add Remote URL endpoint for remote transcription or diarization."""
    try:
        _data = UrlSchema(task=data.task, url=data.url)
    except ValidationError as e:
        raise HTTPException(  # noqa: B904
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Your request is invalid: {e}",
        )

    result: UrlSchema = await asr.add_url(_data)

    if isinstance(result, ProcessException):
        logger.error(result.message)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(result.message),
        )

    return result


@router.post(
    "/remove",
    response_model=Union[UrlSchema, str],
    status_code=http_status.HTTP_200_OK,
)
async def remove_url(data: UrlRequest) -> UrlSchema:
    """Remove Remote URL endpoint for remote transcription or diarization."""
    try:
        _data = UrlSchema(task=data.task, url=data.url)
    except ValidationError as e:
        raise HTTPException(  # noqa: B904
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Your request is invalid: {e}",
        )

    result: UrlSchema = await asr.remove_url(_data)

    if isinstance(result, ProcessException):
        logger.error(result.message)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(result.message),
        )

    return result
