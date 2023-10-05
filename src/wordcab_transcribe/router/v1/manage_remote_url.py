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

from typing import List, Union

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status
from loguru import logger
from pydantic import HttpUrl
from typing_extensions import Literal

from wordcab_transcribe.dependencies import asr
from wordcab_transcribe.models import UrlSchema
from wordcab_transcribe.services.asr_service import ExceptionSource, ProcessException

router = APIRouter()


@router.get(
    "",
    response_model=Union[List[HttpUrl], str],
    status_code=http_status.HTTP_200_OK,
)
async def get_url(task: Literal["transcription", "diarization"]) -> List[HttpUrl]:
    """Get Remote URL endpoint for remote transcription or diarization."""
    result: List[UrlSchema] = await asr.get_url(task)

    if isinstance(result, ProcessException):
        logger.error(result.message)
        if result.source == ExceptionSource.get_url:
            raise HTTPException(
                status_code=http_status.HTTP_405_METHOD_NOT_ALLOWED,
                detail=str(result.message),
            )
        else:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(result.message),
            )

    return result


@router.post(
    "/add",
    response_model=Union[UrlSchema, str],
    status_code=http_status.HTTP_200_OK,
)
async def add_url(data: UrlSchema) -> UrlSchema:
    """Add Remote URL endpoint for remote transcription or diarization."""
    result: UrlSchema = await asr.add_url(data)

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
async def remove_url(data: UrlSchema) -> UrlSchema:
    """Remove Remote URL endpoint for remote transcription or diarization."""
    result: UrlSchema = await asr.remove_url(data)

    if isinstance(result, ProcessException):
        logger.error(result.message)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(result.message),
        )

    return result
