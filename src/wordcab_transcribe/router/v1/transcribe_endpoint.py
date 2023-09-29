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
"""Transcribe endpoint for the Remote Wordcab Transcribe API."""

from typing import Union

from fastapi import APIRouter
from fastapi import status as http_status

from wordcab_transcribe.models import TranscribeResponse

router = APIRouter()


@router.post(
    "",
    response_model=Union[TranscribeResponse, str],
    status_code=http_status.HTTP_200_OK,
)
async def remote_transcription() -> TranscribeResponse:
    """Transcribe endpoint for the Remote Wordcab Transcribe API."""
