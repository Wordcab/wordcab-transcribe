# Copyright 2024 The Wordcab Team. All rights reserved.
#
# Licensed under the MIT License (the "License");
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
"""Routing the requested endpoints to the API."""

from fastapi import APIRouter

from wordcab_transcribe.config import settings
from wordcab_transcribe.router.authentication import router as auth_router  # noqa: F401
from wordcab_transcribe.router.v1.audio_file_endpoint import router as audio_file_router
from wordcab_transcribe.router.v1.audio_url_endpoint import router as audio_url_router
from wordcab_transcribe.router.v1.cortex_endpoint import (  # noqa: F401
    router as cortex_router,
)
from wordcab_transcribe.router.v1.diarize_endpoint import router as diarize_router
from wordcab_transcribe.router.v1.live_endpoint import router as live_router
from wordcab_transcribe.router.v1.manage_remote_url import (
    router as manage_remote_url_router,
)
from wordcab_transcribe.router.v1.transcribe_endpoint import router as transcribe_router
from wordcab_transcribe.router.v1.youtube_endpoint import router as youtube_router

api_router = APIRouter()

async_routers = [
    (audio_file_router, "/audio", "async"),
    (audio_url_router, "/audio-url", "async"),
    (youtube_router, "/youtube", "async"),
]
live_routers = (live_router, "/live", "live")
transcribe_routers = (transcribe_router, "/transcribe", "transcription")
diarize_routers = (diarize_router, "/diarize", "diarization")
manage_remote_url_routers = (
    manage_remote_url_router,
    "/url",
    "remote-url",
)

routers = []
if settings.asr_type == "async":
    routers.extend(async_routers)
    routers.append(manage_remote_url_routers)
elif settings.asr_type == "live":
    routers.append(live_routers)
elif settings.asr_type == "only_transcription":
    routers.append(transcribe_routers)
elif settings.asr_type == "only_diarization":
    routers.append(diarize_routers)


for items in routers:
    router, prefix, tags = items
    api_router.include_router(router, prefix=prefix, tags=[tags])
