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
"""Routing the requested endpoints to the API."""

from fastapi import APIRouter

from wordcab_transcribe.config import settings
from wordcab_transcribe.router.v1.audio_file_endpoint import router as audio_file_router
from wordcab_transcribe.router.v1.audio_url_endpoint import router as audio_url_router
from wordcab_transcribe.router.v1.live_endpoints import router as live_router
from wordcab_transcribe.router.v1.youtube_endpoint import router as youtube_router


api_router = APIRouter()

include_api = api_router.include_router

routers = (
    ("audio_file_endpoint", audio_file_router, "/audio", "async"),
    ("audio_url_endpoint", audio_url_router, "/audio-url", "async"),
    ("youtube_endpoint", youtube_router, "/youtube", "async"),
    ("live_endpoint", live_router, "/live", "live"),
)

for router_items in routers:
    endpoint, router, prefix, tags = router_items

    # If the endpoint is enabled, include it in the API.
    if getattr(settings, endpoint) is True:
        include_api(router, prefix=prefix, tags=[tags])
