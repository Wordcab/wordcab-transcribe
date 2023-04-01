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
"""Test batch request when the API is running, ignored if the API is not running."""

import asyncio

import aiohttp
import pytest

from tests.conftest import is_port_open


async def send_request() -> None:
    """Send a request to the API and print the response status."""
    async with aiohttp.ClientSession() as session:
        with open("tests/sample_1.mp3", "rb") as f:
            audio_file = {"file": f}
            async with session.post(
                "http://localhost:5001/api/v1/audio", data=audio_file
            ) as response:
                print(f"Got a response with status: {response.status}")


@pytest.mark.asyncio
@pytest.mark.skipif(not is_port_open(5001), reason="API is not running.")
async def test_batch_request() -> None:
    """Test batch request when the API is running, ignored if the API is not running."""
    tasks = [asyncio.create_task(send_request()) for _ in range(8)]
    for task in asyncio.as_completed(tasks):
        await task
