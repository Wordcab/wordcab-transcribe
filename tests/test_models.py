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
"""Test the models for requests and responses."""

from wordcab_transcribe.models import ASRRequest, ASRResponse


def test_asr_request() -> None:
    """Test the ASRRequest model."""
    request = ASRRequest()
    assert request.url is None
    assert request.num_speakers is None

    request = ASRRequest(url="http://example.com", num_speakers=2)
    assert request.url == "http://example.com"
    assert request.num_speakers == 2


def test_asr_response() -> None:
    """Test the ASRResponse model."""
    response = ASRResponse(utterances=[])
    assert response.utterances == []

    response = ASRResponse(utterances=["Hello", "world"])
    assert response.utterances == ["Hello", "world"]
