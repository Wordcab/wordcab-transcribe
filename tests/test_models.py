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

import pytest

from wordcab_transcribe.models import ASRResponse, DataRequest


def test_asr_response() -> None:
    """Test the ASRResponse model."""
    response = ASRResponse(
        utterances=[], alignment=False, source_lang="en", timestamps="s"
    )
    assert response.utterances == []
    assert response.alignment is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"

    response = ASRResponse(
        utterances=["Hello", "world"], alignment=True, source_lang="en", timestamps="s"
    )
    assert response.utterances == ["Hello", "world"]
    assert response.alignment is True
    assert response.source_lang == "en"
    assert response.timestamps == "s"


def test_data_request_valid() -> None:
    """Test the DataRequest model with valid data."""
    data = {
        "source_lang": "fr",
        "timestamps": "hms",
    }
    req = DataRequest(**data)
    assert req.source_lang == "fr"
    assert req.timestamps == "hms"


def test_data_request_default() -> None:
    """Test the DataRequest model with default values."""
    req = DataRequest()
    assert req.source_lang == "en"
    assert req.timestamps == "s"


def test_data_request_invalid() -> None:
    """Test the DataRequest model with invalid data."""
    with pytest.raises(ValueError, match="timestamps must be one of 'hms', 'ms', 's'."):
        DataRequest(timestamps="invalid")
