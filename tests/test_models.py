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

from wordcab_transcribe.models import (
    AudioRequest,
    AudioResponse,
    BaseRequest,
    BaseResponse,
    CortexError,
    CortexPayload,
    CortexUrlResponse,
    CortexYoutubeResponse,
    YouTubeResponse,
)


def test_audio_request() -> None:
    """Test the AudioRequest model."""
    request = AudioRequest(
        alignment=True,
        dual_channel=True,
        source_lang="en",
        timestamps="s",
    )
    assert request.alignment is True
    assert request.dual_channel is True
    assert request.source_lang == "en"
    assert request.timestamps == "s"


def test_audio_response() -> None:
    """Test the AudioResponse model."""
    response = AudioResponse(
        utterances=[],
        alignment=False,
        diarization=False,
        dual_channel=False,
        source_lang="en",
        timestamps="s",
        word_timestamps=False,
    )
    assert response.utterances == []
    assert response.alignment is False
    assert response.diarization is False
    assert response.dual_channel is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.word_timestamps is False

    response = AudioResponse(
        utterances=[
            {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
            {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
        ],
        alignment=True,
        diarization=True,
        dual_channel=True,
        source_lang="en",
        timestamps="s",
        word_timestamps=True,
    )
    assert response.utterances == [
        {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
        {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
    ]
    assert response.alignment is True
    assert response.diarization is True
    assert response.dual_channel is True
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.word_timestamps is True


def test_base_request_valid() -> None:
    """Test the BaseRequest model with valid data."""
    data = {
        "alignment": True,
        "source_lang": "fr",
        "timestamps": "hms",
    }
    req = BaseRequest(**data)
    assert req.alignment is True
    assert req.source_lang == "fr"
    assert req.timestamps == "hms"


def test_base_request_default() -> None:
    """Test the BaseRequest model with default values."""
    req = BaseRequest()
    assert req.alignment is False
    assert req.source_lang == "en"
    assert req.timestamps == "s"


def test_base_request_invalid() -> None:
    """Test the BaseRequest model with invalid data."""
    with pytest.raises(ValueError, match="timestamps must be one of 'hms', 'ms', 's'."):
        BaseRequest(timestamps="invalid")


def test_base_response() -> None:
    """Test the BaseResponse model."""
    response = BaseResponse(
        utterances=[
            {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
            {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
        ],
        alignment=True,
        diarization=False,
        source_lang="en",
        timestamps="s",
        word_timestamps=False,
    )
    assert response.utterances == [
        {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
        {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
    ]
    assert response.alignment is True
    assert response.diarization is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.word_timestamps is False


def test_cortex_error() -> None:
    """Test the CortexError model."""
    error = CortexError(
        detail="This is a test error",
    )
    assert error.detail == "This is a test error"


def test_corxet_payload() -> None:
    """Test the CortexPayload model."""
    payload = CortexPayload(
        url_type="youtube",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        api_key="test_api_key",
        alignment=True,
        diarization=False,
        dual_channel=False,
        source_lang="en",
        timestamps="s",
        word_timestamps=False,
        job_name="test_job",
        ping=False,
    )
    assert payload.url_type == "youtube"
    assert payload.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert payload.api_key == "test_api_key"
    assert payload.alignment is True
    assert payload.diarization is False
    assert payload.dual_channel is False
    assert payload.source_lang == "en"
    assert payload.timestamps == "s"
    assert payload.word_timestamps is False
    assert payload.job_name == "test_job"
    assert payload.ping is False


def test_cortex_url_response() -> None:
    """Test the CortexUrlResponse model."""
    response = CortexUrlResponse(
        utterances=[
            {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
            {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
        ],
        alignment=True,
        diarization=False,
        source_lang="en",
        timestamps="s",
        word_timestamps=False,
        dual_channel=False,
        job_name="test_job",
        request_id="test_request_id",
    )
    assert response.utterances == [
        {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
        {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
    ]
    assert response.alignment is True
    assert response.diarization is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.word_timestamps is False
    assert response.dual_channel is False
    assert response.job_name == "test_job"
    assert response.request_id == "test_request_id"


def test_cortex_youtube_response() -> None:
    """Test the CortexYoutubeResponse model."""
    response = CortexYoutubeResponse(
        utterances=[
            {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
            {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
        ],
        alignment=True,
        diarization=False,
        source_lang="en",
        timestamps="s",
        word_timestamps=False,
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        job_name="test_job",
        request_id="test_request_id",
    )
    assert response.utterances == [
        {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
        {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
    ]
    assert response.alignment is True
    assert response.diarization is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.word_timestamps is False
    assert response.video_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert response.job_name == "test_job"
    assert response.request_id == "test_request_id"


def test_youtube_response() -> None:
    """Test the YouTubeResponse model."""
    response = YouTubeResponse(
        utterances=[
            {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
            {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
        ],
        alignment=True,
        diarization=False,
        source_lang="en",
        timestamps="s",
        word_timestamps=False,
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    assert response.utterances == [
        {"text": "Never gonna give you up", "start": 0.0, "end": 3.0},
        {"text": "Never gonna let you down", "start": 3.0, "end": 6.0},
    ]
    assert response.alignment is True
    assert response.diarization is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.word_timestamps is False
    assert response.video_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
