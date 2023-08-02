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
    Utterance,
    Word,
    YouTubeResponse,
)


def test_word() -> None:
    """Test the Word model."""
    word = Word(
        word="test",
        start=0.0,
        end=1.0,
        score=0.9,
    )
    assert word.word == "test"
    assert word.start == 0.0
    assert word.end == 1.0
    assert word.score == 0.9


def test_utterance() -> None:
    """Test the Utterance model."""
    utterance = Utterance(
        text="This is a test.",
        start=0.0,
        end=4.0,
        speaker=0,
        words=[
            Word(
                word="This",
                start=0.0,
                end=1.0,
                score=0.9,
            ),
            Word(
                word="is",
                start=1.0,
                end=2.0,
                score=0.75,
            ),
            Word(
                word="a",
                start=2.0,
                end=3.0,
                score=0.8,
            ),
            Word(
                word="test.",
                start=3.0,
                end=4.0,
                score=0.85,
            ),
        ],
    )
    assert utterance.text == "This is a test."
    assert utterance.start == 0.0
    assert utterance.end == 4.0
    assert utterance.speaker == 0
    assert utterance.words is not None
    assert utterance.words == [
        Word(
            word="This",
            start=0.0,
            end=1.0,
            score=0.9,
        ),
        Word(
            word="is",
            start=1.0,
            end=2.0,
            score=0.75,
        ),
        Word(
            word="a",
            start=2.0,
            end=3.0,
            score=0.8,
        ),
        Word(
            word="test.",
            start=3.0,
            end=4.0,
            score=0.85,
        ),
    ]
    assert isinstance(utterance.words[0], Word)


def test_audio_request() -> None:
    """Test the AudioRequest model."""
    request = AudioRequest(
        alignment=True,
        diarization=True,
        dual_channel=True,
        source_lang="en",
        timestamps="s",
    )
    assert request.alignment is True
    assert request.diarization is True
    assert request.dual_channel is True
    assert request.source_lang == "en"
    assert request.timestamps == "s"
    assert request.use_batch is False
    assert request.vocab == []
    assert request.word_timestamps is False
    assert request.internal_vad is False


def test_audio_response() -> None:
    """Test the AudioResponse model."""
    response = AudioResponse(
        utterances=[],
        audio_duration=0.0,
        alignment=False,
        diarization=False,
        dual_channel=False,
        source_lang="en",
        timestamps="s",
        use_batch=False,
        vocab=["custom company", "custom product"],
        word_timestamps=False,
        internal_vad=False,
    )
    assert response.utterances == []
    assert response.audio_duration == 0.0
    assert response.alignment is False
    assert response.diarization is False
    assert response.dual_channel is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.use_batch is False
    assert response.vocab == ["custom company", "custom product"]
    assert response.word_timestamps is False
    assert response.internal_vad is False

    response = AudioResponse(
        utterances=[
            Utterance(
                text="Never gonna give you up",
                start=0.0,
                end=3.0,
                speaker=0,
                words=[],
            ),
            Utterance(
                text="Never gonna let you down",
                start=3.0,
                end=6.0,
                speaker=1,
                words=[],
            ),
        ],
        audio_duration=6.0,
        alignment=True,
        diarization=True,
        dual_channel=True,
        source_lang="en",
        timestamps="s",
        use_batch=False,
        vocab=["custom company", "custom product"],
        word_timestamps=True,
        internal_vad=False,
    )
    assert response.utterances == [
        Utterance(
            text="Never gonna give you up",
            start=0.0,
            end=3.0,
            speaker=0,
            words=[],
        ),
        Utterance(
            text="Never gonna let you down",
            start=3.0,
            end=6.0,
            speaker=1,
            words=[],
        ),
    ]
    assert response.audio_duration == 6.0
    assert response.alignment is True
    assert response.diarization is True
    assert response.dual_channel is True
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.use_batch is False
    assert response.vocab == ["custom company", "custom product"]
    assert response.word_timestamps is True
    assert response.internal_vad is False


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
    assert req.diarization is False
    assert req.source_lang == "en"
    assert req.timestamps == "s"
    assert req.use_batch is False
    assert req.word_timestamps is False
    assert req.internal_vad is False


def test_base_request_invalid() -> None:
    """Test the BaseRequest model with invalid data."""
    with pytest.raises(
        ValueError, match="`timestamps` must be one of 'hms', 'ms', 's'."
    ):
        BaseRequest(timestamps="invalid")


def test_base_response() -> None:
    """Test the BaseResponse model."""
    response = BaseResponse(
        utterances=[
            Utterance(
                text="Never gonna give you up",
                start=0.0,
                end=3.0,
                speaker=0,
                words=[],
            ),
            Utterance(
                text="Never gonna let you down",
                start=3.0,
                end=6.0,
                speaker=1,
                words=[],
            ),
        ],
        audio_duration=6.0,
        alignment=True,
        diarization=False,
        source_lang="en",
        timestamps="s",
        use_batch=False,
        vocab=["custom company", "custom product"],
        word_timestamps=False,
        internal_vad=False,
    )
    assert response.utterances == [
        Utterance(
            text="Never gonna give you up",
            start=0.0,
            end=3.0,
            speaker=0,
            words=[],
        ),
        Utterance(
            text="Never gonna let you down",
            start=3.0,
            end=6.0,
            speaker=1,
            words=[],
        ),
    ]
    assert response.audio_duration == 6.0
    assert response.alignment is True
    assert response.diarization is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.use_batch is False
    assert response.vocab == ["custom company", "custom product"]
    assert response.word_timestamps is False
    assert response.internal_vad is False


def test_cortex_error() -> None:
    """Test the CortexError model."""
    error = CortexError(
        message="This is a test error",
    )
    assert error.message == "This is a test error"


def test_cortex_payload() -> None:
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
        use_batch=False,
        word_timestamps=False,
        internal_vad=False,
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
    assert payload.use_batch is False
    assert payload.vocab == []
    assert payload.word_timestamps is False
    assert payload.internal_vad is False
    assert payload.job_name == "test_job"
    assert payload.ping is False


def test_cortex_url_response() -> None:
    """Test the CortexUrlResponse model."""
    response = CortexUrlResponse(
        utterances=[
            Utterance(
                text="Never gonna give you up",
                start=0.0,
                end=3.0,
                speaker=0,
                words=[],
            ),
            Utterance(
                text="Never gonna let you down",
                start=3.0,
                end=6.0,
                speaker=1,
                words=[],
            ),
        ],
        audio_duration=6.0,
        alignment=True,
        diarization=False,
        source_lang="en",
        timestamps="s",
        use_batch=False,
        vocab=["custom company", "custom product"],
        word_timestamps=False,
        internal_vad=False,
        dual_channel=False,
        job_name="test_job",
        request_id="test_request_id",
    )
    assert response.utterances == [
        Utterance(
            text="Never gonna give you up",
            start=0.0,
            end=3.0,
            speaker=0,
            words=[],
        ),
        Utterance(
            text="Never gonna let you down",
            start=3.0,
            end=6.0,
            speaker=1,
            words=[],
        ),
    ]
    assert response.audio_duration == 6.0
    assert response.alignment is True
    assert response.diarization is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.use_batch is False
    assert response.vocab == ["custom company", "custom product"]
    assert response.word_timestamps is False
    assert response.internal_vad is False
    assert response.dual_channel is False
    assert response.job_name == "test_job"
    assert response.request_id == "test_request_id"


def test_cortex_youtube_response() -> None:
    """Test the CortexYoutubeResponse model."""
    response = CortexYoutubeResponse(
        utterances=[
            Utterance(
                text="Never gonna give you up",
                start=0.0,
                end=3.0,
                speaker=0,
                words=[],
            ),
            Utterance(
                text="Never gonna let you down",
                start=3.0,
                end=6.0,
                speaker=1,
                words=[],
            ),
        ],
        audio_duration=6.0,
        alignment=True,
        diarization=False,
        source_lang="en",
        timestamps="s",
        use_batch=False,
        vocab=["custom company", "custom product"],
        word_timestamps=False,
        internal_vad=False,
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        job_name="test_job",
        request_id="test_request_id",
    )
    assert response.utterances == [
        Utterance(
            text="Never gonna give you up",
            start=0.0,
            end=3.0,
            speaker=0,
            words=[],
        ),
        Utterance(
            text="Never gonna let you down",
            start=3.0,
            end=6.0,
            speaker=1,
            words=[],
        ),
    ]
    assert response.audio_duration == 6.0
    assert response.alignment is True
    assert response.diarization is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.use_batch is False
    assert response.vocab == ["custom company", "custom product"]
    assert response.word_timestamps is False
    assert response.internal_vad is False
    assert response.video_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert response.job_name == "test_job"
    assert response.request_id == "test_request_id"


def test_youtube_response() -> None:
    """Test the YouTubeResponse model."""
    response = YouTubeResponse(
        utterances=[
            Utterance(
                text="Never gonna give you up",
                start=0.0,
                end=3.0,
                speaker=0,
                words=[],
            ),
            Utterance(
                text="Never gonna let you down",
                start=3.0,
                end=6.0,
                speaker=1,
                words=[],
            ),
        ],
        audio_duration=6.0,
        alignment=True,
        diarization=False,
        source_lang="en",
        timestamps="s",
        use_batch=False,
        vocab=["custom company", "custom product"],
        word_timestamps=False,
        internal_vad=False,
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    assert response.utterances == [
        Utterance(
            text="Never gonna give you up",
            start=0.0,
            end=3.0,
            speaker=0,
            words=[],
        ),
        Utterance(
            text="Never gonna let you down",
            start=3.0,
            end=6.0,
            speaker=1,
            words=[],
        ),
    ]
    assert response.audio_duration == 6.0
    assert response.alignment is True
    assert response.diarization is False
    assert response.source_lang == "en"
    assert response.timestamps == "s"
    assert response.use_batch is False
    assert response.vocab == ["custom company", "custom product"]
    assert response.word_timestamps is False
    assert response.internal_vad is False
    assert response.video_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
