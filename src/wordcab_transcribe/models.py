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
"""Models module of the Wordcab Transcribe."""

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, field_validator


class ProcessTimes(BaseModel):
    """The execution times of the different processes."""

    total: float
    transcription: float
    diarization: Union[float, None]
    post_processing: float


class Timestamps(str, Enum):
    """Timestamps enum for the API."""

    seconds = "s"
    milliseconds = "ms"
    hour_minute_second = "hms"


class Word(BaseModel):
    """Word model for the API."""

    word: str
    start: float
    end: float
    score: float


class Utterance(BaseModel):
    """Utterance model for the API."""

    text: str
    start: float
    end: float
    speaker: Optional[int]
    words: Optional[List[Word]]


class BaseResponse(BaseModel):
    """Base response model, not meant to be used directly."""

    utterances: List[Utterance]
    audio_duration: float
    num_speakers: int
    diarization: bool
    source_lang: str
    timestamps: str
    vocab: List[str]
    word_timestamps: bool
    internal_vad: bool
    repetition_penalty: float
    compression_ratio_threshold: float
    log_prob_threshold: float
    no_speech_threshold: float
    condition_on_previous_text: bool
    process_times: ProcessTimes


class AudioResponse(BaseResponse):
    """Response model for the ASR audio file and url endpoint."""

    dual_channel: bool

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "utterances": [
                    {
                        "text": "Hello World!",
                        "start": 0.345,
                        "end": 1.234,
                        "speaker": 0,
                    },
                    {
                        "text": "Wordcab is awesome",
                        "start": 1.234,
                        "end": 2.678,
                        "speaker": 1,
                    },
                ],
                "audio_duration": 2.678,
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "process_times": {
                    "total": 2.678,
                    "transcription": 2.439,
                    "diarization": None,
                    "post_processing": 0.239,
                },
                "dual_channel": False,
            }
        }


class YouTubeResponse(BaseResponse):
    """Response model for the ASR YouTube endpoint."""

    video_url: str

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "utterances": [
                    {
                        "speaker": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Never gonna give you up!",
                    },
                    {
                        "speaker": 0,
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Never gonna let you down!",
                    },
                ],
                "audio_duration": 2.0,
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "process_times": {
                    "total": 2.678,
                    "transcription": 2.439,
                    "diarization": None,
                    "post_processing": 0.239,
                },
                "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            }
        }


class CortexError(BaseModel):
    """Error model for the Cortex API."""

    message: str

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "message": "Error message here",
            }
        }


class CortexPayload(BaseModel):
    """Request object for Cortex endpoint."""

    url_type: Literal["audio_url", "youtube"]
    url: Optional[str] = None
    api_key: Optional[str] = None
    num_speakers: Optional[int] = -1
    diarization: Optional[bool] = False
    dual_channel: Optional[bool] = False
    source_lang: Optional[str] = "en"
    timestamps: Optional[Timestamps] = Timestamps.seconds
    vocab: Optional[List[str]] = []
    word_timestamps: Optional[bool] = False
    internal_vad: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.2
    compression_ratio_threshold: Optional[float] = 2.4
    log_prob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    condition_on_previous_text: Optional[bool] = True
    job_name: Optional[str] = None
    ping: Optional[bool] = False

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "url_type": "youtube",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "api_key": "1234567890",
                "num_speakers": -1,
                "diarization": False,
                "dual_channel": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "job_name": "job_abc123",
                "ping": False,
            }
        }


class CortexUrlResponse(AudioResponse):
    """Response model for the audio_url type of the Cortex endpoint."""

    job_name: str
    request_id: Optional[str] = None

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "utterances": [
                    {
                        "speaker": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Hello World!",
                    },
                    {
                        "speaker": 0,
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Wordcab is awesome",
                    },
                ],
                "audio_duration": 2.0,
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "process_times": {
                    "total": 2.678,
                    "transcription": 2.439,
                    "diarization": None,
                    "post_processing": 0.239,
                },
                "dual_channel": False,
                "job_name": "job_name",
                "request_id": "request_id",
            }
        }


class CortexYoutubeResponse(YouTubeResponse):
    """Response model for the YouTube type of the Cortex endpoint."""

    job_name: str
    request_id: Optional[str] = None

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "utterances": [
                    {
                        "speaker": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Never gonna give you up!",
                    },
                    {
                        "speaker": 0,
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Never gonna let you down!",
                    },
                ],
                "audio_duration": 2.0,
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "process_times": {
                    "total": 2.678,
                    "transcription": 2.439,
                    "diarization": None,
                    "post_processing": 0.239,
                },
                "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "job_name": "job_name",
                "request_id": "request_id",
            }
        }


class BaseRequest(BaseModel):
    """Base request model for the API."""

    num_speakers: int = -1
    diarization: bool = False
    source_lang: str = "en"
    timestamps: Timestamps = Timestamps.seconds
    vocab: List[str] = []
    word_timestamps: bool = False
    internal_vad: bool = False
    repetition_penalty: float = 1.2
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True

    @field_validator("vocab")
    def validate_each_vocab_value(
        cls, value: List[str]  # noqa: B902, N805
    ) -> List[str]:
        """Validate the value of each vocab field."""
        if not all(isinstance(v, str) for v in value):
            raise ValueError("`vocab` must be a list of strings.")

        return value

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
            }
        }


class AudioRequest(BaseRequest):
    """Request model for the ASR audio file and url endpoint."""

    dual_channel: bool = False

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "num_speakers": -1,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "vocab": [
                    "custom company name",
                    "custom product name",
                    "custom co-worker name",
                ],
                "word_timestamps": False,
                "internal_vad": False,
                "repetition_penalty": 1.2,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "dual_channel": False,
            }
        }


class PongResponse(BaseModel):
    """Response model for the ping endpoint."""

    message: str

    class Config:
        """Pydantic config class."""

        json_schema_extra = {
            "example": {
                "message": "pong",
            },
        }


class Token(BaseModel):
    """Token model for authentication."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """TokenData model for authentication."""

    username: Optional[str] = None
