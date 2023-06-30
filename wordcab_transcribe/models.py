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
"""Models module of the Wordcab Transcribe."""

from typing import List, Optional

from pydantic import BaseModel, validator


class BaseResponse(BaseModel):
    """Base response model, not meant to be used directly."""

    utterances: List[dict]
    audio_duration: float
    alignment: bool
    diarization: bool
    source_lang: str
    timestamps: str
    use_batch: bool
    word_timestamps: bool


class AudioResponse(BaseResponse):
    """Response model for the ASR audio file and url endpoint."""

    dual_channel: bool

    class Config:
        """Pydantic config class."""

        schema_extra = {
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
                "alignment": False,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "use_batch": False,
                "word_timestamps": False,
                "dual_channel": False,
            }
        }


class YouTubeResponse(BaseResponse):
    """Response model for the ASR YouTube endpoint."""

    video_url: str

    class Config:
        """Pydantic config class."""

        schema_extra = {
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
                "alignment": False,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "use_batch": False,
                "word_timestamps": False,
                "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            }
        }


class CortexError(BaseModel):
    """Error model for the Cortex API."""

    message: str

    class Config:
        """Pydantic config class."""

        schema_extra = {
            "example": {
                "message": "Error message here",
            }
        }


class CortexPayload(BaseModel):
    """Request object for Cortex endpoint."""

    url_type: str = "audio_url"
    url: Optional[str] = None
    api_key: Optional[str] = None
    alignment: Optional[bool] = False
    diarization: Optional[bool] = False
    dual_channel: Optional[bool] = False
    source_lang: Optional[str] = "en"
    timestamps: Optional[str] = "s"
    use_batch: Optional[bool] = False
    word_timestamps: Optional[bool] = False
    job_name: Optional[str] = None
    ping: Optional[bool] = False

    @validator("timestamps")
    def validate_timestamps_values(cls, value: str) -> str:  # noqa: B902, N805
        """Validate the value of the timestamps field."""
        if value not in ["hms", "ms", "s"]:
            raise ValueError("timestamps must be one of 'hms', 'ms', 's'.")
        return value

    @validator("url_type")
    def validate_url_type(cls, value: str) -> str:  # noqa: B902, N805
        """Validate the value of the url_type field."""
        if value not in ["audio_url", "youtube"]:
            raise ValueError("Url must be one of 'audio_url', 'youtube'.")
        return value

    class Config:
        """Pydantic config class."""

        schema_extra = {
            "example": {
                "url_type": "youtube",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "api_key": "1234567890",
                "alignment": False,
                "diarization": False,
                "dual_channel": False,
                "source_lang": "en",
                "timestamps": "s",
                "use_batch": False,
                "word_timestamps": False,
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

        schema_extra = {
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
                "alignment": False,
                "diariation": False,
                "source_lang": "en",
                "timestamps": "s",
                "use_batch": False,
                "word_timestamps": False,
                "dual_channel": False,
                "job_name": "job_name",
                "request_id": "request_id",
            }
        }


class CortexYoutubeResponse(YouTubeResponse):
    """Response model for the youtube type of the Cortex endpoint."""

    job_name: str
    request_id: Optional[str] = None

    class Config:
        """Pydantic config class."""

        schema_extra = {
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
                "alignment": False,
                "diariation": False,
                "source_lang": "en",
                "timestamps": "s",
                "use_batch": False,
                "word_timestamps": False,
                "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "job_name": "job_name",
                "request_id": "request_id",
            }
        }


class BaseRequest(BaseModel):
    """Base request model for the API."""

    alignment: bool = False
    diarization: bool = False
    source_lang: str = "en"
    timestamps: str = "s"
    use_batch: bool = False
    word_timestamps: bool = False

    @validator("timestamps")
    def validate_timestamps_values(cls, value: str) -> str:  # noqa: B902, N805
        """Validate the value of the timestamps field."""
        if value not in ["hms", "ms", "s"]:
            raise ValueError("timestamps must be one of 'hms', 'ms', 's'.")
        return value

    class Config:
        """Pydantic config class."""

        schema_extra = {
            "example": {
                "alignment": False,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "use_batch": False,
                "word_timestamps": False,
            }
        }


class AudioRequest(BaseRequest):
    """Request model for the ASR audio file and url endpoint."""

    dual_channel: bool = False

    class Config:
        """Pydantic config class."""

        schema_extra = {
            "example": {
                "alignment": False,
                "diarization": False,
                "source_lang": "en",
                "timestamps": "s",
                "use_batch": False,
                "word_timestamps": False,
                "dual_channel": False,
            }
        }


class PongResponse(BaseModel):
    """Response model for the ping endpoint."""

    message: str

    class Config:
        """Pydantic config class."""

        schema_extra = {
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
