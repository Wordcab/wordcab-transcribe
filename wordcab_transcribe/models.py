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

from typing import Optional

from pydantic import BaseModel, validator


class ASRResponse(BaseModel):
    """Response model for the ASR API."""

    utterances: list

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
                ]
            }
        }


class DataRequest(BaseModel):
    """Request object for the audio file endpoint."""

    source_lang: Optional[str] = "en"
    timestamps: Optional[str] = "s"

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
                "source_lang": "en",
                "timestamps": "s",
            }
        }
