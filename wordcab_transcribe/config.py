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
"""Configuration module of the Wordcab Transcribe."""

from os import getenv
from typing import Union

from dotenv import load_dotenv
from faster_whisper.utils import _MODELS
from pydantic import validator
from pydantic.dataclasses import dataclass


@dataclass
class Settings:
    """Configuration settings for the Wordcab Transcribe API."""

    # Basic API settings
    project_name: str
    version: str
    description: str
    api_prefix: str
    debug: bool
    # Batch request settings
    batch_size: int
    max_wait: float
    # Model settings
    whisper_model: str
    compute_type: str
    nemo_domain_type: str
    nemo_storage_path: str
    nemo_output_path: str
    # ASR service
    asr_type: str
    # API endpoints
    audio_file_endpoint: bool
    audio_url_endpoint: bool
    cortex_endpoint: bool
    youtube_endpoint: bool
    live_endpoint: bool
    # Auth
    cortex_api_key: str
    # Svix
    svix_api_key: str
    svix_app_id: str

    @validator("project_name", "version", "description", "api_prefix")
    def basic_parameters_must_not_be_none(
        cls, value: str, field: str  # noqa: B902, N805
    ):
        """Check that the authentication parameters are not None."""
        if value is None:
            raise ValueError(
                f"{field.name} must not be None, please verify the `.env` file."
            )
        return value

    @validator("batch_size", "max_wait")
    def batch_request_parameters_must_be_positive(
        cls, value: Union[int, float], field: str  # noqa: B902, N805
    ):
        """Check that the model parameters are positive."""
        if value <= 0:
            raise ValueError(f"{field.name} must be positive.")
        return value

    @validator("whisper_model")
    def whisper_model_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the model name is valid."""
        if value not in _MODELS:
            raise ValueError(
                f"{value} is not a valid model name. Choose one of {_MODELS}."
            )
        return value

    @validator("compute_type")
    def compute_type_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the model precision is valid."""
        if value not in {"int8", "int8_float16", "int16", "float_16"}:
            raise ValueError(
                f"{value} is not a valid compute type. Choose one of int8, int8_float16, int16, float_16."
            )
        return value

    @validator("nemo_domain_type")
    def nemo_domain_type_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the model precision is valid."""
        if value not in {"general", "telephonic", "meeting"}:
            raise ValueError(f"{value} is not a valid domain type.")
        return value

    @validator("asr_type")
    def asr_type_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the ASR type is valid."""
        if value not in {"async", "live"}:
            raise ValueError(
                f"{value} is not a valid ASR type. Choose between `async` or `live`."
            )
        return value


load_dotenv()

settings = Settings(
    project_name=getenv("PROJECT_NAME", "Wordcab Transcribe"),
    version=getenv("VERSION", "0.2.0"),
    description=getenv(
        "DESCRIPTION", "💬 ASR FastAPI server using faster-whisper and NVIDIA NeMo."
    ),
    api_prefix=getenv("API_PREFIX", "/api/v1"),
    debug=getenv("DEBUG", True),
    batch_size=getenv("BATCH_SIZE", 1),
    max_wait=getenv("MAX_WAIT", 0.1),
    whisper_model=getenv("WHISPER_MODEL", "large-v2"),
    compute_type=getenv("COMPUTE_TYPE", "int8_float16"),
    nemo_domain_type=getenv("NEMO_DOMAIN_TYPE", "general"),
    nemo_storage_path=getenv("NEMO_STORAGE_PATH", "nemo_storage"),
    nemo_output_path=getenv("NEMO_OUTPUT_PATH", "nemo_outputs"),
    asr_type=getenv("ASR_TYPE", "async"),
    audio_file_endpoint=getenv("AUDIO_FILE_ENDPOINT", True),
    audio_url_endpoint=getenv("AUDIO_URL_ENDPOINT", True),
    cortex_endpoint=getenv("CORTEX_ENDPOINT", True),
    youtube_endpoint=getenv("YOUTUBE_ENDPOINT", True),
    live_endpoint=getenv("LIVE_ENDPOINT", False),
    cortex_api_key=getenv("WORDCAB_TRANSCRIBE_API_KEY", ""),
    svix_api_key=getenv("SVIX_API_KEY", ""),
    svix_app_id=getenv("SVIX_APP_ID", ""),
)
