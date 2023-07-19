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
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from faster_whisper.utils import _MODELS
from loguru import logger
from pydantic import field_validator
from pydantic.dataclasses import dataclass


@dataclass
class Settings:
    """Configuration settings for the Wordcab Transcribe API."""

    # General configuration
    project_name: str
    version: str
    description: str
    api_prefix: str
    debug: bool
    # Batch configuration
    batch_size: int
    max_wait: float
    # Models configuration
    # Whisper
    whisper_model: str
    compute_type: str
    extra_languages: List[str]
    extra_languages_model_paths: Dict[str, str]
    # NVIDIA NeMo
    nemo_domain_type: str
    nemo_storage_path: str
    nemo_output_path: str
    # ASR type configuration
    asr_type: str
    # Endpoints configuration
    audio_file_endpoint: bool
    audio_url_endpoint: bool
    cortex_endpoint: bool
    youtube_endpoint: bool
    live_endpoint: bool
    # API authentication configuration
    username: str
    password: str
    openssl_key: str
    openssl_algorithm: str
    access_token_expire_minutes: int
    # Cortex configuration
    cortex_api_key: str
    # Svix configuration
    svix_api_key: str
    svix_app_id: str

    @field_validator("project_name")
    def project_name_must_not_be_none(cls, value: str):  # noqa: B902, N805
        """Check that the project_name is not None."""
        if value is None:
            raise ValueError(
                "`project_name` must not be None, please verify the `.env` file."
            )

        return value

    @field_validator("version")
    def version_must_not_be_none(cls, value: str):  # noqa: B902, N805
        """Check that the version is not None."""
        if value is None:
            raise ValueError(
                "`version` must not be None, please verify the `.env` file."
            )

        return value

    @field_validator("description")
    def description_must_not_be_none(cls, value: str):  # noqa: B902, N805
        """Check that the description is not None."""
        if value is None:
            raise ValueError(
                "`description` must not be None, please verify the `.env` file."
            )

        return value

    @field_validator("api_prefix")
    def api_prefix_must_not_be_none(cls, value: str):  # noqa: B902, N805
        """Check that the api_prefix is not None."""
        if value is None:
            raise ValueError(
                "`api_prefix` must not be None, please verify the `.env` file."
            )

        return value

    @field_validator("batch_size")
    def batch_size_must_be_positive(cls, value: int):  # noqa: B902, N805
        """Check that the batch_size is positive."""
        if value <= 0:
            raise ValueError(
                "`batch_size` must be positive, please verify the `.env` file."
            )

        return value

    @field_validator("max_wait")
    def max_wait_must_be_positive(cls, value: float):  # noqa: B902, N805
        """Check that the max_wait is positive."""
        if value <= 0:
            raise ValueError(
                "`max_wait` must be positive, please verify the `.env` file."
            )

        return value

    @field_validator("whisper_model")
    def whisper_model_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the model name is valid. It can be a local path or a model name."""
        model_path = Path(value)

        if model_path.exists() is False:
            if value not in _MODELS:
                raise ValueError(
                    f"{value} is not a valid model name. Choose one of {_MODELS}."
                    "If you want to use a local model, please provide a valid path."
                )

        return value

    @field_validator("compute_type")
    def compute_type_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the model precision is valid."""
        if value not in {"int8", "int8_float16", "int16", "float16", "float32"}:
            raise ValueError(
                f"{value} is not a valid compute type. "
                "Choose one of int8, int8_float16, int16, float16, float32."
            )

        return value

    @field_validator("nemo_domain_type")
    def nemo_domain_type_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the model precision is valid."""
        if value not in {"general", "telephonic", "meeting"}:
            raise ValueError(
                f"{value} is not a valid domain type. "
                "Choose one of general, telephonic, meeting."
            )

        return value

    @field_validator("asr_type")
    def asr_type_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the ASR type is valid."""
        if value not in {"async", "live"}:
            raise ValueError(
                f"{value} is not a valid ASR type. Choose between `async` or `live`."
            )

        return value

    @field_validator("openssl_algorithm")
    def openssl_algorithm_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the OpenSSL algorithm is valid."""
        if value not in {"HS256", "HS384", "HS512"}:
            raise ValueError(
                "openssl_algorithm must be a valid algorithm, please verify the `.env` file."
            )

        return value

    @field_validator("access_token_expire_minutes")
    def access_token_expire_minutes_must_be_valid(cls, value: int):  # noqa: B902, N805
        """Check that the access token expiration is valid. Only if debug is False."""
        if value <= 0:
            raise ValueError(
                "access_token_expire_minutes must be positive, please verify the `.env` file."
            )

        return value

    def __post_init__(self):
        """Post initialization checks."""
        endpoints = [
            self.audio_file_endpoint,
            self.audio_url_endpoint,
            self.cortex_endpoint,
            self.youtube_endpoint,
            self.live_endpoint,
        ]
        if not any(endpoints):
            raise ValueError("At least one endpoint configuration must be set to True.")

        if self.debug is False:
            if self.username == "admin" or self.username is None:  # noqa: S105
                logger.warning(
                    f"Username is set to `{self.username}`, which is not secure for production."
                )
            if self.password == "admin" or self.password is None:  # noqa: S105
                logger.warning(
                    f"Password is set to `{self.password}`, which is not secure for production."
                )
            if (
                self.openssl_key == "0123456789abcdefghijklmnopqrstuvwyz"  # noqa: S105
                or self.openssl_key is None
            ):
                logger.warning(
                    f"OpenSSL key is set to `{self.openssl_key}`, which is the default encryption key. "
                    "It's absolutely not secure for production. Please change it in the `.env` file. "
                    "You can generate a new key with `openssl rand -hex 32`."
                )


load_dotenv()

# Extra languages
_extra_languages = getenv("EXTRA_LANGUAGES")
if _extra_languages is not None:
    extra_languages = _extra_languages.split(",")
else:
    extra_languages = []

settings = Settings(
    # General configuration
    project_name=getenv("PROJECT_NAME", "Wordcab Transcribe"),
    version=getenv("VERSION", "0.3.0"),
    description=getenv(
        "DESCRIPTION", "💬 ASR FastAPI server using faster-whisper and NVIDIA NeMo."
    ),
    api_prefix=getenv("API_PREFIX", "/api/v1"),
    debug=getenv("DEBUG", True),
    # Batch configuration
    batch_size=getenv("BATCH_SIZE", 1),
    max_wait=getenv("MAX_WAIT", 0.1),
    # Models configuration
    # Whisper
    whisper_model=getenv("WHISPER_MODEL", "large-v2"),
    compute_type=getenv("COMPUTE_TYPE", "int8_float16"),
    extra_languages=extra_languages,
    extra_languages_model_paths={lang: "" for lang in extra_languages},
    # NeMo
    nemo_domain_type=getenv("NEMO_DOMAIN_TYPE", "general"),
    nemo_storage_path=getenv("NEMO_STORAGE_PATH", "nemo_storage"),
    nemo_output_path=getenv("NEMO_OUTPUT_PATH", "nemo_outputs"),
    # ASR type
    asr_type=getenv("ASR_TYPE", "async"),
    # Endpoints configuration
    audio_file_endpoint=getenv("AUDIO_FILE_ENDPOINT", True),
    audio_url_endpoint=getenv("AUDIO_URL_ENDPOINT", True),
    cortex_endpoint=getenv("CORTEX_ENDPOINT", True),
    youtube_endpoint=getenv("YOUTUBE_ENDPOINT", True),
    live_endpoint=getenv("LIVE_ENDPOINT", False),
    # API authentication configuration
    username=getenv("USERNAME", "admin"),
    password=getenv("PASSWORD", "admin"),
    openssl_key=getenv("OPENSSL_KEY", "0123456789abcdefghijklmnopqrstuvwyz"),
    openssl_algorithm=getenv("OPENSSL_ALGORITHM", "HS256"),
    access_token_expire_minutes=getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30),
    # Cortex configuration
    cortex_api_key=getenv("WORDCAB_TRANSCRIBE_API_KEY", ""),
    # Svix configuration
    svix_api_key=getenv("SVIX_API_KEY", ""),
    svix_app_id=getenv("SVIX_APP_ID", ""),
)
