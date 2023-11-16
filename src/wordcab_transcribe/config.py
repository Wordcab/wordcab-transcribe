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
"""Configuration module of the Wordcab Transcribe."""

import os
from os import getenv
from typing import Dict, List, Union

from dotenv import load_dotenv
from loguru import logger
from pydantic import field_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from wordcab_transcribe import __version__


@dataclass
class Settings:
    """Configuration settings for the Wordcab Transcribe API."""

    # General configuration
    project_name: str
    version: str
    description: str
    api_prefix: str
    debug: bool
    # Models configuration
    # Whisper
    whisper_model: str
    compute_type: str
    extra_languages: Union[List[str], None]
    extra_languages_model_paths: Union[Dict[str, str], None]
    # Diarization
    window_lengths: List[float]
    shift_lengths: List[float]
    multiscale_weights: List[float]
    # ASR type configuration
    asr_type: Literal["async", "live", "only_transcription", "only_diarization"]
    # Endpoint configuration
    cortex_endpoint: bool
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
    # Remote servers configuration
    transcribe_server_urls: Union[List[str], None]
    diarize_server_urls: Union[List[str], None]
    remote_diarization_request_timeout: int

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

    @field_validator("compute_type")
    def compute_type_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the model precision is valid."""
        compute_type_values = [
            "int8",
            "int8_float16",
            "int8_bfloat16",
            "int16",
            "float16",
            "bfloat16",
            "float32",
        ]
        if value not in compute_type_values:
            raise ValueError(
                f"{value} is not a valid compute type. Choose one of"
                f" {compute_type_values}."
            )

        return value

    @field_validator("openssl_algorithm")
    def openssl_algorithm_must_be_valid(cls, value: str):  # noqa: B902, N805
        """Check that the OpenSSL algorithm is valid."""
        if value not in {"HS256", "HS384", "HS512"}:
            raise ValueError(
                "openssl_algorithm must be a valid algorithm, please verify the `.env`"
                " file."
            )

        return value

    @field_validator("access_token_expire_minutes")
    def access_token_expire_minutes_must_be_valid(cls, value: int):  # noqa: B902, N805
        """Check that the access token expiration is valid. Only if debug is False."""
        if value <= 0:
            raise ValueError(
                "access_token_expire_minutes must be positive, please verify the `.env`"
                " file."
            )

        return value

    def __post_init__(self):
        """Post initialization checks."""
        if self.debug is False:
            if self.username == "admin" or self.username is None:  # noqa: S105
                logger.warning(
                    f"Username is set to `{self.username}`, which is not secure for"
                    " production."
                )
            if self.password == "admin" or self.password is None:  # noqa: S105
                logger.warning(
                    f"Password is set to `{self.password}`, which is not secure for"
                    " production."
                )
            if (
                self.openssl_key == "0123456789abcdefghijklmnopqrstuvwyz"  # noqa: S105
                or self.openssl_key is None
            ):
                logger.warning(
                    f"OpenSSL key is set to `{self.openssl_key}`, which is the default"
                    " encryption key. It's absolutely not secure for production."
                    " Please change it in the `.env` file. You can generate a new key"
                    " with `openssl rand -hex 32`."
                )

        if (
            len(self.window_lengths)
            != len(self.shift_lengths)
            != len(self.multiscale_weights)
        ):
            raise ValueError(
                "Length of window_lengths, shift_lengths and multiscale_weights must"
                f" be the same.\nFound: {len(self.window_lengths)},"
                f" {len(self.shift_lengths)}, {len(self.multiscale_weights)}"
            )


load_dotenv()

# Extra languages
_extra_languages = getenv("EXTRA_LANGUAGES", None)
if _extra_languages is not None and _extra_languages != "":
    extra_languages = [lang.strip() for lang in _extra_languages.split(",")]
else:
    extra_languages = None

extra_languages_model_paths = (
    {lang: "" for lang in extra_languages} if extra_languages is not None else None
)

# Diarization scales
_window_lengths = getenv("WINDOW_LENGTHS", None)
if _window_lengths is not None:
    window_lengths = [float(x.strip()) for x in _window_lengths.split(",")]
else:
    window_lengths = [1.5, 1.25, 1.0, 0.75, 0.5]

_shift_lengths = getenv("SHIFT_LENGTHS", None)
if _shift_lengths is not None:
    shift_lengths = [float(x.strip()) for x in _shift_lengths.split(",")]
else:
    shift_lengths = [0.75, 0.625, 0.5, 0.375, 0.25]

_multiscale_weights = getenv("MULTISCALE_WEIGHTS", None)
if _multiscale_weights is not None:
    multiscale_weights = [float(x.strip()) for x in _multiscale_weights.split(",")]
else:
    multiscale_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

# Multi-servers configuration
_transcribe_server_urls = getenv("TRANSCRIBE_SERVER_URLS", None)
if _transcribe_server_urls is not None and _transcribe_server_urls != "":
    transcribe_server_urls = [url.strip() for url in _transcribe_server_urls.split(",")]
else:
    transcribe_server_urls = None

_diarize_server_urls = getenv("DIARIZE_SERVER_URLS", None)
if _diarize_server_urls is not None and _diarize_server_urls != "":
    diarize_server_urls = [url.strip() for url in _diarize_server_urls.split(",")]
else:
    diarize_server_urls = None

settings = Settings(
    # General configuration
    project_name=getenv("PROJECT_NAME", "Wordcab Transcribe"),
    version=getenv("VERSION", __version__),
    description=getenv(
        "DESCRIPTION",
        "💬 ASR FastAPI server using faster-whisper and Auto-Tuning Spectral Clustering"
        " for diarization.",
    ),
    api_prefix=getenv("API_PREFIX", "/api/v1"),
    debug=getenv("DEBUG", True),
    # Models configuration
    # Transcription
    whisper_model=getenv("WHISPER_MODEL", "large-v2"),
    compute_type=getenv("COMPUTE_TYPE", "float16"),
    extra_languages=extra_languages,
    extra_languages_model_paths=extra_languages_model_paths,
    # Diarization
    window_lengths=window_lengths,
    shift_lengths=shift_lengths,
    multiscale_weights=multiscale_weights,
    # ASR type
    asr_type=getenv("ASR_TYPE", "async"),
    # Endpoints configuration
    cortex_endpoint=getenv("CORTEX_ENDPOINT", True),
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
    # Remote servers configuration
    transcribe_server_urls=transcribe_server_urls,
    diarize_server_urls=diarize_server_urls,
    remote_diarization_request_timeout=int(os.getenv("REMOTE_DIARIZE_SERVER_REQUEST_TIMEOUT_SEC", 300)),
)
