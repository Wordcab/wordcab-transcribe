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
"""Test config settings."""

from collections import OrderedDict

import pytest

from wordcab_transcribe.config import Settings, settings


@pytest.fixture
def default_settings() -> OrderedDict:
    """Return the default settings."""
    return OrderedDict(
        project_name="Wordcab Transcribe",
        version="0.2.0",
        description="💬 ASR FastAPI server using faster-whisper and NVIDIA NeMo.",
        api_prefix="/api/v1",
        debug=True,
        batch_size=1,
        max_wait=0.1,
        whisper_model="large-v2",
        compute_type="int8_float16",
        nemo_domain_type="general",
        nemo_storage_path="nemo_storage",
        nemo_output_path="nemo_outputs",
        asr_type="async",
        audio_file_endpoint=True,
        audio_url_endpoint=True,
        cortex_endpoint=True,
        youtube_endpoint=True,
        live_endpoint=False,
        username="admin",
        password="admin",
        openssl_key="0123456789abcdefghijklmnopqrstuvwyz",
        openssl_algorithm="HS256",
        access_token_expire_minutes=30,
        cortex_api_key="",
        svix_api_key="",
        svix_app_id="",
    )


def test_config() -> None:
    """Test default config settings with the .env file."""
    assert settings.project_name == "Wordcab Transcribe"
    assert settings.version == "0.2.0"
    assert (
        settings.description
        == "💬 ASR FastAPI server using faster-whisper and NVIDIA NeMo."
    )
    assert settings.api_prefix == "/api/v1"
    assert settings.debug is True

    assert settings.batch_size == 1
    assert settings.max_wait == 0.1
    assert settings.whisper_model == "large-v2"
    assert settings.compute_type == "int8_float16"
    assert settings.nemo_domain_type == "telephonic"
    assert settings.nemo_storage_path == "nemo_storage"
    assert settings.nemo_output_path == "nemo_outputs"

    assert settings.asr_type == "async"

    assert settings.audio_file_endpoint is True
    assert settings.audio_url_endpoint is True
    assert settings.cortex_endpoint is True
    assert settings.youtube_endpoint is True
    assert settings.live_endpoint is False

    assert settings.username == "admin"  # noqa: S105
    assert settings.password == "admin"  # noqa: S105
    assert settings.openssl_key == "0123456789abcdefghijklmnopqrstuvwyz"  # noqa: S105
    assert settings.openssl_algorithm == "HS256"
    assert settings.access_token_expire_minutes == 30

    assert settings.cortex_api_key == ""
    assert settings.svix_api_key == ""
    assert settings.svix_app_id == ""


def test_general_parameters_validator(default_settings: dict) -> None:
    """Test general parameters validator."""
    wrong_project_name = default_settings.copy()
    wrong_project_name["project_name"] = None
    with pytest.raises(ValueError):
        Settings(**wrong_project_name)

    wrong_version = default_settings.copy()
    wrong_version["version"] = None
    with pytest.raises(ValueError):
        Settings(**wrong_version)

    wrong_description = default_settings.copy()
    wrong_description["description"] = None
    with pytest.raises(ValueError):
        Settings(**wrong_description)

    wrong_api_prefix = default_settings.copy()
    wrong_api_prefix["api_prefix"] = None
    with pytest.raises(ValueError):
        Settings(**wrong_api_prefix)


def test_batch_request_parameters_validator(default_settings: dict) -> None:
    """Test batch request parameters validator."""
    wrong_batch_size = default_settings.copy()
    wrong_batch_size["batch_size"] = 0
    with pytest.raises(ValueError):
        Settings(**wrong_batch_size)

    wrong_max_wait = default_settings.copy()
    wrong_max_wait["max_wait"] = -1
    with pytest.raises(ValueError):
        Settings(**wrong_max_wait)


def test_whisper_model_validator(default_settings: dict) -> None:
    """Test whisper model validator."""
    wrong_whisper_model = default_settings.copy()
    wrong_whisper_model["whisper_model"] = "invalid_model_name"
    with pytest.raises(ValueError):
        Settings(**wrong_whisper_model)

    wrong_whisper_model = default_settings.copy()
    wrong_whisper_model["whisper_model"] = "/path/to/invalid_model"
    with pytest.raises(ValueError):
        Settings(**wrong_whisper_model)


def test_compute_type_validator(default_settings: dict) -> None:
    """Test compute type validator."""
    default_settings["compute_type"] = "invalid_compute_type"
    with pytest.raises(ValueError):
        Settings(**default_settings)


def test_nemo_domain_type_validator(default_settings: dict) -> None:
    """Test nemo domain type validator."""
    default_settings["nemo_domain_type"] = "invalid_domain_type"
    with pytest.raises(ValueError):
        Settings(**default_settings)


def test_asr_type_validator(default_settings: dict) -> None:
    """Test asr type validator."""
    default_settings["asr_type"] = "invalid_asr_type"
    with pytest.raises(ValueError):
        Settings(**default_settings)


def test_openssl_algorithm_validator(default_settings: dict) -> None:
    """Test openssl algorithm validator."""
    default_settings["openssl_algorithm"] = "invalid_algorithm"
    with pytest.raises(ValueError):
        Settings(**default_settings)


def test_access_token_expire_minutes_validator(default_settings: dict) -> None:
    """Test access token expire minutes validator."""
    default_settings["access_token_expire_minutes"] = -1
    with pytest.raises(ValueError):
        Settings(**default_settings)


def test_post_init(default_settings: dict) -> None:
    """Test post init."""
    wrong_endpoint = default_settings.copy()
    wrong_endpoint["audio_file_endpoint"] = False
    wrong_endpoint["audio_url_endpoint"] = False
    wrong_endpoint["cortex_endpoint"] = False
    wrong_endpoint["live_endpoint"] = False
    wrong_endpoint["youtube_endpoint"] = False
    with pytest.raises(ValueError):
        Settings(**wrong_endpoint)
