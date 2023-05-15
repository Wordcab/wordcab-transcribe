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

from wordcab_transcribe.config import settings


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

    assert settings.cortex_api_key == ""
    assert settings.svix_api_key == ""
    assert settings.svix_app_id == ""
