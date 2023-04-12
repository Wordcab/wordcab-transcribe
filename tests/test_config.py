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

from wordcab_transcribe.config import Settings


def test_config() -> None:
    """Test default config settings with the .env file."""
    assert Settings().project_name == "Wordcab Transcribe"
    assert Settings().version == "0.1.0"
    assert (
        Settings().description
        == "ASR FastAPI server using faster-whisper and pyannote-audio."
    )
    assert Settings().api_prefix == "/api/v1"
    assert Settings().debug is True
    assert Settings().batch_size == 1
    assert Settings().max_wait == 0.1
    assert Settings().whisper_model == "large-v2"
    assert Settings().embeddings_model == "speechbrain/spkrec-ecapa-voxceleb"
    assert Settings().compute_type == "int8_float16"
