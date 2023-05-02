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
"""Tests the load_nemo_config function."""

import pytest
import yaml

from wordcab_transcribe.utils import load_nemo_config


@pytest.mark.parametrize("domain_type", ["general", "meeting", "telephonic"])
def test_load_nemo_config(domain_type: str):
    """Test the load_nemo_config function."""
    config = load_nemo_config(domain_type)

    cfg_path = f"config/nemo/diar_infer_{domain_type}.yaml"
    with open(cfg_path) as f:
        data = yaml.safe_load(f)

    assert config == data
    assert isinstance(config, dict)
    assert "name" in config
    assert "num_workers" in config
    assert "sample_rate" in config
    assert "batch_size" in config
    assert "device" in config
    assert "verbose" in config
    assert "diarizer" in config

    diarizer = config["diarizer"]
    assert isinstance(diarizer, dict)
    assert "manifest_filepath" in diarizer
    assert "output_dir" in diarizer
    assert "oracle_vad" in diarizer
    assert "collar" in diarizer
    assert "ignore_overlap" in diarizer

    assert "vad" in diarizer
    assert isinstance(diarizer["vad"], dict)

    assert "speaker_embeddings" in diarizer
    assert isinstance(diarizer["speaker_embeddings"], dict)

    assert "clustering" in diarizer
    assert isinstance(diarizer["clustering"], dict)

    assert "msdd_model" in diarizer
    assert isinstance(diarizer["msdd_model"], dict)

    assert "asr" in config
    assert isinstance(config["asr"], dict)
