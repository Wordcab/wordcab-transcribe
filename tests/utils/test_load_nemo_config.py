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

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from wordcab_transcribe.utils import load_nemo_config


@pytest.mark.parametrize("domain_type", ["general", "meeting", "telephonic"])
def test_load_nemo_config(domain_type: str):
    """Test the load_nemo_config function."""
    cfg = load_nemo_config(domain_type, "storage/path", "output/path")

    cfg_path = f"config/nemo/diar_infer_{domain_type}.yaml"
    with open(cfg_path) as f:
        data = OmegaConf.load(f)

    assert cfg != data

    assert cfg.num_workers == 0
    assert cfg.diarizer.manifest_filepath == str(
        Path.cwd() / "storage/path/infer_manifest.json"
    )
    assert cfg.diarizer.out_dir == str(Path.cwd() / "output/path")
