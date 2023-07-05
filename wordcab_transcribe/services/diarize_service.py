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
"""Diarization Service for audio files."""

from pathlib import Path
from typing import List, NamedTuple, Union

import librosa
import soundfile as sf
import torch
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from wordcab_transcribe.logging import time_and_tell
from wordcab_transcribe.utils import load_nemo_config


class NemoModel(NamedTuple):
    """NeMo Model."""

    model: NeuralDiarizer
    output_path: str
    tmp_audio_path: str
    device: str


class DiarizeService:
    """Diarize Service for audio files."""

    def __init__(
        self,
        domain_type: str,
        storage_path: str,
        output_path: str,
        device: str,
        device_index: List[int],
    ) -> None:
        """Initialize the Diarize Service.

        This service uses the NeuralDiarizer from NeMo to diarize audio files.

        Args:
            domain_type (str): Domain type to use for diarization. Can be "general", "telephonic" or "meeting".
            storage_path (str): Path where the diarization pipeline will save temporary files.
            output_path (str): Path where the diarization pipeline will save the final output files.
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            device_index (Union[int, List[int]]): Index of the device to use for inference.
        """
        self.device = device
        self.models = {}

        for idx in device_index:
            _output_path = Path(output_path) / f"output_{idx}"

            _device = f"cuda:{idx}" if self.device == "cuda" else "cpu"
            cfg, tmp_audio_path = load_nemo_config(
                domain_type=domain_type,
                storage_path=storage_path,
                output_path=_output_path,
                device=_device,
                index=idx,
            )
            model = NeuralDiarizer(cfg=cfg)
            self.models[idx] = NemoModel(
                model=model,
                output_path=_output_path,
                tmp_audio_path=tmp_audio_path,
                device=_device,
            )

    @time_and_tell
    def __call__(
        self, filepath: Union[str, torch.Tensor], model_index: int
    ) -> List[dict]:
        """
        Run inference with the diarization model.

        Args:
            filepath (Union[str, torch.Tensor]): Path to the audio file or waveform.
            model_index (int): Index of the model to use for inference.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "speaker".
        """
        if isinstance(filepath, str):
            waveform, sample_rate = librosa.load(filepath, sr=None)
        else:
            waveform = filepath
            sample_rate = 16000

        sf.write(
            self.models[model_index].tmp_audio_path, waveform, sample_rate, "PCM_16"
        )

        self.models[model_index].model.diarize()

        outputs = self._format_timestamps(self.models[model_index].output_path)

        return outputs

    @staticmethod
    def _format_timestamps(output_path: str) -> List[dict]:
        """
        Format timestamps from the diarization pipeline.

        Args:
            output_path (str): Path where the diarization pipeline saved the final output files.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "speaker".
        """
        speaker_timestamps = []

        with open(f"{output_path}/pred_rttms/mono_file.rttm") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_timestamps.append([s, e, int(line_list[11].split("_")[-1])])

        return speaker_timestamps
