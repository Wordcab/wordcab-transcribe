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

import math
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
            model = NeuralDiarizer(cfg=cfg).to(_device)
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


def real_diarize():
    """"""
    # VAD
    waveform, _ = read_audio("./mono_file.wav")
    vad_service = VadService()
    speech_ts, _ = vad_service(waveform, False)

    # Segmentation
    window_lengths = [1.5, 1.25, 1.0, 0.75, 0.5]
    shift_lengths = [0.75, 0.625, 0.5, 0.375, 0.25]
    multiscale_weights = [1, 1, 1, 1, 1]

    multiscale_args_dict = {'use_single_scale_clustering': False}
    scale_dict = {k: (w, s) for k, (w, s) in enumerate(zip(window_lengths, shift_lengths))}


    # Clustering


    # Scoring

def run_segmentation(
    vad_outputs: List[dict],
    window: float,
    shift: float,
    min_subsegment_duration: float = 0.05,
) -> List[dict]:
    """"""
    scale_segment = []
    for segment in vad_outputs:
        segment_start, segment_end = segment["start"] / 16000, segment["end"] / 16000
        subsegments = get_subsegments(segment_start, segment_end, window, shift)

        for subsegment in subsegments:
            start, duration = subsegment
            if duration > min_subsegment_duration:
                scale_segment.append({"offset": start, "duration": duration})

    return scale_segment

def get_subsegments(segment_start: float, segment_end: float, window: float, shift: float) -> List[List[float]]:
    """
    Return a list of subsegments based on the segment start and end time and the window and shift length.

    Args:
        segment_start (float): Segment start time.
        segment_end (float): Segment end time.
        window (float): Window length.
        shift (float): Shift length.

    Returns:
        List[List[float]]: List of subsegments with start time and duration.
    """
    start = segment_start
    duration = segment_end - segment_start
    base = math.ceil((duration - window) / shift)
    
    subsegments: List[List[float]] = []
    slices = 1 if base < 0 else base + 1
    for slice_id in range(slices):
        end = start + window

        if end > segment_end:
            end = segment_end

        subsegments.append([start, end - start])

        start = segment_start + (slice_id + 1) * shift

    return subsegments