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
"""Voice Activation Detection (VAD) Service for audio files."""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from faster_whisper.vad import SileroVADModel, VadOptions, get_speech_timestamps

from wordcab_transcribe.utils import read_audio


class VadService:
    """VAD Service for audio files."""

    def __init__(self, live=False) -> None:
        """Initialize the VAD Service."""
        self.sample_rate = 16000
        self.options = VadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=30,
            min_silence_duration_ms=100,
            window_size_samples=512,
            speech_pad_ms=400,
        )

        self.live_vad_model = None
        if live:
            vad_model_path = Path(__file__).parent.parent / "assets/silero_vad.onnx"
            self.live_vad_model = SileroVADModel(vad_model_path)

    def __call__(
        self, waveform: torch.Tensor, group_timestamps: Optional[bool] = True
    ) -> Tuple[Union[List[dict], List[List[dict]]], torch.Tensor]:
        """
        Use the VAD model to get the speech timestamps. Multi-channel pipeline.

        Args:
            waveform (torch.Tensor): Audio tensor.
            group_timestamps (Optional[bool], optional): Group timestamps. Defaults to True.

        Returns:
            Tuple[Union[List[dict], List[List[dict]]], torch.Tensor]: Speech timestamps and audio tensor.
        """
        if waveform.size(0) == 1:
            waveform = waveform.squeeze(0)

        speech_timestamps = get_speech_timestamps(
            audio=waveform, vad_options=self.options
        )

        _speech_timestamps_list = [
            {"start": ts["start"], "end": ts["end"]} for ts in speech_timestamps
        ]

        if group_timestamps:
            speech_timestamps_list = self.group_timestamps(_speech_timestamps_list)
        else:
            speech_timestamps_list = _speech_timestamps_list

        return speech_timestamps_list, waveform

    def get_speech_probability(self, audio_chunk):
        """Get speech probability from VAD model."""
        if self.live_vad_model:
            audio_chunk, _ = read_audio(audio_chunk)
            audio_chunk = audio_chunk.float()

            state = self.live_vad_model.get_initial_state(batch_size=1)

            if len(audio_chunk) < self.options.window_size_samples:
                audio_chunk = np.pad(
                    audio_chunk,
                    (0, int(self.options.window_size_samples - len(audio_chunk))),
                )

            speech_prob, _ = self.live_vad_model(audio_chunk, state, self.sample_rate)
            del audio_chunk

            return speech_prob
        return

    def group_timestamps(
        self, timestamps: List[dict], threshold: Optional[float] = 3.0
    ) -> List[List[dict]]:
        """
        Group timestamps based on a threshold.

        Args:
            timestamps (List[dict]): List of timestamps.
            threshold (float, optional): Threshold to use for grouping. Defaults to 3.0.

        Returns:
            List[List[dict]]: List of grouped timestamps.
        """
        grouped_segments = [[]]

        for i in range(len(timestamps)):
            if (
                i > 0
                and (timestamps[i]["start"] - timestamps[i - 1]["end"]) > threshold
            ):
                grouped_segments.append([])

            grouped_segments[-1].append(timestamps[i])

        return grouped_segments

    def save_audio(self, filepath: str, audio: torch.Tensor) -> None:
        """
        Save audio tensor to file.

        Args:
            filepath (str): Path to save the audio file.
            audio (torch.Tensor): Audio tensor.
        """
        torchaudio.save(
            filepath, audio.unsqueeze(0), self.sample_rate, bits_per_sample=16
        )
