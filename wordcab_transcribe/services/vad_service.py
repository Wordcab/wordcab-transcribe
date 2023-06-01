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
"""Voice Activation Detection (VAD) Service for audio files."""

from typing import List, Optional, Tuple, Union

import torch
import torchaudio
from faster_whisper.vad import get_speech_timestamps


class VadService:
    """VAD Service for audio files."""

    def __init__(self) -> None:
        """Initialize the VAD Service."""
        self.sample_rate = 16000

    def __call__(
        self, filepath: str, group_timestamps: Optional[bool] = True
    ) -> Tuple[Union[List[dict], List[List[dict]]], torch.Tensor]:
        """
        Use the VAD model to get the speech timestamps. Dual channel pipeline.

        Args:
            filepath (str): Path to the audio file.
            group_timestamps (Optional[bool], optional): Group timestamps. Defaults to True.

        Returns:
            Tuple[Union[List[dict], List[List[dict]]], torch.Tensor]: Speech timestamps and audio tensor.
        """
        wav, sr = torchaudio.load(filepath)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            wav = transform(wav)
            sr = self.sample_rate

        assert sr == self.sample_rate
        wav = wav.squeeze(0)

        speech_timestamps = get_speech_timestamps(audio=wav)

        speech_timestamps_list = [
            {"start": ts["start"], "end": ts["end"]} for ts in speech_timestamps
        ]

        return speech_timestamps_list, wav

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
