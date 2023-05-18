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

from typing import List, NamedTuple, Optional, Tuple, Union

import torch
import torchaudio
from faster_whisper.vad import get_speech_timestamps


# The code below is adapted from https://github.com/snakers4/silero-vad.
class VadOptions(NamedTuple):
    """VAD options.

    Args:
        threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
            probabilities ABOVE this value are considered as SPEECH. It is better to tune this
            parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
        min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
        max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
            than max_speech_duration_s will be split at the timestamp of the last silence that
            lasts more than 100s (if any), to prevent aggressive cutting. Otherwise, they will be
            split aggressively just before max_speech_duration_s.
        min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
            before separating it.
        window_size_samples: Audio chunks of window_size_samples size are fed to the silero VAD model.
            WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
            For other sample rates the audio is resampled to 16000 and then fed to the model.
        speed_pad_ms: Final speech chunks are padded by speed_pad_ms each side.
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 100
    window_size_samples: int = 512
    speech_pad_ms: int = 400


class VadService:
    """VAD Service for audio files."""

    def __init__(self, vad_options: Optional[dict] = None) -> None:
        """Initialize the VAD Service.

        This service uses the VadOptions from faster-whisper to apply VAD to audio files.

        Args:
            vad_options (Optional[dict], optional): VAD options. Defaults to None.
        """
        self.vad_options = VadOptions() if vad_options is None else VadOptions(**vad_options)
        self.sample_rate = 16000

    def __call__(
        self, filepath: str, group_timestamps: Optional[bool] = True
    ) -> Tuple[Union[List[dict], List[List[dict]]], torch.Tensor]:
        """
        Use the VAD model to get the speech timestamps.

        Args:
            filepath (str): Path to the audio file.
            group_timestamps (Optional[bool], optional): Group timestamps. Defaults to True.

        Returns:
            Tuple[Union[List[dict], List[List[dict]]], torch.Tensor]: Speech timestamps and audio tensor.
        """
        audio = self.read_audio(filepath)
        speech_timestamps = get_speech_timestamps(audio, self.vad_options)

        speech_timestamps_list = [
            {"start": ts["start"], "end": ts["end"]}
            for ts in speech_timestamps
        ]

        if group_timestamps:
            speech_timestamps_list = self.group_timestamps(speech_timestamps_list)

        return speech_timestamps_list, audio

    def read_audio(self, path: str) -> torch.Tensor:
        """
        Read an audio file and return a tensor.

        Args:
            path (str): Path to the audio file.

        Returns:
            torch.Tensor: Tensor containing the audio file.
        """

        wav, sr = torchaudio.load(path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            wav = transform(wav)
            sr = self.sample_rate

        assert sr == self.sample_rate

        return wav.squeeze(0)

    def group_timestamps(self, timestamps: List[dict], threshold: float = 3.0) -> List[List[dict]]:
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
            if i > 0 and (timestamps[i]['start'] - timestamps[i - 1]['end']) > threshold:
                grouped_segments.append([])

            grouped_segments[-1].append(timestamps[i])

        return grouped_segments
