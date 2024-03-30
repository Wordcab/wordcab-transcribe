# Copyright 2024 The Wordcab Team. All rights reserved.
#
# Licensed under the MIT License (the "License");
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
"""Longform diarization Service for audio files."""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import shortuuid
import torch
import torchaudio
from nemo.collections.asr.models import NeuralDiarizer
from tensorshare import Backend, TensorShare

from wordcab_transcribe.models import DiarizationOutput
from wordcab_transcribe.utils import (
    delete_file,
    download_audio_file_sync,
    process_audio_file_sync,
)


class LongFormDiarizeService:
    """Diarize Service for audio files."""

    def __init__(
        self,
        device: str,
    ) -> None:
        """
        Initialize the DiarizeService.

        Args:
            device (str): Device to run the diarization model on.

        Returns:
            None
        """
        self.diarization_model = NeuralDiarizer.from_pretrained(
            model_name="diar_msdd_telephonic"
        ).to(device)
        # TODO: Ability to modify config

    def __call__(
        self,
        oracle_num_speakers: int,
        waveform: Optional[Union[torch.Tensor, TensorShare]] = None,
        url: Optional[str] = None,
        url_type: Optional[str] = None,
    ) -> DiarizationOutput:
        """
        Run inference with the diarization model.

        Args:
            waveform (Union[torch.Tensor, TensorShare]):
                Waveform to run inference on.
            oracle_num_speakers (int):
                Number of speakers in the audio file.

        Returns:
            DiarizationOutput:
                List of segments with the following keys: "start", "end", "speaker".
        """

        if url and url_type:
            audio_filename = f"audio_{shortuuid.ShortUUID().random(length=32)}"
            audio_filepath = download_audio_file_sync(url_type, url, audio_filename)
            processed_audio_filepath = process_audio_file_sync(audio_filepath)
            processed_audio_filepath = (
                Path(__file__).parent / "temp_files" / processed_audio_filepath
            )
            delete_file(audio_filepath)
        else:
            if isinstance(waveform, TensorShare):
                ts = waveform.to_tensors(backend=Backend.TORCH)
                waveform = ts["audio"]
            elif isinstance(waveform, torch.Tensor):
                pass
            else:
                return None

            waveform = waveform.to(torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                torchaudio.save(
                    temp_file.name, waveform, sample_rate=16000, channels_first=True
                )
                processed_audio_filepath = (
                    Path(__file__).parent / "temp_files" / temp_file.name
                )

        temp_dir = (
            Path(processed_audio_filepath).parent / Path(processed_audio_filepath).stem
        )

        annotation = self.diarization_model(
            str(processed_audio_filepath),
            num_speakers=oracle_num_speakers,
            out_dir=temp_dir,
        )

        segments = self.convert_annotation_to_segments(annotation)
        segments = self.get_contiguous_timestamps(segments)
        segments = self.merge_timestamps(segments)

        delete_file(str(processed_audio_filepath))

        return DiarizationOutput(segments=segments)

    def convert_annotation_to_segments(
        self, annotation
    ) -> List[Tuple[float, float, int]]:
        """
        Convert annotation to segments.

        Args:
            annotation: Annotation object.

        Returns:
            List[Tuple[float, float, int]]: List of segments containing the start time, end time and speaker.
        """
        segments = []
        speaker_mapping = {}
        current_speaker_id = 0

        for segment, track in annotation._tracks.items():
            speaker_label = track["_"]
            if speaker_label not in speaker_mapping:
                speaker_mapping[speaker_label] = current_speaker_id
                current_speaker_id += 1

            start = round(segment.start, 2)
            end = round(segment.end, 2)
            speaker = speaker_mapping[speaker_label]

            segments.append((start, end, speaker))

        return segments

    @staticmethod
    def get_contiguous_timestamps(
        stamps: List[Tuple[float, float, int]]
    ) -> List[Tuple[float, float, int]]:
        """
        Return contiguous timestamps.

        Args:
            stamps (List[Tuple[float, float, int]]): List of segments containing the start time, end time and speaker.

        Returns:
            List[Tuple[float, float, int]]: List of segments containing the start time, end time and speaker.
        """
        contiguous_timestamps = []
        for i in range(len(stamps) - 1):
            start, end, speaker = stamps[i]
            next_start, next_end, next_speaker = stamps[i + 1]

            if end > next_start:
                avg = (next_start + end) / 2.0
                stamps[i + 1] = (avg, next_end, next_speaker)
                contiguous_timestamps.append((start, avg, speaker))
            else:
                contiguous_timestamps.append((start, end, speaker))

        start, end, speaker = stamps[-1]
        contiguous_timestamps.append((start, end, speaker))

        return contiguous_timestamps

    @staticmethod
    def merge_timestamps(
        stamps: List[Tuple[float, float, int]]
    ) -> List[Tuple[float, float, int]]:
        """
        Merge timestamps of the same speaker.

        Args:
            stamps (List[Tuple[float, float, int]]): List of segments containing the start time, end time and speaker.

        Returns:
            List[Tuple[float, float, int]]: List of segments containing the start time, end time and speaker.
        """
        overlap_timestamps = []
        for i in range(len(stamps) - 1):
            start, end, speaker = stamps[i]
            next_start, next_end, next_speaker = stamps[i + 1]

            if end == next_start and speaker == next_speaker:
                stamps[i + 1] = (start, next_end, next_speaker)
            else:
                overlap_timestamps.append((start, end, speaker))

        start, end, speaker = stamps[-1]
        overlap_timestamps.append((start, end, speaker))

        return overlap_timestamps
