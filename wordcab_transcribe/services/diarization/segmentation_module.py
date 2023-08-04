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
"""Segmentation module for the diarization service."""

import math
from typing import Dict, List, Tuple

import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from torch.cuda.amp import autocast
from torch.utils.data import Dataset

from wordcab_transcribe.services.diarization.models import MultiscaleEmbeddingsAndTimestamps
from wordcab_transcribe.services.diarization.utils import segmentation_collate_fn


class AudioSegmentDataset(Dataset):
    """Dataset for audio segments used by the SegmentationModule."""

    def __init__(
        self, waveform: torch.Tensor, segments: List[dict], sample_rate=16000
    ) -> None:
        """
        Initialize the dataset for the SegmentationModule.

        Args:
            waveform (torch.Tensor): Waveform of the audio file.
            segments (List[dict]): List of segments with the following keys: "offset", "duration".
            sample_rate (int): Sample rate of the audio file. Defaults to 16000.
        """
        self.waveform = waveform
        self.segments = segments
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to get.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of the audio segment and its length.
        """
        segment_info = self.segments[idx]
        offset_samples = int(segment_info["offset"] * self.sample_rate)
        duration_samples = int(segment_info["duration"] * self.sample_rate)

        segment = self.waveform[offset_samples : offset_samples + duration_samples]

        return segment, torch.tensor(segment.shape[0]).long()


class SegmentationModule:
    """Segmentation module for diariation."""

    def __init__(self, device: str, multiscale_weights: List[float]) -> None:
        """
        Initialize the segmentation module.

        Args:
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            multiscale_weights (List[float]): List of weights for each scale.
        """
        self.multiscale_weights = torch.tensor(multiscale_weights).unsqueeze(0).float()

        if len(multiscale_weights) > 3:
            self.batch_size = 64
        elif len(multiscale_weights) > 1:
            self.batch_size = 128
        else:
            self.batch_size = 256

        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large", map_location=None
        ).to(device)
        self.speaker_model.eval()

    def __call__(
        self,
        waveform: torch.Tensor,
        vad_outputs: List[dict],
        scale_dict: Dict[int, Tuple[float, float]],
    ) -> MultiscaleEmbeddingsAndTimestamps:
        """
        Run the segmentation module.

        Args:
            waveform (torch.Tensor): Waveform of the audio file.
            vad_outputs (List[dict]): List of segments with the following keys: "start", "end".
            scale_dict (Dict[int, Tuple[float, float]]): Dictionary of scales in the format {scale_id: (window, shift)}.

        Returns:
            MultiscaleEmbeddingsAndTimestamps: Embeddings and timestamps of the audio file.

        Raises:
            ValueError: If there is a mismatch of counts between embedding vectors and timestamps.
        """
        embeddings, timestamps = [], []

        for _, (window, shift) in scale_dict.items():
            scale_segments = self.get_audio_segments_from_scale(
                vad_outputs, window, shift
            )

            _embeddings, _timestamps = self.extract_embeddings(waveform, scale_segments)

            if len(_embeddings) != len(_timestamps):
                raise ValueError(
                    "Mismatch of counts between embedding vectors and timestamps"
                )

            embeddings.append(_embeddings)
            timestamps.append(torch.tensor(_timestamps))

        return MultiscaleEmbeddingsAndTimestamps(
            base_scale_index=len(embeddings) - 1,
            embeddings=embeddings,
            timestamps=timestamps,
            multiscale_weights=self.multiscale_weights,
        )

    def get_audio_segments_from_scale(
        self,
        vad_outputs: List[dict],
        window: float,
        shift: float,
        min_subsegment_duration: float = 0.05,
    ) -> List[dict]:
        """
        Return a list of audio segments based on the VAD outputs and the scale window and shift length.

        Args:
            vad_outputs (List[dict]): List of segments with the following keys: "start", "end".
            window (float): Window length. Used to get subsegments.
            shift (float): Shift length. Used to get subsegments.
            min_subsegment_duration (float): Minimum duration of a subsegment in seconds.

        Returns:
            List[dict]: List of audio segments with the following keys: "offset", "duration".
        """
        scale_segment = []
        for segment in vad_outputs:
            segment_start, segment_end = (
                segment["start"] / 16000,
                segment["end"] / 16000,
            )
            subsegments = self.get_subsegments(
                segment_start, segment_end, window, shift
            )

            for subsegment in subsegments:
                start, duration = subsegment
                if duration > min_subsegment_duration:
                    scale_segment.append({"offset": start, "duration": duration})

        return scale_segment

    def extract_embeddings(
        self, waveform: torch.Tensor, scale_segments: List[dict]
    ) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        This method extracts speaker embeddings from the audio file based on the scale segments.

        Args:
            waveform (torch.Tensor): Waveform of the audio file.
            scale_segments (List[dict]): List of segments with the following keys: "offset", "duration".

        Returns:
            Tuple[torch.Tensor, List[List[float]]]: Tuple of embeddings and timestamps.
        """
        all_embs = torch.empty([0])

        dataset = AudioSegmentDataset(waveform, scale_segments)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=segmentation_collate_fn,
        )

        for batch in dataloader:
            _batch = [x.to(self.speaker_model.device) for x in batch]
            audio_signal, audio_signal_len = _batch

            with autocast():
                _, embeddings = self.speaker_model.forward(
                    input_signal=audio_signal, input_signal_length=audio_signal_len
                )
                embeddings = embeddings.view(-1, embeddings.shape[-1])
                all_embs = torch.cat((all_embs, embeddings.cpu().detach()), dim=0)

            del _batch, audio_signal, audio_signal_len, embeddings

        embeddings, time_stamps = [], []
        for i, segment in enumerate(scale_segments):
            if i == 0:
                embeddings = all_embs[i].view(1, -1)
            else:
                embeddings = torch.cat((embeddings, all_embs[i].view(1, -1)))

            time_stamps.append([segment["offset"], segment["duration"]])

        return embeddings, time_stamps

    @staticmethod
    def get_subsegments(
        segment_start: float, segment_end: float, window: float, shift: float
    ) -> List[List[float]]:
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
