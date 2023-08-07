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
"""Diarization Service for audio files."""

import math
from typing import Dict, List, NamedTuple, Tuple, Union

import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering
from torch.cuda.amp import autocast
from torch.utils.data import Dataset

from wordcab_transcribe.logging import time_and_tell
from wordcab_transcribe.services.vad_service import VadService


class MultiscaleEmbeddingsAndTimestamps(NamedTuple):
    """Multiscale embeddings and timestamps outputs of the SegmentationModule."""

    embeddings: torch.Tensor
    timestamps: torch.Tensor
    multiscale_segment_counts: torch.Tensor
    multiscale_weights: torch.Tensor


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


def segmentation_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function used by the dataloader of the SegmentationModule.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): List of audio segments and their lengths.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of the audio segments and their lengths.
    """
    _, audio_lengths = zip(*batch)

    if not audio_lengths[0]:
        return None, None

    fixed_length = int(max(audio_lengths))

    audio_signal, new_audio_lengths = [], []
    for sig, sig_len in batch:
        sig_len = sig_len.item()
        chunck_len = sig_len - fixed_length

        if chunck_len < 0:
            repeat = fixed_length // sig_len
            rem = fixed_length % sig_len
            sub = sig[-rem:] if rem > 0 else torch.tensor([])
            rep_sig = torch.cat(repeat * [sig])
            sig = torch.cat((rep_sig, sub))
        new_audio_lengths.append(torch.tensor(fixed_length))

        audio_signal.append(sig)

    audio_signal = torch.stack(audio_signal)
    audio_lengths = torch.stack(new_audio_lengths)

    return audio_signal, audio_lengths


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
        embeddings, timestamps, segment_indexes = [], [], []

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
            segment_indexes.append(_embeddings.shape[0])
            timestamps.append(torch.tensor(_timestamps))

        return MultiscaleEmbeddingsAndTimestamps(
            embeddings=torch.cat(embeddings, dim=0),
            timestamps=torch.cat(timestamps, dim=0),
            multiscale_segment_counts=torch.tensor(segment_indexes),
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


class ClusteringModule:
    """Clustering module for diariation."""

    def __init__(self, device: str, max_num_speakers: int = 8) -> None:
        """Initialize the clustering module."""
        self.params = dict(
            oracle_num_speakers=False,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=80,
            max_rp_threshold=0.25,
            sparse_search_volume=30,
            maj_vote_spk_count=False,
        )
        self.clustering_model = SpeakerClustering(parallelism=True, cuda=True)
        self.clustering_model.device = device

    def __call__(
        self, ms_emb_ts: MultiscaleEmbeddingsAndTimestamps
    ) -> List[Tuple[float, float, int]]:
        """
        Run the clustering module and return the speaker segments.

        Args:
            ms_emb_ts (MultiscaleEmbeddingsAndTimestamps): Embeddings and timestamps of the audio file in multiscale.
                The multiscale embeddings and timestamps are from the SegmentationModule.

        Returns:
            List[Tuple[float, float, int]]: List of segments with the following keys: "start", "end", "speaker".
        """
        base_scale_idx = ms_emb_ts.multiscale_segment_counts.shape[0] - 1
        cluster_labels = self.clustering_model.forward_infer(
            embeddings_in_scales=ms_emb_ts.embeddings,
            timestamps_in_scales=ms_emb_ts.timestamps,
            multiscale_segment_counts=ms_emb_ts.multiscale_segment_counts,
            multiscale_weights=ms_emb_ts.multiscale_weights,
            oracle_num_speakers=-1,
            max_num_speakers=self.params["max_num_speakers"],
            max_rp_threshold=self.params["max_rp_threshold"],
            sparse_search_volume=self.params["sparse_search_volume"],
        )

        del ms_emb_ts
        torch.cuda.empty_cache()

        timestamps = self.clustering_model.timestamps_in_scales[base_scale_idx]
        cluster_labels = cluster_labels.cpu().numpy()

        if len(cluster_labels) != timestamps.shape[0]:
            raise ValueError(
                "Mismatch of length between cluster_labels and timestamps."
            )

        clustering_labels = []
        for idx, label in enumerate(cluster_labels):
            start, end = timestamps[idx]
            clustering_labels.append((float(start), float(start + end), int(label)))

        return clustering_labels


class DiarizationModels(NamedTuple):
    """Diarization Models."""

    segmentation: SegmentationModule
    clustering: ClusteringModule
    device: str


class DiarizeService:
    """Diarize Service for audio files."""

    def __init__(
        self,
        device: str,
        device_index: List[int],
        window_lengths: List[float],
        shift_lengths: List[float],
        multiscale_weights: List[int],
        max_num_speakers: int = 8,
    ) -> None:
        """Initialize the Diarize Service.

        This service uses the NVIDIA NeMo diarization models.

        Args:
            device (str): Device to use for inference. Can be "cpu" or "cuda".
            device_index (Union[int, List[int]]): Index of the device to use for inference.
            window_lengths (List[float]): List of window lengths.
            shift_lengths (List[float]): List of shift lengths.
            multiscale_weights (List[int]): List of weights for each scale.
            max_num_speakers (int): Maximum number of speakers. Defaults to 8.
        """
        self.device = device
        self.models = {}

        # Multi-scale segmentation diarization
        self.max_num_speakers = max_num_speakers
        self.window_lengths = window_lengths
        self.shift_lengths = shift_lengths
        self.multiscale_weights = multiscale_weights

        self.scale_dict = {
            k: (w, s) for k, (w, s) in enumerate(zip(window_lengths, shift_lengths))
        }

        for idx in device_index:
            _device = f"cuda:{idx}" if self.device == "cuda" else "cpu"

            segmentation_module = SegmentationModule(_device, self.multiscale_weights)
            clustering_module = ClusteringModule(_device, self.max_num_speakers)

            self.models[idx] = DiarizationModels(
                segmentation=segmentation_module,
                clustering=clustering_module,
                device=_device,
            )

    @time_and_tell
    def __call__(
        self,
        filepath: Union[str, torch.Tensor],
        audio_duration: float,
        model_index: int,
        vad_service: VadService,
    ) -> List[dict]:
        """
        Run inference with the diarization model.

        Args:
            filepath (Union[str, torch.Tensor]): Path to the audio file or waveform.
            model_index (int): Index of the model to use for inference.
            vad_service (VadService): VAD service instance to use for Voice Activity Detection.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "speaker".
        """
        vad_outputs, _ = vad_service(filepath, False)

        if audio_duration > 3600:
            window_lengths = [3.0, 2.5, 2.0, 1.5, 1.0]
            self.scale_dict = {
                k: (w, s)
                for k, (w, s) in enumerate(zip(window_lengths, self.shift_lengths))
            }

        ms_emb_ts: MultiscaleEmbeddingsAndTimestamps = self.models[
            model_index
        ].segmentation(
            waveform=filepath,
            vad_outputs=vad_outputs,
            scale_dict=self.scale_dict,
        )

        clustering_outputs = self.models[model_index].clustering(ms_emb_ts)

        _outputs = self.get_contiguous_stamps(clustering_outputs)
        outputs = self.merge_stamps(_outputs)

        return outputs

    @staticmethod
    def get_contiguous_stamps(
        stamps: List[Tuple[float, float, int]]
    ) -> List[Tuple[float, float, int]]:
        """
        Return contiguous timestamps.

        Args:
            stamps (List[Tuple[float, float, int]]): List of segments containing the start time, end time and speaker.

        Returns:
            List[Tuple[float, float, int]]: List of segments containing the start time, end time and speaker.
        """
        contiguous_stamps = []
        for i in range(len(stamps) - 1):
            start, end, speaker = stamps[i]
            next_start, next_end, next_speaker = stamps[i + 1]

            if end > next_start:
                avg = (next_start + end) / 2.0
                stamps[i + 1] = (avg, next_end, next_speaker)
                contiguous_stamps.append((start, avg, speaker))
            else:
                contiguous_stamps.append((start, end, speaker))

        start, end, speaker = stamps[-1]
        contiguous_stamps.append((start, end, speaker))

        return contiguous_stamps

    @staticmethod
    def merge_stamps(
        stamps: List[Tuple[float, float, int]]
    ) -> List[Tuple[float, float, int]]:
        """
        Merge timestamps of the same speaker.

        Args:
            stamps (List[Tuple[float, float, int]]): List of segments containing the start time, end time and speaker.

        Returns:
            List[Tuple[float, float, int]]: List of segments containing the start time, end time and speaker.
        """
        overlap_stamps = []
        for i in range(len(stamps) - 1):
            start, end, speaker = stamps[i]
            next_start, next_end, next_speaker = stamps[i + 1]

            if end == next_start and speaker == next_speaker:
                stamps[i + 1] = (start, next_end, next_speaker)
            else:
                overlap_stamps.append((start, end, speaker))

        start, end, speaker = stamps[-1]
        overlap_stamps.append((start, end, speaker))

        return overlap_stamps
