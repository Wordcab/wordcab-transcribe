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
# from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Union

# import librosa
# import soundfile as sf

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering

import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset

from wordcab_transcribe.logging import time_and_tell
from wordcab_transcribe.services.vad_service import VadService
# from wordcab_transcribe.utils import load_nemo_config


class NemoModel(NamedTuple):
    """NeMo Model."""

    model: NeuralDiarizer
    output_path: str
    tmp_audio_path: str
    device: str


class AudioSegmentDataset(Dataset):
    """Dataset for audio segments used by the SegmentationModule."""
    def __init__(self, waveform: torch.Tensor, segments: List[dict], sample_rate=16000) -> None:
        self.waveform = waveform
        self.segments = segments
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segment_info = self.segments[idx]
        offset_samples = int(segment_info["offset"] * self.sample_rate)
        duration_samples = int(segment_info["duration"] * self.sample_rate)

        segment = self.waveform[offset_samples:offset_samples + duration_samples]

        return segment, torch.tensor(segment.shape[0]).long()


def segmentation_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """"""
    _, audio_lengths = zip(*batch)

    has_audio = audio_lengths[0] is not None
    fixed_length = int(max(audio_lengths))

    audio_signal, new_audio_lengths = [], []
    for sig, sig_len in batch:
        if has_audio:
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

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(new_audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths


class SegmentationModule:
    """Segmentation module for diariation."""
    def __init__(self) -> None:
        """Initialize the segmentation module."""
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large", map_location=None
        )
        self.speaker_model.eval()

    def __call__(
        self,
        waveform: torch.Tensor,
        vad_outputs: List[dict],
        scale_dict: Dict[int, Tuple[float, float]],
    ) -> Dict[str, torch.Tensor]:
        """Run the segmentation module."""
        all_embeddings, all_timestamps, all_segment_indexes = [], [], []

        scales = scale_dict.items()
        for _, (window, shift) in scales:
            scale_segments = self._run_segmentation(vad_outputs, window, shift)

            _embeddings, _timestamps = self._extract_embeddings(waveform, scale_segments)

            if len(_embeddings) != len(_timestamps):
                raise ValueError("Mismatch of counts between embedding vectors and timestamps")

            all_embeddings.append(_embeddings)
            all_segment_indexes.append(_embeddings.shape[0])
            all_timestamps.append(torch.tensor(_timestamps))

        multiscale_embeddings_and_timestamps = {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "timestamps": torch.cat(all_timestamps, dim=0),
            "multiscale_segment_counts": torch.tensor(all_segment_indexes),
            "multiscale_weights": torch.tensor([1, 1, 1, 1, 1]).unsqueeze(0).float(),
        }

        return multiscale_embeddings_and_timestamps

    def _run_segmentation(
        self,
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

    def _extract_embeddings(self, waveform: torch.Tensor, scale_segments: List[dict]):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use. 
        """
        all_embs = torch.empty([0])

        dataset = AudioSegmentDataset(waveform, scale_segments)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, collate_fn=segmentation_collate_fn
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
    def __init__(self, max_num_speakers: int = 8) -> None:
        """Initialize the clustering module."""
        self.params = dict(
            oracle_num_speakers=False,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=80,
            max_rp_threshold=0.25,
            sparse_search_volume=30,
            maj_vote_spk_count=False,
        )
        self.clustering_model = SpeakerClustering(cuda=True)

    def __call__(self, multiscale_embeddings_and_timestamps: Dict[str, torch.Tensor]) -> List[Tuple[float, float, int]]:
        """Run the clustering module."""
        base_scale_idx = multiscale_embeddings_and_timestamps["multiscale_segment_counts"].shape[0] - 1
        cluster_labels = self.clustering_model.forward_infer(
            embeddings_in_scales=multiscale_embeddings_and_timestamps["embeddings"],
            timestamps_in_scales=multiscale_embeddings_and_timestamps["timestamps"],
            multiscale_segment_counts=multiscale_embeddings_and_timestamps["multiscale_segment_counts"],
            multiscale_weights=multiscale_embeddings_and_timestamps["multiscale_weights"],
            oracle_num_speakers=-1,
            max_num_speakers=self.params["max_num_speakers"],
            max_rp_threshold=self.params["max_rp_threshold"],
            sparse_search_volume=self.params["sparse_search_volume"],
        )

        del multiscale_embeddings_and_timestamps
        torch.cuda.empty_cache()

        timestamps = self.clustering_model.timestamps_in_scales[base_scale_idx]
        cluster_labels = cluster_labels.cpu().numpy()
        if len(cluster_labels) != timestamps.shape[0]:
            raise ValueError("Mismatch of length between cluster_labels and timestamps.")

        clustering_labels = []
        for idx, label in enumerate(cluster_labels):
            start, end = timestamps[idx]
            clustering_labels.append((float(start), float(start + end), int(label)))

        return clustering_labels


class DiarizeService:
    """Diarize Service for audio files."""

    def __init__(
        self,
        domain_type: str,
        storage_path: str,
        output_path: str,
        device: str,
        device_index: List[int],
        max_num_speakers: int = 8,
        window_lengths: List[float] = [1.5, 1.25, 1.0, 0.75, 0.5],
        shift_lengths: List[float] = [0.75, 0.625, 0.5, 0.375, 0.25],
        multiscale_weights: List[int] = [1, 1, 1, 1, 1],
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

        # Multi-scale segmentation diarization
        self.max_num_speakers = max_num_speakers
        self.window_lengths = window_lengths
        self.shift_lengths = shift_lengths
        self.multiscale_weights = multiscale_weights
        assert len(self.window_lengths) == len(self.shift_lengths) == len(self.multiscale_weights)
        self.scale_dict = {k: (w, s) for k, (w, s) in enumerate(zip(window_lengths, shift_lengths))}

        # for idx in device_index:
        #     _output_path = Path(output_path) / f"output_{idx}"

        #     _device = f"cuda:{idx}" if self.device == "cuda" else "cpu"
        #     cfg, tmp_audio_path = load_nemo_config(
        #         domain_type=domain_type,
        #         storage_path=storage_path,
        #         output_path=_output_path,
        #         device=_device,
        #         index=idx,
        #     )
        #     model = NeuralDiarizer(cfg=cfg).to(_device)
        #     self.models[idx] = NemoModel(
        #         model=model,
        #         output_path=_output_path,
        #         tmp_audio_path=tmp_audio_path,
        #         device=_device,
        #     )
        self.segmentation_module = SegmentationModule()
        self.clustering_module = ClusteringModule(self.max_num_speakers)

    @time_and_tell
    def __call__(
        self,
        filepath: Union[str, torch.Tensor],
        model_index: int,
        vad_service: VadService,
    ) -> List[dict]:
        """
        Run inference with the diarization model.

        Args:
            filepath (Union[str, torch.Tensor]): Path to the audio file or waveform.
            model_index (int): Index of the model to use for inference.

        Returns:
            List[dict]: List of segments with the following keys: "start", "end", "speaker".
        """
        # if isinstance(filepath, str):
        #     waveform, sample_rate = librosa.load(filepath, sr=None)
        # else:
        #     waveform = filepath
        #     sample_rate = 16000

        # sf.write(
        #     self.models[model_index].tmp_audio_path, waveform, sample_rate, "PCM_16"
        # )

        # self.models[model_index].model.diarize()

        # outputs = self._format_timestamps(self.models[model_index].output_path)

        vad_outputs, _ = vad_service(filepath, False)

        multiscale_embeddings_and_timestamps = self.segmentation_module(
            waveform=filepath,
            vad_outputs=vad_outputs,
            scale_dict=self.scale_dict,
        )

        outputs = self.clustering_module(multiscale_embeddings_and_timestamps)

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