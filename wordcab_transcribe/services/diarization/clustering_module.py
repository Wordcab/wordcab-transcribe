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
"""Clustering module for the diarization service."""

from typing import List, Tuple

import torch
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering

from wordcab_transcribe.services.diarization.models import MultiscaleEmbeddingsAndTimestamps


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
