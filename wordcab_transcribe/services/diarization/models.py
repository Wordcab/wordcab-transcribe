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
"""Models for the diarization service and modules."""

import hashlib
import os
import wget
from pathlib import Path
from typing import List, NamedTuple

import torch

from wordcab_transcribe.services.diarization.utils import resolve_diarization_cache_dir


class MultiscaleEmbeddingsAndTimestamps(NamedTuple):
    """Multiscale embeddings and timestamps outputs of the SegmentationModule."""

    base_scale_index: int
    embeddings: List[torch.Tensor]
    timestamps: List[torch.Tensor]
    multiscale_weights: List[float]


# Inspired from NVIDIA NeMo's EncDecSpeakerLabelModel
# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/label_models.py#L67
class EncDecSpeakerLabelModel:
    """The EncDecSpeakerLabelModel class encapsulates the encoder-decoder speaker label model."""

    def __init__(self, model_name: str = "titanet_large") -> None:
        """Initialize the EncDecSpeakerLabelModel class.

        The EncDecSpeakerLabelModel class encapsulates the encoder-decoder speaker label model.
        Only the "titanet_large" model is supported at the moment.
        For more models: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/label_models.py#L59

        Args:
            model_name (str, optional): The name of the model to use. Defaults to "titanet_large".

        Raises:
            ValueError: If the model name is not supported.
        """
        if model_name != "titanet_large":
            raise ValueError(
                f"Unknown model name: {model_name}. Only 'titanet_large' is supported at the moment."
            )

        self.model_name = model_name
        self.location_in_the_cloud = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/titanet_large/versions/v1/files/titanet-l.nemo"
        self.cache_dir = Path.joinpath(resolve_diarization_cache_dir(), "titanet-l")
        self.cache_subfolder = hashlib.md5((self.location_in_the_cloud).encode("utf-8")).hexdigest()

        nemo_model_file_in_cache = self.download_model_if_required(
            url=self.location_in_the_cloud, cache_dir=self.cache_dir, subfolder=self.cache_subfolder,
        )
        
        # instance = class_.restore_from(
        #     restore_path=nemo_model_file_in_cache,
        #     override_config_path=override_config_path,
        #     map_location=map_location,
        #     strict=strict,
        #     return_config=return_config,
        #     trainer=trainer,
        #     save_restore_connector=save_restore_connector,
        # )


    @staticmethod
    def download_model_if_required(url, subfolder=None, cache_dir=None) -> str:
        """
        Helper function to download pre-trained weights from the cloud.

        Args:
            url: (str) URL to download from.
            cache_dir: (str) a cache directory where to download. If not present, this function will attempt to create it.
                If None (default), then it will be $HOME/.cache/torch/NeMo
            subfolder: (str) subfolder within cache_dir. The file will be stored in cache_dir/subfolder. Subfolder can
                be empty

        Returns:
            If successful - absolute local path to the downloaded file
            else - empty string
        """
        destination = Path.joinpath(cache_dir, subfolder)

        if not destination.exists():
            destination.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        destination_file = Path.joinpath(destination, filename)

        if destination_file.exists():
            return str(destination_file)

        i = 0
        while i < 10:  # try 10 times
            i += 1

            try:
                wget.download(url, str(destination_file))
                if os.path.exists(destination_file):
                    return destination_file

            except:
                continue

        raise ValueError("Not able to download the diarization model, please try again later.")
