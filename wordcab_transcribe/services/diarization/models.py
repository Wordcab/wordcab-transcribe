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
import math
import os
import random
import tarfile
import wget
import yaml
from pathlib import Path
from typing import List, NamedTuple, Tuple

import librosa
import torch
import torch.nn as nn

from wordcab_transcribe.services.diarization.utils import resolve_diarization_cache_dir


# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L59
def normalize_batch(x, seq_len, normalize_type):
    """Normalize batch of audio features."""
    x_mean = None
    x_std = None

    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            if x[i, :, : seq_len[i]].shape[1] == 1:
                raise ValueError(
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                    "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
                    "feature (ex. at least `hop_length` for Mel Spectrograms)."
                )
            x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += 1e-5

        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std

    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += 1e-5

        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)

        return (
            (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2),
            x_mean,
            x_std,
        )
    else:
        return x, x_mean, x_std


# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L117
def splice_frames(x, frame_splicing):
    """Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames
    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))

    return torch.cat(seq, dim=1)


class MultiscaleEmbeddingsAndTimestamps(NamedTuple):
    """Multiscale embeddings and timestamps outputs of the SegmentationModule."""

    base_scale_index: int
    embeddings: List[torch.Tensor]
    timestamps: List[torch.Tensor]
    multiscale_weights: List[float]


# This code is from NVIDIA NeMo toolkit package `FilterbankFeaturesTA` class:
# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L469
class MelSpectrogramPreprocessor(nn.Module):
    """Mel Spectrogram extraction."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        window: str = "hann",
        normalize: str = "per_feature",
        n_fft: int = None,
        preemph: float = 0.97,
        features: int = 64,
        lowfreq: int = 0,
        highfreq: int = None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=1e-5,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
    ):
        super().__init__()

        self.log_zero_guard_value = log_zero_guard_value

        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(window_stride * sample_rate)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = None

        torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "none": None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)
        self.stft = lambda x: torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window.to(dtype=torch.float),
            return_complex=True,
        )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = features
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=features, fmin=lowfreq, fmax=highfreq),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        self.log_zero_guard_type = log_zero_guard_type

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor((seq_len + pad_amount - self.n_fft) / self.hop_length) + 1
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x, seq_len, linear_spec=False):
        seq_len = self.get_seq_len(seq_len.float())

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect"
            ).squeeze(1)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # disable autocast to get full range of stft values
        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else 1e-5
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to

        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
    
        return x, seq_len


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
        cache_subfolder = hashlib.md5((self.location_in_the_cloud).encode("utf-8")).hexdigest()

        self.nemo_model_folder, self.nemo_model_file = self.download_model_if_required(
            url=self.location_in_the_cloud, cache_dir=self.cache_dir, subfolder=cache_subfolder,
        )

        self.model_files = Path.joinpath(self.nemo_model_folder, "model_files")
        if not self.model_files.exists():
            self.model_files.mkdir(parents=True, exist_ok=True)
            self.unpack_nemo_file(self.nemo_model_file, self.model_files)

        model_config_file_path = Path.joinpath(self.model_files, "model_config.yaml")
        with open(model_config_file_path, "r") as config_file:
            model_config = yaml.safe_load(config_file)

        self.preprocessor = MelSpectrogramPreprocessor(**model_config["preprocessor"])
        self.encoder = None
        self.decoder = None

    def forward() -> None:
        """Forward pass of the model."""
        pass


    @staticmethod
    def download_model_if_required(url, subfolder=None, cache_dir=None) -> Tuple[str, str]:
        """
        Helper function to download pre-trained weights from the cloud.

        Args:
            url: (str) URL to download from.
            cache_dir: (str) a cache directory where to download. If not present, this function will attempt to create it.
                If None (default), then it will be $HOME/.cache/torch/diarization
            subfolder: (str) subfolder within cache_dir. The file will be stored in cache_dir/subfolder. Subfolder can
                be empty

        Returns:
            Tuple[str, str]: cache_dir and filepath to the downloaded file.
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
                    return destination, destination_file

            except:
                continue

        raise ValueError("Not able to download the diarization model, please try again later.")
    
    @staticmethod
    def unpack_nemo_file(filepath: Path, out_folder: Path) -> str:
        """
        Unpacks a .nemo file into a folder.

        Args:
            filepath (Path): path to the .nemo file (can be compressed or uncompressed)
            out_folder (Path): path to the folder where the .nemo file should be unpacked

        Returns:
            path to the unpacked folder
        """
        try:
            tar = tarfile.open(filepath, "r:")  # try uncompressed
        except tarfile.ReadError:
            tar = tarfile.open(filepath, "r:gz")  # try compressed
        finally:
            tar.extractall(path=out_folder)
            tar.close()
